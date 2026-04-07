from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import wave

from asr_cli.core.config import ASRConfig
from asr_cli.core.errors import ProviderDependencyError, ProviderError
from asr_cli.core.models import PreparedMedia, TranscriptDocument, TranscriptSegment, TranscriptWord
from asr_cli.core.progress import ProgressListener


@dataclass(slots=True)
class _AudioChunk:
    path: Path
    offset_seconds: float
    duration_seconds: float


class GigaAMASRProvider:
    provider_id = "gigaam"

    def __init__(self, config: ASRConfig) -> None:
        self.config = config
        try:
            import gigaam  # type: ignore
        except ImportError as exc:
            raise ProviderDependencyError(
                "GigaAM provider requires the optional 'gigaam' package."
            ) from exc
        self._gigaam = gigaam
        self._model = gigaam.load_model(config.model_name)

    def transcribe(
        self,
        media: PreparedMedia,
        config: ASRConfig,
        language: str,
        progress_listener: ProgressListener | None = None,
    ) -> TranscriptDocument:
        use_longform = (
            config.longform_mode == "always"
            or (
                config.longform_mode == "auto"
                and media.duration_seconds >= config.longform_threshold_seconds
            )
        )
        try:
            if use_longform:
                try:
                    segments, longform_metadata = self._transcribe_longform(
                        media,
                        config,
                        progress_listener=progress_listener,
                    )
                except Exception as exc:
                    if not self._should_fallback_to_shortform(exc):
                        raise
                    segments, longform_metadata = self._transcribe_chunked_shortform(
                        media,
                        config,
                        progress_listener=progress_listener,
                    )
                    longform_metadata['longform_fallback_reason'] = str(exc)
            else:
                result = self._model.transcribe(
                    str(media.prepared_path), word_timestamps=True
                )
                segments = self._parse_shortform_result(result)
                longform_metadata = {
                    'longform_chunked': False,
                    'longform_chunk_count': 1,
                    'longform_chunk_seconds': config.longform_chunk_seconds,
                    'longform_backend': 'shortform',
                }
        except Exception as exc:
            raise ProviderError(f"GigaAM transcription failed: {exc}") from exc

        return TranscriptDocument(
            title=media.original_path.stem,
            language=language,
            segments=segments,
            metadata={
                "provider": self.provider_id,
                "model_name": config.model_name,
                "device": config.device,
                "backend": config.backend,
                "longform_used": use_longform,
                **longform_metadata,
            },
        )

    def _transcribe_longform(
        self,
        media: PreparedMedia,
        config: ASRConfig,
        progress_listener: ProgressListener | None = None,
    ) -> tuple[list[TranscriptSegment], dict[str, float | int | bool | str]]:
        chunks = self._split_wav_chunks(
            media.prepared_path,
            chunk_duration_seconds=config.longform_chunk_seconds,
        )
        total_chunks = len(chunks)
        segments: list[TranscriptSegment] = []

        try:
            if progress_listener is not None:
                progress_listener.on_stage_progress(
                    'transcription',
                    completed=0,
                    total=max(total_chunks, 1),
                    path=media.original_path,
                )
            for index, chunk in enumerate(chunks, start=1):
                result = self._model.transcribe_longform(str(chunk.path))
                segments.extend(
                    self._parse_longform_result(result, offset_seconds=chunk.offset_seconds)
                )
                if progress_listener is not None:
                    progress_listener.on_stage_progress(
                        'transcription',
                        completed=index,
                        total=max(total_chunks, 1),
                        path=media.original_path,
                    )
        finally:
            for chunk in chunks:
                chunk.path.unlink(missing_ok=True)

        return segments, {
            'longform_chunked': total_chunks > 1,
            'longform_chunk_count': total_chunks,
            'longform_chunk_seconds': config.longform_chunk_seconds,
            'longform_backend': 'transcribe_longform',
        }

    def _transcribe_chunked_shortform(
        self,
        media: PreparedMedia,
        config: ASRConfig,
        progress_listener: ProgressListener | None = None,
    ) -> tuple[list[TranscriptSegment], dict[str, float | int | bool | str]]:
        chunk_seconds = min(config.longform_chunk_seconds, config.longform_threshold_seconds, 25.0)
        if chunk_seconds <= 0:
            chunk_seconds = 25.0
        chunks = self._split_wav_chunks(
            media.prepared_path,
            chunk_duration_seconds=chunk_seconds,
        )
        total_chunks = len(chunks)
        segments: list[TranscriptSegment] = []

        try:
            if progress_listener is not None:
                progress_listener.on_stage_progress(
                    'transcription',
                    completed=0,
                    total=max(total_chunks, 1),
                    path=media.original_path,
                )
            for index, chunk in enumerate(chunks, start=1):
                result = self._model.transcribe(str(chunk.path), word_timestamps=True)
                chunk_segments = self._parse_shortform_result(result)
                segments.extend(
                    self._offset_segments(chunk_segments, offset_seconds=chunk.offset_seconds)
                )
                if progress_listener is not None:
                    progress_listener.on_stage_progress(
                        'transcription',
                        completed=index,
                        total=max(total_chunks, 1),
                        path=media.original_path,
                    )
        finally:
            for chunk in chunks:
                chunk.path.unlink(missing_ok=True)

        return segments, {
            'longform_chunked': total_chunks > 1,
            'longform_chunk_count': total_chunks,
            'longform_chunk_seconds': chunk_seconds,
            'longform_backend': 'shortform_chunked_fallback',
        }

    def _split_wav_chunks(
        self,
        prepared_path: Path,
        *,
        chunk_duration_seconds: float,
    ) -> list[_AudioChunk]:
        if chunk_duration_seconds <= 0:
            raise ProviderError('longform_chunk_seconds must be greater than zero')

        chunks: list[_AudioChunk] = []
        with wave.open(str(prepared_path), 'rb') as reader:
            params = reader.getparams()
            frame_rate = reader.getframerate()
            total_frames = reader.getnframes()
            if total_frames <= 0:
                raise ProviderError('prepared audio is empty')
            frames_per_chunk = max(int(chunk_duration_seconds * frame_rate), 1)
            offset_frames = 0
            chunk_index = 0

            while offset_frames < total_frames:
                reader.setpos(offset_frames)
                frames_to_read = min(frames_per_chunk, total_frames - offset_frames)
                chunk_frames = reader.readframes(frames_to_read)
                chunk_path = prepared_path.parent / f'{prepared_path.stem}.chunk{chunk_index:04d}.wav'
                with wave.open(str(chunk_path), 'wb') as writer:
                    writer.setparams(params)
                    writer.writeframes(chunk_frames)
                chunks.append(
                    _AudioChunk(
                        path=chunk_path,
                        offset_seconds=offset_frames / frame_rate,
                        duration_seconds=frames_to_read / frame_rate,
                    )
                )
                offset_frames += frames_to_read
                chunk_index += 1

        return chunks

    def _parse_longform_result(
        self,
        result: object,
        *,
        offset_seconds: float,
    ) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        for segment in result:
            text = str(segment.text).strip()
            segments.append(
                TranscriptSegment(
                    start=float(segment.start) + offset_seconds,
                    end=float(segment.end) + offset_seconds,
                    text=text,
                    raw_text=text,
                )
            )
        return segments

    def _offset_segments(
        self,
        segments: list[TranscriptSegment],
        *,
        offset_seconds: float,
    ) -> list[TranscriptSegment]:
        shifted: list[TranscriptSegment] = []
        for segment in segments:
            shifted.append(
                replace(
                    segment,
                    start=segment.start + offset_seconds,
                    end=segment.end + offset_seconds,
                    words=[
                        replace(
                            word,
                            start=word.start + offset_seconds,
                            end=word.end + offset_seconds,
                        )
                        for word in segment.words
                    ],
                )
            )
        return shifted

    def _should_fallback_to_shortform(self, exc: Exception) -> bool:
        message = str(exc).lower()
        fallback_markers = (
            'audiodecoder',
            'torchcodec',
            'could not load libtorchcodec',
            'name \'audiodecoder\' is not defined',
        )
        return any(marker in message for marker in fallback_markers)

    def _parse_shortform_result(self, result: object) -> list[TranscriptSegment]:
        if isinstance(result, str):
            return [TranscriptSegment(start=0.0, end=0.0, text=result, raw_text=result)]
        words = getattr(result, "words", None)
        text = str(getattr(result, "text", "")).strip()
        if words:
            parsed_words = [
                TranscriptWord(
                    start=float(word.start),
                    end=float(word.end),
                    text=str(word.text),
                )
                for word in words
            ]
            start = parsed_words[0].start
            end = parsed_words[-1].end
            return [
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text or " ".join(word.text for word in parsed_words),
                    raw_text=text or " ".join(word.text for word in parsed_words),
                    words=parsed_words,
                )
            ]
        if text:
            return [TranscriptSegment(start=0.0, end=0.0, text=text, raw_text=text)]
        raise ProviderError("GigaAM returned an unsupported transcription result")