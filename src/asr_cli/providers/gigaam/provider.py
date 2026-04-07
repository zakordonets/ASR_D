from __future__ import annotations

from asr_cli.core.config import ASRConfig
from asr_cli.core.errors import ProviderDependencyError, ProviderError
from asr_cli.core.models import PreparedMedia, TranscriptDocument, TranscriptSegment, TranscriptWord


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
        self, media: PreparedMedia, config: ASRConfig, language: str
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
                result = self._model.transcribe_longform(str(media.prepared_path))
                segments = [
                    TranscriptSegment(
                        start=float(segment.start),
                        end=float(segment.end),
                        text=str(segment.text).strip(),
                        raw_text=str(segment.text).strip(),
                    )
                    for segment in result
                ]
            else:
                result = self._model.transcribe(
                    str(media.prepared_path), word_timestamps=True
                )
                segments = self._parse_shortform_result(result)
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
            },
        )

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
