from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from asr_cli.core.models import SourceOffset, SpeakerTurn, TranscriptDocument, TranscriptSegment


def combine_documents(
    documents: list[TranscriptDocument], source_paths: list[Path], title: str = "combined"
) -> TranscriptDocument:
    combined_segments: list[TranscriptSegment] = []
    combined_turns: list[SpeakerTurn] = []
    source_offsets: list[SourceOffset] = []
    offset = 0.0
    metadata_sources: list[dict[str, str | float]] = []

    for source_path, document in zip(source_paths, documents, strict=True):
        max_end = offset
        for segment in document.segments:
            shifted_words = [
                replace(word, start=word.start + offset, end=word.end + offset)
                for word in segment.words
            ]
            shifted_segment = replace(
                segment,
                start=segment.start + offset,
                end=segment.end + offset,
                words=shifted_words,
            )
            combined_segments.append(shifted_segment)
            max_end = max(max_end, shifted_segment.end)
        for turn in document.speaker_turns:
            combined_turns.append(
                replace(turn, start=turn.start + offset, end=turn.end + offset)
            )
            max_end = max(max_end, turn.end + offset)
        source_duration = max_end - offset
        source_offsets.append(
            SourceOffset(
                path=source_path,
                start_offset=offset,
                duration_seconds=source_duration,
            )
        )
        metadata_sources.append(
            {
                "path": str(source_path),
                "start_offset": offset,
                "duration_seconds": source_duration,
            }
        )
        offset = max_end

    return TranscriptDocument(
        title=title,
        language=documents[0].language if documents else "ru",
        segments=combined_segments,
        speaker_turns=combined_turns,
        source_offsets=source_offsets,
        metadata={"sources": metadata_sources},
        warnings=[],
    )
