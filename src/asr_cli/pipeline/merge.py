from __future__ import annotations

from dataclasses import replace

from asr_cli.core.models import SpeakerTurn, TranscriptDocument, TranscriptSegment


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(
    document: TranscriptDocument, speaker_turns: list[SpeakerTurn]
) -> TranscriptDocument:
    if not speaker_turns:
        return document
    updated_segments = []
    for segment in document.segments:
        speaker = _best_speaker(segment, speaker_turns)
        updated_segments.append(
            replace(
                segment,
                speaker=speaker,
                words=[replace(word, speaker=speaker) for word in segment.words],
            )
        )
    return replace(document, segments=updated_segments, speaker_turns=speaker_turns)


def _best_speaker(
    segment: TranscriptSegment, speaker_turns: list[SpeakerTurn]
) -> str | None:
    winner: str | None = None
    winner_overlap = 0.0
    for turn in speaker_turns:
        overlap = _overlap(segment.start, segment.end, turn.start, turn.end)
        if overlap > winner_overlap:
            winner = turn.speaker
            winner_overlap = overlap
    return winner

