from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asr_cli.core.enums import JobStatus


@dataclass(slots=True)
class PreparedMedia:
    original_path: Path
    prepared_path: Path
    duration_seconds: float
    source_kind: str
    sample_rate: int = 16000
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranscriptWord:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[TranscriptWord] = field(default_factory=list)
    raw_text: str | None = None
    normalized_text: str | None = None


@dataclass(slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(slots=True)
class SourceOffset:
    path: Path
    start_offset: float
    duration_seconds: float


@dataclass(slots=True)
class TranscriptDocument:
    title: str
    language: str
    segments: list[TranscriptSegment]
    speaker_turns: list[SpeakerTurn] = field(default_factory=list)
    source_offsets: list[SourceOffset] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def raw_text(self) -> str:
        return "\n".join(segment.raw_text or segment.text for segment in self.segments)

    @property
    def normalized_text(self) -> str:
        return "\n".join(segment.normalized_text or segment.text for segment in self.segments)


@dataclass(slots=True)
class JobResult:
    status: JobStatus
    input_path: Path | None
    document: TranscriptDocument | None = None
    error: str | None = None
    output_files: list[Path] = field(default_factory=list)


@dataclass(slots=True)
class BatchResult:
    total: int
    succeeded: int
    failed: int
    skipped: int
    results: list[JobResult]

