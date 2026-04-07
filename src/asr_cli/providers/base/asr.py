from __future__ import annotations

from typing import Protocol

from asr_cli.core.config import ASRConfig
from asr_cli.core.models import PreparedMedia, TranscriptDocument
from asr_cli.core.progress import ProgressListener


class ASRProvider(Protocol):
    provider_id: str

    def transcribe(
        self,
        media: PreparedMedia,
        config: ASRConfig,
        language: str,
        progress_listener: ProgressListener | None = None,
    ) -> TranscriptDocument:
        ...