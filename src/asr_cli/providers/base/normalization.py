from __future__ import annotations

from typing import Protocol

from asr_cli.core.config import NormalizationConfig
from asr_cli.core.models import TranscriptDocument


class NormalizationProvider(Protocol):
    provider_id: str

    def normalize(
        self, document: TranscriptDocument, config: NormalizationConfig, language: str
    ) -> TranscriptDocument:
        ...
