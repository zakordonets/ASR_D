from __future__ import annotations

from typing import Protocol

from asr_cli.core.config import DiarizationConfig
from asr_cli.core.models import PreparedMedia, SpeakerTurn


class DiarizationProvider(Protocol):
    provider_id: str

    def diarize(
        self, media: PreparedMedia, config: DiarizationConfig
    ) -> list[SpeakerTurn]:
        ...

