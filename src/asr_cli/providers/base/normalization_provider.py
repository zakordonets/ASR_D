from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from asr_cli.core.config import NormalizationConfig
from asr_cli.core.models import TranscriptDocument
from asr_cli.core.progress import ProgressListener
from asr_cli.providers.openai_compatible.client import OpenAICompatibleClient

BATCH_SIZE = 10


class BaseNormalizationProvider(ABC):
    """Shared normalization logic for OpenAI-compatible providers."""

    provider_id: str
    default_base_url: str

    def __init__(self, config: NormalizationConfig) -> None:
        base_url = config.base_url or self.default_base_url
        self._client = OpenAICompatibleClient(
            api_key=config.api_key,
            base_url=base_url,
            headers=config.headers,
        )

    def normalize(
        self,
        document: TranscriptDocument,
        config: NormalizationConfig,
        language: str,
        progress_listener: ProgressListener | None = None,
    ) -> TranscriptDocument:
        normalized_segments = []
        total = len(document.segments)
        completed = 0

        for batch_start in range(0, total, BATCH_SIZE):
            batch = document.segments[batch_start : batch_start + BATCH_SIZE]
            texts = [seg.text for seg in batch]
            normalized_texts = self._client.normalize_texts(
                model=config.model_name,
                language=language,
                texts=texts,
                reasoning_enabled=config.reasoning_enabled,
            )
            assert len(normalized_texts) == len(batch), (
                f'Expected {len(batch)} normalized texts, got {len(normalized_texts)}'
            )
            for segment, norm_text in zip(batch, normalized_texts):
                normalized_segments.append(replace(segment, normalized_text=norm_text))
            completed += len(batch)
            if progress_listener is not None:
                progress_listener.on_stage_progress(
                    'normalization',
                    completed=completed,
                    total=total,
                )

        return replace(
            document,
            segments=normalized_segments,
            metadata={
                **document.metadata,
                'normalization': self._build_metadata(config),
            },
        )

    @abstractmethod
    def _build_metadata(self, config: NormalizationConfig) -> dict[str, object]:
        """Return provider-specific metadata for the normalized document."""
        ...
