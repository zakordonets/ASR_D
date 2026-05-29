from __future__ import annotations

from dataclasses import replace

from asr_cli.core.config import NormalizationConfig
from asr_cli.core.models import TranscriptDocument
from asr_cli.core.progress import ProgressListener
from asr_cli.providers.openai_compatible.client import OpenAICompatibleClient

BATCH_SIZE = 10


class DeepSeekNormalizationProvider:
    provider_id = "deepseek"

    def __init__(self, config: NormalizationConfig) -> None:
        base_url = config.base_url or "https://api.deepseek.com"
        self._client = OpenAICompatibleClient(
            api_key=config.api_key,
            base_url=base_url,
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
                "normalization": {
                    "provider": self.provider_id,
                    "model_name": config.model_name,
                    "batch_size": BATCH_SIZE,
                },
            },
        )
