from __future__ import annotations

from dataclasses import replace

from asr_cli.core.config import NormalizationConfig
from asr_cli.core.models import TranscriptDocument
from asr_cli.core.progress import ProgressListener
from asr_cli.providers.openai_compatible.client import OpenAICompatibleClient


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
        for index, segment in enumerate(document.segments, start=1):
            normalized_text = self._client.normalize_text(
                model=config.model_name,
                language=language,
                text=segment.text,
            )
            normalized_segments.append(
                replace(
                    segment,
                    normalized_text=normalized_text,
                )
            )
            if progress_listener is not None:
                progress_listener.on_stage_progress(
                    'normalization',
                    completed=index,
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
                },
            },
        )
