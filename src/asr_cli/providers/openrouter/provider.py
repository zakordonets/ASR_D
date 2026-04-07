from __future__ import annotations

from dataclasses import replace

from asr_cli.core.config import NormalizationConfig
from asr_cli.core.models import TranscriptDocument
from asr_cli.providers.openai_compatible.client import OpenAICompatibleClient


class OpenRouterNormalizationProvider:
    provider_id = 'openrouter'

    def __init__(self, config: NormalizationConfig) -> None:
        base_url = config.base_url or 'https://openrouter.ai/api/v1'
        self._client = OpenAICompatibleClient(
            api_key=config.api_key,
            base_url=base_url,
            headers=config.headers,
        )

    def normalize(
        self, document: TranscriptDocument, config: NormalizationConfig, language: str
    ) -> TranscriptDocument:
        normalized_segments = []
        for segment in document.segments:
            normalized_text = self._client.normalize_text(
                model=config.model_name,
                language=language,
                text=segment.text,
                reasoning_enabled=config.reasoning_enabled,
            )
            normalized_segments.append(
                replace(
                    segment,
                    normalized_text=normalized_text,
                )
            )
        return replace(
            document,
            segments=normalized_segments,
            metadata={
                **document.metadata,
                'normalization': {
                    'provider': self.provider_id,
                    'model_name': config.model_name,
                    'reasoning_enabled': config.reasoning_enabled,
                },
            },
        )