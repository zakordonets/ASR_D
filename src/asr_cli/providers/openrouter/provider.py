from __future__ import annotations

from asr_cli.core.config import NormalizationConfig
from asr_cli.providers.base.normalization_provider import BaseNormalizationProvider


class OpenRouterNormalizationProvider(BaseNormalizationProvider):
    provider_id = 'openrouter'
    default_base_url = 'https://openrouter.ai/api/v1'

    def _build_metadata(self, config: NormalizationConfig) -> dict[str, object]:
        return {
            'provider': self.provider_id,
            'model_name': config.model_name,
            'reasoning_enabled': config.reasoning_enabled,
        }
