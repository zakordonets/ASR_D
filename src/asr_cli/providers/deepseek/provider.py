from __future__ import annotations

from asr_cli.core.config import NormalizationConfig
from asr_cli.providers.base.normalization_provider import BaseNormalizationProvider


class DeepSeekNormalizationProvider(BaseNormalizationProvider):
    provider_id = 'deepseek'
    default_base_url = 'https://api.deepseek.com'

    def _build_metadata(self, config: NormalizationConfig) -> dict[str, object]:
        return {
            'provider': self.provider_id,
            'model_name': config.model_name,
        }
