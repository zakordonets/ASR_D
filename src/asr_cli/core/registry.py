from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from asr_cli.core.errors import ProviderError


ProviderFactory = Callable[[Any], Any]


@dataclass
class ProviderRegistry:
    asr_factories: dict[str, ProviderFactory] = field(default_factory=dict)
    diarization_factories: dict[str, ProviderFactory] = field(default_factory=dict)
    normalization_factories: dict[str, ProviderFactory] = field(default_factory=dict)

    def register_asr(self, provider_id: str, factory: ProviderFactory) -> None:
        self.asr_factories[provider_id] = factory

    def register_diarization(self, provider_id: str, factory: ProviderFactory) -> None:
        self.diarization_factories[provider_id] = factory

    def register_normalization(
        self, provider_id: str, factory: ProviderFactory
    ) -> None:
        self.normalization_factories[provider_id] = factory

    def create_asr(self, provider_id: str, config: Any) -> Any:
        return self._create(self.asr_factories, provider_id, config)

    def create_diarization(self, provider_id: str, config: Any) -> Any:
        return self._create(self.diarization_factories, provider_id, config)

    def create_normalization(self, provider_id: str, config: Any) -> Any:
        return self._create(self.normalization_factories, provider_id, config)

    def _create(
        self, factories: dict[str, ProviderFactory], provider_id: str, config: Any
    ) -> Any:
        try:
            factory = factories[provider_id]
        except KeyError as exc:
            available = ', '.join(sorted(factories)) or '<none>'
            raise ProviderError(
                f"Unknown provider '{provider_id}'. Available: {available}"
            ) from exc
        return factory(config)


def build_default_registry() -> ProviderRegistry:
    from asr_cli.providers.deepseek.provider import DeepSeekNormalizationProvider
    from asr_cli.providers.gigaam.provider import GigaAMASRProvider
    from asr_cli.providers.openrouter.provider import OpenRouterNormalizationProvider
    from asr_cli.providers.pyannote.provider import PyannoteDiarizationProvider

    registry = ProviderRegistry()
    registry.register_asr('gigaam', GigaAMASRProvider)
    registry.register_diarization('pyannote', PyannoteDiarizationProvider)
    registry.register_normalization('deepseek', DeepSeekNormalizationProvider)
    registry.register_normalization('openrouter', OpenRouterNormalizationProvider)
    return registry