from asr_cli.core.config import NormalizationConfig
from asr_cli.core.models import TranscriptDocument, TranscriptSegment
from asr_cli.providers.openrouter.provider import OpenRouterNormalizationProvider


class StubClient:
    def __init__(self, *, api_key, base_url, headers) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.headers = headers
        self.calls = []

    def normalize_text(self, *, model, language, text, reasoning_enabled=None) -> str:
        self.calls.append(
            {
                'model': model,
                'language': language,
                'text': text,
                'reasoning_enabled': reasoning_enabled,
            }
        )
        return text.upper()


def test_openrouter_provider_uses_openrouter_client(monkeypatch) -> None:
    holder = {}

    def factory(*, api_key, base_url, headers):
        client = StubClient(api_key=api_key, base_url=base_url, headers=headers)
        holder['client'] = client
        return client

    monkeypatch.setattr(
        'asr_cli.providers.openrouter.provider.OpenAICompatibleClient',
        factory,
    )

    config = NormalizationConfig(
        provider_id='openrouter',
        enabled=True,
        model_name='xiaomi/mimo-v2-flash',
        api_key='or-key',
        base_url='https://openrouter.ai/api/v1',
        headers={
            'HTTP-Referer': 'https://example.test',
            'X-Title': 'asr-cli-tests',
        },
        reasoning_enabled=False,
    )
    document = TranscriptDocument(
        title='demo',
        language='ru',
        segments=[TranscriptSegment(start=0.0, end=1.0, text='тест', raw_text='тест')],
    )

    provider = OpenRouterNormalizationProvider(config)
    result = provider.normalize(document, config, 'ru')

    assert holder['client'].api_key == 'or-key'
    assert holder['client'].base_url == 'https://openrouter.ai/api/v1'
    assert holder['client'].headers['X-Title'] == 'asr-cli-tests'
    assert holder['client'].calls[0]['model'] == 'xiaomi/mimo-v2-flash'
    assert holder['client'].calls[0]['reasoning_enabled'] is False
    assert result.segments[0].normalized_text == 'ТЕСТ'
    assert result.metadata['normalization']['provider'] == 'openrouter'