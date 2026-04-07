from types import SimpleNamespace

from asr_cli.providers.openai_compatible.client import OpenAICompatibleClient


class StubCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class StubOpenAI:
    def __init__(self, *, content: str) -> None:
        self.completions = StubCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def test_normalization_prompt_requests_output_only(monkeypatch) -> None:
    holder = {}

    def factory(*, api_key, base_url, default_headers=None):
        client = StubOpenAI(content='Нормализованный текст.')
        holder['client'] = client
        return client

    monkeypatch.setattr('openai.OpenAI', factory)

    client = OpenAICompatibleClient(api_key='x', base_url='https://example.test')
    result = client.normalize_text(
        model='demo-model',
        language='ru',
        text='сырое предложение',
    )

    assert result == 'Нормализованный текст.'
    messages = holder['client'].completions.last_kwargs['messages']
    assert 'Return only the final normalized transcript text' in messages[0]['content']
    assert 'Output only the normalized text.' in messages[1]['content']


def test_normalization_client_rejects_meta_response(monkeypatch) -> None:
    def factory(*, api_key, base_url, default_headers=None):
        return StubOpenAI(
            content='Вот исправленный вариант текста:\n\n**Что было изменено:**\n1. ...'
        )

    monkeypatch.setattr('openai.OpenAI', factory)

    client = OpenAICompatibleClient(api_key='x', base_url='https://example.test')
    original = 'исходный текст'
    result = client.normalize_text(
        model='demo-model',
        language='ru',
        text=original,
    )

    assert result == original
