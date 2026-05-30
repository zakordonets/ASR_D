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


def test_normalize_text_returns_original_on_none_content(monkeypatch) -> None:
    def factory(*, api_key, base_url, default_headers=None):
        return StubOpenAI(content=None)

    monkeypatch.setattr('openai.OpenAI', factory)

    client = OpenAICompatibleClient(api_key='x', base_url='https://example.test')
    result = client.normalize_text(model='m', language='ru', text='original')

    assert result == 'original'


def test_normalize_text_rejects_markdown_heading(monkeypatch) -> None:
    def factory(*, api_key, base_url, default_headers=None):
        return StubOpenAI(content='# Heading\nSome text')

    monkeypatch.setattr('openai.OpenAI', factory)

    client = OpenAICompatibleClient(api_key='x', base_url='https://example.test')
    result = client.normalize_text(model='m', language='ru', text='original')

    assert result == 'original'


def test_normalize_text_rejects_blockquote(monkeypatch) -> None:
    def factory(*, api_key, base_url, default_headers=None):
        return StubOpenAI(content='> quoted text')

    monkeypatch.setattr('openai.OpenAI', factory)

    client = OpenAICompatibleClient(api_key='x', base_url='https://example.test')
    result = client.normalize_text(model='m', language='ru', text='original')

    assert result == 'original'


def test_normalize_text_rejects_bold_markers(monkeypatch) -> None:
    def factory(*, api_key, base_url, default_headers=None):
        return StubOpenAI(content='**bold** text')

    monkeypatch.setattr('openai.OpenAI', factory)

    client = OpenAICompatibleClient(api_key='x', base_url='https://example.test')
    result = client.normalize_text(model='m', language='ru', text='original')

    assert result == 'original'


def test_normalize_texts_empty_list(monkeypatch) -> None:
    monkeypatch.setattr('openai.OpenAI', lambda **kw: StubOpenAI(content=''))

    client = OpenAICompatibleClient(api_key='x', base_url='https://example.test')
    result = client.normalize_texts(model='m', language='ru', texts=[])

    assert result == []


def test_normalize_texts_single_item(monkeypatch) -> None:
    def factory(*, api_key, base_url, default_headers=None):
        return StubOpenAI(content='NORMALIZED')

    monkeypatch.setattr('openai.OpenAI', factory)

    client = OpenAICompatibleClient(api_key='x', base_url='https://example.test')
    result = client.normalize_texts(model='m', language='ru', texts=['test'])

    assert result == ['NORMALIZED']


def test_parse_numbered_response_valid() -> None:
    content = '[1] Hello.\n[2] World.\n[3] Foo.'
    result = OpenAICompatibleClient._parse_numbered_response(content, count=3)

    assert result == ['Hello.', 'World.', 'Foo.']


def test_parse_numbered_response_wrong_count() -> None:
    content = '[1] Hello.\n[2] World.'
    result = OpenAICompatibleClient._parse_numbered_response(content, count=3)

    assert result is None


def test_parse_numbered_response_out_of_range() -> None:
    content = '[1] Hello.\n[5] World.'
    result = OpenAICompatibleClient._parse_numbered_response(content, count=2)

    assert result is None


def test_parse_numbered_response_empty() -> None:
    result = OpenAICompatibleClient._parse_numbered_response('', count=1)

    assert result is None
