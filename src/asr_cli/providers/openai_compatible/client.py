from __future__ import annotations

from asr_cli.core.errors import ProviderDependencyError


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        headers: dict[str, str] | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ProviderDependencyError(
                "Normalization provider requires the optional 'openai' package."
            ) from exc
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers or None,
        )

    def normalize_text(
        self,
        *,
        model: str,
        language: str,
        text: str,
        reasoning_enabled: bool | None = None,
    ) -> str:
        request_kwargs = {}
        if reasoning_enabled is not None:
            request_kwargs['extra_body'] = {'reasoning': {'enabled': reasoning_enabled}}
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You normalize Russian transcripts. '
                        'Improve punctuation, casing, and readability without '
                        'inventing content.'
                    ),
                },
                {
                    'role': 'user',
                    'content': f'Language: {language}\nText:\n{text}',
                },
            ],
            **request_kwargs,
        )
        return response.choices[0].message.content or text