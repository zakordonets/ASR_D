from __future__ import annotations

from asr_cli.core.errors import ProviderDependencyError


META_RESPONSE_MARKERS = (
    'что было изменено',
    'что было исправлено',
    'альтернативный вариант',
    'исходный вариант',
    'вот исправленный вариант',
    'вот отредактированный вариант',
    'улучшенный вариант',
)


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
                        'You normalize transcript text. '
                        'Return only the final normalized transcript text and nothing else. '
                        'Do not add explanations, comments, bullet lists, markdown, '
                        'headings, quotes, alternative variants, or notes about changes. '
                        'Do not address the user. '
                        'Preserve the original meaning. '
                        'Fix punctuation, casing, spacing, and obvious ASR artifacts only.'
                    ),
                },
                {
                    'role': 'user',
                    'content': (
                        f'Language: {language}\n'
                        'Task: normalize the transcript segment below.\n'
                        'Rules:\n'
                        '- Output only the normalized text.\n'
                        '- No explanations.\n'
                        '- No markdown.\n'
                        '- No lists.\n'
                        '- No alternative versions.\n'
                        '- If the text is already fine, return it as-is.\n'
                        f'Transcript:\n{text}'
                    ),
                },
            ],
            **request_kwargs,
        )
        content = response.choices[0].message.content or text
        return self._sanitize_normalized_text(content, original_text=text)

    def _sanitize_normalized_text(self, content: str, *, original_text: str) -> str:
        normalized = content.strip()
        lowered = normalized.lower()

        if any(marker in lowered for marker in META_RESPONSE_MARKERS):
            return original_text

        if (
            normalized.startswith('#')
            or normalized.startswith('>')
            or '**' in normalized
            or '```' in normalized
        ):
            return original_text

        return normalized or original_text
