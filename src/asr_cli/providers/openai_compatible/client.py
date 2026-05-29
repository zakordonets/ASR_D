from __future__ import annotations

import logging
import re

from asr_cli.core.errors import ProviderDependencyError

logger = logging.getLogger(__name__)


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

    def normalize_texts(
        self,
        *,
        model: str,
        language: str,
        texts: list[str],
        reasoning_enabled: bool | None = None,
    ) -> list[str]:
        """Normalize multiple segments in a single LLM request.

        Falls back to individual requests if batch parsing fails.
        """
        if not texts:
            return []
        if len(texts) == 1:
            return [
                self.normalize_text(
                    model=model,
                    language=language,
                    text=texts[0],
                    reasoning_enabled=reasoning_enabled,
                )
            ]

        numbered_input = '\n'.join(
            f'[{i + 1}] {text}' for i, text in enumerate(texts)
        )
        request_kwargs = {}
        if reasoning_enabled is not None:
            request_kwargs['extra_body'] = {'reasoning': {'enabled': reasoning_enabled}}

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            'You normalize transcript text. '
                            'You will receive numbered transcript segments. '
                            'Return each normalized segment on its own line with the same numbering. '
                            'Example input:\n'
                            '[1] hello world\n[2] foo bar\n'
                            'Example output:\n'
                            '[1] Hello, world.\n[2] Foo bar.\n'
                            'Return ONLY the numbered normalized segments. '
                            'Do not add explanations, comments, markdown, or notes.'
                        ),
                    },
                    {
                        'role': 'user',
                        'content': (
                            f'Language: {language}\n'
                            f'Normalize these {len(texts)} segments:\n'
                            f'{numbered_input}'
                        ),
                    },
                ],
                **request_kwargs,
            )
            content = response.choices[0].message.content or ''
            parsed = self._parse_numbered_response(content, count=len(texts))
            if parsed is not None:
                return [
                    self._sanitize_normalized_text(normalized, original_text=original)
                    for normalized, original in zip(parsed, texts)
                ]
        except Exception as exc:
            logger.warning('Batch normalization failed, falling back to individual: %s', exc)

        # Fallback: process individually
        return [
            self.normalize_text(
                model=model,
                language=language,
                text=text,
                reasoning_enabled=reasoning_enabled,
            )
            for text in texts
        ]

    @staticmethod
    def _parse_numbered_response(content: str, *, count: int) -> list[str] | None:
        """Parse '[1] text\n[2] text\n...' response into a list.

        Returns None if parsing fails (wrong count or no matches).
        """
        pattern = re.compile(r'^\[(\d+)\]\s*(.+)$', re.MULTILINE)
        matches = pattern.findall(content.strip())
        if len(matches) != count:
            return None
        # Verify sequential numbering
        results: list[str] = [''] * count
        for num_str, text in matches:
            idx = int(num_str) - 1
            if idx < 0 or idx >= count:
                return None
            results[idx] = text.strip()
        return results

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
