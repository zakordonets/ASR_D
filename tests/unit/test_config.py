from pathlib import Path

from asr_cli.core.config import build_app_config


def test_build_app_config_uses_openrouter_env(monkeypatch, workspace_tmp) -> None:
    env_file = workspace_tmp / '.env'
    env_file.write_text(
        '\n'.join(
            [
                'NORMALIZATION_PROVIDER=openrouter',
                'OPENROUTER_API_KEY=test-openrouter-key',
                'OPENROUTER_BASE_URL=https://openrouter.ai/api/v1',
                'OPENROUTER_MODEL=xiaomi/mimo-v2-flash',
                'OPENROUTER_HTTP_REFERER=https://example.test',
                'OPENROUTER_APP_NAME=asr-cli-tests',
                'OPENROUTER_REASONING_ENABLED=true',
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    for key in [
        'NORMALIZATION_PROVIDER',
        'LLM_PROVIDER',
        'OPENROUTER_API_KEY',
        'OPENROUTER_BASE_URL',
        'OPENROUTER_MODEL',
        'OPENROUTER_HTTP_REFERER',
        'OPENROUTER_APP_NAME',
        'OPENROUTER_REASONING_ENABLED',
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(workspace_tmp)

    config = build_app_config(
        output_dir=Path('out'),
        formats=[],
    )

    assert config.normalization.provider_id == 'openrouter'
    assert config.normalization.model_name == 'xiaomi/mimo-v2-flash'
    assert config.normalization.api_key == 'test-openrouter-key'
    assert config.normalization.base_url == 'https://openrouter.ai/api/v1'
    assert config.normalization.headers == {
        'HTTP-Referer': 'https://example.test',
        'X-Title': 'asr-cli-tests',
    }
    assert config.normalization.reasoning_enabled is True


def test_build_app_config_prefers_cli_llm_provider(monkeypatch, workspace_tmp) -> None:
    env_file = workspace_tmp / '.env'
    env_file.write_text(
        'NORMALIZATION_PROVIDER=openrouter\nOPENROUTER_API_KEY=test-key\n',
        encoding='utf-8',
    )
    monkeypatch.delenv('NORMALIZATION_PROVIDER', raising=False)
    monkeypatch.chdir(workspace_tmp)

    config = build_app_config(
        output_dir=Path('out'),
        formats=[],
        llm_provider='deepseek',
    )

    assert config.normalization.provider_id == 'deepseek'