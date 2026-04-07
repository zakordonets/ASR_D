from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from asr_cli.core.enums import OutputFormat


@dataclass(slots=True)
class ASRConfig:
    provider_id: str = 'gigaam'
    model_name: str = 'v3_e2e_rnnt'
    device: str = 'cpu'
    backend: str = 'onnx-cpu'
    longform_mode: str = 'auto'
    longform_threshold_seconds: float = 25.0
    longform_chunk_seconds: float = 300.0


@dataclass(slots=True)
class DiarizationConfig:
    provider_id: str = 'pyannote'
    enabled: bool = True
    hf_token: str | None = None
    model_name: str = 'pyannote/speaker-diarization-3.1'


@dataclass(slots=True)
class NormalizationConfig:
    provider_id: str = 'deepseek'
    enabled: bool = False
    apply_to_subtitles: bool = False
    model_name: str = 'deepseek-chat'
    api_key: str | None = None
    base_url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    reasoning_enabled: bool = False


@dataclass(slots=True)
class ExportConfig:
    output_dir: Path = Path('out')
    formats: list[OutputFormat] = field(
        default_factory=lambda: [OutputFormat.TXT, OutputFormat.JSON]
    )


@dataclass(slots=True)
class AppConfig:
    language: str = 'ru'
    recursive: bool = False
    continue_on_error: bool = True
    workers: int = 1
    asr: ASRConfig = field(default_factory=ASRConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def load_config_file(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open('rb') as handle:
        return tomllib.load(handle)


def load_dotenv(dotenv_path: Path | None = None) -> dict[str, str]:
    path = dotenv_path or Path('.env')
    if not path.exists() or not path.is_file():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            loaded[key] = value
            os.environ.setdefault(key, value)
    return loaded


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _load_from_env() -> dict:
    load_dotenv()
    return {
        'diarization': {
            'hf_token': os.getenv('HF_TOKEN'),
        },
        'normalization': {
            'provider_id': os.getenv('NORMALIZATION_PROVIDER') or os.getenv('LLM_PROVIDER'),
            'api_key': os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY'),
            'base_url': os.getenv('OPENAI_BASE_URL'),
            'headers': {},
            'reasoning_enabled': False,
        },
        'openrouter': {
            'api_key': os.getenv('OPENROUTER_API_KEY'),
            'base_url': os.getenv('OPENROUTER_BASE_URL') or 'https://openrouter.ai/api/v1',
            'model_name': os.getenv('OPENROUTER_MODEL') or 'xiaomi/mimo-v2-flash',
            'app_name': os.getenv('OPENROUTER_APP_NAME') or 'asr-cli',
            'headers': {
                key: value
                for key, value in {
                    'HTTP-Referer': os.getenv('OPENROUTER_HTTP_REFERER'),
                    'X-OpenRouter-Title': os.getenv('OPENROUTER_APP_NAME') or 'asr-cli',
                    'X-Title': os.getenv('OPENROUTER_APP_NAME') or 'asr-cli',
                }.items()
                if value
            },
            'reasoning_enabled': _env_bool('OPENROUTER_REASONING_ENABLED', False),
        },
    }


def build_app_config(
    *,
    output_dir: Path,
    formats: list[OutputFormat],
    recursive: bool = False,
    continue_on_error: bool = True,
    normalize: bool = False,
    apply_normalization_to_subtitles: bool = False,
    asr_provider: str = 'gigaam',
    diarizer_provider: str = 'pyannote',
    llm_provider: str | None = None,
    config_file: Path | None = None,
) -> AppConfig:
    defaults_asr = ASRConfig()
    defaults_diarization = DiarizationConfig()
    defaults_normalization = NormalizationConfig()
    file_config = load_config_file(config_file)
    env_config = _load_from_env()

    selected_llm_provider = (
        llm_provider
        or file_config.get('normalization', {}).get('provider_id')
        or env_config['normalization']['provider_id']
        or defaults_normalization.provider_id
    )

    if selected_llm_provider == 'openrouter':
        normalization_defaults = {
            'model_name': env_config['openrouter']['model_name'],
            'api_key': env_config['openrouter']['api_key'],
            'base_url': env_config['openrouter']['base_url'],
            'headers': env_config['openrouter']['headers'],
            'reasoning_enabled': env_config['openrouter']['reasoning_enabled'],
        }
    else:
        normalization_defaults = {
            'model_name': defaults_normalization.model_name,
            'api_key': env_config['normalization']['api_key'],
            'base_url': env_config['normalization']['base_url'],
            'headers': env_config['normalization']['headers'],
            'reasoning_enabled': env_config['normalization']['reasoning_enabled'],
        }

    asr_cfg = ASRConfig(
        provider_id=asr_provider,
        model_name=file_config.get('asr', {}).get('model_name', defaults_asr.model_name),
        device=file_config.get('asr', {}).get('device', defaults_asr.device),
        backend=file_config.get('asr', {}).get('backend', defaults_asr.backend),
        longform_mode=file_config.get('asr', {}).get(
            'longform_mode', defaults_asr.longform_mode
        ),
        longform_threshold_seconds=file_config.get('asr', {}).get(
            'longform_threshold_seconds', defaults_asr.longform_threshold_seconds
        ),
        longform_chunk_seconds=file_config.get('asr', {}).get(
            'longform_chunk_seconds', defaults_asr.longform_chunk_seconds
        ),
    )
    diarization_cfg = DiarizationConfig(
        provider_id=diarizer_provider,
        enabled=file_config.get('diarization', {}).get('enabled', defaults_diarization.enabled),
        hf_token=file_config.get('diarization', {}).get('hf_token')
        or env_config['diarization']['hf_token'],
        model_name=file_config.get('diarization', {}).get(
            'model_name', defaults_diarization.model_name
        ),
    )
    normalization_cfg = NormalizationConfig(
        provider_id=selected_llm_provider,
        enabled=normalize,
        apply_to_subtitles=apply_normalization_to_subtitles,
        model_name=file_config.get('normalization', {}).get(
            'model_name', normalization_defaults['model_name']
        ),
        api_key=file_config.get('normalization', {}).get('api_key')
        or normalization_defaults['api_key'],
        base_url=file_config.get('normalization', {}).get('base_url')
        or normalization_defaults['base_url'],
        headers=file_config.get('normalization', {}).get('headers')
        or normalization_defaults['headers'],
        reasoning_enabled=file_config.get('normalization', {}).get(
            'reasoning_enabled', normalization_defaults['reasoning_enabled']
        ),
    )
    return AppConfig(
        language=file_config.get('app', {}).get('language', 'ru'),
        recursive=recursive,
        continue_on_error=continue_on_error,
        workers=file_config.get('app', {}).get('workers', 1),
        asr=asr_cfg,
        diarization=diarization_cfg,
        normalization=normalization_cfg,
        export=ExportConfig(output_dir=output_dir, formats=formats),
    )