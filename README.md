# ASR CLI

Python CLI utility for transcription, diarization, subtitle export, and optional LLM-based transcript normalization.

The project is built around a provider-based architecture:
- default ASR: `GigaAM`
- default diarization: `pyannote`
- normalization providers: `DeepSeek` and `OpenRouter`
- default OpenRouter model: `xiaomi/mimo-v2-flash`

## Features

- Transcription for audio and video files
- Speaker diarization with `pyannote`
- Optional text normalization via LLM
- Output formats: `txt`, `json`, `srt`, `vtt`
- Processing modes:
  - single file
  - combined files into one transcript
  - folder batch processing
- Batch-safe execution: one bad file does not stop the whole folder run
- Provider selection via CLI, config file, or `.env`
- Windows-first implementation with CPU-first defaults
- Automated tests for pipeline, exporters, config resolution, and provider selection

## Status

Implemented and verified:
- real `GigaAM + pyannote` transcription flow
- real `DeepSeek` normalization flow
- real `OpenRouter` normalization flow
- `combine` mode
- `batch` mode with continue-on-error behavior

Still recommended for further validation before production-heavy use:
- longform transcription on multi-hour inputs
- large folder workloads
- provider-specific performance tuning

## Requirements

- Python `>= 3.10`
- `ffmpeg` and `ffprobe` available in `PATH`
- Hugging Face access token for `pyannote`
- API key for the selected normalization backend if normalization is enabled

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

If you want to use the live normalization providers and diarization stack, install the optional runtime dependencies in the same environment.

## Quick Start

Run environment diagnostics:

```powershell
$env:PYTHONPATH='src'
python -m asr_cli doctor
```

List available providers:

```powershell
$env:PYTHONPATH='src'
python -m asr_cli providers list
```

Transcribe a file:

```powershell
$env:PYTHONPATH='src'
python -m asr_cli transcribe .\sample.wav --output-dir .\out
```

Transcribe with all output formats:

```powershell
$env:PYTHONPATH='src'
python -m asr_cli transcribe .\sample.wav --output-dir .\out --txt --json --srt --vtt
```

Transcribe and normalize with the default normalization provider:

```powershell
$env:PYTHONPATH='src'
python -m asr_cli transcribe .\sample.wav --output-dir .\out --json --normalize
```

Override the normalization provider explicitly:

```powershell
$env:PYTHONPATH='src'
python -m asr_cli transcribe .\sample.wav --output-dir .\out --json --normalize --llm deepseek
$env:PYTHONPATH='src'
python -m asr_cli transcribe .\sample.wav --output-dir .\out --json --normalize --llm openrouter
```

Combine multiple files into one transcript:

```powershell
$env:PYTHONPATH='src'
python -m asr_cli combine .\part1.wav .\part2.wav --output-dir .\out --json
```

Process a folder:

```powershell
$env:PYTHONPATH='src'
python -m asr_cli batch .\media --output-dir .\out --json --continue-on-error
```

## Configuration

Configuration sources are applied in this priority order:

1. CLI flags
2. config file
3. `.env`
4. built-in defaults

### Normalization Provider Selection

Normalization provider selection is resolved in this order:

1. `--llm`
2. `[normalization].provider_id` in config file
3. `NORMALIZATION_PROVIDER` or `LLM_PROVIDER` in `.env`
4. fallback to `deepseek`

Supported normalization providers right now:
- `deepseek`
- `openrouter`

### `.env`

Start from [`.env.example`](./.env.example).

Important variables:
- `HF_TOKEN`: required for `pyannote` diarization
- `NORMALIZATION_PROVIDER`: default normalization provider, e.g. `deepseek` or `openrouter`
- `DEEPSEEK_API_KEY`: required when using `deepseek`
- `OPENROUTER_API_KEY`: required when using `openrouter`
- `OPENROUTER_MODEL`: default OpenRouter model, e.g. `xiaomi/mimo-v2-flash`
- `OPENROUTER_REASONING_ENABLED`: usually keep `false` for normalization

Example:

```env
HF_TOKEN=...
NORMALIZATION_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=xiaomi/mimo-v2-flash
OPENROUTER_APP_NAME=asr-cli
OPENROUTER_REASONING_ENABLED=false
```

### Config File Example

```toml
[diarization]
enabled = true
model_name = "pyannote/speaker-diarization-3.1"

[normalization]
provider_id = "openrouter"
model_name = "xiaomi/mimo-v2-flash"
reasoning_enabled = false
```

Use it like this:

```powershell
$env:PYTHONPATH='src'
python -m asr_cli transcribe .\sample.wav --output-dir .\out --normalize --config .\config.toml
```

## Output Formats

- `txt`: readable transcript, using normalized text when available
- `json`: full structured output with metadata, timings, speakers, and normalization details
- `srt`: subtitle export
- `vtt`: WebVTT subtitle export

## Batch Behavior

- `batch` mode is failure-isolated by design
- a bad file becomes a failed job entry in `batch_report.json`
- the rest of the folder continues when `--continue-on-error` is enabled
- `combine` mode is fail-fast: if one input fails, the whole combined run fails

## Testing

Run the automated test suite:

```powershell
.\.venv\Scripts\python -m pytest -q
```

The repository also supports live smoke testing against the installed providers, but those tests depend on local credentials and provider access.

## Notes

- Current implementation is Windows-first, but the structure is intended to stay cross-platform.
- CPU-first execution is the default path.
- `pyannote` access requires accepting model terms on Hugging Face.
- On Windows, Hugging Face cache symlink warnings may appear; they are not fatal.