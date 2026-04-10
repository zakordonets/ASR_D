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
- Runtime status output for `transcribe`, `combine`, and `batch`
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
- Internet access for downloading Python packages and model assets on first run
- Hugging Face access token for `pyannote` if diarization is enabled
- API key for the selected normalization backend if normalization is enabled

## Installation

### Windows PowerShell

#### 1. Install system dependencies

Install:
- Python `3.10+`
- `ffmpeg` with `ffprobe`

Install `ffmpeg` on Windows using one of these options:

```powershell
# winget
winget install Gyan.FFmpeg
```

```powershell
# Chocolatey
choco install ffmpeg -y
```

If you install `ffmpeg` manually from a zip archive, add the directory containing `ffmpeg.exe` and `ffprobe.exe` to `PATH`, then reopen PowerShell.

Verify both commands are available in `PATH`:

```powershell
python --version
ffmpeg -version
ffprobe -version
```

#### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### 3. Install the CLI and runtime dependencies

Fast path for a clean Windows machine:

```powershell
.\setup.ps1
```

Add `-IncludeDev` if you also want the test dependencies:

```powershell
.\setup.ps1 -IncludeDev
```

Manual installation:

Install the package itself:

```powershell
pip install .
```

Install optional dependencies for normalization and diarization:

```powershell
pip install .[llm,pyannote]
```

If you want the test dependencies too:

```powershell
pip install .[dev]
```

#### 4. Create `.env`

Start from the example file:

```powershell
Copy-Item .env.example .env
```

Then fill in the variables you actually need.

Minimum for the default full pipeline:
- `HF_TOKEN` for `pyannote` diarization

Additional variables only if you use normalization:
- `DEEPSEEK_API_KEY` for `deepseek`
- `OPENROUTER_API_KEY` for `openrouter`

#### 5. Accept the pyannote model terms

If you want diarization, accept the terms for `pyannote/speaker-diarization-3.1` on Hugging Face before the first run.

Without this, diarization initialization will fail even if `HF_TOKEN` is set.

#### 6. Verify the environment

Run:

```powershell
asr-cli doctor
asr-cli providers list
```

### Linux

#### 1. Install system dependencies

Install:
- Python `3.10+`
- `ffmpeg` with `ffprobe`

Install `ffmpeg` on Linux using your distribution package manager:

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y python3 python3-venv ffmpeg
```

```bash
# Fedora
sudo dnf install -y python3 ffmpeg
```

```bash
# Arch Linux
sudo pacman -S --needed python ffmpeg
```

If your distribution does not package `ffmpeg` in the default repositories, install it from the distro-supported multimedia repository and make sure both `ffmpeg` and `ffprobe` are in `PATH`.

Verify both commands are available in `PATH`:

```bash
python3 --version
ffmpeg -version
ffprobe -version
```

#### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install the CLI and runtime dependencies

Fast path on Linux:

```bash
chmod +x ./setup.sh
./setup.sh
```

Add `--include-dev` if you also want the test dependencies:

```bash
chmod +x ./setup.sh
./setup.sh --include-dev
```

Manual installation:

```bash
python -m pip install --upgrade pip
pip install .
pip install '.[llm,pyannote]'
```

If you want the test dependencies too:

```bash
pip install '.[dev]'
```

#### 4. Create `.env`

Start from the example file:

```bash
cp .env.example .env
```

Then fill in the variables you actually need.

Minimum for the default full pipeline:
- `HF_TOKEN` for `pyannote` diarization

Additional variables only if you use normalization:
- `DEEPSEEK_API_KEY` for `deepseek`
- `OPENROUTER_API_KEY` for `openrouter`

#### 5. Accept the pyannote model terms

If you want diarization, accept the terms for `pyannote/speaker-diarization-3.1` on Hugging Face before the first run.

Without this, diarization initialization will fail even if `HF_TOKEN` is set.

#### 6. Verify the environment

Run:

```bash
asr-cli doctor
asr-cli providers list
```

## Quick Start

The examples below use Windows-style paths. On Linux, replace `.\sample.wav` with `./sample.wav` and so on.

### First run on a clean machine

The console entry point installed by the package is `asr-cli`.

If you have configured `HF_TOKEN` and accepted the `pyannote` terms, you can run the default pipeline directly:

```powershell
asr-cli transcribe .\sample.wav --output-dir .\out
```

### First run without diarization

Diarization is enabled by default. If you do not want to configure Hugging Face yet, use `--no-diarization`:

```powershell
asr-cli transcribe .\sample.wav --output-dir .\out --no-diarization
```

Example status output:

```text
Starting transcription
Target: .\sample.wav
ASR: gigaam
Diarization: disabled
Normalization: disabled
Formats: txt, json
Output dir: out
OK: .\sample.wav
out\sample.txt
out\sample.json
```

Transcribe with all output formats:

```powershell
asr-cli transcribe .\sample.wav --output-dir .\out --txt --json --srt --vtt
```

Disable diarization explicitly:

```powershell
asr-cli transcribe .\sample.wav --output-dir .\out --json --no-diarization
```

Transcribe and normalize with the default normalization provider:

```powershell
asr-cli transcribe .\sample.wav --output-dir .\out --json --normalize
```

Override the normalization provider explicitly:

```powershell
asr-cli transcribe .\sample.wav --output-dir .\out --json --normalize --llm deepseek
asr-cli transcribe .\sample.wav --output-dir .\out --json --normalize --llm openrouter
```

Combine multiple files into one transcript:

```powershell
asr-cli combine .\part1.wav .\part2.wav --output-dir .\out --json
asr-cli combine .\part1.wav .\part2.wav --output-dir .\out --json --no-diarization
```

Process a folder:

```powershell
asr-cli batch .\media --output-dir .\out --json --continue-on-error
asr-cli batch .\media --output-dir .\out --json --continue-on-error --no-diarization
```

Example status output:

```text
Starting batch
Target: .\media
Items: 3
ASR: gigaam
Diarization: enabled (pyannote)
Normalization: disabled
Formats: json
Output dir: out
[OK] .\media\a.wav
[FAILED] .\media\broken.wav: ...
[OK] .\media\c.wav
Processed=3 Succeeded=2 Failed=1
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
- `OPENROUTER_APP_NAME`: value used by OpenRouter to populate the `App` field in request logs; default is `asr-cli`
- `OPENROUTER_HTTP_REFERER`: optional site URL for OpenRouter attribution
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

OpenRouter attribution:
- `OPENROUTER_APP_NAME` is sent as `X-OpenRouter-Title` and `X-Title`
- this is what OpenRouter uses to populate the `App` field in request logs
- `OPENROUTER_HTTP_REFERER` is optional but recommended
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
asr-cli transcribe .\sample.wav --output-dir .\out --normalize --config .\config.toml
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
