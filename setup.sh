#!/usr/bin/env bash
set -euo pipefail

include_dev=0
if [[ "${1-}" == "--include-dev" ]]; then
  include_dev=1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not available in PATH. Install Python 3.10+ and rerun setup.sh." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is not available in PATH. Install ffmpeg and rerun setup.sh." >&2
  exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe is not available in PATH. Install ffmpeg/ffprobe and rerun setup.sh." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

venv_python=".venv/bin/python"

echo "Upgrading pip..."
"$venv_python" -m pip install --upgrade pip

echo "Installing CLI package..."
"$venv_python" -m pip install .

echo "Installing optional runtime dependencies..."
"$venv_python" -m pip install '.[llm,pyannote]'

if [[ "$include_dev" -eq 1 ]]; then
  echo "Installing development dependencies..."
  "$venv_python" -m pip install '.[dev]'
fi

if [[ -f ".env.example" && ! -f ".env" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Fill in tokens before using diarization or normalization."
fi

echo "Running environment checks..."
"$venv_python" -m asr_cli doctor

echo
echo "Setup completed."
echo "Activate the environment with: source .venv/bin/activate"
echo "If you need diarization, set HF_TOKEN in .env and accept the pyannote model terms on Hugging Face."
