[CmdletBinding()]
param(
    [switch]$IncludeDev
)

$ErrorActionPreference = 'Stop'

function Test-CommandAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName
    )

    return $null -ne (Get-Command $CommandName -ErrorAction SilentlyContinue)
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if (-not (Test-CommandAvailable -CommandName 'python')) {
    throw 'Python is not available in PATH. Install Python 3.10+ and rerun setup.ps1.'
}

if (-not (Test-CommandAvailable -CommandName 'ffmpeg')) {
    throw 'ffmpeg is not available in PATH. Install ffmpeg and rerun setup.ps1.'
}

if (-not (Test-CommandAvailable -CommandName 'ffprobe')) {
    throw 'ffprobe is not available in PATH. Install ffmpeg/ffprobe and rerun setup.ps1.'
}

$venvDir = Join-Path $repoRoot '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host 'Creating virtual environment...'
    python -m venv $venvDir
}

Write-Host 'Upgrading pip...'
& $venvPython -m pip install --upgrade pip

Write-Host 'Installing CLI package...'
& $venvPython -m pip install .

Write-Host 'Installing optional runtime dependencies...'
& $venvPython -m pip install '.[llm,pyannote]'

if ($IncludeDev) {
    Write-Host 'Installing development dependencies...'
    & $venvPython -m pip install '.[dev]'
}

$envExamplePath = Join-Path $repoRoot '.env.example'
$envPath = Join-Path $repoRoot '.env'
if ((Test-Path -LiteralPath $envExamplePath) -and -not (Test-Path -LiteralPath $envPath)) {
    Copy-Item -LiteralPath $envExamplePath -Destination $envPath
    Write-Host 'Created .env from .env.example. Fill in tokens before using diarization or normalization.'
}

Write-Host 'Running environment checks...'
& $venvPython -m asr_cli doctor

Write-Host ''
Write-Host 'Setup completed.'
Write-Host 'Activate the environment with: .\.venv\Scripts\Activate.ps1'
Write-Host 'If you need diarization, set HF_TOKEN in .env and accept the pyannote model terms on Hugging Face.'
