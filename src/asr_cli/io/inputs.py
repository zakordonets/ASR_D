from __future__ import annotations

from pathlib import Path

from asr_cli.core.errors import InputResolutionError

SUPPORTED_MEDIA_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
}


def ensure_supported_media_file(path: Path) -> Path:
    if not path.exists():
        raise InputResolutionError(f"Input does not exist: {path}")
    if not path.is_file():
        raise InputResolutionError(f"Expected a file, got: {path}")
    if path.suffix.lower() not in SUPPORTED_MEDIA_SUFFIXES:
        raise InputResolutionError(f"Unsupported media file type: {path.suffix}")
    return path


def discover_media_files(folder: Path, recursive: bool) -> list[Path]:
    if not folder.exists():
        raise InputResolutionError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise InputResolutionError(f"Expected a folder, got: {folder}")
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_SUFFIXES
    ]
    return sorted(files)

