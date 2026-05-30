from __future__ import annotations

from pathlib import Path

import pytest

from asr_cli.core.errors import InputResolutionError
from asr_cli.io.inputs import discover_media_files, ensure_supported_media_file


def test_ensure_supported_media_file_accepts_wav(workspace_tmp: Path) -> None:
    path = workspace_tmp / 'test.wav'
    path.write_bytes(b'RIFF')
    assert ensure_supported_media_file(path) == path


def test_ensure_supported_media_file_accepts_mp3(workspace_tmp: Path) -> None:
    path = workspace_tmp / 'test.mp3'
    path.write_bytes(b'ID3')
    assert ensure_supported_media_file(path) == path


def test_ensure_supported_media_file_rejects_missing() -> None:
    with pytest.raises(InputResolutionError, match='does not exist'):
        ensure_supported_media_file(Path('/nonexistent/file.wav'))


def test_ensure_supported_media_file_rejects_directory(workspace_tmp: Path) -> None:
    with pytest.raises(InputResolutionError, match='Expected a file'):
        ensure_supported_media_file(workspace_tmp)


def test_ensure_supported_media_file_rejects_unsupported(workspace_tmp: Path) -> None:
    path = workspace_tmp / 'test.txt'
    path.write_text('hello')
    with pytest.raises(InputResolutionError, match='Unsupported'):
        ensure_supported_media_file(path)


def test_discover_media_files_finds_supported(workspace_tmp: Path) -> None:
    (workspace_tmp / 'a.wav').write_bytes(b'RIFF')
    (workspace_tmp / 'b.mp3').write_bytes(b'ID3')
    (workspace_tmp / 'c.txt').write_text('nope')
    (workspace_tmp / 'd.m4a').write_bytes(b'ftyp')

    files = discover_media_files(workspace_tmp, recursive=False)

    names = [f.name for f in files]
    assert 'a.wav' in names
    assert 'b.mp3' in names
    assert 'd.m4a' in names
    assert 'c.txt' not in names


def test_discover_media_files_sorted(workspace_tmp: Path) -> None:
    (workspace_tmp / 'c.wav').write_bytes(b'RIFF')
    (workspace_tmp / 'a.wav').write_bytes(b'RIFF')
    (workspace_tmp / 'b.wav').write_bytes(b'RIFF')

    files = discover_media_files(workspace_tmp, recursive=False)

    assert files == sorted(files)


def test_discover_media_files_recursive(workspace_tmp: Path) -> None:
    sub = workspace_tmp / 'sub'
    sub.mkdir()
    (workspace_tmp / 'root.wav').write_bytes(b'RIFF')
    (sub / 'nested.mp3').write_bytes(b'ID3')

    flat = discover_media_files(workspace_tmp, recursive=False)
    deep = discover_media_files(workspace_tmp, recursive=True)

    assert len(flat) == 1
    assert len(deep) == 2


def test_discover_media_files_rejects_missing() -> None:
    with pytest.raises(InputResolutionError, match='does not exist'):
        discover_media_files(Path('/nonexistent'), recursive=False)


def test_discover_media_files_rejects_file(workspace_tmp: Path) -> None:
    path = workspace_tmp / 'file.wav'
    path.write_bytes(b'RIFF')
    with pytest.raises(InputResolutionError, match='Expected a folder'):
        discover_media_files(path, recursive=False)


def test_discover_media_files_empty_folder(workspace_tmp: Path) -> None:
    files = discover_media_files(workspace_tmp, recursive=False)
    assert files == []
