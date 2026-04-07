from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from asr_cli.core.errors import PreprocessingError
from asr_cli.core.models import PreparedMedia


class FfmpegPreprocessor:
    def __init__(self, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin

    def is_available(self) -> bool:
        return shutil.which(self.ffmpeg_bin) is not None and shutil.which(
            self.ffprobe_bin
        ) is not None

    def inspect(self, path: Path) -> dict:
        cmd = [
            self.ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise PreprocessingError("ffprobe is not available in PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise PreprocessingError(
                f"ffprobe failed for {path}: {exc.stderr.strip()}"
            ) from exc
        return json.loads(completed.stdout)

    def prepare(self, path: Path, workspace: Path) -> PreparedMedia:
        workspace.mkdir(parents=True, exist_ok=True)
        probe = self.inspect(path)
        duration = float(probe.get("format", {}).get("duration", 0.0))
        normalized_path = workspace / f"{path.stem}.wav"
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            str(normalized_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise PreprocessingError("ffmpeg is not available in PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise PreprocessingError(
                f"ffmpeg failed for {path}: {exc.stderr.strip()}"
            ) from exc
        source_kind = "video" if any(
            stream.get("codec_type") == "video" for stream in probe.get("streams", [])
        ) else "audio"
        return PreparedMedia(
            original_path=path,
            prepared_path=normalized_path,
            duration_seconds=duration,
            source_kind=source_kind,
            sample_rate=16000,
            metadata={"probe": probe},
        )
