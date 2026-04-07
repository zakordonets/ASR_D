from __future__ import annotations

from pathlib import Path

from asr_cli.core.models import TranscriptDocument


class TxtWriter:
    extension = "txt"

    def write(self, document: TranscriptDocument, destination: Path) -> Path:
        lines: list[str] = []
        for segment in document.segments:
            speaker = f"[{segment.speaker}] " if segment.speaker else ""
            text = segment.normalized_text or segment.text
            lines.append(f"{speaker}{text}")
        destination.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return destination

