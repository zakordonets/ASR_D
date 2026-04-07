from __future__ import annotations

from pathlib import Path

from asr_cli.core.models import TranscriptDocument
from asr_cli.utils.timecodes import format_srt_timestamp


class SrtWriter:
    extension = "srt"

    def write(self, document: TranscriptDocument, destination: Path) -> Path:
        blocks: list[str] = []
        for index, segment in enumerate(document.segments, start=1):
            speaker = f"[{segment.speaker}] " if segment.speaker else ""
            blocks.append(
                "\n".join(
                    [
                        str(index),
                        (
                            f"{format_srt_timestamp(segment.start)} --> "
                            f"{format_srt_timestamp(segment.end)}"
                        ),
                        f"{speaker}{segment.text}",
                    ]
                )
            )
        destination.write_text("\n\n".join(blocks).strip() + "\n", encoding="utf-8")
        return destination

