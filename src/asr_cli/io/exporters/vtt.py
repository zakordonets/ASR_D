from __future__ import annotations

from pathlib import Path

from asr_cli.core.models import TranscriptDocument
from asr_cli.utils.timecodes import format_vtt_timestamp


class VttWriter:
    extension = "vtt"

    def write(
        self,
        document: TranscriptDocument,
        destination: Path,
        *,
        use_normalized: bool = False,
    ) -> Path:
        blocks = ["WEBVTT"]
        for segment in document.segments:
            speaker = f"[{segment.speaker}] " if segment.speaker else ""
            text = (
                segment.normalized_text
                if use_normalized and segment.normalized_text
                else segment.text
            )
            blocks.append(
                "\n".join(
                    [
                        (
                            f"{format_vtt_timestamp(segment.start)} --> "
                            f"{format_vtt_timestamp(segment.end)}"
                        ),
                        f"{speaker}{text}",
                    ]
                )
            )
        destination.write_text("\n\n".join(blocks).strip() + "\n", encoding="utf-8")
        return destination
