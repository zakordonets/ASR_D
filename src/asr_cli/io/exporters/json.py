from __future__ import annotations

import json
from pathlib import Path

from asr_cli.core.models import TranscriptDocument
from asr_cli.utils.json import to_jsonable


class JsonWriter:
    extension = "json"

    def write(self, document: TranscriptDocument, destination: Path) -> Path:
        payload = to_jsonable(document)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

