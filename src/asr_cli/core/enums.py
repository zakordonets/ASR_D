from enum import Enum


class OutputFormat(str, Enum):
    TXT = "txt"
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"


class JobStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"

