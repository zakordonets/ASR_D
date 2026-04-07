def _split_seconds(total_seconds: float) -> tuple[int, int, int, int]:
    milliseconds = int(round(total_seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return hours, minutes, seconds, milliseconds


def format_srt_timestamp(total_seconds: float) -> str:
    hours, minutes, seconds, milliseconds = _split_seconds(total_seconds)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def format_vtt_timestamp(total_seconds: float) -> str:
    hours, minutes, seconds, milliseconds = _split_seconds(total_seconds)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

