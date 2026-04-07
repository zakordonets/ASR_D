from asr_cli.utils.timecodes import format_srt_timestamp, format_vtt_timestamp


def test_timecode_formatters() -> None:
    assert format_srt_timestamp(3661.275) == "01:01:01,275"
    assert format_vtt_timestamp(3661.275) == "01:01:01.275"
