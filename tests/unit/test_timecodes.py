from asr_cli.utils.timecodes import format_duration, format_srt_timestamp, format_vtt_timestamp


def test_timecode_formatters() -> None:
    assert format_srt_timestamp(3661.275) == '01:01:01,275'
    assert format_vtt_timestamp(3661.275) == '01:01:01.275'


def test_timecode_zero() -> None:
    assert format_srt_timestamp(0.0) == '00:00:00,000'
    assert format_vtt_timestamp(0.0) == '00:00:00.000'


def test_timecode_sub_second() -> None:
    assert format_srt_timestamp(0.5) == '00:00:00,500'
    assert format_vtt_timestamp(0.123) == '00:00:00.123'


def test_timecode_large_values() -> None:
    assert format_srt_timestamp(90000.0) == '25:00:00,000'
    assert format_vtt_timestamp(90000.0) == '25:00:00.000'


def test_timecode_rounding() -> None:
    assert format_srt_timestamp(1.9999) == '00:00:02,000'


def test_format_duration_seconds() -> None:
    assert format_duration(0.0) == '0.00s'
    assert format_duration(5.5) == '5.50s'
    assert format_duration(59.99) == '59.99s'


def test_format_duration_minutes() -> None:
    assert format_duration(60.0) == '1m 0.0s'
    assert format_duration(90.5) == '1m 30.5s'
    assert format_duration(3599.0) == '59m 59.0s'


def test_format_duration_hours() -> None:
    assert format_duration(3600.0) == '1h 0m 0.0s'
    assert format_duration(3661.5) == '1h 1m 1.5s'
    assert format_duration(86400.0) == '24h 0m 0.0s'
