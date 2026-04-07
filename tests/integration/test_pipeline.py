import json

from typer.testing import CliRunner

from asr_cli.cli.main import app
from asr_cli.core.config import build_app_config
from asr_cli.core.enums import JobStatus, OutputFormat
from asr_cli.core.errors import CombineProcessingError


cli_runner = CliRunner()


def _write_media_file(path):
    path.write_text('media', encoding='utf-8')
    return path


def test_transcribe_single_file_outputs_all(fake_runner, monkeypatch, workspace_tmp) -> None:
    monkeypatch.setattr('asr_cli.cli.main.build_runner', lambda: fake_runner)
    input_path = _write_media_file(workspace_tmp / 'sample.wav')
    output_dir = workspace_tmp / 'out'

    result = cli_runner.invoke(
        app,
        [
            'transcribe',
            str(input_path),
            '--output-dir',
            str(output_dir),
            '--txt',
            '--json',
            '--normalize',
            '--asr',
            'fake-asr',
            '--diarizer',
            'fake-diarizer',
            '--llm',
            'fake-llm',
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output_dir / 'sample.txt').exists()
    assert (output_dir / 'sample.json').exists()
    payload = json.loads((output_dir / 'sample.json').read_text(encoding='utf-8'))
    assert payload['metadata']['normalized'] is True
    assert payload['segments'][0]['speaker'] == 'SPEAKER_00'
    assert 'RAW SAMPLE' in (output_dir / 'sample.txt').read_text(encoding='utf-8')


def test_combine_fail_fast_on_bad_file(fake_runner, workspace_tmp) -> None:
    output_dir = workspace_tmp / 'out'
    config = build_app_config(
        output_dir=output_dir,
        formats=[OutputFormat.JSON],
        asr_provider='fake-asr',
        diarizer_provider='fake-diarizer',
        llm_provider='fake-llm',
    )
    good = _write_media_file(workspace_tmp / 'good.wav')
    bad = _write_media_file(workspace_tmp / 'bad.wav')

    try:
        fake_runner.combine_files([good, bad], config)
    except CombineProcessingError as exc:
        assert 'fake asr failure' in str(exc)
    else:
        raise AssertionError('combine was expected to fail')

    assert not (output_dir / 'combined.json').exists()


def test_batch_continues_when_one_file_fails(fake_runner, workspace_tmp) -> None:
    folder = workspace_tmp / 'batch'
    folder.mkdir()
    _write_media_file(folder / 'a.wav')
    _write_media_file(folder / 'bad.wav')
    _write_media_file(folder / 'c.wav')
    output_dir = workspace_tmp / 'out'
    config = build_app_config(
        output_dir=output_dir,
        formats=[OutputFormat.JSON],
        continue_on_error=True,
        asr_provider='fake-asr',
        diarizer_provider='fake-diarizer',
        llm_provider='fake-llm',
    )

    result = fake_runner.batch_folder(folder, config)

    assert result.total == 3
    assert result.succeeded == 2
    assert result.failed == 1
    assert any(item.status == JobStatus.FAILED for item in result.results)
    report = json.loads((output_dir / 'batch_report.json').read_text(encoding='utf-8'))
    assert report['failed'] == 1


def test_combine_offsets_segments(fake_runner, workspace_tmp) -> None:
    output_dir = workspace_tmp / 'out'
    config = build_app_config(
        output_dir=output_dir,
        formats=[OutputFormat.JSON],
        asr_provider='fake-asr',
        diarizer_provider='fake-diarizer',
        llm_provider='fake-llm',
    )
    first = _write_media_file(workspace_tmp / 'first.wav')
    second = _write_media_file(workspace_tmp / 'second.wav')

    result = fake_runner.combine_files([first, second], config)

    assert result.status == JobStatus.SUCCEEDED
    assert result.document is not None
    assert len(result.document.source_offsets) == 2
    assert result.document.segments[2].start >= result.document.source_offsets[1].start_offset
