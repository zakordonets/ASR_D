import json

from typer.testing import CliRunner

from asr_cli.cli.main import app
from asr_cli.core.config import build_app_config
from asr_cli.core.enums import JobStatus, OutputFormat
from asr_cli.core.errors import CombineProcessingError
from asr_cli.core.progress import ProgressListener


cli_runner = CliRunner()


class RecordingProgressListener(ProgressListener):
    def __init__(self) -> None:
        self.events = []

    def on_batch_started(self, total: int) -> None:
        self.events.append(('batch_started', total))

    def on_batch_advanced(self, completed: int, total: int, *, path=None, status=None) -> None:
        self.events.append(
            ('batch_advanced', completed, total, path.name if path else None, status)
        )

    def on_file_started(self, path, *, operation: str) -> None:
        self.events.append(('file_started', path.name, operation))

    def on_stage_started(self, stage: str, *, path=None, total=None) -> None:
        self.events.append(('stage_started', stage, path.name if path else None, total))

    def on_stage_progress(self, stage: str, *, completed: int, total: int, path=None) -> None:
        self.events.append(
            ('stage_progress', stage, completed, total, path.name if path else None)
        )

    def on_stage_completed(self, stage: str, *, path=None, elapsed_seconds=None) -> None:
        self.events.append(
            ('stage_completed', stage, path.name if path else None, elapsed_seconds)
        )

    def on_file_completed(self, path, *, status, error=None, elapsed_seconds=None) -> None:
        self.events.append(
            ('file_completed', path.name if path else None, status, error, elapsed_seconds)
        )


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
    assert 'Starting transcription' in result.stdout
    assert 'ASR: fake-asr' in result.stdout
    assert 'Normalization: enabled (fake-llm:deepseek-chat)' in result.stdout
    assert 'Timings:' in result.stdout
    assert 'transcription=' in result.stdout
    assert (output_dir / 'sample.txt').exists()
    assert (output_dir / 'sample.json').exists()
    payload = json.loads((output_dir / 'sample.json').read_text(encoding='utf-8'))
    assert payload['metadata']['normalized'] is True
    assert 'timings' in payload['metadata']
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
    assert result.elapsed_seconds >= 0
    assert any(item.status == JobStatus.FAILED for item in result.results)
    report = json.loads((output_dir / 'batch_report.json').read_text(encoding='utf-8'))
    assert report['failed'] == 1
    assert report['elapsed_seconds'] >= 0


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
    assert 'timings' in result.document.metadata
    assert 'combine' in result.timings
    assert len(result.document.source_offsets) == 2
    assert result.document.segments[2].start >= result.document.source_offsets[1].start_offset


def test_batch_cli_prints_per_file_status(fake_runner, monkeypatch, workspace_tmp) -> None:
    monkeypatch.setattr('asr_cli.cli.main.build_runner', lambda: fake_runner)
    folder = workspace_tmp / 'batch'
    folder.mkdir()
    _write_media_file(folder / 'ok.wav')
    _write_media_file(folder / 'bad.wav')
    output_dir = workspace_tmp / 'out'

    result = cli_runner.invoke(
        app,
        [
            'batch',
            str(folder),
            '--output-dir',
            str(output_dir),
            '--json',
            '--asr',
            'fake-asr',
            '--diarizer',
            'fake-diarizer',
            '--llm',
            'fake-llm',
            '--continue-on-error',
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert 'Starting batch' in result.stdout
    assert '[OK]' in result.stdout
    assert 'total=' in result.stdout
    assert '[FAILED]' in (result.stdout + result.stderr)
    assert 'Processed=2 Succeeded=1 Failed=1 Elapsed=' in result.stdout


def test_transcribe_reports_progress_stages(fake_runner, workspace_tmp) -> None:
    input_path = _write_media_file(workspace_tmp / 'sample.wav')
    output_dir = workspace_tmp / 'out'
    config = build_app_config(
        output_dir=output_dir,
        formats=[OutputFormat.JSON],
        normalize=True,
        asr_provider='fake-asr',
        diarizer_provider='fake-diarizer',
        llm_provider='fake-llm',
    )
    listener = RecordingProgressListener()

    result = fake_runner.transcribe_file(input_path, config, progress_listener=listener)

    assert result.status == JobStatus.SUCCEEDED
    assert result.timings['total'] >= 0
    assert ('file_started', 'sample.wav', 'transcribe') in listener.events
    assert ('stage_started', 'preprocess', 'sample.wav', None) in listener.events
    assert ('stage_started', 'transcription', 'sample.wav', None) in listener.events
    assert ('stage_started', 'diarization', 'sample.wav', None) in listener.events
    assert ('stage_started', 'normalization', 'sample.wav', 2) in listener.events
    assert ('stage_started', 'export', 'sample.wav', 1) in listener.events
    normalization_progress = [
        event for event in listener.events if event[:2] == ('stage_progress', 'normalization')
    ]
    assert normalization_progress[-1][2:4] == (2, 2)
    file_completed = [event for event in listener.events if event[0] == 'file_completed']
    assert file_completed[-1][1] == 'sample.wav'
    assert file_completed[-1][2] == JobStatus.SUCCEEDED
    assert file_completed[-1][4] is not None
    stage_completed = [event for event in listener.events if event[0] == 'stage_completed']
    assert all(event[3] is not None for event in stage_completed)


def test_batch_reports_progress(fake_runner, workspace_tmp) -> None:
    folder = workspace_tmp / 'batch'
    folder.mkdir()
    _write_media_file(folder / 'ok.wav')
    _write_media_file(folder / 'bad.wav')
    output_dir = workspace_tmp / 'out'
    config = build_app_config(
        output_dir=output_dir,
        formats=[OutputFormat.JSON],
        continue_on_error=True,
        asr_provider='fake-asr',
        diarizer_provider='fake-diarizer',
        llm_provider='fake-llm',
    )
    listener = RecordingProgressListener()

    result = fake_runner.batch_folder(folder, config, progress_listener=listener)

    assert result.total == 2
    assert result.elapsed_seconds >= 0
    assert ('batch_started', 2) in listener.events
    batch_advanced = [event for event in listener.events if event[0] == 'batch_advanced']
    assert len(batch_advanced) == 2


def test_transcribe_cli_disables_diarization(fake_runner, monkeypatch, workspace_tmp) -> None:
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
            '--json',
            '--no-diarization',
            '--asr',
            'fake-asr',
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert 'Diarization: disabled' in result.stdout
    payload = json.loads((output_dir / 'sample.json').read_text(encoding='utf-8'))
    assert all(segment['speaker'] is None for segment in payload['segments'])

