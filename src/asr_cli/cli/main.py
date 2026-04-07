from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import typer

from asr_cli.cli.progress import CliProgressReporter
from asr_cli.cli.runtime import build_runner
from asr_cli.core.config import build_app_config, load_dotenv
from asr_cli.core.enums import JobStatus, OutputFormat
from asr_cli.core.errors import CombineProcessingError
from asr_cli.io.inputs import discover_media_files

app = typer.Typer(help='Transcribe and diarize audio/video files.')
providers_app = typer.Typer(help='Provider diagnostics and listing.')
app.add_typer(providers_app, name='providers')


def _parse_formats(
    txt: bool,
    json_output: bool,
    srt: bool,
    vtt: bool,
    all_formats: bool,
) -> list[OutputFormat]:
    if all_formats:
        return [
            OutputFormat.TXT,
            OutputFormat.JSON,
            OutputFormat.SRT,
            OutputFormat.VTT,
        ]
    selected = []
    if txt:
        selected.append(OutputFormat.TXT)
    if json_output:
        selected.append(OutputFormat.JSON)
    if srt:
        selected.append(OutputFormat.SRT)
    if vtt:
        selected.append(OutputFormat.VTT)
    return selected or [OutputFormat.TXT, OutputFormat.JSON]


def _build_config(
    *,
    output_dir: Path,
    formats: list[OutputFormat],
    recursive: bool = False,
    continue_on_error: bool = True,
    normalize: bool = False,
    apply_normalization_to_subtitles: bool = False,
    asr_provider: str = 'gigaam',
    diarizer_provider: str = 'pyannote',
    llm_provider: str | None = None,
    config_file: Path | None = None,
):
    return build_app_config(
        output_dir=output_dir,
        formats=formats,
        recursive=recursive,
        continue_on_error=continue_on_error,
        normalize=normalize,
        apply_normalization_to_subtitles=apply_normalization_to_subtitles,
        asr_provider=asr_provider,
        diarizer_provider=diarizer_provider,
        llm_provider=llm_provider,
        config_file=config_file,
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _formats_label(formats: list[OutputFormat]) -> str:
    return ', '.join(fmt.value for fmt in formats)


def _normalization_label(config) -> str:
    if not config.normalization.enabled:
        return 'disabled'
    return (
        f"enabled ({config.normalization.provider_id}:{config.normalization.model_name})"
    )


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.2f}s'
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f'{int(minutes)}m {remainder:.1f}s'
    hours, minutes = divmod(int(minutes), 60)
    return f'{hours}h {minutes}m {remainder:.1f}s'


def _format_timings(timings: dict[str, float]) -> str:
    order = ['preprocess', 'transcription', 'diarization', 'normalization', 'combine', 'export', 'total']
    parts = [
        f'{stage}={_format_duration(timings[stage])}'
        for stage in order
        if stage in timings
    ]
    extras = [
        f'{stage}={_format_duration(seconds)}'
        for stage, seconds in timings.items()
        if stage not in order
    ]
    return ', '.join(parts + extras)


def _print_timings(timings: dict[str, float]) -> None:
    if not timings:
        return
    typer.echo(f'Timings: {_format_timings(timings)}')


def _print_start_status(kind: str, config, *, target: str, count: int | None = None) -> None:
    typer.echo(f'Starting {kind}')
    typer.echo(f'Target: {target}')
    if count is not None:
        typer.echo(f'Items: {count}')
    typer.echo(f'ASR: {config.asr.provider_id}')
    typer.echo(
        f"Diarization: {'enabled (' + config.diarization.provider_id + ')' if config.diarization.enabled else 'disabled'}"
    )
    typer.echo(f'Normalization: {_normalization_label(config)}')
    typer.echo(f'Formats: {_formats_label(config.export.formats)}')
    typer.echo(f'Output dir: {config.export.output_dir}')


def _print_batch_job_status(result) -> None:
    path_label = str(result.input_path) if result.input_path is not None else '<unknown>'
    timing_suffix = f' ({_format_timings(result.timings)})' if result.timings else ''
    if result.status == JobStatus.SUCCEEDED:
        typer.echo(f'[OK] {path_label}{timing_suffix}')
    else:
        typer.echo(f'[FAILED] {path_label}: {result.error}{timing_suffix}', err=True)


@app.command()
def transcribe(
    input_path: Path = typer.Argument(..., exists=False),
    output_dir: Path = typer.Option(Path('out')),
    txt: bool = typer.Option(False, '--txt'),
    json_output: bool = typer.Option(False, '--json'),
    srt: bool = typer.Option(False, '--srt'),
    vtt: bool = typer.Option(False, '--vtt'),
    all_formats: bool = typer.Option(False, '--all-formats'),
    normalize: bool = typer.Option(False, '--normalize'),
    normalize_subtitles: bool = typer.Option(False, '--normalize-subtitles'),
    asr_provider: str = typer.Option('gigaam', '--asr'),
    diarizer_provider: str = typer.Option('pyannote', '--diarizer'),
    llm_provider: str | None = typer.Option(None, '--llm'),
    config_file: Path | None = typer.Option(None, '--config'),
) -> None:
    runner = build_runner()
    formats = _parse_formats(txt, json_output, srt, vtt, all_formats)
    config = _build_config(
        output_dir=output_dir,
        formats=formats,
        normalize=normalize,
        apply_normalization_to_subtitles=normalize_subtitles,
        asr_provider=asr_provider,
        diarizer_provider=diarizer_provider,
        llm_provider=llm_provider,
        config_file=config_file,
    )
    _print_start_status('transcription', config, target=str(input_path))
    with CliProgressReporter() as progress_reporter:
        result = runner.transcribe_file(
            input_path,
            config,
            progress_listener=progress_reporter,
        )
    if result.status == JobStatus.SUCCEEDED:
        typer.echo(f'OK: {input_path}')
        _print_timings(result.timings)
        for output_file in result.output_files:
            typer.echo(str(output_file))
        return
    typer.echo(result.error or 'transcription failed', err=True)
    _print_timings(result.timings)
    raise typer.Exit(code=1)


@app.command()
def combine(
    input_paths: list[Path] = typer.Argument(...),
    output_dir: Path = typer.Option(Path('out')),
    txt: bool = typer.Option(False, '--txt'),
    json_output: bool = typer.Option(False, '--json'),
    srt: bool = typer.Option(False, '--srt'),
    vtt: bool = typer.Option(False, '--vtt'),
    all_formats: bool = typer.Option(False, '--all-formats'),
    normalize: bool = typer.Option(False, '--normalize'),
    normalize_subtitles: bool = typer.Option(False, '--normalize-subtitles'),
    asr_provider: str = typer.Option('gigaam', '--asr'),
    diarizer_provider: str = typer.Option('pyannote', '--diarizer'),
    llm_provider: str | None = typer.Option(None, '--llm'),
    config_file: Path | None = typer.Option(None, '--config'),
) -> None:
    runner = build_runner()
    formats = _parse_formats(txt, json_output, srt, vtt, all_formats)
    config = _build_config(
        output_dir=output_dir,
        formats=formats,
        normalize=normalize,
        apply_normalization_to_subtitles=normalize_subtitles,
        asr_provider=asr_provider,
        diarizer_provider=diarizer_provider,
        llm_provider=llm_provider,
        config_file=config_file,
    )
    _print_start_status('combine', config, target='multiple files', count=len(input_paths))
    try:
        with CliProgressReporter() as progress_reporter:
            result = runner.combine_files(
                input_paths,
                config,
                progress_listener=progress_reporter,
            )
    except CombineProcessingError as exc:
        typer.echo(f'Combine failed: {exc}', err=True)
        raise typer.Exit(code=1) from exc
    if result.status == JobStatus.SUCCEEDED:
        typer.echo('OK: combined')
        _print_timings(result.timings)
        for output_file in result.output_files:
            typer.echo(str(output_file))


@app.command()
def batch(
    folder: Path = typer.Argument(..., exists=False),
    output_dir: Path = typer.Option(Path('out')),
    recursive: bool = typer.Option(False, '--recursive'),
    continue_on_error: bool = typer.Option(True, '--continue-on-error/--fail-fast'),
    txt: bool = typer.Option(False, '--txt'),
    json_output: bool = typer.Option(False, '--json'),
    srt: bool = typer.Option(False, '--srt'),
    vtt: bool = typer.Option(False, '--vtt'),
    all_formats: bool = typer.Option(False, '--all-formats'),
    normalize: bool = typer.Option(False, '--normalize'),
    normalize_subtitles: bool = typer.Option(False, '--normalize-subtitles'),
    asr_provider: str = typer.Option('gigaam', '--asr'),
    diarizer_provider: str = typer.Option('pyannote', '--diarizer'),
    llm_provider: str | None = typer.Option(None, '--llm'),
    config_file: Path | None = typer.Option(None, '--config'),
) -> None:
    runner = build_runner()
    formats = _parse_formats(txt, json_output, srt, vtt, all_formats)
    config = _build_config(
        output_dir=output_dir,
        formats=formats,
        recursive=recursive,
        continue_on_error=continue_on_error,
        normalize=normalize,
        apply_normalization_to_subtitles=normalize_subtitles,
        asr_provider=asr_provider,
        diarizer_provider=diarizer_provider,
        llm_provider=llm_provider,
        config_file=config_file,
    )
    files = discover_media_files(folder, recursive=config.recursive)
    _print_start_status('batch', config, target=str(folder), count=len(files))
    with CliProgressReporter() as progress_reporter:
        result = runner.batch_folder(
            folder,
            config,
            on_job_complete=_print_batch_job_status,
            progress_listener=progress_reporter,
        )
    typer.echo(
        f'Processed={result.total} Succeeded={result.succeeded} Failed={result.failed} Elapsed={_format_duration(result.elapsed_seconds)}'
    )
    if result.failed and not continue_on_error:
        raise typer.Exit(code=1)


@providers_app.command('list')
def list_providers() -> None:
    runner = build_runner()
    typer.echo('ASR:')
    for provider_id in sorted(runner.registry.asr_factories):
        typer.echo(f'  {provider_id}')
    typer.echo('Diarization:')
    for provider_id in sorted(runner.registry.diarization_factories):
        typer.echo(f'  {provider_id}')
    typer.echo('Normalization:')
    for provider_id in sorted(runner.registry.normalization_factories):
        typer.echo(f'  {provider_id}')


@app.command()
def doctor() -> None:
    load_dotenv()
    runner = build_runner()
    ffmpeg_ok = runner.preprocessor.is_available()
    typer.echo(f"ffmpeg: {'OK' if shutil.which('ffmpeg') else 'MISSING'}")
    typer.echo(f"ffprobe: {'OK' if shutil.which('ffprobe') else 'MISSING'}")
    typer.echo(f"preprocessor: {'OK' if ffmpeg_ok else 'CHECK PATH'}")
    typer.echo(f"gigaam-import: {'OK' if _module_available('gigaam') else 'MISSING'}")
    typer.echo(f"pyannote-import: {'OK' if _module_available('pyannote.audio') else 'MISSING'}")
    typer.echo(f"openai-import: {'OK' if _module_available('openai') else 'MISSING'}")
    typer.echo(f"HF_TOKEN: {'SET' if os.getenv('HF_TOKEN') else 'MISSING'}")
    typer.echo(f"DEEPSEEK_API_KEY: {'SET' if os.getenv('DEEPSEEK_API_KEY') else 'MISSING'}")
    typer.echo(f"OPENROUTER_API_KEY: {'SET' if os.getenv('OPENROUTER_API_KEY') else 'MISSING'}")
    typer.echo(
        f"NORMALIZATION_PROVIDER: {os.getenv('NORMALIZATION_PROVIDER') or os.getenv('LLM_PROVIDER') or 'deepseek'}"
    )
    typer.echo(
        f"OPENROUTER_MODEL: {os.getenv('OPENROUTER_MODEL') or 'xiaomi/mimo-v2-flash'}"
    )