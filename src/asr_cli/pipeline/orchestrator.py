from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from asr_cli.core.config import AppConfig
from asr_cli.core.enums import JobStatus, OutputFormat
from asr_cli.core.errors import CombineProcessingError
from asr_cli.core.models import BatchResult, JobResult, TranscriptDocument
from asr_cli.core.progress import ProgressListener
from asr_cli.core.registry import ProviderRegistry
from asr_cli.io.exporters.json import JsonWriter
from asr_cli.io.exporters.srt import SrtWriter
from asr_cli.io.exporters.txt import TxtWriter
from asr_cli.io.exporters.vtt import VttWriter
from asr_cli.io.ffmpeg import FfmpegPreprocessor
from asr_cli.io.inputs import discover_media_files, ensure_supported_media_file
from asr_cli.pipeline.combine import combine_documents
from asr_cli.pipeline.merge import assign_speakers
from asr_cli.utils.json import to_jsonable


class PipelineRunner:
    def __init__(
        self,
        registry: ProviderRegistry,
        preprocessor: FfmpegPreprocessor,
    ) -> None:
        self.registry = registry
        self.preprocessor = preprocessor
        self._writers = {
            OutputFormat.TXT: TxtWriter(),
            OutputFormat.JSON: JsonWriter(),
            OutputFormat.SRT: SrtWriter(),
            OutputFormat.VTT: VttWriter(),
        }

    def transcribe_file(
        self,
        input_path: Path,
        config: AppConfig,
        progress_listener: ProgressListener | None = None,
    ) -> JobResult:
        progress = progress_listener or ProgressListener()
        timings: dict[str, float] = {}
        try:
            source_path = ensure_supported_media_file(input_path)
            progress.on_file_started(source_path, operation='transcribe')
            document = self._process_document(
                source_path,
                config,
                progress,
                timings=timings,
            )
            output_files = self._run_export(
                document,
                source_path.stem,
                config,
                progress_listener=progress,
                timings=timings,
                progress_path=source_path,
            )
            finalized_timings = self._finalize_timings(timings)
            self._attach_document_timings(document, finalized_timings)
            self._refresh_json_outputs(document, output_files)
            result = JobResult(
                status=JobStatus.SUCCEEDED,
                input_path=source_path,
                document=document,
                output_files=output_files,
                timings=finalized_timings,
            )
            progress.on_file_completed(
                source_path,
                status=result.status,
                elapsed_seconds=finalized_timings.get('total'),
            )
            return result
        except Exception as exc:
            finalized_timings = self._finalize_timings(timings)
            progress.on_file_completed(
                input_path,
                status=JobStatus.FAILED,
                error=str(exc),
                elapsed_seconds=finalized_timings.get('total'),
            )
            return JobResult(
                status=JobStatus.FAILED,
                input_path=input_path,
                error=str(exc),
                timings=finalized_timings,
            )

    def combine_files(
        self,
        input_paths: list[Path],
        config: AppConfig,
        progress_listener: ProgressListener | None = None,
    ) -> JobResult:
        progress = progress_listener or ProgressListener()
        aggregate_timings: dict[str, float] = {}
        documents: list[TranscriptDocument] = []
        resolved_paths: list[Path] = []
        try:
            for input_path in input_paths:
                source_path = ensure_supported_media_file(input_path)
                resolved_paths.append(source_path)
                progress.on_file_started(source_path, operation='combine')
                file_timings: dict[str, float] = {}
                try:
                    document = self._process_document(
                        source_path,
                        config,
                        progress,
                        timings=file_timings,
                    )
                except Exception as exc:
                    finalized_file_timings = self._finalize_timings(file_timings)
                    progress.on_file_completed(
                        source_path,
                        status=JobStatus.FAILED,
                        error=str(exc),
                        elapsed_seconds=finalized_file_timings.get('total'),
                    )
                    raise
                finalized_file_timings = self._finalize_timings(file_timings)
                self._merge_timings(aggregate_timings, finalized_file_timings, include_total=False)
                progress.on_file_completed(
                    source_path,
                    status=JobStatus.SUCCEEDED,
                    elapsed_seconds=finalized_file_timings.get('total'),
                )
                documents.append(document)

            progress.on_stage_started('combine', total=len(documents))
            combine_started = perf_counter()
            combined = combine_documents(documents, resolved_paths, title='combined')
            self._finish_stage(
                'combine',
                combine_started,
                aggregate_timings,
                progress_listener=progress,
            )

            if config.normalization.enabled:
                progress.on_stage_started(
                    'normalization',
                    total=len(combined.segments),
                )
                normalization_started = perf_counter()
                combined = self._normalize_document(
                    combined,
                    config,
                    progress_listener=progress,
                )
                self._finish_stage(
                    'normalization',
                    normalization_started,
                    aggregate_timings,
                    progress_listener=progress,
                )

            output_files = self._run_export(
                combined,
                'combined',
                config,
                progress_listener=progress,
                timings=aggregate_timings,
            )
            finalized_timings = self._finalize_timings(aggregate_timings)
            self._attach_document_timings(combined, finalized_timings)
            self._refresh_json_outputs(combined, output_files)
            return JobResult(
                status=JobStatus.SUCCEEDED,
                input_path=None,
                document=combined,
                output_files=output_files,
                timings=finalized_timings,
            )
        except Exception as exc:
            raise CombineProcessingError(str(exc)) from exc

    def batch_folder(
        self,
        folder: Path,
        config: AppConfig,
        on_job_complete: Callable[[JobResult], None] | None = None,
        progress_listener: ProgressListener | None = None,
    ) -> BatchResult:
        progress = progress_listener or ProgressListener()
        files = discover_media_files(folder, recursive=config.recursive)
        results: list[JobResult] = []
        succeeded = 0
        failed = 0
        batch_started = perf_counter()
        progress.on_batch_started(len(files))

        for index, file_path in enumerate(files, start=1):
            result = self.transcribe_file(file_path, config, progress_listener=progress)
            results.append(result)
            if on_job_complete is not None:
                on_job_complete(result)
            if result.status == JobStatus.SUCCEEDED:
                succeeded += 1
            else:
                failed += 1
                progress.on_batch_advanced(
                    index,
                    len(files),
                    path=file_path,
                    status=result.status,
                )
                if not config.continue_on_error:
                    break
                continue
            progress.on_batch_advanced(
                index,
                len(files),
                path=file_path,
                status=result.status,
            )

        batch_result = BatchResult(
            total=len(files),
            succeeded=succeeded,
            failed=failed,
            skipped=0,
            results=results,
            elapsed_seconds=perf_counter() - batch_started,
        )
        self._write_batch_report(batch_result, config)
        return batch_result

    def _process_document(
        self,
        source_path: Path,
        config: AppConfig,
        progress_listener: ProgressListener | None = None,
        timings: dict[str, float] | None = None,
    ) -> TranscriptDocument:
        progress = progress_listener or ProgressListener()
        stage_timings = timings if timings is not None else {}
        temp_root = config.export.output_dir / '.tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        workspace = temp_root / f'asr_cli_{uuid.uuid4().hex}'
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            progress.on_stage_started('preprocess', path=source_path)
            preprocess_started = perf_counter()
            prepared = self.preprocessor.prepare(source_path, workspace)
            self._finish_stage(
                'preprocess',
                preprocess_started,
                stage_timings,
                progress_listener=progress,
                path=source_path,
            )

            asr_provider = self.registry.create_asr(config.asr.provider_id, config.asr)
            progress.on_stage_started('transcription', path=source_path)
            transcription_started = perf_counter()
            document = asr_provider.transcribe(
                prepared,
                config.asr,
                config.language,
                progress_listener=progress,
            )
            self._finish_stage(
                'transcription',
                transcription_started,
                stage_timings,
                progress_listener=progress,
                path=source_path,
            )

            if config.diarization.enabled:
                diarizer = self.registry.create_diarization(
                    config.diarization.provider_id, config.diarization
                )
                progress.on_stage_started('diarization', path=source_path)
                diarization_started = perf_counter()
                speaker_turns = diarizer.diarize(prepared, config.diarization)
                document = assign_speakers(document, speaker_turns)
                self._finish_stage(
                    'diarization',
                    diarization_started,
                    stage_timings,
                    progress_listener=progress,
                    path=source_path,
                )

            if config.normalization.enabled:
                progress.on_stage_started(
                    'normalization',
                    path=source_path,
                    total=len(document.segments),
                )
                normalization_started = perf_counter()
                document = self._normalize_document(
                    document,
                    config,
                    progress_listener=progress,
                )
                self._finish_stage(
                    'normalization',
                    normalization_started,
                    stage_timings,
                    progress_listener=progress,
                    path=source_path,
                )
            return document
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def _normalize_document(
        self,
        document: TranscriptDocument,
        config: AppConfig,
        progress_listener: ProgressListener | None = None,
    ) -> TranscriptDocument:
        provider = self.registry.create_normalization(
            config.normalization.provider_id, config.normalization
        )
        return provider.normalize(
            document,
            config.normalization,
            config.language,
            progress_listener=progress_listener,
        )

    def _run_export(
        self,
        document: TranscriptDocument,
        basename: str,
        config: AppConfig,
        *,
        progress_listener: ProgressListener | None,
        timings: dict[str, float],
        progress_path: Path | None = None,
    ) -> list[Path]:
        progress = progress_listener or ProgressListener()
        progress.on_stage_started(
            'export',
            path=progress_path,
            total=len(config.export.formats),
        )
        export_started = perf_counter()
        output_files = self._export_document(
            document,
            basename,
            config,
            progress_listener=progress,
            progress_path=progress_path,
        )
        self._finish_stage(
            'export',
            export_started,
            timings,
            progress_listener=progress,
            path=progress_path,
        )
        return output_files

    def _export_document(
        self,
        document: TranscriptDocument,
        basename: str,
        config: AppConfig,
        progress_listener: ProgressListener | None = None,
        progress_path: Path | None = None,
    ) -> list[Path]:
        config.export.output_dir.mkdir(parents=True, exist_ok=True)
        output_files: list[Path] = []
        total = len(config.export.formats)
        for index, output_format in enumerate(config.export.formats, start=1):
            writer = self._writers[output_format]
            destination = config.export.output_dir / f'{basename}.{output_format.value}'
            output_files.append(writer.write(document, destination))
            if progress_listener is not None:
                progress_listener.on_stage_progress(
                    'export',
                    completed=index,
                    total=total,
                    path=progress_path,
                )
        return output_files

    def _finish_stage(
        self,
        stage: str,
        started_at: float,
        timings: dict[str, float],
        *,
        progress_listener: ProgressListener,
        path: Path | None = None,
    ) -> float:
        elapsed = perf_counter() - started_at
        timings[stage] = timings.get(stage, 0.0) + elapsed
        progress_listener.on_stage_completed(
            stage,
            path=path,
            elapsed_seconds=elapsed,
        )
        return elapsed

    def _finalize_timings(self, timings: dict[str, float]) -> dict[str, float]:
        finalized = {
            stage: round(seconds, 3)
            for stage, seconds in timings.items()
            if stage != 'total'
        }
        finalized['total'] = round(sum(finalized.values()), 3)
        return finalized

    def _attach_document_timings(
        self,
        document: TranscriptDocument,
        timings: dict[str, float],
    ) -> None:
        document.metadata = {
            **document.metadata,
            'timings': timings,
        }

    def _refresh_json_outputs(
        self,
        document: TranscriptDocument,
        output_files: list[Path],
    ) -> None:
        json_writer = self._writers[OutputFormat.JSON]
        for output_file in output_files:
            if output_file.suffix.lower() == '.json':
                json_writer.write(document, output_file)

    def _merge_timings(
        self,
        target: dict[str, float],
        source: dict[str, float],
        *,
        include_total: bool,
    ) -> None:
        for stage, seconds in source.items():
            if stage == 'total' and not include_total:
                continue
            target[stage] = target.get(stage, 0.0) + seconds

    def _write_batch_report(self, batch_result: BatchResult, config: AppConfig) -> None:
        config.export.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = config.export.output_dir / 'batch_report.json'
        report_path.write_text(
            json.dumps(to_jsonable(batch_result), ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )