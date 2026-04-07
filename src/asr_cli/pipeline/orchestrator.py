from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from asr_cli.core.config import AppConfig
from asr_cli.core.enums import JobStatus, OutputFormat
from asr_cli.core.errors import CombineProcessingError
from asr_cli.core.models import BatchResult, JobResult, TranscriptDocument
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

    def transcribe_file(self, input_path: Path, config: AppConfig) -> JobResult:
        try:
            source_path = ensure_supported_media_file(input_path)
            document = self._process_document(source_path, config)
            output_files = self._export_document(document, source_path.stem, config)
            return JobResult(
                status=JobStatus.SUCCEEDED,
                input_path=source_path,
                document=document,
                output_files=output_files,
            )
        except Exception as exc:
            return JobResult(
                status=JobStatus.FAILED,
                input_path=input_path,
                error=str(exc),
            )

    def combine_files(self, input_paths: list[Path], config: AppConfig) -> JobResult:
        documents: list[TranscriptDocument] = []
        resolved_paths: list[Path] = []
        try:
            for input_path in input_paths:
                source_path = ensure_supported_media_file(input_path)
                resolved_paths.append(source_path)
                documents.append(self._process_document(source_path, config))
            combined = combine_documents(documents, resolved_paths, title="combined")
            if config.normalization.enabled:
                combined = self._normalize_document(combined, config)
            output_files = self._export_document(combined, "combined", config)
            return JobResult(
                status=JobStatus.SUCCEEDED,
                input_path=None,
                document=combined,
                output_files=output_files,
            )
        except Exception as exc:
            raise CombineProcessingError(str(exc)) from exc

    def batch_folder(self, folder: Path, config: AppConfig) -> BatchResult:
        files = discover_media_files(folder, recursive=config.recursive)
        results: list[JobResult] = []
        succeeded = 0
        failed = 0

        for file_path in files:
            result = self.transcribe_file(file_path, config)
            results.append(result)
            if result.status == JobStatus.SUCCEEDED:
                succeeded += 1
            else:
                failed += 1
                if not config.continue_on_error:
                    break

        batch_result = BatchResult(
            total=len(files),
            succeeded=succeeded,
            failed=failed,
            skipped=0,
            results=results,
        )
        self._write_batch_report(batch_result, config)
        return batch_result

    def _process_document(self, source_path: Path, config: AppConfig) -> TranscriptDocument:
        temp_root = config.export.output_dir / '.tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        workspace = temp_root / f'asr_cli_{uuid.uuid4().hex}'
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            prepared = self.preprocessor.prepare(source_path, workspace)
            asr_provider = self.registry.create_asr(config.asr.provider_id, config.asr)
            document = asr_provider.transcribe(prepared, config.asr, config.language)
            if config.diarization.enabled:
                diarizer = self.registry.create_diarization(
                    config.diarization.provider_id, config.diarization
                )
                speaker_turns = diarizer.diarize(prepared, config.diarization)
                document = assign_speakers(document, speaker_turns)
            if config.normalization.enabled:
                document = self._normalize_document(document, config)
            return document
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def _normalize_document(
        self, document: TranscriptDocument, config: AppConfig
    ) -> TranscriptDocument:
        provider = self.registry.create_normalization(
            config.normalization.provider_id, config.normalization
        )
        return provider.normalize(document, config.normalization, config.language)

    def _export_document(
        self, document: TranscriptDocument, basename: str, config: AppConfig
    ) -> list[Path]:
        config.export.output_dir.mkdir(parents=True, exist_ok=True)
        output_files: list[Path] = []
        for output_format in config.export.formats:
            writer = self._writers[output_format]
            destination = config.export.output_dir / f"{basename}.{output_format.value}"
            output_files.append(writer.write(document, destination))
        return output_files

    def _write_batch_report(self, batch_result: BatchResult, config: AppConfig) -> None:
        config.export.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = config.export.output_dir / 'batch_report.json'
        report_path.write_text(
            json.dumps(to_jsonable(batch_result), ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
