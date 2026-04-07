from __future__ import annotations

import sys
from pathlib import Path

from asr_cli.core.enums import JobStatus
from asr_cli.core.progress import ProgressListener


class CliProgressReporter(ProgressListener):
    def __init__(self) -> None:
        self._enabled = False
        self._progress = None
        self._batch_task_id: int | None = None
        self._stage_task_id: int | None = None
        self._normalization_task_id: int | None = None

        if not sys.stdout.isatty():
            return

        try:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )
        except ImportError:
            return

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn('{task.description}'),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
        )
        self._enabled = True

    def __enter__(self) -> CliProgressReporter:
        if self._enabled and self._progress is not None:
            self._progress.start()
            self._batch_task_id = self._progress.add_task('Batch', total=1, visible=False)
            self._stage_task_id = self._progress.add_task(
                'Current stage', total=None, visible=False
            )
            self._normalization_task_id = self._progress.add_task(
                'Normalization', total=1, visible=False
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._enabled and self._progress is not None:
            self._progress.stop()

    def on_batch_started(self, total: int) -> None:
        if not self._enabled or self._progress is None or self._batch_task_id is None:
            return
        self._progress.update(
            self._batch_task_id,
            description='Batch progress',
            total=max(total, 1),
            completed=0,
            visible=True,
        )

    def on_batch_advanced(
        self,
        completed: int,
        total: int,
        *,
        path: Path | None = None,
        status: JobStatus | None = None,
    ) -> None:
        if not self._enabled or self._progress is None or self._batch_task_id is None:
            return
        label = f'Batch progress ({completed}/{total})'
        if path is not None and status is not None:
            label = f'Batch progress ({completed}/{total}) {path.name}: {status.value}'
        self._progress.update(
            self._batch_task_id,
            description=label,
            total=max(total, 1),
            completed=completed,
            visible=True,
        )

    def on_file_started(self, path: Path, *, operation: str) -> None:
        if not self._enabled or self._progress is None or self._stage_task_id is None:
            return
        self._progress.update(
            self._stage_task_id,
            description=f'{operation.title()}: {path.name}',
            total=None,
            completed=0,
            visible=True,
        )

    def on_stage_started(
        self,
        stage: str,
        *,
        path: Path | None = None,
        total: int | None = None,
    ) -> None:
        if not self._enabled or self._progress is None or self._stage_task_id is None:
            return
        label = self._stage_label(stage, path)
        if (
            stage == 'normalization'
            and total is not None
            and self._normalization_task_id is not None
        ):
            self._progress.update(
                self._normalization_task_id,
                description=label,
                total=max(total, 1),
                completed=0,
                visible=True,
            )
            return
        self._progress.update(
            self._stage_task_id,
            description=label,
            total=total,
            completed=0,
            visible=True,
        )

    def on_stage_progress(
        self,
        stage: str,
        *,
        completed: int,
        total: int,
        path: Path | None = None,
    ) -> None:
        if not self._enabled or self._progress is None:
            return
        if stage == 'normalization' and self._normalization_task_id is not None:
            self._progress.update(
                self._normalization_task_id,
                description=self._stage_label(stage, path),
                total=max(total, 1),
                completed=completed,
                visible=True,
            )
            return
        if self._stage_task_id is None:
            return
        self._progress.update(
            self._stage_task_id,
            description=self._stage_label(stage, path),
            total=max(total, 1),
            completed=completed,
            visible=True,
        )

    def on_stage_completed(
        self,
        stage: str,
        *,
        path: Path | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        if not self._enabled or self._progress is None:
            return
        if stage == 'normalization' and self._normalization_task_id is not None:
            task = self._progress.tasks[self._normalization_task_id]
            self._progress.update(
                self._normalization_task_id,
                description=self._completion_label(stage, path, elapsed_seconds),
                completed=task.total or task.completed,
                visible=False,
            )
            return
        if self._stage_task_id is None:
            return
        task = self._progress.tasks[self._stage_task_id]
        self._progress.update(
            self._stage_task_id,
            description=self._completion_label(stage, path, elapsed_seconds),
            completed=task.total or task.completed,
            visible=True,
        )

    def on_file_completed(
        self,
        path: Path | None,
        *,
        status: JobStatus,
        error: str | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        if not self._enabled or self._progress is None or self._stage_task_id is None:
            return
        if path is None:
            return
        description = f'{path.name}: {status.value}'
        if elapsed_seconds is not None:
            description = f'{description} ({self._format_duration(elapsed_seconds)})'
        if error:
            description = f'{description} ({error})'
        self._progress.update(
            self._stage_task_id,
            description=description,
            total=None,
            completed=0,
            visible=True,
        )

    def _completion_label(
        self,
        stage: str,
        path: Path | None,
        elapsed_seconds: float | None,
    ) -> str:
        label = f'{self._stage_label(stage, path)} done'
        if elapsed_seconds is not None:
            return f'{label} ({self._format_duration(elapsed_seconds)})'
        return label

    def _stage_label(self, stage: str, path: Path | None) -> str:
        labels = {
            'preprocess': 'Preprocessing',
            'transcription': 'Transcription',
            'diarization': 'Diarization',
            'normalization': 'Normalization',
            'combine': 'Combining files',
            'export': 'Exporting',
        }
        label = labels.get(stage, stage.replace('_', ' ').title())
        if path is None:
            return label
        return f'{label}: {path.name}'

    def _format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f'{seconds:.2f}s'
        minutes, remainder = divmod(seconds, 60)
        if minutes < 60:
            return f'{int(minutes)}m {remainder:.1f}s'
        hours, minutes = divmod(int(minutes), 60)
        return f'{hours}h {minutes}m {remainder:.1f}s'