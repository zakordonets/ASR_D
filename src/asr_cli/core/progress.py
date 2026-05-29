from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from asr_cli.core.enums import JobStatus


class ProgressListener(ABC):
    """Abstract base class for progress reporting."""

    @abstractmethod
    def on_batch_started(self, total: int) -> None:
        ...

    @abstractmethod
    def on_batch_advanced(
        self,
        completed: int,
        total: int,
        *,
        path: Path | None = None,
        status: JobStatus | None = None,
    ) -> None:
        ...

    @abstractmethod
    def on_file_started(self, path: Path, *, operation: str) -> None:
        ...

    @abstractmethod
    def on_stage_started(
        self,
        stage: str,
        *,
        path: Path | None = None,
        total: int | None = None,
    ) -> None:
        ...

    @abstractmethod
    def on_stage_progress(
        self,
        stage: str,
        *,
        completed: int,
        total: int,
        path: Path | None = None,
    ) -> None:
        ...

    @abstractmethod
    def on_stage_completed(
        self,
        stage: str,
        *,
        path: Path | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        ...

    @abstractmethod
    def on_file_completed(
        self,
        path: Path | None,
        *,
        status: JobStatus,
        error: str | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        ...


class NullProgressListener(ProgressListener):
    """No-op progress listener for use as a default."""

    def on_batch_started(self, total: int) -> None:
        return None

    def on_batch_advanced(
        self,
        completed: int,
        total: int,
        *,
        path: Path | None = None,
        status: JobStatus | None = None,
    ) -> None:
        return None

    def on_file_started(self, path: Path, *, operation: str) -> None:
        return None

    def on_stage_started(
        self,
        stage: str,
        *,
        path: Path | None = None,
        total: int | None = None,
    ) -> None:
        return None

    def on_stage_progress(
        self,
        stage: str,
        *,
        completed: int,
        total: int,
        path: Path | None = None,
    ) -> None:
        return None

    def on_stage_completed(
        self,
        stage: str,
        *,
        path: Path | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        return None

    def on_file_completed(
        self,
        path: Path | None,
        *,
        status: JobStatus,
        error: str | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        return None
