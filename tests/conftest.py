from __future__ import annotations

import shutil
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from asr_cli.core.config import ASRConfig, DiarizationConfig, NormalizationConfig
from asr_cli.core.models import PreparedMedia, SpeakerTurn, TranscriptDocument, TranscriptSegment
from asr_cli.core.progress import ProgressListener
from asr_cli.core.registry import ProviderRegistry
from asr_cli.pipeline.orchestrator import PipelineRunner


class FakePreprocessor:
    def is_available(self) -> bool:
        return True

    def prepare(self, path: Path, workspace: Path) -> PreparedMedia:
        duration = 65.0 if 'long' in path.stem else 5.0
        prepared = workspace / f'{path.stem}.wav'
        prepared.write_text('prepared', encoding='utf-8')
        return PreparedMedia(
            original_path=path,
            prepared_path=prepared,
            duration_seconds=duration,
            source_kind='audio',
            sample_rate=16000,
        )


class FakeASRProvider:
    provider_id = 'fake-asr'

    def __init__(self, config: ASRConfig) -> None:
        self.config = config

    def transcribe(
        self,
        media: PreparedMedia,
        config: ASRConfig,
        language: str,
        progress_listener: ProgressListener | None = None,
    ) -> TranscriptDocument:
        if 'bad' in media.original_path.stem:
            raise RuntimeError('fake asr failure')
        text = f'raw {media.original_path.stem}'
        segments = [
            TranscriptSegment(start=0.0, end=1.5, text=text, raw_text=text),
            TranscriptSegment(
                start=1.5,
                end=3.0,
                text=f'tail {media.original_path.stem}',
                raw_text=f'tail {media.original_path.stem}',
            ),
        ]
        return TranscriptDocument(
            title=media.original_path.stem,
            language=language,
            segments=segments,
            metadata={
                'provider': self.provider_id,
                'longform_used': media.duration_seconds >= config.longform_threshold_seconds,
            },
        )


class FakeDiarizationProvider:
    provider_id = 'fake-diarizer'

    def __init__(self, config: DiarizationConfig) -> None:
        self.config = config

    def diarize(
        self, media: PreparedMedia, config: DiarizationConfig
    ) -> list[SpeakerTurn]:
        return [
            SpeakerTurn(start=0.0, end=1.6, speaker='SPEAKER_00'),
            SpeakerTurn(start=1.6, end=5.0, speaker='SPEAKER_01'),
        ]


class FakeNormalizationProvider:
    provider_id = 'fake-llm'

    def __init__(self, config: NormalizationConfig) -> None:
        self.config = config

    def normalize(
        self,
        document: TranscriptDocument,
        config: NormalizationConfig,
        language: str,
        progress_listener: ProgressListener | None = None,
    ) -> TranscriptDocument:
        normalized_segments = [
            replace(segment, normalized_text=segment.text.upper())
            for segment in document.segments
        ]
        if progress_listener is not None:
            total = len(document.segments)
            for index, _segment in enumerate(document.segments, start=1):
                progress_listener.on_stage_progress(
                    'normalization',
                    completed=index,
                    total=total,
                )
        return replace(
            document,
            segments=normalized_segments,
            metadata={**document.metadata, 'normalized': True},
        )


@pytest.fixture()
def fake_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register_asr('fake-asr', FakeASRProvider)
    registry.register_diarization('fake-diarizer', FakeDiarizationProvider)
    registry.register_normalization('fake-llm', FakeNormalizationProvider)
    return registry


@pytest.fixture()
def fake_runner(fake_registry: ProviderRegistry) -> PipelineRunner:
    return PipelineRunner(fake_registry, FakePreprocessor())


@pytest.fixture()
def workspace_tmp() -> Path:
    root = Path('test_artifacts') / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)
