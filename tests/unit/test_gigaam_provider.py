import sys
import wave
from pathlib import Path
from types import SimpleNamespace

from asr_cli.core.config import ASRConfig
from asr_cli.core.models import PreparedMedia
from asr_cli.core.progress import ProgressListener
from asr_cli.providers.gigaam.provider import GigaAMASRProvider


class RecordingProgressListener(ProgressListener):
    def __init__(self) -> None:
        self.events = []

    def on_stage_progress(self, stage: str, *, completed: int, total: int, path=None) -> None:
        self.events.append((stage, completed, total, path.name if path else None))


class DummyLongformModel:
    def __init__(self) -> None:
        self.calls = []

    def transcribe_longform(self, path: str):
        chunk_name = Path(path).stem
        self.calls.append(chunk_name)
        return [SimpleNamespace(start=0.0, end=0.5, text=f'text {chunk_name}')]

    def transcribe(self, path: str, word_timestamps: bool = True):
        raise AssertionError('shortform path should not be used for this test')


def _write_wav(path: Path, duration_seconds: int, sample_rate: int = 16000) -> None:
    total_frames = duration_seconds * sample_rate
    with wave.open(str(path), 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b'\x00\x00' * total_frames)


def test_gigaam_longform_reports_chunk_progress_and_offsets(monkeypatch, workspace_tmp) -> None:
    model = DummyLongformModel()
    monkeypatch.setitem(
        sys.modules,
        'gigaam',
        SimpleNamespace(load_model=lambda _model_name: model),
    )

    prepared_path = workspace_tmp / 'long.wav'
    _write_wav(prepared_path, duration_seconds=3)
    media = PreparedMedia(
        original_path=prepared_path,
        prepared_path=prepared_path,
        duration_seconds=3.0,
        source_kind='audio',
        sample_rate=16000,
    )
    config = ASRConfig(
        longform_mode='always',
        longform_threshold_seconds=1.0,
        longform_chunk_seconds=1.0,
    )
    listener = RecordingProgressListener()

    provider = GigaAMASRProvider(config)
    document = provider.transcribe(
        media,
        config,
        'ru',
        progress_listener=listener,
    )

    assert [segment.start for segment in document.segments] == [0.0, 1.0, 2.0]
    assert document.metadata['longform_used'] is True
    assert document.metadata['longform_chunked'] is True
    assert document.metadata['longform_chunk_count'] == 3
    assert listener.events == [
        ('transcription', 0, 3, 'long.wav'),
        ('transcription', 1, 3, 'long.wav'),
        ('transcription', 2, 3, 'long.wav'),
        ('transcription', 3, 3, 'long.wav'),
    ]
    assert model.calls == ['long.chunk0000', 'long.chunk0001', 'long.chunk0002']
    assert not list(workspace_tmp.glob('*.chunk*.wav'))