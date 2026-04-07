from __future__ import annotations

from asr_cli.core.config import DiarizationConfig
from asr_cli.core.errors import ProviderDependencyError, ProviderError
from asr_cli.core.models import PreparedMedia, SpeakerTurn


class PyannoteDiarizationProvider:
    provider_id = 'pyannote'

    def __init__(self, config: DiarizationConfig) -> None:
        self.config = config
        try:
            import torch  # type: ignore
            import torchaudio  # type: ignore
            from pyannote.audio import Pipeline  # type: ignore
        except ImportError as exc:
            raise ProviderDependencyError(
                "Pyannote provider requires the optional 'pyannote.audio' dependencies."
            ) from exc
        self._torch = torch
        self._torchaudio = torchaudio
        try:
            self._pipeline = Pipeline.from_pretrained(
                config.model_name,
                token=config.hf_token,
            )
            self._pipeline.to(torch.device('cpu'))
        except Exception as exc:
            raise ProviderError(f'Failed to initialize pyannote pipeline: {exc}') from exc

    def diarize(
        self, media: PreparedMedia, config: DiarizationConfig
    ) -> list[SpeakerTurn]:
        try:
            waveform, sample_rate = self._torchaudio.load(str(media.prepared_path))
            diarization_output = self._pipeline(
                {
                    'waveform': waveform,
                    'sample_rate': sample_rate,
                }
            )
            annotation = getattr(
                diarization_output,
                'speaker_diarization',
                diarization_output,
            )
            return [
                SpeakerTurn(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker=str(speaker),
                )
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            ]
        except Exception as exc:
            raise ProviderError(f'pyannote diarization failed: {exc}') from exc