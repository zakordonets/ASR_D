from __future__ import annotations

from asr_cli.core.registry import build_default_registry
from asr_cli.io.ffmpeg import FfmpegPreprocessor
from asr_cli.pipeline.orchestrator import PipelineRunner


def build_runner() -> PipelineRunner:
    return PipelineRunner(
        registry=build_default_registry(),
        preprocessor=FfmpegPreprocessor(),
    )
