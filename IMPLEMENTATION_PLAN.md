# ASR CLI Implementation Plan

## Goal

Build a Python CLI utility for transcription and diarization of audio and video files with:

- pluggable ASR models
- pluggable diarization backends
- optional LLM-based normalization
- outputs to `txt`, `json`, `srt`, `vtt`
- support for single files, combined files, and folders
- batch-safe execution where one bad file does not stop the whole run
- strong automated test coverage

Default stack:

- ASR: `GigaAM`
- diarization: `pyannote`
- normalization: `DeepSeek` via OpenAI-compatible API

Primary target:

- Windows first
- architecture kept cross-platform from day one
- CPU-first execution, no GPU dependency in the default path

---

## Upstream Constraints Confirmed

Based on the upstream `salute-developers/GigaAM` repository:

- `model.transcribe(...)` is intended for short audio only and the README explicitly states it is applicable up to 25 seconds.
- Long audio should use `model.transcribe_longform(...)`.
- Longform support requires additional `pyannote.audio` dependencies.
- The upstream package dependencies include `onnx` and `onnxruntime`, so an ONNX/CPU execution path is realistic and should be the default integration direction for Windows on Radeon hardware.
- The longform extra currently depends on `pyannote.audio==4.0`, `torch==2.8.*`, and `torchaudio==2.8.*`.

Design implication:

- Long recordings must be handled by a dedicated longform pipeline.
- CPU execution must be a first-class path, not a fallback.
- Provider initialization must expose a clear backend mode such as `auto`, `cpu`, `onnx-cpu`.

---

## Proposed Repository Layout

```text
asr_cli/
  pyproject.toml
  README.md
  .env.example
  src/asr_cli/
    __init__.py
    cli/
      __init__.py
      main.py
      commands.py
      options.py
    core/
      config.py
      logging.py
      errors.py
      models.py
      enums.py
      registry.py
    io/
      inputs.py
      media.py
      ffmpeg.py
      exporters/
        txt.py
        json.py
        srt.py
        vtt.py
    pipeline/
      orchestrator.py
      single_file.py
      combined_files.py
      batch.py
      merge.py
      normalization.py
      progress.py
    providers/
      base/
        asr.py
        diarization.py
        normalization.py
      gigaam/
        provider.py
        adapter.py
      pyannote/
        provider.py
      deepseek/
        provider.py
      openai_compatible/
        client.py
    utils/
      paths.py
      timecodes.py
      tempfiles.py
  tests/
    unit/
    integration/
    fixtures/
    golden/
```

---

## Core Architecture

### 1. Domain Interfaces

Define stable interfaces that isolate the core pipeline from specific models:

- `ASRProvider`
- `DiarizationProvider`
- `NormalizationProvider`
- `OutputWriter`

Each provider receives standard config objects and returns normalized internal data models.

### 2. Internal Data Model

Use one canonical representation across the whole app:

- `MediaInput`
- `ResolvedInput`
- `TranscriptWord`
- `TranscriptSegment`
- `SpeakerTurn`
- `TranscriptDocument`
- `JobResult`
- `BatchResult`
- `ProcessingError`

Important fields in `TranscriptDocument`:

- source file metadata
- combined-input metadata
- language
- full text
- segment list
- word-level timings when available
- speaker labels
- normalization metadata
- warnings and recoverable errors

### 3. Provider Registry

Use a registry/factory so CLI and pipeline only refer to provider IDs:

- `gigaam`
- `pyannote`
- `deepseek`
- future providers like `whisper`, `faster-whisper`, `vosk`, `openai`

This allows:

- default provider selection
- capability checks
- provider-specific option validation
- future plugin loading if needed

---

## Processing Modes

### 1. Single File

Input:

- one audio or video file

Flow:

- normalize media with `ffmpeg`
- pick short or longform ASR path
- run diarization
- merge transcript with speaker turns
- optionally normalize text with LLM
- export requested formats

### 2. Combined Files

Input:

- multiple files provided explicitly

Goal:

- produce one combined transcript across all files

Plan:

- preprocess each file independently
- preserve per-file duration
- concatenate in a virtual combined timeline
- maintain source boundaries in metadata
- offset transcript and diarization timestamps by cumulative duration
- run export as one final `TranscriptDocument`

Output metadata should include:

- source file order
- start/end offsets for each source file
- per-source warnings or failures

Failure policy for combined mode:

- if any source file in the combine set fails, the entire combined job fails
- no partial combined transcript is emitted
- the error report must identify the failing source file and processing stage

### 3. Folder / Batch Mode

Input:

- directory, optionally recursive

Behavior:

- enumerate supported media files
- process each file as an independent job
- continue on error by default
- produce a final summary and machine-readable error report

Failure isolation:

- one file failure must not abort the folder run
- each job writes its own logs and status

---

## Media and Preprocessing Layer

### FFmpeg Wrapper

Implement a typed wrapper around `ffmpeg` and `ffprobe`:

- extract audio from video
- convert to mono PCM WAV
- resample to provider-required sample rate
- inspect duration, codec, channels, sample rate
- optionally split huge files for emergency recovery workflows

Target default normalized format:

- WAV
- mono
- 16 kHz unless provider requires another rate

### Temporary Workspace

Each job gets an isolated temp directory containing:

- normalized audio
- optional VAD artifacts
- provider intermediate files
- logs

This prevents collisions in parallel or repeated runs.

---

## ASR Strategy

### Default: GigaAM

Implement `GigaAMASRProvider` with two execution paths:

- short-form path using `transcribe`
- long-form path using `transcribe_longform`

Selection rules:

- use longform when duration exceeds a configurable threshold
- allow explicit override from CLI

### CPU-First Backend Policy

For the first implementation:

- default to CPU execution
- prefer ONNX runtime where supported by the selected GigaAM model path
- expose backend selection in config

Recommended config values:

- `backend = auto`
- `device = cpu`
- `execution_provider = cpu`

### Longform Requirements

Large files over one hour should default to longform mode using VAD-aware transcription:

- `transcribe_longform`
- external VAD via `pyannote`
- chunked processing with stitched output

This should be treated as the normal path for long recordings, not an edge case.

### Future ASR Providers

The interface should support adding:

- `whisper`
- `faster-whisper`
- other OpenAI-compatible transcription APIs
- internal HTTP-based providers

---

## Diarization Strategy

### Default: pyannote

Implement `PyannoteDiarizationProvider` as the default diarizer.

Responsibilities:

- run speaker segmentation on normalized audio
- return speaker turns in canonical format
- support Hugging Face token via environment variables

### Merge Logic

Need a dedicated merge/alignment module that:

- aligns transcript segments or words to speaker turns
- assigns speaker labels by overlap ratio
- supports unknown speaker fallback when overlap is weak
- keeps deterministic output for tests

If word-level timestamps are available from ASR, use them for higher-quality speaker assignment.

---

## Optional LLM Normalization

### Default: DeepSeek

Implement normalization as a post-processing stage over the canonical transcript.

Behavior:

- disabled by default unless requested
- uses an OpenAI-compatible client abstraction
- default provider configuration targets DeepSeek
- enabled by default primarily for `txt` and `json` oriented post-processing flows

Normalization can modify:

- punctuation
- casing
- formatting
- light cleanup
- optional style normalization

It does not need to preserve text byte-for-byte, but it must preserve enough structure to keep exports coherent.

### Safety Rules for Normalization

To avoid corruption of structured output:

- normalization should operate segment-by-segment or chunk-by-chunk
- original timestamps and speaker labels remain attached to the segment unless explicit remapping is implemented
- store both raw and normalized text in JSON output when normalization is enabled
- normalized text should not be applied to `srt` or `vtt` unless explicitly requested by CLI/config

### Future LLM Backends

The same abstraction should support:

- DeepSeek
- OpenAI
- local OpenAI-compatible gateways
- other vendors exposing OpenAI API semantics

---

## Output Formats

### TXT

- plain readable transcript
- optional speakers
- optional timestamps

### JSON

- full-fidelity internal structure
- metadata
- per-segment timings
- speakers
- normalization details
- warnings/errors

### SRT

- segment-based subtitle export
- configurable speaker prefix format

### VTT

- WebVTT export with similar segmentation policy to SRT

Exporters should all consume the same `TranscriptDocument`.

---

## CLI Design

Use `Typer` or `Click`. `Typer` is a strong fit for typed options and testability.

Proposed commands:

```text
asr-cli transcribe <input>
asr-cli combine <file1> <file2> ...
asr-cli batch <folder>
asr-cli providers list
asr-cli doctor
```

Important options:

- `--output-dir`
- `--format txt`
- `--format json`
- `--format srt`
- `--format vtt`
- `--all-formats`
- `--recursive`
- `--normalize`
- `--asr gigaam`
- `--diarizer pyannote`
- `--llm deepseek`
- `--device cpu`
- `--backend onnx-cpu`
- `--longform auto`
- `--continue-on-error`
- `--config path/to/config.toml`
- `--workers N`
- `--log-level`

`doctor` should validate:

- `ffmpeg` and `ffprobe` in `PATH`
- optional provider imports
- API tokens present when required
- runtime hints for CPU-only deployment

---

## Error Handling and Batch Safety

Errors should be classified:

- input validation errors
- media preprocessing errors
- provider initialization errors
- inference errors
- export errors

Rules:

- independent jobs must never share failure state
- all recoverable exceptions become `JobResult(status="failed")`
- batch mode returns a summary at the end
- folder mode writes an error report file such as `batch_report.json`

Recommended summary fields:

- total files
- succeeded
- failed
- skipped
- total duration processed
- wall-clock processing time

---

## Configuration Model

Support config from:

1. CLI flags
2. config file
3. environment variables

Recommended env vars:

- `HF_TOKEN`
- `DEEPSEEK_API_KEY`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

Split config into:

- `AppConfig`
- `ASRConfig`
- `DiarizationConfig`
- `NormalizationConfig`
- `ExportConfig`

---

## Testing Strategy

### Unit Tests

Cover:

- provider registry and factory logic
- input resolution
- file type detection
- ffmpeg command construction
- timestamp offset logic for combined files
- merge/alignment logic
- format exporters
- config precedence
- CLI argument parsing

### Integration Tests

Use fake or mocked providers for deterministic testing:

- single file transcription flow
- combined files into one transcript with correct offsets
- batch folder where one file fails and others continue
- optional normalization stage
- mixed requested output formats

### Golden Tests

Keep snapshot-like expected outputs for:

- `txt`
- `json`
- `srt`
- `vtt`

### Live Tests

Keep real-provider tests behind markers such as:

- `@pytest.mark.live`
- `@pytest.mark.gigaam`
- `@pytest.mark.pyannote`

These should be excluded from default CI.

### CI

Recommended baseline CI:

- lint
- unit tests
- integration tests with fake providers

Windows CI should be added early because it is a primary target.

---

## Implementation Phases

### Phase 1. Project Skeleton

- initialize package
- add CLI entrypoint
- add base config and logging
- add test tooling

### Phase 2. Canonical Models and Core Pipeline

- define domain models
- define provider interfaces
- build orchestrator for single-job flow

### Phase 3. Media Layer

- implement `ffmpeg`/`ffprobe` wrapper
- normalize audio/video inputs
- add duration and metadata extraction

### Phase 4. Exporters

- implement `txt`, `json`, `srt`, `vtt`
- add golden tests

### Phase 5. Default Providers

- implement `GigaAMASRProvider`
- implement `PyannoteDiarizationProvider`
- implement `DeepSeekNormalizationProvider`

### Phase 6. Longform and Combined Files

- add automatic longform routing
- add combined timeline offset handling
- add large-file tests

### Phase 7. Batch Processing

- folder enumeration
- continue-on-error behavior
- reporting and summary

### Phase 8. Hardening

- provider diagnostics
- retry policy for API-backed normalization
- documentation
- Windows validation

---

## Key Technical Decisions

1. Use a provider-based architecture from the start.
2. Treat longform transcription as a first-class flow, not a later enhancement.
3. Keep CPU-first compatibility mandatory for the default stack.
4. Use one canonical transcript model for all exporters and post-processing.
5. Store structured metadata so failures and normalization remain auditable.
6. Build deterministic fake providers early to keep tests fast and reliable.

---

## Risks and Mitigations

### Risk: upstream provider instability on Windows

Mitigation:

- isolate provider adapters
- add `doctor` command
- support alternative ASR backends later without core rewrites

### Risk: long recordings consume too much RAM or take too long

Mitigation:

- use longform path by default for large files
- stream progress
- keep intermediate artifacts on disk, not only in memory

### Risk: normalization corrupts subtitle segmentation

Mitigation:

- normalize per segment
- keep raw text in JSON
- make normalized subtitle export configurable

### Risk: combined mode becomes hard to debug

Mitigation:

- preserve per-source offsets in metadata
- write clear manifest into JSON output

---

## Recommended First Deliverable

The first milestone should already be usable:

- single-file CPU transcription with GigaAM
- pyannote diarization
- `txt/json/srt/vtt` export
- fake-provider integration tests
- Windows-friendly setup docs
- Russian-first defaults for prompts, formatting rules, and model assumptions

After that:

- longform routing
- combined files
- batch mode
- optional DeepSeek normalization

---

## Finalized Product Decisions

1. Normalization is primarily for `txt` and `json`. For `srt` and `vtt` it is opt-in only.
2. Combined mode is fail-fast at the job level. If any source file fails, the whole combined run fails.
3. The product is Russian-first. Language-specific defaults may be generalized later through configuration without changing the core architecture.
