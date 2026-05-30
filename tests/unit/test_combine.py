from __future__ import annotations

from pathlib import Path

from asr_cli.core.models import (
    SourceOffset,
    SpeakerTurn,
    TranscriptDocument,
    TranscriptSegment,
    TranscriptWord,
)
from asr_cli.pipeline.combine import combine_documents


def test_combine_offsets_segments() -> None:
    doc1 = TranscriptDocument(
        title='first',
        language='ru',
        segments=[
            TranscriptSegment(start=0.0, end=1.0, text='hello'),
            TranscriptSegment(start=1.0, end=2.0, text='world'),
        ],
    )
    doc2 = TranscriptDocument(
        title='second',
        language='ru',
        segments=[
            TranscriptSegment(start=0.0, end=1.5, text='foo'),
        ],
    )

    result = combine_documents([doc1, doc2], [Path('a.wav'), Path('b.wav')])

    assert len(result.segments) == 3
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 1.0
    assert result.segments[1].start == 1.0
    assert result.segments[1].end == 2.0
    assert result.segments[2].start == 2.0
    assert result.segments[2].end == 3.5


def test_combine_offsets_words() -> None:
    doc = TranscriptDocument(
        title='test',
        language='ru',
        segments=[
            TranscriptSegment(
                start=0.0,
                end=1.0,
                text='hi',
                words=[TranscriptWord(start=0.0, end=0.5, text='hi')],
            ),
        ],
    )

    result = combine_documents([doc, doc], [Path('a.wav'), Path('b.wav')])

    assert result.segments[1].words[0].start == 1.0
    assert result.segments[1].words[0].end == 1.5


def test_combine_preserves_speaker_turns() -> None:
    doc = TranscriptDocument(
        title='test',
        language='ru',
        segments=[TranscriptSegment(start=0.0, end=2.0, text='test')],
        speaker_turns=[SpeakerTurn(start=0.0, end=2.0, speaker='SPEAKER_00')],
    )

    result = combine_documents([doc, doc], [Path('a.wav'), Path('b.wav')])

    assert len(result.speaker_turns) == 2
    assert result.speaker_turns[0].start == 0.0
    assert result.speaker_turns[1].start == 2.0


def test_combine_source_offsets() -> None:
    doc1 = TranscriptDocument(
        title='first',
        language='ru',
        segments=[TranscriptSegment(start=0.0, end=3.0, text='a')],
    )
    doc2 = TranscriptDocument(
        title='second',
        language='ru',
        segments=[TranscriptSegment(start=0.0, end=2.0, text='b')],
    )

    result = combine_documents(
        [doc1, doc2], [Path('a.wav'), Path('b.wav')], title='merged'
    )

    assert result.title == 'merged'
    assert len(result.source_offsets) == 2
    assert result.source_offsets[0].path == Path('a.wav')
    assert result.source_offsets[0].start_offset == 0.0
    assert result.source_offsets[1].path == Path('b.wav')
    assert result.source_offsets[1].start_offset == 3.0


def test_combine_empty_segments() -> None:
    doc = TranscriptDocument(
        title='empty',
        language='ru',
        segments=[],
    )

    result = combine_documents([doc], [Path('a.wav')])

    assert result.segments == []
    assert result.title == 'combined'
