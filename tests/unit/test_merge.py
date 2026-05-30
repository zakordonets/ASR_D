from __future__ import annotations

from asr_cli.core.models import SpeakerTurn, TranscriptDocument, TranscriptSegment, TranscriptWord
from asr_cli.pipeline.merge import assign_speakers


def test_assign_speakers_matches_by_overlap() -> None:
    doc = TranscriptDocument(
        title='test',
        language='ru',
        segments=[
            TranscriptSegment(start=0.0, end=2.0, text='hello'),
            TranscriptSegment(start=2.0, end=4.0, text='world'),
        ],
    )
    turns = [
        SpeakerTurn(start=0.0, end=1.5, speaker='SPEAKER_00'),
        SpeakerTurn(start=1.5, end=5.0, speaker='SPEAKER_01'),
    ]

    result = assign_speakers(doc, turns)

    assert result.segments[0].speaker == 'SPEAKER_00'
    assert result.segments[1].speaker == 'SPEAKER_01'
    assert result.speaker_turns == turns


def test_assign_speakers_propagates_to_words() -> None:
    doc = TranscriptDocument(
        title='test',
        language='ru',
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text='hello world',
                words=[
                    TranscriptWord(start=0.0, end=1.0, text='hello'),
                    TranscriptWord(start=1.0, end=2.0, text='world'),
                ],
            ),
        ],
    )
    turns = [SpeakerTurn(start=0.0, end=2.0, speaker='SPEAKER_00')]

    result = assign_speakers(doc, turns)

    assert result.segments[0].words[0].speaker == 'SPEAKER_00'
    assert result.segments[0].words[1].speaker == 'SPEAKER_00'


def test_assign_speakers_empty_turns_returns_unchanged() -> None:
    doc = TranscriptDocument(
        title='test',
        language='ru',
        segments=[TranscriptSegment(start=0.0, end=1.0, text='hi')],
    )

    result = assign_speakers(doc, [])

    assert result is doc
    assert result.segments[0].speaker is None


def test_assign_speakers_no_overlap_returns_none() -> None:
    doc = TranscriptDocument(
        title='test',
        language='ru',
        segments=[TranscriptSegment(start=10.0, end=12.0, text='far away')],
    )
    turns = [SpeakerTurn(start=0.0, end=1.0, speaker='SPEAKER_00')]

    result = assign_speakers(doc, turns)

    assert result.segments[0].speaker is None


def test_assign_speakers_does_not_mutate_input() -> None:
    original_segment = TranscriptSegment(start=0.0, end=2.0, text='test')
    doc = TranscriptDocument(
        title='test',
        language='ru',
        segments=[original_segment],
    )
    turns = [SpeakerTurn(start=0.0, end=2.0, speaker='SPEAKER_00')]

    result = assign_speakers(doc, turns)

    assert original_segment.speaker is None
    assert result.segments[0].speaker == 'SPEAKER_00'
    assert result.segments is not doc.segments


def test_assign_speakers_multiple_turns_picks_best() -> None:
    doc = TranscriptDocument(
        title='test',
        language='ru',
        segments=[TranscriptSegment(start=0.0, end=5.0, text='long segment')],
    )
    turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker='SPEAKER_00'),   # overlap 1.0
        SpeakerTurn(start=0.0, end=4.0, speaker='SPEAKER_01'),   # overlap 4.0
    ]

    result = assign_speakers(doc, turns)

    assert result.segments[0].speaker == 'SPEAKER_01'
