import json

from asr_cli.core.models import TranscriptDocument, TranscriptSegment
from asr_cli.io.exporters.json import JsonWriter
from asr_cli.io.exporters.srt import SrtWriter
from asr_cli.io.exporters.txt import TxtWriter
from asr_cli.io.exporters.vtt import VttWriter


def test_exporters_write_expected_formats(workspace_tmp) -> None:
    document = TranscriptDocument(
        title='demo',
        language='ru',
        segments=[
            TranscriptSegment(
                start=0.0,
                end=1.2,
                text='hello',
                raw_text='hello',
                normalized_text='HELLO',
                speaker='SPEAKER_00',
            )
        ],
    )

    txt_path = TxtWriter().write(document, workspace_tmp / 'demo.txt')
    json_path = JsonWriter().write(document, workspace_tmp / 'demo.json')
    srt_path = SrtWriter().write(document, workspace_tmp / 'demo.srt')
    vtt_path = VttWriter().write(document, workspace_tmp / 'demo.vtt')

    assert txt_path.read_text(encoding='utf-8') == '[SPEAKER_00] HELLO\n'
    payload = json.loads(json_path.read_text(encoding='utf-8'))
    assert payload['segments'][0]['normalized_text'] == 'HELLO'
    assert '00:00:00,000 --> 00:00:01,200' in srt_path.read_text(encoding='utf-8')
    assert vtt_path.read_text(encoding='utf-8').startswith('WEBVTT')
