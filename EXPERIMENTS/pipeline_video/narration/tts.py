"""Narration -> one wav per beat with Kokoro (KPipeline lang_code='a', voice am_adam, 24 kHz).

    python narration/tts.py            # synthesise beats whose spoken text changed
    python narration/tts.py --force    # all beats

Fills each beat's {placeholders} from values(), applies the pronunciation pass, synthesises, and
writes audio/<beat>.wav (16-bit) plus audio/beats.json (filled text, spoken text, phonemes,
duration). Run in the video venv; set HF_HOME to keep the Kokoro model with that venv.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
VIDEO = HERE.parent
AUDIO = VIDEO / 'audio'
sys.path.insert(0, str(HERE))
from pronounce import speak                                   # noqa: E402
from values import values                                     # noqa: E402

SR = 24000


def filled_beats():
    spec = json.loads((HERE / 'narration.json').read_text(encoding='utf-8'))
    V = values()
    out = []
    for b in spec['beats']:
        try:
            text = b['text'].format(**V)
        except KeyError as e:
            text, missing = None, str(e)
        else:
            missing = None
        out.append(dict(id=b['id'], template=b['text'], text=text, missing=missing))
    return spec, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    spec, beats = filled_beats()
    AUDIO.mkdir(exist_ok=True)
    index_path = AUDIO / 'beats.json'
    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {}
    missing = [b['id'] + ': ' + b['missing'] for b in beats if b['missing']]
    if missing:
        print('beats with values not yet extracted (skipped):', missing)
    from kokoro import KPipeline
    pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    for b in beats:
        if b['text'] is None or (a.only and b['id'] not in a.only):
            continue
        spoken = speak(b['text'])
        key = hashlib.sha256(f"{spec['voice']}|{spec['speed']}|{spoken}".encode()).hexdigest()[:16]
        wav = AUDIO / f"{b['id']}.wav"
        if not a.force and wav.exists() and index.get(b['id'], {}).get('key') == key:
            continue
        chunks, phonemes = [], []
        for gs, ps, audio in pipe(spoken, voice=spec['voice'], speed=spec['speed']):
            chunks.append(np.asarray(audio))
            phonemes.append(ps)
        y = np.concatenate(chunks)
        sf.write(wav, y, SR, subtype='PCM_16')
        index[b['id']] = dict(key=key, text=b['text'], spoken=spoken, phonemes=' | '.join(phonemes),
                              seconds=len(y) / SR)
        print(f"{b['id']}: {len(y) / SR:5.1f} s  {spoken[:90]}", flush=True)
        index_path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding='utf-8')
    total = sum(v['seconds'] for v in index.values())
    print(f'{len(index)} beats, {total / 60:.1f} min of narration')


if __name__ == '__main__':
    main()
