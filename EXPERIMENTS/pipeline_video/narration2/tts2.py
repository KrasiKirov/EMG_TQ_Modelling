"""v2 narration -> one wav per beat with Kokoro (KPipeline lang_code='a', voice af_sarah, speed 1.0, 24 kHz).

    python narration2/tts2.py            # synthesise beats whose spoken text changed
    python narration2/tts2.py --force    # all beats
    python narration2/tts2.py --only c01_b01 c01_b02

Reads narration2/narration.json (built by parse_storyboard.py), applies the pronunciation pass
(narration/pronounce.py plus the v2 additions below) and writes audio2/<beat>.wav (16-bit) and
audio2/beats.json (text, spoken text, phonemes, duration). Run in the video venv with HF_HOME set.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
VIDEO = HERE.parent
AUDIO = VIDEO / 'audio2'
sys.path.insert(0, str(VIDEO / 'narration'))
from pronounce import speak as speak_v1                       # noqa: E402

SR = 24000
# v2 tokens the v1 pass does not know (whole words, case-sensitive), applied before it
EXTRA = [('X00f', 'X zero zero F'), ('X00', 'X zero zero'), ('X02', 'X zero two'), ('X04', 'X zero four'),
         ('X05', 'X zero five'), ('X06', 'X zero six'), ('D02', 'D zero two'), ('MAE', 'M A E'), ('RMSE', 'R M S E'),
         ('Nadam', 'N-Adam'), ('vs', 'versus'),
         ('2019', 'twenty nineteen'), ('2020', 'twenty twenty'), ('2022', 'twenty twenty-two')]


def speak(text):
    for a, b in EXTRA:
        text = re.sub(rf'(?<![\w\[/]){re.escape(a)}(?![\w\]])', b, text)
    text = text.replace('"', '').replace('“', '').replace('”', '')
    # the v1 number pattern swallows a comma that follows a number ("session 1, to"): drop thousands
    # separators here, then keep every other comma out of the number token
    text = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', text)
    text = re.sub(r'(\d),', r'\1 ,', text)
    return speak_v1(text).replace(' ,', ',')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    spec = json.loads((HERE / 'narration.json').read_text(encoding='utf-8'))
    AUDIO.mkdir(exist_ok=True)
    index_path = AUDIO / 'beats.json'
    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {}
    from kokoro import KPipeline
    pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    for b in spec['beats']:
        if a.only and b['id'] not in a.only:
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
    ids = {b['id'] for b in spec['beats']}
    total = sum(v['seconds'] for k, v in index.items() if k in ids)
    print(f'{len(ids)} beats, {total / 60:.1f} min of narration (+ {len(ids) * spec["gap_s"] / 60:.1f} min of gaps)')


if __name__ == '__main__':
    main()
