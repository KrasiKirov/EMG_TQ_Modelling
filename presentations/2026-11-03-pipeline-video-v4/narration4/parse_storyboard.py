"""STORYBOARD4.md -> narration4/narration.json (the single source of the v2 narration).

    python narration4/parse_storyboard.py

`## cNN | Title` starts a chapter, `### cNN_bMM` a beat; inside a beat, `VIS:`, `SAY:` and `SRC:` start fields
that run until the next field or heading. Placeholders `[[...]]` left in SAY stop the script, so no beat is
synthesised with a missing number.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEO = HERE.parent
SPEC = dict(voice='af_sarah', speed=1.0, gap_s=0.6)


def parse(text):
    chapters, beats, ch, beat, field = [], [], None, None, None
    for line in text.splitlines():
        if line.startswith('## '):
            m = re.match(r'## (c\d\d) \| (.+)', line)
            if not m:
                continue
            ch = dict(id=m.group(1), title=m.group(2).strip(), beats=[])
            chapters.append(ch)
            beat = field = None
        elif line.startswith('### '):
            beat = dict(id=line[4:].strip(), chapter=ch['id'], vis='', text='', src='')
            beats.append(beat)
            ch['beats'].append(beat['id'])
            field = None
        elif beat is not None:
            m = re.match(r'(VIS|SAY|SRC): ?(.*)', line)
            if m:
                field = dict(VIS='vis', SAY='text', SRC='src')[m.group(1)]
                beat[field] = m.group(2).strip()
            elif field and line.strip() and not line.startswith('---'):
                beat[field] += ' ' + line.strip()
    return chapters, beats


def main():
    chapters, beats = parse((VIDEO / 'STORYBOARD4.md').read_text(encoding='utf-8'))
    left = [b['id'] for b in beats if '[[' in b['text']]
    if left:
        sys.exit(f'unfilled placeholders in SAY: {left}')
    out = dict(SPEC, chapters=chapters, beats=beats)
    (HERE / 'narration.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    words = sum(len(b['text'].split()) for b in beats)
    print(f'{len(chapters)} chapters, {len(beats)} beats, {words} words')


if __name__ == '__main__':
    main()
