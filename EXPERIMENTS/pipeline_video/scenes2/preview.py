"""Low-quality preview of one chapter, plus one still per beat and a contact sheet.

    <video venv>/python scenes2/preview.py c05            # render 480p15, grab stills
    <video venv>/python scenes2/preview.py c05 --hq       # 1080p30 instead (slow)
    <video venv>/python scenes2/preview.py c05 --at 0.5   # stills at 50 % of each beat (default 0.92)

Writes renders2/preview/<cNN>/<beat>.png and renders2/preview/<cNN>/sheet.png (all stills tiled).
Run from anywhere; uses the manim next to this interpreter.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEO = HERE.parent
MANIM = str(Path(sys.executable).with_name('manim.exe'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('chapter')
    ap.add_argument('--hq', action='store_true')
    ap.add_argument('--at', type=float, default=0.92)
    a = ap.parse_args()
    mod, cls = a.chapter, f'Ch{a.chapter[1:]}'
    q, res = ('-qh', '1080p30') if a.hq else ('-ql', '480p15')
    media = VIDEO / 'renders2' / 'preview_media'
    subprocess.run([MANIM, 'render', q, '--disable_caching', '--media_dir', str(media), f'{mod}.py', cls],
                   cwd=HERE, check=True)
    mp4 = media / 'videos' / mod / res / f'{cls}.mp4'
    shutil.rmtree(mp4.parent / 'partial_movie_files', ignore_errors=True)
    log = json.loads((VIDEO / 'renders2' / 'beatlog' / f'{cls}.json').read_text(encoding='utf-8'))
    out = VIDEO / 'renders2' / 'preview' / mod
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    pngs = []
    for bid, t0, t1 in log:
        t = t0 + a.at * (t1 - t0)
        png = out / f'{bid}.png'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{t:.2f}', '-i', str(mp4), '-frames:v', '1', str(png)],
                       check=True)
        pngs.append(png)
    cols = 3
    rows = (len(pngs) + cols - 1) // cols
    inputs = sum((['-i', str(p)] for p in pngs), [])
    layout = '|'.join(f"{'+'.join(['w0'] * (i % cols)) or '0'}_{'+'.join(['h0'] * (i // cols)) or '0'}"
                      for i in range(len(pngs)))
    if len(pngs) == 1:
        shutil.copy(pngs[0], out / 'sheet.png')
    else:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex',
                        f'xstack=inputs={len(pngs)}:layout={layout}:fill=black', str(out / 'sheet.png')], check=True)
    print(f'{mp4}\n{len(pngs)} stills in {out} (sheet.png: {cols} x {rows})')
    print(json.dumps(log))


if __name__ == '__main__':
    main()
