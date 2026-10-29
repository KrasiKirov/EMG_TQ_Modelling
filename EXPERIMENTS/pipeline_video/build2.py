"""Rebuild the v2 explainer video (the baseline, in plain English).

    python build2.py extract     # data2/*.npz + data2/facts_*.json  (.venv311; no training)
    python build2.py narration   # STORYBOARD2.md -> narration2/narration.json
    python build2.py tts         # audio2/<beat>.wav + audio2/beats.json  (video venv, Kokoro af_sarah)
    python build2.py render      # one 1080p30 mp4 per chapter  (video venv, manim; at most 8 at a time)
    python build2.py concat      # renders2/EMG_torque_baseline_explained.mp4 with chapter markers
    python build2.py all
    python build2.py render c05 c12   # only some chapters

Interpreters: PROJECT_PY (default .venv311) and VIDEO_PY (manim + kokoro, default EMG_TQ_Modelling_video_venv).
"""
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECTS = Path(os.environ.get('KRASI_ML', r'C:/Users/dramo/Desktop/Projects/krasi_ml'))
PROJECT_PY = os.environ.get('PROJECT_PY', str(PROJECTS / 'EMG_TQ_Modelling/.venv311/Scripts/python.exe'))
VIDEO_VENV = Path(os.environ.get('VIDEO_VENV', PROJECTS / 'EMG_TQ_Modelling_video_venv'))
VIDEO_PY = str(VIDEO_VENV / 'Scripts/python.exe')
MANIM = str(VIDEO_VENV / 'Scripts/manim.exe')
EXTRACT = ['y01_baseline_scores', 'y02_inputs', 'y03_predictions', 'y04_learning', 'y05_passive', 'y06_techniques',
           'y07_reference_checks', 'c04_trim']
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           EMG_DATA_DIR=os.environ.get('EMG_DATA_DIR', str(PROJECTS / 'EMG_TQ_Modelling/ML/subjects')),
           HF_HOME=os.environ.get('HF_HOME', str(VIDEO_VENV / 'hf_home')))
RENDERS = HERE / 'renders2'
OUT = RENDERS / 'EMG_torque_baseline_explained.mp4'


def chapters():
    spec = json.loads((HERE / 'narration2' / 'narration.json').read_text(encoding='utf-8'))
    return [(c['id'], f"Ch{c['id'][1:]}", c['title']) for c in spec['chapters']]   # (module, scene class, title)


def run(cmd, cwd):
    print('>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True)


def extract(_=None):
    for name in EXTRACT:
        run([PROJECT_PY, f'{name}.py'], HERE / 'extract2')


def narration(_=None):
    run([VIDEO_PY, 'narration2/parse_storyboard.py'], HERE)


def tts(_=None):
    run([VIDEO_PY, 'narration2/tts2.py'], HERE)


def chapter_file(mod, cls):
    return RENDERS / 'media' / 'videos' / mod / '1080p30' / f'{cls}.mp4'


def render(only=None):
    todo = [c for c in chapters() if not only or c[0] in only]

    def one(ch):
        mod, cls, _ = ch
        run([MANIM, 'render', '-qh', '--fps', '30', '--disable_caching', '--media_dir', '../renders2/media',
             f'{mod}.py', cls], HERE / 'scenes2')
        partial = RENDERS / 'media' / 'videos' / mod / '1080p30' / 'partial_movie_files'
        shutil.rmtree(partial, ignore_errors=True)                 # the C: drive is small
    with ThreadPoolExecutor(max_workers=int(os.environ.get('RENDER_JOBS', 8))) as ex:
        list(ex.map(one, todo))


def duration(f):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(f)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def concat(_=None):
    chs = chapters()
    files = [chapter_file(m, c) for m, c, _ in chs]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f'missing chapter renders: {missing}')
    lst = RENDERS / 'concat.txt'
    lst.write_text(''.join(f"file '{f.as_posix()}'\n" for f in files), encoding='utf-8')
    meta, t, rows = [';FFMETADATA1', 'title=EMG to ankle torque: the baseline, in plain English'], 0.0, []
    for (m, c, title), f in zip(chs, files):
        d = duration(f)
        meta += ['[CHAPTER]', 'TIMEBASE=1/1000', f'START={int(t * 1000)}', f'END={int((t + d) * 1000)}', f'title={title}']
        rows.append((title, t, d))
        t += d
    (RENDERS / 'chapters.ffmeta').write_text('\n'.join(meta) + '\n', encoding='utf-8')
    run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(lst), '-i',
         str(RENDERS / 'chapters.ffmeta'), '-map_metadata', '1', '-map_chapters', '1', '-c', 'copy',
         '-movflags', '+faststart', str(OUT)], HERE)
    stamps = [f'| {k + 1} | {title} | {int(s // 60)}:{int(s % 60):02d} | {d / 60:.1f} min |'
              for k, (title, s, d) in enumerate(rows)]
    (RENDERS / 'chapters.md').write_text('\n'.join(stamps) + '\n', encoding='utf-8')
    print('\n'.join(stamps))
    print(f'total {t / 60:.1f} min -> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
    steps = {'extract': extract, 'narration': narration, 'tts': tts, 'render': render, 'concat': concat}
    args = sys.argv[1:] or ['all']
    if args[0] == 'render' and len(args) > 1:
        render(set(args[1:]))
    else:
        for a in args:
            for name in (steps if a == 'all' else [a]):
                steps[name]()
