"""Rebuild the explainer video.

    python build.py extract     # data/*.npz + data/facts_*.json  (.venv311; x06 trains for ~40 min)
    python build.py tts         # audio/<beat>.wav + audio/beats.json  (video venv, Kokoro am_adam)
    python build.py render      # one 1080p30 mp4 per chapter  (video venv, manim; 8 at a time)
    python build.py concat      # renders/EMG_torque_pipeline_explained.mp4 with chapter markers
    python build.py all

Interpreters: PROJECT_PY (TensorFlow, default .venv311) and VIDEO_PY (manim + kokoro, default
EMG_TQ_Modelling_video_venv). Set HF_HOME to keep the Kokoro model with the video venv.
"""
import json
import os
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
CHAPTERS = [('ch01_data', 'Ch01', 'The data'), ('ch02_problem', 'Ch02', 'The problem'),
            ('ch03_tree', 'Ch03', 'The pipeline in one picture'), ('ch04_preproc', 'Ch04', 'Preprocessing, step by step'),
            ('ch05_autogain', 'Ch05', 'EMG scaling and auto-gain'), ('ch06_passive', 'Ch06', 'Passive map and target'),
            ('ch07_windows', 'Ch07', 'Windows and splits'), ('ch08_balance', 'Ch08', 'Balanced positions'),
            ('ch09_lstm', 'Ch09', 'Inside the LSTM'), ('ch10_predict', 'Ch10', 'Prediction'),
            ('ch11_scores', 'Ch11', 'Evaluation, every number'), ('ch12_recap', 'Ch12', "Last week's pipeline, and recap")]
EXTRACT = ['code_excerpts', 'x01_data', 'x02_envelope', 'x03_scaling', 'x04_passive', 'x05_windows', 'x08_scores',
           'x09_last_week', 'x06_train_probe --mode probe', 'x07_lstm_window']
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', TF_CPP_MIN_LOG_LEVEL='2', PIP_NO_CACHE_DIR='1',
           HF_HOME=os.environ.get('HF_HOME', str(VIDEO_VENV / 'hf_home')))
OUT = HERE / 'renders' / 'EMG_torque_pipeline_explained.mp4'


def run(cmd, cwd):
    print('>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True)


def extract():
    for s in EXTRACT:
        name, *args = s.split()
        run([PROJECT_PY, f'{name}.py', *args], HERE / 'extract')


def tts():
    run([VIDEO_PY, 'narration/tts.py'], HERE)


def chapter_file(mod, cls):
    return HERE / 'renders' / 'media' / 'videos' / mod / '1080p30' / f'{cls}.mp4'


def render():
    def one(ch):
        mod, cls, _ = ch
        run([MANIM, 'render', '-qh', '--fps', '30', '--disable_caching', '--media_dir', '../renders/media',
             f'{mod}.py', cls], HERE / 'scenes')
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, CHAPTERS))


def duration(f):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(f)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def concat():
    files = [chapter_file(m, c) for m, c, _ in CHAPTERS]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f'missing chapter renders: {missing}')
    lst = HERE / 'renders' / 'concat.txt'
    lst.write_text(''.join(f"file '{f.as_posix()}'\n" for f in files), encoding='utf-8')
    meta, t, rows = [';FFMETADATA1', 'title=EMG to ankle torque: what the code actually does'], 0.0, []
    for (m, c, title), f in zip(CHAPTERS, files):
        d = duration(f)
        meta += ['[CHAPTER]', 'TIMEBASE=1/1000', f'START={int(t * 1000)}', f'END={int((t + d) * 1000)}', f'title={title}']
        rows.append((title, t, d))
        t += d
    (HERE / 'renders' / 'chapters.ffmeta').write_text('\n'.join(meta) + '\n', encoding='utf-8')
    run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(lst), '-i',
         str(HERE / 'renders' / 'chapters.ffmeta'), '-map_metadata', '1', '-map_chapters', '1', '-c', 'copy',
         '-movflags', '+faststart', str(OUT)], HERE)
    stamps = [f'| {k + 1} | {title} | {int(s // 60)}:{int(s % 60):02d} | {d / 60:.1f} min |' for k, (title, s, d) in enumerate(rows)]
    (HERE / 'renders' / 'chapters.md').write_text('\n'.join(stamps) + '\n', encoding='utf-8')
    print('\n'.join(stamps))
    print(f'total {t / 60:.1f} min -> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
    steps = {'extract': extract, 'tts': tts, 'render': render, 'concat': concat}
    for a in sys.argv[1:] or ['all']:
        for name in (steps if a == 'all' else [a]):
            steps[name]()
