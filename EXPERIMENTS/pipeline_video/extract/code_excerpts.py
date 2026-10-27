"""Snapshot of every code excerpt shown in the video: path, line range, the lines, the file's SHA-256.

The D01 code is uncommitted in the dorsiflexion worktree, so the snapshot keeps the video traceable
if those files change. Scenes render data/code_excerpts.json only, never the source files.
Run with any Python (no imports beyond the standard library).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

from common import D01_WT, DATA, REPO

KERAS = Path(os.environ.get('KERAS_SRC', Path(sys.executable).parent.parent / 'Lib/site-packages/keras/src'))
ROOTS = {'d01': D01_WT, 'repo': REPO, 'keras': KERAS}
SHOWN = {'d01': '', 'repo': '', 'keras': 'keras/src/'}

EXCERPTS = {
    # key: (root, path, first line, last line)
    'data_load': ('d01', 'ML/research/data.py', 161, 162),
    'parse_comment': ('d01', 'ML/preprocessing/dataset_builder.py', 87, 100),
    'canonical_order': ('d01', 'ML/preprocessing/emg_envelope.py', 26, 26),
    'demean': ('d01', 'ML/preprocessing/emg_envelope.py', 122, 124),
    'bandpass': ('d01', 'ML/preprocessing/emg_envelope.py', 126, 130),
    'rectify': ('d01', 'ML/preprocessing/emg_envelope.py', 132, 133),
    'lowpass': ('d01', 'ML/preprocessing/emg_envelope.py', 135, 140),
    'decim': ('d01', 'ML/research/data.py', 62, 63),
    'trim_decimate': ('d01', 'ML/research/data.py', 94, 98),
    'env_max1k': ('d01', 'ML/research/data.py', 100, 103),
    'trainmax': ('d01', 'ML/research/emg_norm.py', 85, 91),
    'agc_auto': ('d01', 'ML/research/emg_norm.py', 96, 103),
    'push_dirs': ('d01', 'ML/research/emg_norm.py', 141, 151),
    'agc_consts': ('d01', 'ML/preprocessing/emg_agc.py', 25, 33),
    'causal_q': ('d01', 'ML/preprocessing/emg_agc.py', 36, 51),
    'priors': ('d01', 'ML/preprocessing/emg_agc.py', 86, 91),
    'agc_call': ('d01', 'ML/research/emg_norm.py', 104, 108),
    'agc_fn': ('d01', 'ML/research/emg_norm.py', 60, 70),
    'despike': ('d01', 'ML/preprocessing/emg_agc.py', 66, 72),
    'split_passive': ('d01', 'ML/preprocessing/dataset_builder.py', 156, 169),
    'still_test': ('d01', 'ML/preprocessing/active_torque.py', 64, 71),
    'map_merge': ('d01', 'ML/preprocessing/active_torque.py', 72, 80),
    'yes_table': ('d01', 'ML/config.py', 47, 50),
    'worksheet_override': ('d01', 'ML/research/data.py', 147, 149),
    'empty_map': ('d01', 'ML/research/passive.py', 45, 49),
    'targets': ('d01', 'ML/research/run.py', 70, 71),
    'map_by_session': ('d01', 'ML/research/passive.py', 100, 109),
    'lookup': ('d01', 'ML/preprocessing/dataset_builder.py', 269, 275),
    'run_exp': ('d01', 'EXPERIMENTS/code/run_experiment.py', 113, 123),
    'prepare_steps': ('d01', 'ML/research/run.py', 62, 73),
    'passive_fit_noop': ('d01', 'ML/research/passive.py', 141, 145),
    'models_map': ('d01', 'EXPERIMENTS/code/presentation_metrics.py', 54, 56),
    'windows': ('d01', 'ML/research/splits.py', 35, 41),
    'split': ('d01', 'ML/research/splits.py', 66, 72),
    'drop_last': ('d01', 'EXPERIMENTS/code/run_experiment.py', 65, 74),
    'lastweek_blocks': ('repo', 'ML/preprocessing/pipeline.py', 75, 85),
    'fit_call': ('d01', 'ML/research/train.py', 177, 181),
    'shuffle_tf': ('keras', 'trainers/data_adapters/array_data_adapter.py', 127, 134),
    'knots': ('d01', 'ML/research/train.py', 28, 39),
    'sd': ('d01', 'ML/research/train.py', 144, 152),
    'pred_scale': ('d01', 'ML/research/train.py', 182, 184),
    'builder': ('d01', 'EXPERIMENTS/code/run_experiment.py', 51, 61),
    'lstm_cell': ('keras', 'layers/rnn/lstm.py', 221, 228),
    'lstm_call': ('keras', 'layers/rnn/lstm.py', 277, 286),
    'seed': ('d01', 'ML/research/train.py', 136, 137),
    'seed_keras': ('keras', 'utils/rng_utils.py', 63, 66),
    'default_seed': ('keras', 'random/seed_generator.py', 145, 146),
    'train_step': ('keras', 'backend/tensorflow/trainer.py', 57, 68),
    'apply_grads': ('keras', 'backend/tensorflow/trainer.py', 78, 84),
    'nadam_u': ('keras', 'optimizers/nadam.py', 95, 105),
    'nadam': ('keras', 'optimizers/nadam.py', 114, 146),
    'protocol': ('d01', 'ML/research/config.py', 38, 41),
    'callbacks': ('d01', 'ML/research/train.py', 173, 176),
    'es_end': ('keras', 'callbacks/early_stopping.py', 123, 140),
    'rlr': ('keras', 'callbacks/reduce_lr_on_plateau.py', 105, 118),
    'pm_key': ('d01', 'EXPERIMENTS/code/presentation_metrics.py', 70, 78),
    'pm_metrics': ('d01', 'EXPERIMENTS/code/presentation_metrics.py', 81, 88),
    'pm_sample': ('d01', 'EXPERIMENTS/code/presentation_metrics.py', 107, 112),
    'dash_floor': ('repo', 'EXPERIMENTS/dashboard_react/src/lib/metrics.ts', 32, 35),
    'dash_score': ('repo', 'EXPERIMENTS/dashboard_react/src/tabs/ExperimentTab.tsx', 29, 34),
    'dash_toes': ('repo', 'EXPERIMENTS/dashboard_react/src/lib/data.ts', 91, 94),
    'dash_quantile': ('repo', 'EXPERIMENTS/dashboard_react/src/lib/stats.ts', 3, 11),
    'dash_group': ('repo', 'EXPERIMENTS/dashboard_react/src/lib/aggregate.ts', 46, 57),
    'lw_config': ('repo', 'ML/preprocessing/pipeline.py', 17, 30),
    'lw_train': ('repo', 'ML/training/benchmark.py', 69, 78),
    'lw_args': ('repo', 'ML/training/parallel_benchmark.py', 28, 29),
}

out, hashes = {}, {}
for key, (root, rel, a, b) in EXCERPTS.items():
    path = ROOTS[root] / rel
    raw = path.read_bytes()
    lines = raw.decode('utf-8').splitlines()
    if b > len(lines):
        raise SystemExit(f'{key}: {rel} has only {len(lines)} lines')
    hashes[rel] = hashlib.sha256(raw).hexdigest()
    out[key] = dict(root=root, path=SHOWN[root] + rel, first=a, last=b, lines=lines[a - 1:b], sha256=hashes[rel])
DATA.mkdir(exist_ok=True)
(DATA / 'code_excerpts.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'{len(out)} excerpts from {len(hashes)} files')
