"""
Run ONE experiment configuration for one subject and one seed, trained on session 1 and tested on session 2.

    python -m EXPERIMENTS.pipeline.run --exp FE_A0 --subject JM \
        --seed 0 --threads 1

The configuration is read from EXPERIMENTS/configs/<exp>.json.  Every
configuration starts from the LSTM-8 baseline and lists only what it changes
(see configs/X00f_baseline_framework.json for the full baseline definition).

Data building, passive-torque models, EMG normalisation and the training loop come from ML/research, which
builds the same data as the frozen original pipeline. As in the original split, the last validation window of
every training trial is dropped. Training settings default to the original ones (Nadam 0.003, batch 8,
200 epochs, ReduceLROnPlateau 0.7/10, EarlyStopping 25).

Outputs: EXPERIMENTS/results/<exp>/<subject>/<direction>/seed<k>.{npz,json}
Runs of a pre-registered study (EXPERIMENTS/protocol/protocol.STUDIES; study v2: ids TS2_ / FE_) write to the
study's folder (EXPERIMENTS/results_v2/) instead, refuse to start on uncommitted code, and record the hash of the
study's protocol file. Every run records its provenance and config hash. --out-root DIR writes this run under DIR
instead (checks; the study's rules still apply).

Settings: settings.py. Model building and fitting: fitters.py. Saving: outputs.py.
"""
import argparse
import os
import sys
import time

import numpy as np

from EXPERIMENTS import paths as P
from EXPERIMENTS.pipeline.fitters import add_passive_input, drop_last_val_window, fit
from EXPERIMENTS.pipeline.settings import BASE, DIRECTION, load_config  # noqa: F401  (re-exported)
from EXPERIMENTS.protocol.protocol import STUDIES, study_of

RESULTS = str(P.RESULTS)
RESULTS_V2 = str(P.RESULTS_V2)
RCFG_KEYS = ('passive', 'emg', 'window', 'envelope', 'trim_end_s', 'state', 'lp_order',
             'val_gap', 'metadata_passive', 'scale_part')


def is_study(exp):
    return study_of(exp) is not None


def results_root(exp):
    """Study runs live apart from v1, so v1 scripts never read them."""
    st = study_of(exp)
    if st is None:
        return RESULTS
    return RESULTS_V2 if st.results == P.RESULTS_V2 else str(st.results)


def protocol_hash(protocol_file=STUDIES['study_v2'].protocol_file):
    from EXPERIMENTS.pipeline.provenance import sha256_file
    h = sha256_file(str(protocol_file))
    if h is None:
        raise SystemExit(f'study runs need the protocol file {protocol_file}')
    return h


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--save-weights', action='store_true', help='also save the trained weights next to the results')
    ap.add_argument('--out-root', default=None,
                    help='write this run under DIR/<exp>/... instead of the results folder (checks)')
    a = ap.parse_args(argv)
    a.direction = DIRECTION          # always session 1 → session 2; not an option

    from EXPERIMENTS.pipeline import provenance as PV
    spec, cfg = load_config(a.exp)
    st = study_of(a.exp)
    prov = PV.collect(threads=a.threads, protocol_hash=protocol_hash(st.protocol_file) if st else None,
                      subject=a.subject, spec_path=os.path.join(str(P.CONFIGS), f'{a.exp}.json'))
    if st and prov['git_dirty']:
        raise SystemExit(f'{a.exp}: study runs refuse uncommitted code (ML/, EXPERIMENTS/pipeline|protocol|tuning, '
                         'or the spec itself); commit first')

    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(a.threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, a.threads))
    except RuntimeError:          # TensorFlow already running in this process (tests): keep its settings
        pass

    from ML.research.run import prepare

    t0 = time.time()
    data, details = prepare(a.subject, a.direction, {k: cfg[k] for k in RCFG_KEYS})
    if st:
        from ML.research.splits import check_common_targets
        try:
            check_common_targets(data['train_recs'])
        except ValueError as e:
            raise SystemExit(f'{a.subject}: {e}')
    data = drop_last_val_window(data)
    if cfg.get('extra_input') == 'passive_trial_value':
        data = add_passive_input(data, a.subject)
    t_prep = time.time() - t0
    out = os.path.join(a.out_root or results_root(a.exp), a.exp, a.subject, a.direction)
    os.makedirs(out, exist_ok=True)
    pred, info, data = fit(data, cfg, a.seed,
                           save_weights=os.path.join(out, f'seed{a.seed}.weights.h5') if a.save_weights else None,
                           protocol_file=st.protocol_file if st else STUDIES['study_v2'].protocol_file)
    t_train = time.time() - t0 - t_prep

    val_pred = info.pop('val_pred')
    res = {'experiment': a.exp, 'subject': a.subject, 'direction': a.direction, 'seed': a.seed,
           'config': cfg, 'hypothesis': spec.get('hypothesis'), 'details': details,
           'train_info': {k: v for k, v in info.items() if k != 'init_weights'},
           'n_windows': {p: int(len(data[f'y_{p}'])) for p in ('train', 'val', 'test')},
           'prep_s': t_prep, 'train_s': t_train, 'config_hash': PV.config_hash(cfg), 'provenance': prov,
           'versions': {'tensorflow': tf.__version__, 'numpy': np.__version__, 'python': sys.version.split()[0]}}
    from EXPERIMENTS.pipeline.outputs import write_run
    write_run(os.path.join(out, f'seed{a.seed}'), data, pred, val_pred, res)
    print(f'[{a.exp} {a.subject} {a.direction} s{a.seed}] {info["epochs"]} ep (best {info.get("best_epoch", "-")}) {t_train:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
