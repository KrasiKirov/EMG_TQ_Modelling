"""
Cross-subject pre-training (transfer learning), leave-one-subject-out.

Why
---
At the most plantarflexed angles a subject's training session contains only
one 90 s trial with a very small torque range (YES: about 2 Nm), so the
network sees little evidence of how weak EMG maps to weak torque. The
other subjects contain many more such examples. Pre-training on other people
and then fine-tuning on the target is the standard transfer-learning recipe
for EMG models that have little subject-specific data.

Protocol for target subject T (no data of T is used before fine-tuning)
------------------------------------------------------------------------
1. Source set = every OTHER subject in config.ALL_SUBJECTS, both sessions,
   all active trials.  Each source SESSION is treated like a training
   session of its own:
     * EMG normalised with that session's statistics (same ``emg`` kind
       as the configuration),
     * target = measured torque - passive model (same ``passive`` kind)
       fitted on that session's EMG-detected rest periods,
     * first 80 % of each trial -> pre-training set, last 20 % -> its
       validation set (early stopping).
   Windows are taken with a stride of 2 samples (neighbouring windows
   overlap by 95 %, so this halves the cost and loses almost nothing).
   Loss weight per window = 1 / variance of its trial's target, so that every
   trial, and therefore every angle, weighs the same (as balance='weights').
2. Pre-train the configuration's model on the source set.
3. Fine-tune ALL weights on T's training session with the normal
   train.fit_predict (same balance, same early stopping), starting from the
   pre-trained weights, at a lower learning rate (FINETUNE_LR).
4. Test on T's other session, exactly like every other configuration.

The pre-trained weights depend only on (T, configuration, seed), so both folds
of T share them.  They are cached in CACHE_DIR/pretrain/; a lock file stops
two processes from pre-training the same model at the same time.
"""

import os
import time

import numpy as np

from . import config as C
from .data import load_subject
from .emg_norm import normalise
from .passive import PassiveModel, rest_segments
from .splits import windows
from .models import build as build_model

PRETRAIN = dict(batch=256, lr=0.003, lr_factor=0.5, lr_patience=2, early=5, epochs=40, stride=2)
FINETUNE_LR = 0.001
LOCK_STALE_S = 2 * 3600   # a pre-training lock older than this belongs to a dead process


def source_session_data(subject, session, cfg, stride=PRETRAIN['stride']):
    """Windows, targets, trial weights and split flags of ONE source session."""
    bundle = load_subject(subject)
    recs = [r for r in bundle['active'] if r['session'] == session]
    feats, _ = normalise(cfg['emg'], bundle, recs, recs)
    base_env, _ = normalise('trainmax', bundle, recs, recs)
    tr_segs, va_segs = rest_segments(recs, base_env)
    pm = PassiveModel(cfg['passive'], bundle).fit(recs, tr_segs, va_segs)
    X, y, w, is_val = [], [], [], []
    for r in recs:
        Xr, yr, _ = windows(feats[r['idx']], r['pos'], r['tq'] - pm.predict(r))
        n_tr = int(len(Xr) * C.TRAIN_FRAC)
        sel = np.arange(0, len(Xr), stride)
        X.append(Xr[sel]); y.append(yr[sel])
        w.append(np.full(len(sel), 1.0 / max(float(np.var(yr[:n_tr])), 1e-3)))
        is_val.append(sel >= n_tr)
    return np.concatenate(X), np.concatenate(y), np.concatenate(w), np.concatenate(is_val)


def source_set(target, cfg):
    """Concatenate every other subject's sessions (see module docstring)."""
    parts = [source_session_data(s, sess, cfg) for s in C.ALL_SUBJECTS if s != target for sess in ('S1', 'S2')]
    X, y, w, v = (np.concatenate(z) for z in zip(*parts))
    w = w / w[~v].mean()
    return X, y, w, v


def _path(target, cfg, seed):
    tag = f"{target}_{cfg['passive']}_{cfg['emg']}_{cfg['model']}_{cfg['balance']}_s{seed}".replace('+', '-')
    return os.path.join(C.CACHE_DIR, 'pretrain', tag + '.weights.h5')


def pretrained_model_path(target, cfg, seed, n_steps=C.WINDOW, n_features=5):
    """Path of the pre-trained weights for (target, cfg, seed); trains them first
    if they do not exist yet (with a lock against duplicate work)."""
    path = _path(target, cfg, seed)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = path + '.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:                       # someone else is pre-training it
        while not os.path.exists(path):
            if not os.path.exists(lock):
                return pretrained_model_path(target, cfg, seed, n_steps, n_features)
            if time.time() - os.path.getmtime(lock) > LOCK_STALE_S:   # owner died: take over
                try:
                    os.remove(lock)
                except OSError:
                    pass
                return pretrained_model_path(target, cfg, seed, n_steps, n_features)
            time.sleep(10)
        return path
    try:
        _pretrain(target, cfg, seed, path, n_steps, n_features)
    finally:
        os.close(fd)
        os.remove(lock)
    return path


def _pretrain(target, cfg, seed, path, n_steps, n_features):
    import tensorflow as tf
    from tensorflow.keras.optimizers import Nadam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

    if cfg['model'].split('+')[0] != 'lstm8':
        raise ValueError('cross-subject pre-training is implemented for lstm8 models only '
                         '(the MoE head is tied to the target subject\'s angles)')
    tf.keras.utils.set_random_seed(seed)
    X, y, w, v = source_set(target, cfg)
    p = PRETRAIN
    model = build_model(cfg['model'], n_steps, n_features, knots=None)
    model.compile(optimizer=Nadam(learning_rate=p['lr']), loss='mse')
    hist = model.fit(X[~v], y[~v], sample_weight=w[~v].astype(np.float32),
                     validation_data=(X[v], y[v], w[v].astype(np.float32)),
                     epochs=p['epochs'], batch_size=p['batch'], shuffle=True, verbose=0,
                     callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=p['lr_factor'],
                                                  patience=p['lr_patience'], min_lr=1e-5),
                                EarlyStopping(monitor='val_loss', patience=p['early'],
                                              restore_best_weights=True)])
    tmp = path + '.tmp.weights.h5'
    model.save_weights(tmp)
    os.replace(tmp, path)
    with open(path[:-len('.weights.h5')] + '.txt', 'w') as f:
        f.write(f"sources={[s for s in C.ALL_SUBJECTS if s != target]} n_train={int((~v).sum())} "
                f"n_val={int(v.sum())} epochs={len(hist.history['loss'])} "
                f"best_val={min(hist.history['val_loss']):.4f}\n")
