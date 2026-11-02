"""
Zero-anchored models:  ŷ = F(emg, pos) − F(emg_rest, pos)

Physiological prior: with the muscles silent there is no ACTIVE torque, at
every ankle angle.  The baseline LSTM has to learn "rest → 0" separately for
each position and, under MSE, barely bothers at low-torque positions (YES
−0.486 has 0.4 Nm std vs 15 Nm elsewhere), leaving a ~0.5 Nm rest bias that
alone makes R² negative there.  Anchoring makes that bias structurally zero.

``emg_rest`` is all-zero EMG, so pair these models with an input plugin that
maps rest to 0 (core.floor_subtract).  Extra features (position one-hot etc.)
are kept in the anchor copy.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def floor_train(trainval, retest, ctx):
    """[CLEAN] Subtract a per-channel EMG noise floor estimated from the
    TRAINING part of the test session only (pooled 5th percentile), clip ≥ 0.
    Unlike core.floor_subtract it uses no statistics of the retest EMG, so
    'rest' maps to ≈0 for the anchored models without being transductive."""
    from ML.preprocessing import dataset_builder as db
    q = ctx['opts'].get('params', {}).get('floor_q', 5.0)
    parts = [r['env'][:int(len(r['env']) * db.TRAIN_FRAC)] for r in trainval]
    floor = np.percentile(np.concatenate(parts), q, axis=0)
    ctx['emg_floor_train'] = floor
    for r in trainval + retest:
        r['env'] = np.clip(r['env'] - floor, 0.0, None)


def _rest_copy(x):
    return tf.concat([tf.zeros_like(x[:, :, :4]), x[:, :, 4:]], axis=-1)


def _core_lstm(n_steps, n_features, units, dropout):
    inp = layers.Input(shape=(n_steps, n_features))
    h = layers.LSTM(units, return_sequences=True)(inp)
    h = layers.Dropout(dropout)(h)
    h = layers.LSTM(units)(h)
    h = layers.Dropout(dropout)(h)
    return Model(inp, layers.Dense(1)(h), name='core')


def anchored_lstm(n_steps, n_features, units=8, dropout=0.3, **kw):
    """Baseline-sized LSTM (8-8) with zero anchoring."""
    core = _core_lstm(n_steps, n_features, units, dropout)
    inp = layers.Input(shape=(n_steps, n_features))
    rest = layers.Lambda(_rest_copy)(inp)
    out = layers.Subtract()([core(inp), core(rest)])
    return Model(inp, out, name='anchored_lstm')


def anchored_lstm16(n_steps, n_features, **kw):
    return anchored_lstm(n_steps, n_features, units=16, dropout=0.2)


def anchored_gain(n_steps, n_features, units=8, dropout=0.3, **kw):
    """Anchored + explicit position gain:  ŷ = g(pos) · [F(emg,pos) − F(0,pos)]
    with g = softplus(MLP(pos)) > 0, so the network can scale its output down
    at weak (plantarflexed) angles without having to shrink its features."""
    core = _core_lstm(n_steps, n_features, units, dropout)
    inp = layers.Input(shape=(n_steps, n_features))
    rest = layers.Lambda(_rest_copy)(inp)
    act = layers.Subtract()([core(inp), core(rest)])
    pos = layers.Lambda(lambda x: x[:, -1, 4:5])(inp)
    g = layers.Dense(8, activation='tanh')(pos)
    g = layers.Dense(1, activation='softplus')(g)
    return Model(inp, layers.Multiply()([act, g]), name='anchored_gain')
