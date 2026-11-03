"""Open the trained final-model LSTM (V0 = N0 settings, EG, seed 0, weights saved) and look inside.

    python extract4/w02_inside.py        # .venv311, PYTHONPATH = repository root, EMG_DATA_DIR set

1. Recompute the LSTM by hand (numpy, Keras gate order i, f, c, o) for windows of EG's session-2 trial and check
   that it gives the same output as the saved model.
2. Record, for one window at the start of a push, the 20 steps of layer 1: input, forget / input / output gates,
   the 8-number cell memory and the 8-number output.
3. Order test: predict every window of that trial in time order, and again with the windows shuffled, then put
   the predictions back by time stamp; report the largest difference.
4. Check that this retrained model reproduces the saved N0 run for EG seed 0.
Writes data4/w02.npz and data4/facts_w02.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

VIDEO = Path(__file__).resolve().parents[1]
ROOT = VIDEO.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'EXPERIMENTS' / 'code'))
RES = ROOT / 'EXPERIMENTS' / 'results'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lstm_layer(X, kernel, rec, bias, units=8):
    """X (T, d) -> per-step h, c and gates, Keras LSTM equations (activation tanh, recurrent sigmoid)."""
    h, c = np.zeros(units), np.zeros(units)
    out = {k: [] for k in ('h', 'c', 'i', 'f', 'o', 'g')}
    for x in X:
        z = x @ kernel + h @ rec + bias
        i, f, g, o = sigmoid(z[:units]), sigmoid(z[units:2 * units]), np.tanh(z[2 * units:3 * units]), sigmoid(z[3 * units:])
        c = f * c + i * g
        h = o * np.tanh(c)
        for k, v in zip(('h', 'c', 'i', 'f', 'o', 'g'), (h, c, i, f, o, g)):
            out[k].append(v.copy())
    return {k: np.array(v) for k, v in out.items()}


def main():
    import tensorflow as tf
    from run_experiment import load_config, make_builder
    from ML.research.run import prepare
    from ML.research import train as RT

    spec, cfg = load_config('V0_saved_weights')
    data, _ = prepare('EG', 'S1toS2', {k: cfg[k] for k in ('passive', 'emg', 'window')})
    info = json.loads((RES / 'V0_saved_weights' / 'EG' / 'S1toS2' / 'seed0.json').read_text())
    knots, sd = np.array(info['train_info']['knots']), np.array(info['train_info']['sd_per_knot'])
    model = make_builder(cfg)('lstm', 20, 5, knots)
    model.load_weights(str(RES / 'V0_saved_weights' / 'EG' / 'S1toS2' / 'seed0.weights.h5'))
    lstms = [l for l in model.layers if 'lstm' in l.name]
    dense = [l for l in model.layers if 'dense' in l.name][0]
    (k1, r1, b1), (k2, r2, b2) = [l.get_weights() for l in lstms]
    wd, bd = dense.get_weights()

    X, meta = data['X_test'], data['meta_test']
    ang = meta['angle']
    a0 = min(np.unique(ang), key=lambda a: abs(a - (-0.038)))
    m = np.where(np.isclose(ang, a0))[0]
    Xt, tt, yt = X[m], meta['t'][m] - meta['t'][m][0], data['y_test'][m]
    scale = sd[RT.snap(np.array([a0]), knots)[0]]

    # 1. by hand vs the model, on 200 windows spread over the trial
    idx = np.linspace(0, len(m) - 1, 200).astype(int)
    keras_out = model.predict(Xt[idx], verbose=0, batch_size=256).ravel()
    hand = []
    for w in Xt[idx]:
        l1 = lstm_layer(w, k1, r1, b1)
        l2 = lstm_layer(l1['h'], k2, r2, b2)
        hand.append(float(l2['h'][-1] @ wd[:, 0] + bd[0]))
    hand = np.array(hand)

    # 2. one window at the start of a push (largest torque rise over 0.2 s)
    rise = np.abs(yt[20:] - yt[:-20])
    j = int(np.argmax(rise)) + 20
    w = Xt[j]
    l1 = lstm_layer(w, k1, r1, b1)
    l2 = lstm_layer(l1['h'], k2, r2, b2)
    out_j = float(l2['h'][-1] @ wd[:, 0] + bd[0]) * scale

    # 3. order test over the whole trial
    p_time = model.predict(Xt, verbose=0, batch_size=4096).ravel() * scale
    perm = np.random.default_rng(7).permutation(len(Xt))
    p_shuf = np.empty_like(p_time)
    p_shuf[perm] = model.predict(Xt[perm], verbose=0, batch_size=4096).ravel() * scale

    # 4. reproduces the saved N0 run?
    z = np.load(RES / 'N0_total_trainmax' / 'EG' / 'S1toS2' / 'seed0.npz')
    zv = np.load(RES / 'V0_saved_weights' / 'EG' / 'S1toS2' / 'seed0.npz')
    same_as_n0 = float(np.max(np.abs(z['y_pred'].astype(float) - zv['y_pred'].astype(float))))

    # memory at the end of every window (layer-2 output) along the trial, for a heat map
    step = 10
    mem = []
    for w_ in Xt[::step]:
        mem.append(lstm_layer(lstm_layer(w_, k1, r1, b1)['h'], k2, r2, b2)['h'][-1])
    np.savez_compressed(VIDEO / 'data4' / 'w02.npz', win=w, win_t=tt[j] - (19 - np.arange(20)) / 100,
                        l1_h=l1['h'], l1_c=l1['c'], l1_f=l1['f'], l1_i=l1['i'], l1_o=l1['o'], l2_h=l2['h'],
                        trial_t=tt, trial_y=yt, p_time=p_time, p_shuf=p_shuf, mem=np.array(mem), mem_t=tt[::step],
                        idx=idx, keras_out=keras_out, hand_out=hand)
    facts = dict(subject='EG', session=2, angle_rad=round(float(a0), 3), seed=0, epochs=info['train_info']['epochs'],
                 best_epoch=info['train_info']['best_epoch'], n_params=info['train_info']['n_params'],
                 hand_vs_model_max_abs=float(np.max(np.abs(hand - keras_out))), hand_checked_windows=len(idx),
                 order_test_windows=int(len(Xt)), order_test_max_abs_Nm=float(np.max(np.abs(p_time - p_shuf))),
                 window_time_s=round(float(tt[j]), 2), window_torque_Nm=round(float(yt[j]), 2),
                 window_prediction_Nm=round(out_j, 2), swing_scale=round(float(scale), 3),
                 v0_vs_saved_n0_max_abs_Nm=same_as_n0, units=8, layers=2)
    (VIDEO / 'data4' / 'facts_w02.json').write_text(json.dumps(facts, indent=1), encoding='utf-8')
    print(json.dumps(facts, indent=1))


if __name__ == '__main__':
    main()
