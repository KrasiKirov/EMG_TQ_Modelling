"""
LSTM with carried state (experiment setting ``"state": "carry"``); ``train.fit_predict`` is the stateless version.

training   the training trials run side by side, one batch lane each, in chunks of ``chunk`` samples; the weights
           update after each chunk and the hidden and cell states carry into the next chunk (truncated BPTT).
           Every training sample is a target. The state is zero at the start of every trial and epoch.
validation after each epoch the whole training-session trial runs from a zero state; loss on the validation part.
test       each test trial runs once from a zero state.

Loss (plain MSE), optimiser, LR schedule and early stopping follow ``train.fit_predict`` (ML.config.TRAINING).
"""
import numpy as np

from ML.config import CLIPNORM, MIN_LR, TRAINING


def build_seq(n_features, units, layers, dropout):
    """Sequence model: (x, initial states) -> (y for every step, final states)."""
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    x = Input(shape=(None, n_features))
    states_in, states_out, h = [], [], x
    for _ in range(layers):
        s0, s1 = Input(shape=(units,)), Input(shape=(units,))
        states_in += [s0, s1]
        h, hf, cf = LSTM(units, activation='tanh', return_sequences=True, return_state=True)(h, initial_state=[s0, s1])
        states_out += [hf, cf]
        h = Dropout(dropout)(h)
    y = Dense(1)(h)
    return Model([x] + states_in, [y] + states_out, name='lstm_carry')


def _zeros(model, B):
    return [np.zeros((B, int(t.shape[-1])), np.float32) for t in model.inputs[1:]]


def run_streams(model, X, chunk=200):
    """Forward pass over (B, T, F) streams, chunk by chunk, carrying the state. Returns (B, T)."""
    X = np.asarray(X, np.float32)
    states, out = _zeros(model, len(X)), []
    for a in range(0, X.shape[1], chunk):
        res = model([X[:, a:a + chunk]] + states, training=False)
        out.append(np.asarray(res[0])[..., 0])
        states = [np.asarray(s) for s in res[1:]]
    return np.concatenate(out, axis=1)


def _pad(streams, key, T, dtype=np.float32):
    first = np.asarray(streams[0][key])
    out = np.zeros((len(streams), T) + first.shape[1:], dtype)
    for i, s in enumerate(streams):
        v = np.asarray(s[key]); out[i, :len(v)] = v
    return out


def fit_predict_carry(train, test, chunk, units, layers, dropout, seed=0, lr=None, verbose=0):
    """train: list of {'X' (T, F), 'y' (T,), 'train_mask', 'val_mask'} (one per session-1 trial);
    test : list of {'X' (T, F)}. lr replaces the schedule's learning rate.
    Returns ([predictions per test trial, in Nm], info)."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    sch = TRAINING
    T = max(len(s['y']) for s in train)
    X, y = _pad(train, 'X', T), _pad(train, 'y', T, np.float64)
    m_tr, m_va = _pad(train, 'train_mask', T), _pad(train, 'val_mask', T)
    t = y.astype(np.float32)
    model = build_seq(X.shape[2], units, layers, dropout)
    opt = tf.keras.optimizers.Nadam(learning_rate=lr or sch['lr'], clipnorm=CLIPNORM)

    @tf.function(reduce_retracing=True)
    def step(x, yt, m, *states):
        with tf.GradientTape() as tape:
            res = model([x] + list(states), training=True)
            err = tf.square(res[0][..., 0] - yt) * m
            loss = tf.reduce_sum(err) / tf.maximum(tf.reduce_sum(m), 1.0)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return [loss] + list(res[1:])

    last_train = int(np.max(np.nonzero(m_tr.any(axis=0))[0])) + 1
    hist, best, best_w, wait, plateau = [], np.inf, None, 0, 0
    for ep in range(sch['epochs']):
        states = _zeros(model, len(X))
        for a in range(0, last_train, chunk):
            if not m_tr[:, a:a + chunk].any():
                continue
            out = step(X[:, a:a + chunk], t[:, a:a + chunk], m_tr[:, a:a + chunk], *states)
            states = [s.numpy() for s in out[1:]]
        pv = run_streams(model, X, chunk=max(chunk, 200))
        val = float(np.sum(m_va * (pv - t) ** 2) / max(m_va.sum(), 1))
        hist.append(val)
        if verbose:
            print(f'epoch {ep + 1}: val {val:.4f}', flush=True)
        if val < best - 1e-12:
            best, best_w, wait, plateau = val, model.get_weights(), 0, 0
        else:
            wait += 1; plateau += 1
            if plateau >= sch['lr_patience']:
                opt.learning_rate.assign(max(float(opt.learning_rate.numpy()) * sch['lr_factor'], MIN_LR)); plateau = 0
            if wait >= sch['early']:
                break
    model.set_weights(best_w)
    pv = run_streams(model, X, chunk=max(chunk, 200))
    val_streams = [pv[i, :len(s['y'])].astype(np.float64) for i, s in enumerate(train)]
    preds = []
    for s in test:
        p = run_streams(model, np.asarray(s['X'], np.float32)[None], chunk=max(chunk, 200))[0]
        preds.append(p.astype(np.float64))
    info = {'state': 'carry', 'chunk': chunk, 'clipnorm': CLIPNORM, 'epochs': len(hist), 'best_epoch': int(np.argmin(hist)) + 1,
            'best_val_loss': float(best), 'first_val_loss': float(hist[0]),
            'n_params': int(model.count_params()), 'max_epochs': sch['epochs'], 'shuffle': False,
            'val_streams': val_streams}
    return preds, info
