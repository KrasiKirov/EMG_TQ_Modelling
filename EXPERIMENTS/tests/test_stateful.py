"""Carried LSTM state (state='carry'): the state passes from chunk to chunk within a trial and resets per trial."""
import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')
from ML.research import stateful as ST  # noqa: E402


def _streams(B=3, T=400, F=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((B, T, F)).astype(np.float32)
    return X


def test_chunks_with_carried_state_equal_one_pass():
    tf.keras.utils.set_random_seed(0)
    m = ST.build_seq(5)
    X = _streams()
    one = ST.run_streams(m, X, chunk=X.shape[1])
    chunked = ST.run_streams(m, X, chunk=20)
    assert np.allclose(one, chunked, atol=1e-5)


def test_trials_do_not_leak_into_each_other():
    tf.keras.utils.set_random_seed(0)
    m = ST.build_seq(5)
    X = _streams()
    alone = ST.run_streams(m, X[:1], chunk=20)
    together = ST.run_streams(m, X, chunk=20)
    assert np.allclose(alone[0], together[0], atol=1e-5)


def test_fit_learns_and_predicts_every_test_sample():
    rng = np.random.default_rng(1)
    def stream(T, k):
        x = rng.random((T, 5)).astype(np.float32)
        y = np.convolve(x[:, 0], np.ones(30) / 30, mode='full')[:T] * 3     # depends on the past 0.3 s
        return {'X': x, 'y': y, 'k': k}
    train = []
    for k in range(3):
        s = stream(600, k); s['train_mask'] = np.arange(600) < 480; s['val_mask'] = ~s['train_mask']; train.append(s)
    test = [stream(500, 0), stream(450, 2)]
    preds, info = ST.fit_predict_carry(train, test, n_knots=3, chunk=20, seed=0, epochs=40, early=10,
                                       lr_patience=5, lr=0.01, balance='scale')
    assert [len(p) for p in preds] == [500, 450]
    assert info['best_val_loss'] < info['first_val_loss']
    assert info['epochs'] <= 40 and info['state'] == 'carry'
