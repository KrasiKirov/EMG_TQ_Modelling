"""Chapters 8 and 9 data from the instrumented run (data/x06_probe.npz).

* one real session-2 window of EG through the trained LSTM in NumPy, gate by gate
  (keras/src/layers/rnn/lstm.py:221-228, 277-286), checked against Keras' own prediction;
* the initial weights (Glorot limit, orthogonal recurrent weights, forget bias 1);
* step 1: the recorded batch, its loss and one weight's Nadam update recomputed from the formula
  (keras/src/optimizers/nadam.py:95-146) and checked against the recorded weights;
* the first five steps, the epoch history, and each angle's share of the training loss.
Writes data/x07.npz, data/facts_x07.json.
"""
import json

import numpy as np

from common import C01, DATA, save_facts, save_npz, use_d01_code

use_d01_code()
import tensorflow as tf                                      # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(2)
import run_experiment as RE                                  # noqa: E402
from ML.research.run import prepare                          # noqa: E402
from ML.research import train as RT                          # noqa: E402

P = np.load(DATA / 'x06_probe.npz')
shapes = json.loads(str(P['shapes']))
names = [str(n) for n in P['names']]
sizes = [int(np.prod(s)) for s in shapes]
offsets = np.cumsum([0] + sizes)


def unflat(v):
    return [v[offsets[i]:offsets[i + 1]].reshape(shapes[i]) for i in range(len(shapes))]


W = unflat(P['w_final'])
W0 = unflat(P['w_init'])
facts, arrays, disp = {}, {}, {}

_, cfg = RE.load_config(C01)
data, _ = prepare('EG', 'S1toS2', {k: cfg[k] for k in ('passive', 'emg', 'window')})
data = RE.drop_last_val_window(data)
knots = RT.angle_groups(data['train_angles'])
k_tr = RT.snap(data['meta_train']['angle'], knots)
k_te = RT.snap(data['meta_test']['angle'], knots)
y64 = data['y_train'].astype(np.float64)
sd = np.array([max(np.std(y64[k_tr == j]), 0.1) for j in range(len(knots))])

model = RE.make_builder(cfg)('lstm', 20, 5, knots)
model.set_weights(W)


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def lstm(xs, kernel, rec, bias):
    u = rec.shape[0]
    h, c = np.zeros(u), np.zeros(u)
    out = {k: [] for k in ('z', 'i', 'f', 'g', 'o', 'c', 'h')}
    for x in xs:
        z = x @ kernel + h @ rec + bias
        i, f, g, o = sig(z[:u]), sig(z[u:2 * u]), np.tanh(z[2 * u:3 * u]), sig(z[3 * u:])
        c = f * c + i * g
        h = o * np.tanh(c)
        for k, v in (('z', z), ('i', i), ('f', f), ('g', g), ('o', o), ('c', c), ('h', h)):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


# ── one session-2 window: EG at -0.038 rad, 0.15 s after a push starts ──────
mt = data['meta_test']
m = mt['trial'] == 37
idx37 = np.flatnonzero(m)
y37 = data['y_test'][m]
mid = 0.5 * (np.percentile(y37, 5) + np.percentile(y37, 95))
below = np.r_[False, y37 < mid, False]
edges = np.flatnonzero(np.diff(below.astype(int)))
runs = [(a, e) for a, e in zip(edges[::2], edges[1::2]) if e - a >= 50]
w_local = runs[2][0] + 15
gi = int(idx37[w_local])                                     # index into X_test
xw = data['X_test'][gi].astype(np.float64)
L1 = lstm(xw, *[w.astype(np.float64) for w in W[0:3]])
L2 = lstm(L1['h'], *[w.astype(np.float64) for w in W[3:6]])
yhat = float(L2['h'][-1] @ W[6].astype(np.float64)[:, 0] + W[7][0])
knot = int(k_te[gi])
pred_np = yhat * sd[knot]
pred_keras = float(model.predict(xw[None].astype(np.float32), verbose=0)[0, 0]) * sd[knot]
pred_run = float(P['pred'][gi])
unit = int(np.argmax(np.abs(L1['h'][0])))
U = 8
cols = {g: slice(j * U, (j + 1) * U) for j, g in enumerate('ifgo')}
facts['window'] = dict(test_index=gi, t=float(mt['t'][gi]), knot=knot, sd=sd[knot], yhat=yhat, pred_numpy=pred_np,
                       pred_keras=pred_keras, pred_run=pred_run, numpy_vs_keras=abs(pred_np - pred_keras),
                       keras_vs_run=abs(pred_keras - pred_run), target=float(data['y_test'][gi]), unit=unit + 1,
                       x1=xw[0], kernel_unit={g: W[0][:, cols[g]][:, unit] for g in 'ifgo'},
                       bias_unit={g: float(W[2][cols[g]][unit]) for g in 'ifgo'},
                       z1={g: float(L1['z'][0][cols[g]][unit]) for g in 'ifgo'},
                       gates1=dict(i=float(L1['i'][0][unit]), f=float(L1['f'][0][unit]), g=float(L1['g'][0][unit]),
                                   o=float(L1['o'][0][unit]), c=float(L1['c'][0][unit]), h=float(L1['h'][0][unit])),
                       h2_last=L2['h'][-1], dense_w=W[6][:, 0], dense_b=float(W[7][0]))
arrays.update(window_x=xw, L1_h=L1['h'], L1_c=L1['c'], L1_i=L1['i'], L1_f=L1['f'], L1_o=L1['o'], L1_g=L1['g'],
              L2_h=L2['h'], L2_c=L2['c'], trial37_y=y37, trial37_t=mt['t'][m], window_local=w_local)
w = facts['window']
disp.update(cell_unit=str(unit + 1), cell_zi=f"{w['z1']['i']:.2f}".replace('-', '−'),
            cell_i=f"{w['gates1']['i']:.2f}", cell_f=f"{w['gates1']['f']:.2f}",
            cell_g=f"{w['gates1']['g']:.2f}".replace('-', '−'), cell_o=f"{w['gates1']['o']:.2f}",
            cell_c=f"{w['gates1']['c']:.3f}".replace('-', '−'), cell_h=f"{w['gates1']['h']:.3f}".replace('-', '−'),
            cell_yhat=f'{yhat:.3f}'.replace('-', '−'), cell_sd=f'{sd[knot]:.2f}',
            cell_pred=f'{pred_np:.2f}'.replace('-', '−'), cell_true=f"{w['target']:.2f}".replace('-', '−'))

# ── initial weights ──────────────────────────────────────────────────────────
lim = float(np.sqrt(6 / (5 + 32)))
rk = W0[1].astype(np.float64)                                 # (8, 32): orthonormal rows
facts['init'] = dict(glorot_limit_l1=lim, kernel_l1_absmax=float(np.max(np.abs(W0[0]))),
                     forget_bias=W0[2][8:16], other_bias_absmax=float(np.max(np.abs(np.r_[W0[2][:8], W0[2][16:]]))),
                     orthogonal_err=float(np.max(np.abs(rk @ rk.T - np.eye(8)))))
arrays.update(w0_kernel1=W0[0], w0_rec1=W0[1], w0_bias1=W0[2])
disp['glorot1'] = f'{lim:.2f}'

# ── step 1: the batch, its loss, one Nadam update ────────────────────────────
bi = P['step_batch_index'][0]
ang = data['meta_train']['angle'][bi]
kb = k_tr[bi]
yb, pb = P['step_y'][0].astype(np.float64), P['step_y_pred'][0].astype(np.float64)
facts['batch1'] = dict(index=bi, trial=data['meta_train']['trial'][bi], angle=ang, t=data['meta_train']['t'][bi],
                       knot=kb, sd=sd[kb], y_Nm=y64[bi], y_scaled=yb, y_pred=pb, sq=(pb - yb) ** 2,
                       loss_recorded=float(P['step_loss'][0]), loss_recomputed=float(np.mean((pb - yb) ** 2)),
                       n_angles=int(len(set(kb.tolist()))), drop1_zero_frac=float(np.mean(P['step_drop1'][0] == 0)),
                       drop2_zero_frac=float(np.mean(P['step_drop2'][0] == 0)))
disp.update(batch_n_angles=str(facts['batch1']['n_angles']), batch_loss=f"{facts['batch1']['loss_recorded']:.3f}")
facts['batches'] = dict(knot=[k_tr[b].tolist() for b in P['step_batch_index']],
                        trial=[data['meta_train']['trial'][b].tolist() for b in P['step_batch_index']],
                        t=[data['meta_train']['t'][b].tolist() for b in P['step_batch_index']])
arrays.update(batch1_x=P['step_x'][0], batch1_drop2=P['step_drop2'][0])

g1, m1, v1 = P['step_grads'][0].astype(np.float32), P['step_m'][0].astype(np.float32), P['step_v'][0].astype(np.float32)
w_after1 = P['check_w'][0].astype(np.float32)
w_init = P['w_init'].astype(np.float32)
lr, b1, b2, eps = np.float32(0.003), np.float32(0.9), np.float32(0.999), np.float32(1e-7)
u1 = b1 * (np.float32(1) - np.float32(0.5) * np.float32(0.96) ** np.float32(1))
u2 = b1 * (np.float32(1) - np.float32(0.5) * np.float32(0.96) ** np.float32(2))
up1 = np.float32(1) * u1
up2 = up1 * u2
m_hat = u2 * m1 / (np.float32(1) - up2) + (np.float32(1) - u1) * g1 / (np.float32(1) - up1)
v_hat = v1 / (np.float32(1) - b2 ** np.float32(1))
step = lr * m_hat / (np.sqrt(v_hat) + eps)
w_pred = w_init - step
d1 = np.abs(w_after1 - w_init)
dense0 = int(offsets[6])                                      # first dense kernel weight
k_show = dense0 + int(np.argmax(np.abs(g1[offsets[6]:offsets[7]])))
facts['nadam1'] = dict(index=k_show, name=names[6], g=float(g1[k_show]), m=float(m1[k_show]), v=float(v1[k_show]),
                       u1=float(u1), u2=float(u2), m_hat=float(m_hat[k_show]), v_hat=float(v_hat[k_show]),
                       step=float(step[k_show]), w0=float(w_init[k_show]), w1=float(w_after1[k_show]),
                       w1_formula=float(w_pred[k_show]), formula_max_err=float(np.max(np.abs(w_pred - w_after1))),
                       abs_step_min=float(d1.min()), abs_step_median=float(np.median(d1)), abs_step_max=float(d1.max()),
                       recorded_u_product=P['step_u_product'])
arrays.update(step1_abs=d1)
disp.update(step1_size=f'{np.median(d1):.4f}', nd_g=f"{g1[k_show]:.3g}".replace('-', '−'),
            nd_w0=f'{w_init[k_show]:.4f}'.replace('-', '−'), nd_w1=f'{w_after1[k_show]:.4f}'.replace('-', '−'))

# ── first five steps: losses and three weights ───────────────────────────────
picks = {'dense kernel (chosen above)': k_show,
         f'layer 1 SOL -> input gate, unit {unit + 1}': int(offsets[0] + 2 * 32 + unit),
         'dense bias': int(offsets[7])}
traj = {k: [float(w_init[i])] + [float(P['check_w'][s][i]) for s in range(5)] for k, i in picks.items()}
facts['first_steps'] = dict(loss=P['step_loss'], running_loss=P['check_running_loss'][:5], weights=traj,
                            picks=picks)

# ── epochs ───────────────────────────────────────────────────────────────────
cuts = [int(e) for e, a, b in zip(P['epoch'], P['lr_begin'], P['lr_end']) if b < a - 1e-9]
facts['epochs'] = dict(loss=P['loss'], val_loss=P['val_loss'], lr_begin=P['lr_begin'], lr_end=P['lr_end'], cuts=cuts,
                       best=int(np.argmin(P['val_loss']) + 1))
disp.update(lr_cut1=str(cuts[0]), lr_cut2=str(cuts[1]) if len(cuts) > 1 else '')

# ── each angle's share of the training loss with the trained weights ─────────
pt = model.predict(data['X_train'], verbose=0, batch_size=4096)[:, 0].astype(np.float64) * sd[k_tr]
ss = np.array([np.sum((pt[k_tr == j] - y64[k_tr == j]) ** 2) for j in range(len(knots))])
n = np.array([np.sum(k_tr == j) for j in range(len(knots))])
r2_tr = 1 - ss / (n * np.array([np.var(y64[k_tr == j]) for j in range(len(knots))]))
plain, scaled = ss / ss.sum(), (ss / sd ** 2) / np.sum(ss / sd ** 2)
pte = model.predict(data['X_test'], verbose=0, batch_size=4096)[:, 0].astype(np.float64) * sd[k_te]
facts['shares'] = dict(knots=knots, n=n, ss_res=ss, r2_train=r2_tr, plain=plain, scaled=scaled,
                       keras_test_vs_run=float(np.max(np.abs(pte - P['pred'].astype(np.float64)))))
disp.update(share_plain_low=f'{100 * plain[0]:.1f}', share_scaled_low=f'{100 * scaled[0]:.1f}')

facts['display'] = disp
save_npz('x07', **arrays)
save_facts('x07', {'x07': facts})
print(json.dumps({k: facts['window'][k] for k in ('t', 'yhat', 'pred_numpy', 'pred_keras', 'pred_run', 'target', 'unit',
                                                   'numpy_vs_keras', 'keras_vs_run')}, default=float))
print('init', {k: (v if not isinstance(v, np.ndarray) else np.round(v, 3).tolist()) for k, v in facts['init'].items()})
print('batch1 loss', facts['batch1']['loss_recorded'], facts['batch1']['loss_recomputed'], 'angles', facts['batch1']['n_angles'])
print('nadam', {k: v for k, v in facts['nadam1'].items() if k != 'recorded_u_product'})
print('cuts', cuts, 'shares plain', np.round(plain, 3), 'scaled', np.round(scaled, 3), 'r2 train', np.round(r2_tr, 3))
print('display', disp)
