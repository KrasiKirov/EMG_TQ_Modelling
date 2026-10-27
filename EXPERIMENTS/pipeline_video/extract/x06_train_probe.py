"""Instrumented rerun of D01's training: EG, learn session 1 -> test session 2, seed 0.

    python x06_train_probe.py --mode probe   # records the first steps in full
    python x06_train_probe.py --mode plain   # unchanged training, read-only callback only

Both modes make exactly the calls of EXPERIMENTS/code/run_experiment.py main() (dorsiflexion
worktree): load_config, make_builder, prepare, drop_last_val_window, fit_predict, one thread.
EG's D01 runs are stored as C01 (same passive tables for EG), so the C01 config is used and
its saved results/C01_autogain+equalweight/EG/S1toS2/seed0.npz is the reference.

probe: Keras' train_step (keras/src/backend/tensorflow/trainer.py:53-87) is replaced, on this
model instance only, by an identical copy that also writes the batch, the prediction, the loss
and the gradients into tf.Variables; both Dropout layers also write their output (zeros = the
dropout mask). Nothing touches a random-number generator, so the run should be the saved run.
plain: no changes at all, only a callback that reads weights and optimizer state.

A run passes when epochs, best epoch and best validation loss equal the saved run JSON and its
test predictions equal the saved ones. Writes data/x06_<mode>.npz and data/facts_x06_<mode>.json.
"""
import argparse
import json
import time
import warnings

import numpy as np

from common import C01, RESULTS, save_facts, save_npz, use_d01_code

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['probe', 'plain'], required=True)
ap.add_argument('--smoke', action='store_true', help='1 epoch only, to test the instrumentation (outputs *_smoke)')
a = ap.parse_args()
TAG = f'{a.mode}_smoke' if a.smoke else a.mode
SUBJECT, DIRECTION, SEED, EXP_ID = 'EG', 'S1toS2', 0, C01
K = 5                                                    # steps recorded in full
CHECK = sorted(set(range(1, 21)) | {30, 50, 100, 200, 500, 1000, 2000, 4000, 7112})

use_d01_code()
import tensorflow as tf                                  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(1)  # run_experiment.py:107-108 with --threads 1
tf.config.threading.set_inter_op_parallelism_threads(1)

import run_experiment as RE                              # noqa: E402  (dorsiflexion EXPERIMENTS/code)
from ML.research.run import prepare                      # noqa: E402
from ML.research import train as RT                      # noqa: E402
from keras.src import tree                               # noqa: E402
from keras.src.losses import loss as loss_module         # noqa: E402
from keras.src.trainers.data_adapters import data_adapter_utils  # noqa: E402

PV = {}          # probe variables (probe mode)
HOLD = {}        # the model built inside fit_predict


def probe_train_step(self, data):
    """keras/src/backend/tensorflow/trainer.py:53-87, plus the four PV lines marked 'probe'."""
    x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

    # Forward pass
    with tf.GradientTape() as tape:
        if self._call_has_training_arg:
            y_pred = self(x, training=True)
        else:
            y_pred = self(x)
        loss = self._compute_loss(
            x=x,
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            training=True,
        )
        self._loss_tracker.update_state(
            loss_module.unscale_loss_for_distribution(loss),
            sample_weight=tf.shape(
                next(i for i in tree.flatten(x) if i is not None)
            )[0],
        )
        if self.optimizer is not None:
            loss = self.optimizer.scale_loss(loss)

    # Compute gradients
    if self.trainable_weights:
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, trainable_weights)

        PV['x'].assign(tf.cast(x, tf.float32))                                  # probe
        PV['y'].assign(tf.reshape(tf.cast(y, tf.float32), [-1]))                # probe
        PV['y_pred'].assign(tf.reshape(tf.cast(y_pred, tf.float32), [-1]))      # probe
        PV['loss'].assign(tf.cast(loss, tf.float32))                            # probe
        for v, g in zip(PV['grads'], gradients):                                # probe
            v.assign(g)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))
    else:
        warnings.warn("The model does not have any trainable weights.")

    return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)


def wrap_dropout(layer, var):
    orig = layer.call

    def call(inputs, training=None):
        out = orig(inputs, training=training)
        if training:
            var.assign(out)                                                     # probe
        return out
    object.__setattr__(layer, 'call', call)


class Recorder(tf.keras.callbacks.Callback):
    """Read-only: copies weights / optimizer state at chosen steps and the per-epoch logs."""

    def __init__(self):
        super().__init__()
        self.step, self.snaps, self.epochs, self.epoch_w = 0, {}, [], []

    def on_train_begin(self, logs=None):
        self.w0 = [w.copy() for w in self.model.get_weights()]
        self.t0 = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.lr_begin = float(np.asarray(self.model.optimizer.learning_rate))
        self.te = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step not in CHECK:
            return
        opt = self.model.optimizer
        s = {'w': np.concatenate([w.ravel() for w in self.model.get_weights()]),
             'running_loss': float(np.asarray(logs['loss']))}
        if self.step <= K:
            s.update(m=np.concatenate([np.asarray(v).ravel() for v in opt._momentums]),
                     v=np.concatenate([np.asarray(v).ravel() for v in opt._velocities]),
                     u_product=float(np.asarray(opt._u_product)), iterations=int(np.asarray(opt.iterations)),
                     lr=float(np.asarray(opt.learning_rate)))
            if PV:
                s.update(x=PV['x'].numpy(), y=PV['y'].numpy(), y_pred=PV['y_pred'].numpy(),
                         loss=float(PV['loss'].numpy()),
                         grads=np.concatenate([v.numpy().ravel() for v in PV['grads']]),
                         drop1=PV['drop1'].numpy(), drop2=PV['drop2'].numpy())
        self.snaps[self.step] = s

    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        self.epochs.append(dict(epoch=epoch + 1, loss=float(np.asarray(logs['loss'])),
                                val_loss=float(np.asarray(logs['val_loss'])), lr_begin=self.lr_begin,
                                lr_end=float(np.asarray(opt.learning_rate)), steps=int(np.asarray(opt.iterations)),
                                seconds=time.time() - self.te))
        self.epoch_w.append(np.concatenate([w.ravel() for w in self.model.get_weights()]))
        print(f"epoch {epoch + 1}: loss {self.epochs[-1]['loss']:.5f} val {self.epochs[-1]['val_loss']:.5f} "
              f"lr {self.epochs[-1]['lr_end']:.6f} ({self.epochs[-1]['seconds']:.0f} s)", flush=True)


REC = Recorder()
spec, cfg = RE.load_config(EXP_ID)
builder = RE.make_builder(cfg)


def build(name, n_steps, n_features, knots):
    """RE.make_builder's model, then (probe mode) the instrumentation; never touches an RNG."""
    model = builder(name, n_steps, n_features, knots)
    if a.mode == 'probe':
        PV.update(x=tf.Variable(tf.zeros((8, n_steps, n_features)), trainable=False),
                  y=tf.Variable(tf.zeros((8,)), trainable=False),
                  y_pred=tf.Variable(tf.zeros((8,)), trainable=False),
                  loss=tf.Variable(0.0, trainable=False),
                  grads=[tf.Variable(tf.zeros(w.shape), trainable=False) for w in model.trainable_weights],
                  drop1=tf.Variable(tf.zeros((8, n_steps, 8)), trainable=False),
                  drop2=tf.Variable(tf.zeros((8, 8)), trainable=False))
        drops = [l for l in model.layers if isinstance(l, tf.keras.layers.Dropout)]
        wrap_dropout(drops[0], PV['drop1'])
        wrap_dropout(drops[1], PV['drop2'])
        object.__setattr__(model, 'train_step', probe_train_step.__get__(model))
    fit = model.fit

    def fit_with_recorder(*args, callbacks=None, **kw):
        return fit(*args, callbacks=list(callbacks or []) + [REC], **kw)
    object.__setattr__(model, 'fit', fit_with_recorder)
    HOLD['model'] = model
    return model


RT.build_model = build                                   # as run_experiment.py:114 does with make_builder
if a.smoke:
    RT.C.PROTOCOLS['orig']['epochs'] = 1
t0 = time.time()
data, details = prepare(SUBJECT, DIRECTION, {k: cfg[k] for k in ('passive', 'emg', 'window')})
data = RE.drop_last_val_window(data)
pred, info = RT.fit_predict(data, cfg['model'], cfg['balance'], cfg['pf_extra'], cfg['train_protocol'], SEED,
                            lr=cfg['lr'], rest_weight=cfg['rest_weight'])
seconds = time.time() - t0
model = HOLD['model']

# ── checks against the saved run ────────────────────────────────────────────
stem = RESULTS / EXP_ID / SUBJECT / DIRECTION / f'seed{SEED}'
ref = np.load(f'{stem}.npz')
ref_info = json.loads(open(f'{stem}.json').read())['train_info']
pred_diff = float(np.max(np.abs(pred.astype(np.float32) - ref['y_pred'])))
target_diff = float(np.max(np.abs(data['y_test'].astype(np.float32) - ref['y_own_target'])))
same = dict(epochs=info['epochs'] == ref_info['epochs'], best_epoch=info['best_epoch'] == ref_info['best_epoch'],
            best_val_loss=abs(info['best_val_loss'] - ref_info['best_val_loss']) < 1e-12, predictions=pred_diff == 0.0)
print('checks', same, 'max |pred - saved| =', pred_diff, 'Nm; max |target - saved| =', target_diff, 'Nm', flush=True)

# which training windows were in the recorded batches
index = {data['X_train'][i].tobytes(): i for i in range(len(data['X_train']))}
names = [w.path for w in model.weights]
shapes = [list(w.shape) for w in model.weights]
arrays = dict(names=np.array(names), shapes=np.array(json.dumps(shapes)),
              w_init=np.concatenate([w.ravel() for w in REC.w0]),
              w_final=np.concatenate([w.ravel() for w in model.get_weights()]),
              check_steps=np.array(sorted(REC.snaps)), check_w=np.stack([REC.snaps[k]['w'] for k in sorted(REC.snaps)]),
              check_running_loss=np.array([REC.snaps[k]['running_loss'] for k in sorted(REC.snaps)]),
              epoch=np.array([e['epoch'] for e in REC.epochs]), loss=np.array([e['loss'] for e in REC.epochs]),
              val_loss=np.array([e['val_loss'] for e in REC.epochs]),
              lr_begin=np.array([e['lr_begin'] for e in REC.epochs]), lr_end=np.array([e['lr_end'] for e in REC.epochs]),
              epoch_seconds=np.array([e['seconds'] for e in REC.epochs]), epoch_w=np.stack(REC.epoch_w),
              pred=pred.astype(np.float32), knots=np.array(info['knots']), sd=np.array(info['sd_per_knot']))
first = [REC.snaps[k] for k in range(1, K + 1)]
for key in ('m', 'v', 'u_product', 'iterations', 'lr'):
    arrays[f'step_{key}'] = np.array([s[key] for s in first])
if a.mode == 'probe':
    for key in ('x', 'y', 'y_pred', 'loss', 'grads', 'drop1', 'drop2'):
        arrays[f'step_{key}'] = np.array([s[key] for s in first])
    arrays['step_batch_index'] = np.array([[index.get(xx.tobytes(), -1) for xx in s['x']] for s in first])
save_npz(f'x06_{TAG}', **arrays)
save_facts(f'x06_{TAG}', {f'x06_{TAG}': dict(
    checks=same, max_abs_pred_diff_Nm=pred_diff, max_abs_target_diff_Nm=target_diff, epochs=info['epochs'],
    best_epoch=info['best_epoch'], best_val_loss=info['best_val_loss'], n_params=info['n_params'],
    seconds=seconds, steps_per_epoch=int(len(data['y_train']) // 8), weight_names=names, weight_shapes=shapes,
    tf=tf.__version__)})
print(f'done in {seconds:.0f} s', flush=True)
