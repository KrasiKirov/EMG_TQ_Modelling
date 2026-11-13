import numpy as np
import keras
from keras import layers, ops, regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

N_EMG = 4            # EMG channels come first in the feature vector
POS_INDEX = 4        # ankle position column


@keras.saving.register_keras_serializable(package='emg_tq')
class ZeroEMG(layers.Layer):
    """Copy of the input window with the EMG channels set to 0 (= rest)."""
    def call(self, x):
        return ops.concatenate([ops.zeros_like(x[:, :, :N_EMG]), x[:, :, N_EMG:]], axis=-1)


@keras.saving.register_keras_serializable(package='emg_tq')
class AngleGate(layers.Layer):
    """Hat-basis gate over the training operating angles, read from the
    last-step position: one-hot at a training angle, linear blend between
    neighbouring angles, clamped at the ends."""
    def __init__(self, knots, **kw):
        super().__init__(**kw)
        self.knots = [float(k) for k in knots]
        k = np.asarray(self.knots, np.float32)
        self._k = k
        self._kl = np.concatenate([[k[0] - 1.0], k[:-1]]).astype(np.float32)
        self._kr = np.concatenate([k[1:], [k[-1] + 1.0]]).astype(np.float32)

    def call(self, x):
        p = x[:, -1, POS_INDEX:POS_INDEX + 1]
        if len(self._k) == 1:
            return ops.ones_like(p)
        p = ops.clip(p, float(self._k[0]), float(self._k[-1]))
        left = ops.clip((p - self._kl) / (self._k - self._kl), 0.0, 1.0)
        right = ops.clip((self._kr - p) / (self._kr - self._k), 0.0, 1.0)
        return ops.minimum(left, right)

    def get_config(self):
        return {**super().get_config(), 'knots': self.knots}


@keras.saving.register_keras_serializable(package='emg_tq')
class GatedSum(layers.Layer):
    def call(self, inputs):
        z, g = inputs
        return ops.sum(z * g, axis=-1, keepdims=True)


def _lstm_stack(n_steps, n_features, name='ankle_torque_lstm'):
    return Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(8, activation='tanh', return_sequences=True),
        Dropout(0.3),
        LSTM(8, activation='tanh', return_sequences=False),
        Dropout(0.3),
        Dense(1),
    ], name=name)


def build_model(n_steps: int = 20, n_features: int = 5,
                anchored: bool = False, moe_knots=None,
                moe_expert_bias: bool = True) -> Model:
    """Two-layer stacked LSTM for ankle torque regression (Nm).

    Architecture
    ------------
    Input  : [batch, 20, 5]
    LSTM 1 : 8 units, tanh, return_sequences=True
    Dropout: 30%
    LSTM 2 : 8 units, tanh, return_sequences=False
    Dropout: 30%
    Dense  : 1 unit  → predicted ankle torque (Nm)

    moe_knots (list of training operating angles) → mixture-of-experts head:
        shared LSTM(16)×2 trunk, ẑ = shared(h) + Σ_k g_k(angle)·expert_k(h),
        one L2-shrunk, zero-initialised linear expert per angle.  Each angle
        gets its own correction of the EMG→torque read-out (the gain and
        muscle mix change with muscle length; a shared read-out is otherwise
        fitted to the strong dorsiflexed angles).  Train it on per-angle
        standardised targets (train_model(position_balance='standardize')).

    anchored=True  →  ŷ = F(emg, pos) − F(0, pos)  with the baseline F.
        Enforces "silent muscles → zero active torque" at every angle.
        Requires EMG inputs where rest ≈ 0 (build_dataset(emg_floor='train')).
    """
    if moe_knots is not None:
        inp = Input(shape=(n_steps, n_features))
        h = LSTM(16, activation='tanh', return_sequences=True)(inp)
        h = Dropout(0.2)(h)
        h = LSTM(16, activation='tanh')(h)
        h = Dropout(0.2)(h)
        z_shared = Dense(1)(h)
        z_expert = Dense(len(moe_knots), kernel_initializer='zeros',
                         use_bias=moe_expert_bias,
                         kernel_regularizer=regularizers.l2(1e-3),
                         bias_regularizer=(regularizers.l2(1e-3)
                                           if moe_expert_bias else None))(h)
        gate = AngleGate(moe_knots)(inp)
        out = layers.Add()([z_shared, GatedSum()([z_expert, gate])])
        return Model(inp, out, name='ankle_torque_lstm_moe')

    core = _lstm_stack(n_steps, n_features)
    if not anchored:
        return core
    inp = Input(shape=(n_steps, n_features))
    rest = ZeroEMG(name='emg_at_rest')(inp)
    out = layers.Subtract(name='active_torque')([core(inp), core(rest)])
    return Model(inp, out, name='ankle_torque_lstm_anchored')
