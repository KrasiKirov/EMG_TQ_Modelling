"""
Model builders around ML.models.lstm.build_model.

lstm8   LSTM(8) -> Dropout(0.3) -> LSTM(8) -> Dropout(0.3) -> Dense(1), the original model (1,001 parameters)
moe     shared LSTM(16)x2 trunk + one zero-initialised linear expert per training ankle angle, selected by the angle

Suffix ``+ga<NN>`` (e.g. ``lstm8+ga50``) adds training-only EMG gain augmentation: each EMG channel of each window is
multiplied by exp(N(0, NN/100)); the angle input is untouched and the layer has no weights. It targets the
between-session amplitude changes (electrode contact, placement).
"""

import keras
from keras import layers, ops
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from ML.models.lstm import build_model, N_EMG


@keras.saving.register_keras_serializable(package='emg_tq_research')
class RandomChannelGain(layers.Layer):
    """Training-time log-normal EMG gain, constant over a window and independent between channels."""

    def __init__(self, sigma=0.3, n_emg=N_EMG, **kw):
        super().__init__(**kw)
        self.sigma = float(sigma)
        self.n_emg = int(n_emg)
        self.seed_gen = keras.random.SeedGenerator()   # seeded by keras.utils.set_random_seed

    def call(self, x, training=None):
        if not training or self.sigma <= 0:
            return x
        g = ops.exp(keras.random.normal((ops.shape(x)[0], 1, self.n_emg), stddev=self.sigma,
                                        seed=self.seed_gen))
        return ops.concatenate([x[:, :, :self.n_emg] * g, x[:, :, self.n_emg:]], axis=-1)

    def get_config(self):
        return {**super().get_config(), 'sigma': self.sigma, 'n_emg': self.n_emg}


def _base(name, n_steps, n_features, knots):
    if name == 'lstm8':
        return build_model(n_steps=n_steps, n_features=n_features)
    if name == 'moe':
        return build_model(n_steps=n_steps, n_features=n_features, moe_knots=list(knots),
                           moe_expert_bias=False)
    raise ValueError(name)


def build(name, n_steps, n_features, knots):
    """``name`` = base model, optionally followed by ``+ga<NN>``."""
    base_name, _, aug = name.partition('+')
    base = _base(base_name, n_steps, n_features, knots)
    if not aug:
        return base
    if not aug.startswith('ga'):
        raise ValueError(name)
    inp = Input(shape=(n_steps, n_features))
    x = RandomChannelGain(sigma=int(aug[2:]) / 100.0, name='emg_gain_augmentation')(inp)
    return Model(inp, base(x), name=f'{base.name}_gainaug')
