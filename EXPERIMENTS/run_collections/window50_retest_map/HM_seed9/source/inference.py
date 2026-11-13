"""Load a complete offline inference package, including the fitted preprocessing."""
import json
from pathlib import Path
import numpy as np

from ML.models.baselines import load_baseline
from ML.preprocessing.calibration import PassiveCalibration
from ML.preprocessing.pipeline import PipelineConfig, process_block
from ML.preprocessing.trial_manifest import CHANNELS


class Predictor:
    def __init__(self, folder):
        folder = Path(folder)
        self.package = json.loads((folder / 'package.json').read_text())
        if self.package.get('schema_version') != 1:
            raise ValueError('Unsupported package version')
        self.preprocess = self.package['preprocessing']
        offset, scale = self.package['target_offset'], self.package['target_scale']
        if not np.isfinite([offset, scale]).all() or scale <= 0:
            raise ValueError('Invalid fitted target scaling')
        if self.package['kind'] not in ('position', 'ridge', 'mlp', 'lstm'):
            raise ValueError('Unsupported model kind')
        if self.preprocess['channels'] != list(CHANNELS) or self.preprocess.get('mode') != 'offline':
            raise ValueError('Incompatible feature schema/preprocessing mode')
        self.config = PipelineConfig(**self.preprocess['config'])
        if self.package['input_shape'] != [self.config.window, 5]:
            raise ValueError('Package and preprocessing input shapes disagree')
        scales = np.asarray(self.preprocess['emg_scales'])
        if scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('Invalid fitted EMG scales')
        if self.config.target_mode == 'active_torque' and not self.preprocess.get('passive'):
            raise ValueError('Active-torque package lacks calibration support')
        if self.package['kind'] in ('ridge', 'position'):
            self.model = load_baseline(folder / 'model.npz')
        else:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(folder / 'model.keras', compile=False)
            if list(self.model.input_shape[1:]) != self.package['input_shape']:
                raise ValueError('Weights and package input shapes disagree')
            # Cache a graph with a variable batch axis: recurrent Python/eager
            # dispatch otherwise dominates latency for this 1,001-parameter model.
            self._forward = tf.function(lambda x: self.model(x, training=False),
                input_signature=[tf.TensorSpec([None, *self.package['input_shape']], tf.float32)])

    def predict_windows(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or list(X.shape[1:]) != self.package['input_shape']:
            raise ValueError('Incompatible window layout')
        if not np.isfinite(X).all():
            raise ValueError('Non-finite input')
        if self.package.get('omit_muscle'):
            X = X.copy()
            X[:, :, CHANNELS.index(self.package['omit_muscle'])] = 0
        if not len(X):
            return np.array([], dtype=float)
        if self.package['kind'] in ('ridge', 'position'):
            return self.model.predict(X)
        output = np.concatenate([np.asarray(self._forward(X[i:i+512])).ravel()
                                 for i in range(0, len(X), 512)])
        return output * self.package['target_scale'] + self.package['target_offset']

    def predict_recording(self, frame):
        """Offline only. Input must contain named EMG/position and domainIncr attrs.

        Returns indices for the retained predictions, never invents edge outputs.
        Torque is optional at inference and is not a model input.
        """
        frame = frame.copy(deep=True)
        if 'domainIncr' not in frame.attrs:
            raise ValueError('Input sample interval (domainIncr) is required')
        if 'torque' not in frame:
            frame['torque'] = 0.
        signals, index = process_block(frame, 0, len(frame), self.config)
        signals[:, :4] /= np.asarray(self.preprocess['emg_scales'])
        X = np.lib.stride_tricks.sliding_window_view(signals[:, :5], self.config.window, axis=0)
        X = np.transpose(X, (0, 2, 1)).copy()
        yp = self.predict_windows(X)
        supported = np.ones(len(yp), bool)
        if self.config.target_mode == 'active_torque':
            calibration = PassiveCalibration(**self.preprocess['passive'])
            supported = np.isfinite(calibration.predict(X[:, -1, -1]))
            yp = np.where(supported, yp, np.nan)
        return dict(sample_index=index[self.config.window-1:], prediction_nm=yp,
                    supported=supported, target_mode=self.config.target_mode, processing='offline')
