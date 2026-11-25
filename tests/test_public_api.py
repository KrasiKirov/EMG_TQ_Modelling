"""Names imported from outside the repository (Capstone/LSTM_Presentation) must stay importable and keep their shape."""
import importlib
import inspect

import numpy as np
import pytest

PUBLIC = {
    'EXPERIMENTS.pipeline.run': ('load_config', 'make_builder', 'drop_last_val_window'),
    'EXPERIMENTS.pipeline.scoring': ('score_trial', 'person_scale'),
    'ML.research.run': ('prepare', 'rest_flags'),
    'ML.research.data': ('load_subject',),
    'ML.research.splits': ('common_start', 'COMMON_MAX_WINDOW'),
    'ML.research.train': ('fit_predict', 'build_model'),
    'ML.research.config': ('PROTOCOLS',),
    'ML.preprocessing.flb_reader': ('read_flb',),
    'ML.preprocessing.emg_envelope': ('detect_emg_columns',),
    'ML.config': ('SUBJECTS', 'DATA_DIR'),
}


@pytest.mark.parametrize('module,names', sorted(PUBLIC.items()))
def test_names_importable(module, names):
    m = importlib.import_module(module)
    missing = [n for n in names if not hasattr(m, n)]
    assert not missing, f'{module} lost {missing}'


def test_fit_predict_positional_signature():
    from ML.research import train as RT
    params = list(inspect.signature(RT.fit_predict).parameters)
    assert params[:6] == ['data', 'model_name', 'balance', 'pf_extra', 'protocol', 'seed']


def test_monkeypatched_build_model_is_used(patch_subject, monkeypatch):
    """LSTM_Presentation/training_run.py sets RT.build_model = make_builder(cfg) and calls fit_predict positionally."""
    from EXPERIMENTS.pipeline import run as PR
    from ML.research import config as RC, train as RT
    from ML.research.run import prepare
    monkeypatch.setitem(RC.PROTOCOLS, 'unit', dict(batch=256, lr=0.003, lr_factor=0.5, lr_patience=6, early=15, epochs=1))
    _, cfg = PR.load_config('FE_A0')
    rcfg = {k: cfg[k] for k in ('passive', 'emg', 'window', 'envelope', 'trim_end_s', 'input_step',
                                'state', 'lp_order', 'val_gap', 'metadata_passive', 'scale_part')}
    data = PR.drop_last_val_window(prepare('SYN', 'S1toS2', rcfg)[0])
    built = []
    real = PR.make_builder(cfg)

    def stub(*args):
        built.append(args[0])
        return real(*args)
    monkeypatch.setattr(RT, 'build_model', stub)
    pred, _ = RT.fit_predict(data, cfg['model'], cfg['balance'], cfg['pf_extra'], 'unit', 10)
    assert built == ['lstm'] and np.isfinite(pred).all()
