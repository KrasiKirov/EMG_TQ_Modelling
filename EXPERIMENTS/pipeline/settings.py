"""Run settings: the defaults every spec starts from, the values each setting may take, and loading a spec.

BASE is frozen: confirm.py recomputes config_hash(BASE + changes) for every final run and refuses one whose stored
hash differs, so a key or default changed here would invalidate the study's results. Options no spec uses any more
(pf_extra, rest_weight) stay in BASE for that reason, and must stay None.
"""
from __future__ import annotations

from numbers import Integral

from EXPERIMENTS.pipeline.spec import SpecError, load

BASE = dict(passive='legacy', emg='trainmax', units=8, layers=2, dropout=0.3, model='lstm',
            balance='none', pf_extra=None, window=20, lr=None, train_protocol='orig', rest_weight=None,
            extra_input=None, shuffle=True, envelope='zerophase', trim_end_s=0.0, input_step=1, train_stride=1,
            state='reset', lp_order=4, val_gap=False, metadata_passive=False, chunk=None, clipnorm=None, epoch_windows=None,
            scale_part='train_val', batch=None)

# every run learns from session 1 and predicts session 2 (study v2 is S1→S2 only); there is no other direction
DIRECTION = 'S1toS2'


def _allowed() -> dict[str, tuple]:
    from ML.research import config as RC
    from ML.research.data import ENVELOPES
    from ML.research.emg_norm import KINDS as EMG_KINDS
    from ML.research.passive import KINDS as PASSIVE_KINDS
    return {
        'passive': PASSIVE_KINDS + ('none',),
        'emg': EMG_KINDS,
        'balance': ('none', 'scale'),
        'model': ('lstm', 'ridge', 'ridge_lagged'),
        'state': ('reset', 'carry'),
        'envelope': ENVELOPES,
        'train_protocol': tuple(RC.PROTOCOLS),
        'scale_part': ('train_val', 'train'),
        'extra_input': (None, 'passive_trial_value'),
        'pf_extra': (None,),
        'rest_weight': (None,),
    }


def validate(cfg: dict) -> None:
    """Raise SpecError naming the setting if ``cfg`` has an unknown key or a value the pipeline does not run."""
    unknown = sorted(set(cfg) - set(BASE))
    if unknown:
        raise SpecError(f'unknown setting(s) {", ".join(unknown)}')
    for key, values in _allowed().items():
        if cfg[key] not in values:
            raise SpecError(f'{key}={cfg[key]!r} is not supported (removed 2026-11; see tag pre-cleanup-2026-11)'
                            if key in ('passive', 'emg', 'balance', 'model', 'pf_extra', 'rest_weight', 'train_protocol')
                            else f'{key} must be one of {values}, got {cfg[key]!r}')
    ew = cfg['epoch_windows']
    if not (ew is None or ew == 'full' or (isinstance(ew, Integral) and not isinstance(ew, bool) and ew > 0)):
        raise SpecError(f"epoch_windows must be None, 'full' or a positive integer, got {ew!r}")
    if cfg['state'] == 'carry':
        if cfg['model'] != 'lstm' or cfg['extra_input'] or cfg['train_stride'] != 1:
            raise SpecError("state='carry' supports the plain LSTM with no extra input and train_stride 1")
        if ew is not None:
            raise SpecError("epoch_windows is not implemented for state='carry'")


def load_config(exp, configs=None):
    """(raw spec dict, run settings = BASE updated with the spec's changes), validated."""
    s = load(exp, configs)
    cfg = dict(BASE)
    cfg.update(s.changes)
    validate(cfg)
    return s.raw, cfg
