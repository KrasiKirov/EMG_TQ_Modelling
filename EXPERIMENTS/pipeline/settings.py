"""Run settings: the defaults every spec starts from, the values each setting may take, and loading a spec.

Changing BASE changes config_hash for every spec. Final runs are checked against the config stored with them
(analysis/confirm.py), not a hash recomputed from BASE, so a change here does not invalidate saved results. Options
no spec uses any more (pf_extra, rest_weight) stay in BASE and must stay None.

Retired settings (RETIRED): a config may still carry them, but only with the one value the pipeline now always uses;
strip_retired drops them and refuses any other value. Tag study-v2-final keeps the removed code.
  2026-11-24 (full overlap only): train_stride and epoch_windows
  2026-11-26 (plain MSE only): balance; its 'scale' option and the 18 older specs that used it were removed
  2026-11-26 (consecutive 10 ms samples only): input_step; the 10 older specs with input_step 5 were removed
"""
from __future__ import annotations

from EXPERIMENTS.pipeline.spec import SpecError, load

BASE = dict(passive='legacy', emg='trainmax', units=8, layers=2, dropout=0.3, model='lstm',
            pf_extra=None, window=20, lr=None, train_protocol='orig', rest_weight=None,
            extra_input=None, shuffle=True, envelope='zerophase', trim_end_s=0.0,
            state='reset', lp_order=4, val_gap=False, metadata_passive=False, chunk=None, clipnorm=None,
            scale_part='train_val', batch=None)

# retired settings and the only values still accepted: full overlap (one window per sample, every window each epoch),
# plain MSE and windows of consecutive samples
RETIRED = {'train_stride': (1,), 'epoch_windows': (None, 'full'), 'balance': ('none',), 'input_step': (1,)}
_RETIRED_ON = {'train_stride': '2026-11-24 (full overlap only)', 'epoch_windows': '2026-11-24 (full overlap only)',
               'balance': '2026-11-26 (plain MSE only)', 'input_step': '2026-11-26 (consecutive samples only)'}

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
                            if key in ('passive', 'emg', 'model', 'pf_extra', 'rest_weight', 'train_protocol')
                            else f'{key} must be one of {values}, got {cfg[key]!r}')
    if cfg['state'] == 'carry' and (cfg['model'] != 'lstm' or cfg['extra_input']):
        raise SpecError("state='carry' supports the plain LSTM with no extra input")


def strip_retired(cfg: dict) -> dict:
    """``cfg`` without the retired settings; SpecError if one holds anything but its accepted value."""
    for key, ok in RETIRED.items():
        if key in cfg and cfg[key] not in ok:
            raise SpecError(f'{key}={cfg[key]!r}: {key} was retired on {_RETIRED_ON[key]}; '
                            'tag study-v2-final keeps it')
    return {k: v for k, v in cfg.items() if k not in RETIRED}


def load_config(exp, configs=None):
    """(raw spec dict, run settings = BASE updated with the spec's changes), validated."""
    s = load(exp, configs)
    cfg = strip_retired(dict(BASE, **s.changes))
    validate(cfg)
    return s.raw, cfg
