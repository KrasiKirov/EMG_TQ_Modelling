"""Run settings: the defaults every spec starts from, the values each setting may take, and loading a spec.

Changing BASE changes config_hash for every spec. Final runs are checked against the config stored with them
(analysis/confirm.py), not a hash recomputed from BASE, so a change here does not invalidate saved results.

Retired settings (RETIRED): a config may still carry them, but only with a value the pipeline now always uses;
strip_retired drops them and refuses any other value. The tag named with each date keeps the removed code.
  2026-11-24 (full overlap only; tag study-v2-final): train_stride and epoch_windows
  2026-11-26 (plain MSE only; tag study-v2-final): balance; its 'scale' option and the 18 older specs that used it
  2026-11-26 (consecutive 10 ms samples only; tag study-v2-final): input_step; the 10 older specs with input_step 5
  2026-11-26 (never set by any spec; tag study-v2-final): pf_extra and rest_weight
  2026-11-26 (study v2 values only; tag older-options-2026-11-26): emg, scale_part, shuffle, extra_input, trim_end_s,
             lp_order, val_gap, metadata_passive, train_protocol and clipnorm, with the passive option 'legacy' and the
             model 'ridge'; the 84 older specs that used other values were removed
  2026-11-26 (causal envelope only; tag older-options-2026-11-26): envelope; its spec FE_A0_zerophase (a descriptive
             arm, deviation 2026-11-23, whose saved runs stay) was removed
"""
from __future__ import annotations

from EXPERIMENTS.pipeline.spec import SpecError, load

BASE = dict(passive='static', units=8, layers=2, dropout=0.3, model='lstm', window=20, lr=None, state='reset',
            chunk=None, batch=None)

# retired settings and the only values still accepted. train_protocol 'orig' was the old default: the lagged-ridge
# run FE_A1 (no training schedule) stored it
RETIRED = {'train_stride': (1,), 'epoch_windows': (None, 'full'), 'balance': ('none',), 'input_step': (1,),
           'pf_extra': (None,), 'rest_weight': (None,),
           'emg': ('trainmax',), 'scale_part': ('train',), 'shuffle': (False,), 'extra_input': (None,),
           'trim_end_s': (1.0,), 'lp_order': (2,), 'val_gap': (True,), 'metadata_passive': (True,),
           'train_protocol': ('study400', 'orig'), 'clipnorm': (1.0,), 'envelope': ('causal',)}
_V2_ONLY = ('2026-11-26 (study v2 values only)', 'older-options-2026-11-26')
_RETIRED_ON = {'train_stride': ('2026-11-24 (full overlap only)', 'study-v2-final'),
               'epoch_windows': ('2026-11-24 (full overlap only)', 'study-v2-final'),
               'balance': ('2026-11-26 (plain MSE only)', 'study-v2-final'),
               'input_step': ('2026-11-26 (consecutive samples only)', 'study-v2-final'),
               'pf_extra': ('2026-11-26 (never set by any spec)', 'study-v2-final'),
               'rest_weight': ('2026-11-26 (never set by any spec)', 'study-v2-final'),
               **{k: _V2_ONLY for k in ('emg', 'scale_part', 'shuffle', 'extra_input', 'trim_end_s', 'lp_order',
                                        'val_gap', 'metadata_passive', 'train_protocol', 'clipnorm')},
               'envelope': ('2026-11-26 (causal envelope only)', 'older-options-2026-11-26')}

# every run learns from session 1 and predicts session 2 (study v2 is S1→S2 only); there is no other direction
DIRECTION = 'S1toS2'


def _allowed() -> dict[str, tuple]:
    from ML.research.passive import KINDS as PASSIVE_KINDS
    return {
        'passive': PASSIVE_KINDS + ('none',),
        'model': ('lstm', 'ridge_lagged'),
        'state': ('reset', 'carry'),
    }


def validate(cfg: dict) -> None:
    """Raise SpecError naming the setting if ``cfg`` has an unknown key or a value the pipeline does not run."""
    unknown = sorted(set(cfg) - set(BASE))
    if unknown:
        raise SpecError(f'unknown setting(s) {", ".join(unknown)}')
    for key, values in _allowed().items():
        if cfg[key] not in values:
            raise SpecError(f'{key} must be one of {values}, got {cfg[key]!r} (removed options: tags '
                            'pre-cleanup-2026-11 and older-options-2026-11-26)')
    if cfg['state'] == 'carry' and cfg['model'] != 'lstm':
        raise SpecError("state='carry' supports the LSTM only")


def strip_retired(cfg: dict) -> dict:
    """``cfg`` without the retired settings; SpecError if one holds anything but its accepted value."""
    for key, ok in RETIRED.items():
        if key in cfg and cfg[key] not in ok:
            when, tag = _RETIRED_ON[key]
            raise SpecError(f'{key}={cfg[key]!r}: {key} was retired on {when}; tag {tag} keeps it')
    if cfg.get('train_protocol') == 'orig' and cfg.get('model') != 'ridge_lagged':
        raise SpecError("train_protocol='orig' was retired on 2026-11-26 (study v2 values only); "
                        'tag older-options-2026-11-26 keeps it')
    return {k: v for k, v in cfg.items() if k not in RETIRED}


def load_config(exp, configs=None):
    """(raw spec dict, run settings = BASE updated with the spec's changes), validated."""
    s = load(exp, configs)
    cfg = strip_retired(dict(BASE, **s.changes))
    validate(cfg)
    return s.raw, cfg
