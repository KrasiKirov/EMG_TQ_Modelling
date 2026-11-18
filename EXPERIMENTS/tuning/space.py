"""Study v2 search space (decisions D3, D5): cells, coarse and fine random draws, the anchor; read from the protocol.

Coarse (R1): every cell gets the same 6 draws plus the anchor, so cells differ only in the window setting.
Fine (R2): 4 draws (2 after the budget step fine_2) in a narrowed range around the cell's best coarse config (learning rate x/÷ 3, dropout ± 0.1,
units and batch the same or one step either side, layers the same), inside the coarse space; the same random
numbers in every cell. Carried-state cells train on whole trials and have no batch size, so batch is dropped.
"""
from __future__ import annotations

import numpy as np

from EXPERIMENTS.protocol import protocol as PT

RUN_KEYS = ('window', 'train_stride', 'epoch_windows', 'state', 'chunk')   # cell keys that are run settings
_P = PT.load()
ANCHOR = dict(_P['anchor'])


def cells() -> dict[str, dict]:
    """{cell: protocol entry} for the 15 always-run and 12 gated cells."""
    return {k: dict(v) for k, v in _P['cells'].items()}


def changes(cell: str) -> dict:
    """The run settings of one cell (the keys run.py reads)."""
    return {k: v for k, v in _P['cells'][cell].items() if k in RUN_KEYS}


def draws(n: int, seed: int) -> list[dict]:
    """n configurations drawn from the protocol's space with numpy's default_rng(seed)."""
    rng, sp = np.random.default_rng(seed), _P['space']
    lo, hi = np.log(sp['lr']['log_uniform'])
    out = []
    for _ in range(n):
        out.append(dict(units=int(rng.choice(sp['units']['choice'])),
                        layers=int(rng.choice(sp['layers']['choice'])),
                        dropout=round(float(rng.uniform(*sp['dropout']['uniform'])), 3),
                        lr=float(f'{np.exp(rng.uniform(lo, hi)):.3g}'),
                        batch=int(rng.choice(sp['batch']['choice']))))
    return out


def configs(cell: str) -> dict[str, dict]:
    """{'anchor', 'd0'..'d5'} → network settings for this cell (batch dropped for carried state)."""
    out = {'anchor': dict(ANCHOR)}
    out.update({f'd{i}': d for i, d in enumerate(draws(_P['n_draws'], _P['draw_seed']))})
    if _P['cells'][cell].get('state') == 'carry':
        out = {k: {kk: vv for kk, vv in v.items() if kk != 'batch'} for k, v in out.items()}
    return out


def _near(choices, v, rng):
    i = choices.index(v)
    return int(rng.choice(choices[max(0, i - 1):i + 2]))


def fine_draws(center: dict, n: int, seed: int) -> list[dict]:
    """n configurations in the narrowed range around ``center`` (an R1 config), drawn with default_rng(seed)."""
    rng, sp, f = np.random.default_rng(seed), _P['space'], _P['fine']
    lo_s, hi_s = sp['lr']['log_uniform']
    lo, hi = np.log(max(lo_s, center['lr'] / f['lr_factor'])), np.log(min(hi_s, center['lr'] * f['lr_factor']))
    d_lo, d_hi = sp['dropout']['uniform']
    out = []
    for _ in range(n):
        d = dict(units=_near(sp['units']['choice'], center['units'], rng), layers=int(center['layers']),
                 dropout=round(float(rng.uniform(max(d_lo, center['dropout'] - f['dropout_pm']),
                                                 min(d_hi, center['dropout'] + f['dropout_pm']))), 3),
                 lr=float(f'{np.exp(rng.uniform(lo, hi)):.3g}'))
        b = _near(sp['batch']['choice'], center.get('batch', 64), rng)    # drawn always, so cells share numbers
        if 'batch' in center:
            d['batch'] = b
        out.append(d)
    return out


def fine_configs(cell: str, center: dict, n: int | None = None) -> dict[str, dict]:
    """{'f0'..} (n of them, default the protocol's) for this cell around its best R1 config (batch dropped for
    carried state). The draws are sequential, so n = 2 gives the first 2 of n = 4."""
    carry = _P['cells'][cell].get('state') == 'carry'
    c = dict(center, batch=center.get('batch', 64))
    out = {f'f{i}': d for i, d in enumerate(fine_draws(c, n or _P['fine']['n'], _P['fine']['seed']))}
    return {k: {kk: vv for kk, vv in v.items() if not (carry and kk == 'batch')} for k, v in out.items()}
