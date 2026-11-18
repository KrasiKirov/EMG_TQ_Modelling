"""Study v2 search space (decisions D3, D5): the cells, the shared random draws and the anchor, read from the protocol.

Every cell gets the same 8 draws plus the anchor, so cells differ only in the window setting. Carried-state cells
train on whole trials and have no batch size, so the drawn batch is dropped for them.
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
    """{'anchor', 'd0'..'d7'} → network settings for this cell (batch dropped for carried state)."""
    out = {'anchor': dict(ANCHOR)}
    out.update({f'd{i}': d for i, d in enumerate(draws(_P['n_draws'], _P['draw_seed']))})
    if _P['cells'][cell].get('state') == 'carry':
        out = {k: {kk: vv for kk, vv in v.items() if kk != 'batch'} for k, v in out.items()}
    return out
