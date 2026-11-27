"""Statistics of study v2 (decision D9): the nMAE scale, per-person effects, a descriptive t-interval and the decision rule.

Effect of an arm for one person = log(median nMAE of the arm over its seeds / median nMAE of the comparator).
"Differs" only when all people agree in sign and the absolute median effect is at least log(1 + min_effect);
with 6 people this is reported as "consistent across all 6 people", not as significant.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats as _st


def person_scale(y_measured) -> float:
    """Normalising scale of one person and target: 99th percentile of |measured target| (never a prediction)."""
    return float(np.percentile(np.abs(np.asarray(y_measured, float)), 99))


def person_effects(arm: dict[str, dict[int, float]], ref: dict[str, dict[int, float]]) -> dict[str, float]:
    """{person: log(median_seeds(arm) / median_seeds(ref))} for the people both have."""
    return {p: float(np.log(np.median(list(arm[p].values())) / np.median(list(ref[p].values()))))
            for p in arm if p in ref}


def t_interval(effects: dict[str, float], level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi): mean of the per-person effects ± t(n - 1) · SD / √n. Descriptive only."""
    v = np.asarray(list(effects.values()), float)
    m = float(v.mean())
    half = float(_st.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, m - half, m + half


def decide(effects: dict[str, float], min_agree: int = 6,
           min_effect: float = 0.05) -> Literal['better', 'worse', 'no clear difference']:
    """'better' (lower error) / 'worse' when at least ``min_agree`` people share the sign and the absolute median
    effect is at least log(1 + min_effect); otherwise 'no clear difference'."""
    v = np.asarray(list(effects.values()), float)
    med = float(np.median(v))
    if abs(med) < np.log1p(min_effect):
        return 'no clear difference'
    agree = int(np.sum(np.sign(v) == np.sign(med)))
    if agree < min_agree:
        return 'no clear difference'
    return 'better' if med < 0 else 'worse'
