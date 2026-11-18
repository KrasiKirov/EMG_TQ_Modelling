"""Cuts inside a cell and the long-window gates (decisions D4, D23) on validation nMAE tables.

A table is {config: {person: validation nMAE}} (median over seeds). Nothing here reads test predictions.
"""
from __future__ import annotations

import numpy as np

from EXPERIMENTS.protocol import protocol as PT

_G = PT.load()['gates']


def score(val: dict[str, dict[str, float]]) -> dict[str, float]:
    """config → mean over people of log validation nMAE (lower is better)."""
    return {c: float(np.mean(np.log(list(v.values())))) for c, v in val.items()}


def keep(val: dict[str, dict[str, float]], keep: int) -> list[str]:
    """The ``keep`` best configs by score, best first (ties broken by name)."""
    s = score(val)
    return sorted(s, key=lambda c: (s[c], c))[:keep]


def best(val: dict[str, dict[str, float]]) -> dict[str, float]:
    return val[keep(val, 1)[0]]


def gate(screen: dict[str, dict[str, float]], shorter: dict[str, dict[str, dict[str, float]]],
         margin: float = _G['margin'], min_people: int = _G['min_people']) -> tuple[bool, str, dict[str, float]]:
    """D23: one comparison. The reference is the shorter length (full overlap) whose best R1 config has the best
    score; the screened length stops when its best R1 config is worse than the reference's by more than ``margin``
    in the median over people of the ratio, and for at least ``min_people`` people (on D24 common targets).
    ``shorter`` is {length cell: R1 table}. Returns (continues, reference cell, per-person ratios screened / ref)."""
    s = best(screen)
    bests = {c: best(t) for c, t in shorter.items()}
    ref = min(bests, key=lambda c: (float(np.mean(np.log(list(bests[c].values())))), c))
    rat = {p: s[p] / bests[ref][p] for p in s if p in bests[ref]}
    stops = (np.median(list(rat.values())) > 1 + margin
             and sum(r > 1 + margin for r in rat.values()) >= min_people)
    return not stops, ref, rat


def gated_lengths_to_run(decisions: dict[str, bool], order: list[str] = _G['order']) -> list[str]:
    """Gated lengths to run next, in order: every length that passed, then the next unscreened one; a stopped length
    stops every longer one."""
    out = []
    for cell in order:
        d = decisions.get(cell)
        if d is False:
            break
        out.append(cell)
        if d is None:
            break
    return out
