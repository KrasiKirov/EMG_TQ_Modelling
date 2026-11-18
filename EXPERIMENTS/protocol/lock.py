"""The session-2 lock (decision D7): test predictions of study v2 are opened only after A0 is chosen.

``results_v2/choice.json`` is written by ``select.choose_a0``. Before it exists, ``load_test`` refuses to open any
``seedN.npz`` (the session-2 predictions); selection code reads ``seedN.json`` and ``seedN_val.npz`` only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from EXPERIMENTS.pipeline.run import RESULTS_V2


class TestLocked(RuntimeError):
    """Session-2 predictions were requested before A0 was chosen."""


def test_open(results_dir: Path | str = RESULTS_V2) -> bool:
    return (Path(results_dir) / 'choice.json').exists()


def load_test(path: Path | str, results_dir: Path | str = RESULTS_V2):
    if not test_open(results_dir):
        raise TestLocked(f'{path}: session-2 predictions stay closed until {Path(results_dir) / "choice.json"} exists')
    return np.load(path)
