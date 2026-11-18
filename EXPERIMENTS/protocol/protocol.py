"""The study v2 protocol (pre-registration): EXPERIMENTS/protocol/study_v2.json, loaded as a plain dict.

The committed JSON is the pre-registration; its commit time is the evidence. ``load`` adds ``hash`` (SHA-256 of the
file bytes) and ``deviations`` (EXPERIMENTS/protocol/deviations.json, kept by hand, outside the hash).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE / 'study_v2.json'
DEVIATIONS = HERE / 'deviations.json'


def load(path: Path | str = PATH) -> dict:
    path = Path(path)
    raw = path.read_bytes()
    out = json.loads(raw)
    out['hash'] = hashlib.sha256(raw).hexdigest()
    dev = path.parent / 'deviations.json'
    out['deviations'] = json.loads(dev.read_text()) if dev.exists() else []
    return out
