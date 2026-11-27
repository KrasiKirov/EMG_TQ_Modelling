"""The study v2 protocol (pre-registration): EXPERIMENTS/protocol/study_v2.json, loaded as a plain dict.

The committed JSON is the pre-registration; its commit time is the evidence. ``load`` adds ``hash`` (SHA-256 of the
file bytes) and ``deviations`` (EXPERIMENTS/protocol/deviations.json, kept by hand, outside the hash).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from EXPERIMENTS.paths import PROTOCOL_DIR, RESULTS_V2

PATH = PROTOCOL_DIR / 'study_v2.json'


@dataclass(frozen=True)
class Study:
    """A pre-registered study: its protocol file, the spec ids that belong to it, and where its runs are saved.
    Study runs refuse uncommitted code and record the protocol file's hash."""
    name: str
    protocol_file: Path
    prefixes: tuple
    results: Path


STUDIES = {
    'study_v2': Study('study_v2', PATH, ('TS2_', 'FE_'), RESULTS_V2),
}


def retired_cell(entry: dict) -> bool:
    """True for a protocol cell with a window overlap other than full (o75, o50, o0): retired 2026-11-24, when the
    team kept full overlap only. Their finished runs are still reported (F2); they are never tuned or run again."""
    return entry.get('overlap', 'full') != 'full'


def live_cells(protocol: dict) -> dict[str, dict]:
    """The protocol's cells without the retired overlap cells (the protocol file itself is never edited)."""
    return {c: v for c, v in protocol['cells'].items() if not retired_cell(v)}


def study_of(exp_id: str) -> Study | None:
    """The study a spec id belongs to (by its prefix), or None for a v1 experiment."""
    for st in STUDIES.values():
        if exp_id.startswith(st.prefixes):
            return st
    return None


def load(path: Path | str = PATH) -> dict:
    path = Path(path)
    raw = path.read_bytes()
    out = json.loads(raw)
    out['hash'] = hashlib.sha256(raw).hexdigest()
    dev = path.parent / 'deviations.json'
    out['deviations'] = json.loads(dev.read_text()) if dev.exists() else []
    return out
