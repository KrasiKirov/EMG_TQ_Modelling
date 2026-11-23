"""Helpers shared by the study v2 exports: the training computer's description and the waveform packing.

Moved unchanged from the retired v1 scripts study_scores.py (machine, read_machine) and study_waves.py (the rest).
"""
from __future__ import annotations

import base64
import importlib.metadata
import json
import platform
import subprocess

import numpy as np

SLOTS = (1.0, 0.2, 441)         # first slot start (s), slot width (s), slots: covers each trial's 1.2-89.1 s
STEP = 0.01                     # sample period of the saved predictions (s)
MISSING = -32768
WORST = 2                       # positions shown per person


def machine(raw, workers, threads):
    """The computer the runs trained on, from raw system values (see read_machine), and how the runs shared it."""
    return {'computer': raw.get('machine_name'), 'processor': raw.get('chip_type'),
            'cores': int(raw['hw.physicalcpu']), 'performance_cores': int(raw['hw.perflevel0.physicalcpu']),
            'efficiency_cores': int(raw['hw.perflevel1.physicalcpu']),
            'ram_gb': int(round(int(raw['hw.memsize']) / 2 ** 30)), 'os': f"macOS {raw['os']}",
            'accelerator': ', '.join(raw['gpus']) if raw.get('gpus') else 'none (CPU only)',
            'parallel_runs': workers, 'threads_per_run': threads}


def read_machine():
    """Raw values for machine() on this Mac: model name and chip (system_profiler, no serial numbers), core counts
    and memory (sysctl), macOS version, and any GPU plug-in TensorFlow could use (tensorflow-metal)."""
    hw = json.loads(subprocess.run(['system_profiler', 'SPHardwareDataType', '-json'], capture_output=True,
                                   text=True, check=True).stdout)['SPHardwareDataType'][0]
    raw = {k: hw.get(k) for k in ('machine_name', 'chip_type')}
    for key in ('hw.physicalcpu', 'hw.perflevel0.physicalcpu', 'hw.perflevel1.physicalcpu', 'hw.memsize'):
        raw[key] = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True, check=True).stdout.strip()
    raw['os'] = platform.mac_ver()[0]
    raw['gpus'] = []
    for pkg in ('tensorflow-metal',):
        try:
            raw['gpus'].append(f'{pkg} {importlib.metadata.version(pkg)}')
        except importlib.metadata.PackageNotFoundError:
            pass
    return raw


def envelope(ts, v):
    """per slot, the lowest and the highest sample in time order: (values, offsets in 10 ms steps)"""
    t0, width, n = SLOTS
    if np.any(np.diff(ts) > 1.5 * STEP) or np.any(np.diff(ts) <= 0):
        raise ValueError('trial has gaps or unsorted times; the slot format assumes one sample every 10 ms')
    per = int(round(width / STEP))
    tick = np.round((ts - t0) / STEP).astype(int)          # samples since the first slot start (times are float32)
    if np.any(np.abs(tick * STEP + t0 - ts) > STEP / 4):
        raise ValueError('sample times are not on the 10 ms grid')
    slot, off = tick // per, tick % per
    if slot.min() < 0 or slot.max() >= n:
        raise ValueError('trial runs outside the slot range')
    vals = np.full(2 * n, np.nan)
    offs = np.zeros(2 * n, dtype=np.uint8)
    for s in np.unique(slot):
        idx = np.flatnonzero(slot == s)
        lo, hi = idx[np.argmin(v[idx])], idx[np.argmax(v[idx])]
        for j, i in enumerate(sorted((lo, hi))):
            vals[2 * s + j] = v[i]
            offs[2 * s + j] = off[i]
    return vals, offs


def packed(a):
    q = np.where(np.isfinite(a), np.round(np.nan_to_num(a) * 100), MISSING)
    if np.any(q[q != MISSING] <= MISSING) or np.any(q > 32767):
        raise ValueError('torque outside the int16 range of 0.01 Nm')
    return base64.b64encode(q.astype('<i2').tobytes()).decode()
