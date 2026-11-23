"""Immutable trial inventory; uncertain labels are never resolved using model scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ML.config import DATA_DIR, SUBJECTS
from ML.preprocessing.dataset_builder import _parse_comment
from ML.preprocessing.flb_reader import read_flb

CHANNELS = ('gm', 'gl', 'sol', 'ta')
SCHEMA_VERSION = 1
# These records have conflicting labels/position sequences. Neither the worksheet
# nor waveform similarity independently establishes their exact active-trial ID.
AMBIGUOUS_YES = {27, 29, 31, 33}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(subject, trials, source_hash, overrides=None):
    """Overrides require evidence and use one-based trial numbers; raw frames stay untouched."""
    records = []
    overrides = overrides or {}
    for frame in trials:
        number = int(frame.attrs['trial_index']) + 1
        kind, session, position = _parse_comment(frame.attrs.get('comment', ''))
        evidence = 'FLB comment; automated signal checks (not independent label verification)'
        confidence = 'comment'
        if subject == 'IES' and kind == 'active' and session is None:
            session = 'test'
            evidence = 'IES metadata: first PUSH series is Test; second series is Re-Test'
            confidence = 'metadata'
        status = 'included'
        reason = ''
        if subject == 'YES' and number in AMBIGUOUS_YES:
            status, reason = 'ambiguous', 'Conflicting passive label / position sequence; unresolved trial identity'
        override = overrides.get(str(number), {})
        if override:
            if not override.get('evidence'):
                raise ValueError(f'{subject} trial {number}: override requires evidence')
            kind = override.get('kind', kind)
            session = override.get('session', session)
            position = override.get('position_id', position)
            status = override.get('status', status)
            reason = override.get('reason', '' if override.get('status') == 'included' else reason)
            evidence, confidence = override['evidence'], 'override'
        if kind not in ('active', 'passive', 'mvc') or session not in (None, 'test', 'retest'):
            raise ValueError('Invalid manifest type/session')
        if status not in ('included', 'excluded', 'ambiguous'):
            raise ValueError('Invalid inclusion status')
        if position is not None and (not isinstance(position, str) or not position):
            raise ValueError('Position identity must be a nonempty string')
        intervals = override.get('valid_intervals', [[0, len(frame)]])
        previous = 0
        if not intervals:
            raise ValueError('At least one valid interval must be specified')
        for start, stop in intervals:
            if not (isinstance(start, int) and isinstance(stop, int)
                    and previous <= start < stop <= len(frame)):
                raise ValueError('Manifest intervals must be ordered, disjoint raw sample ranges')
            previous = stop
        import pandas as pd
        qc_frame = pd.concat([frame.iloc[start:stop] for start, stop in intervals])
        required = ['position', 'torque', *CHANNELS]
        missing = set(required) - set(frame.columns)
        finite = not missing and bool(np.isfinite(qc_frame[required].to_numpy()).all())
        span = float(np.ptp(np.quantile(qc_frame.position, [.05, .95]))) if finite else None
        signal_qc = {}
        if finite:
            for channel in required:
                values = qc_frame[channel].to_numpy()
                signal_qc[channel] = dict(minimum=float(values.min()), maximum=float(values.max()),
                    sd=float(values.std()), rms=float(np.sqrt(np.mean(values ** 2))),
                    endpoint_fraction=float(np.mean((values == values.min()) | (values == values.max()))),
                    max_adjacent_step=float(max(np.max(np.abs(np.diff(frame[channel].iloc[a:b])), initial=0.)
                                                for a, b in intervals)))
        if not finite:
            status, reason = 'excluded', 'Missing channels or non-finite signal values'
        elif kind == 'active' and span > .03:
            status, reason = 'excluded', 'Position 5th–95th percentile span exceeds 0.03 rad'
        elif kind == 'active' and (session is None or position is None):
            status, reason = 'ambiguous', 'Active trial lacks explicit session/position'
        records.append(dict(subject=subject, trial=number, kind=kind, session=session,
                            position_id=position, status=status, reason=reason,
                            confidence=confidence, evidence=evidence,
                            comment=frame.attrs.get('comment', ''), samples=len(frame),
                            sample_interval_s=float(frame.attrs['domainIncr']),
                            domain_start_s=float(frame.attrs.get('domainStart', 0.)),
                            mean_position_rad=float(qc_frame.position.mean()) if finite else None,
                            position_span_rad=span,
                            torque_sd_nm=float(qc_frame.torque.std(ddof=0)) if finite else None,
                            signal_qc=signal_qc,
                            valid_intervals=intervals))
    return dict(schema_version=SCHEMA_VERSION, subject=subject, source_sha256=source_hash,
                passive_calibration=overrides.get('passive_calibration'),
                quality_policy={'active_position_span_rad': .03, 'quantiles': [.05, .95]},
                trials=records)


def load_inventory(subject, data_dir=DATA_DIR, overrides_path=None):
    filename, subject_id = SUBJECTS[subject]
    source = Path(data_dir) / filename
    overrides = {}
    if overrides_path:
        overrides = json.loads(Path(overrides_path).read_text()).get(subject, {})
    trials = read_flb(source, subject_id=subject_id)
    return trials, inventory(subject, trials, file_hash(source), overrides)


def coverage(manifest):
    return [{k: record[k] for k in ('subject', 'trial', 'session', 'kind', 'position_id',
                                   'status', 'reason')}
            for record in manifest['trials']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--output', required=True)
    parser.add_argument('--overrides')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for subject in SUBJECTS:
        _, manifest = load_inventory(subject, args.data_dir, args.overrides)
        (output / f'{subject}.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
