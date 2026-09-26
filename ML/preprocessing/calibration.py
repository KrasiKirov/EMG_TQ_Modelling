"""Passive calibration from source-session plateaus or documented existing measurements."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

from ML.preprocessing.trial_manifest import file_hash


@dataclass
class PassiveCalibration:
    positions: list
    torques: list
    evidence: list
    support_margin_rad: float = .015

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float).tolist()
        self.torques = np.asarray(self.torques, dtype=float).tolist()
        if len(self.positions) != len(self.torques) or not np.isfinite(self.positions + self.torques).all():
            raise ValueError('Invalid calibration arrays')
        if len(self.positions) > 1 and not np.all(np.diff(self.positions) > 0):
            raise ValueError('Calibration positions must be strictly increasing')
        if not 0 <= self.support_margin_rad <= .05:
            raise ValueError('Invalid support tolerance')

    def predict(self, positions):
        """Clamp only within measurement tolerance; unsupported positions are NaN."""
        positions = np.asarray(positions)
        if len(self.positions) < 2:
            return np.full(positions.shape, np.nan)
        result = np.interp(positions, self.positions, self.torques)
        supported = ((positions >= self.positions[0] - self.support_margin_rad)
                     & (positions <= self.positions[-1] + self.support_margin_rad))
        return np.where(supported, result, np.nan)

    def to_dict(self):
        return asdict(self)


def _documented_ies(data_dir):
    """Read the existing DOCX table, not estimated from active outcomes."""
    source = Path(data_dir) / 'experiment_metadata_IES.docx'
    if not source.exists():
        return None
    # Confirm the table still contains the reviewed paired measurements rather than
    # trusting a filename if a user replaces the metadata document.
    import zipfile
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(source) as archive:
        root = ET.fromstring(archive.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    found = {}
    for table in root.findall('.//w:tbl', ns):
        rows = [[''.join(t.text or '' for t in c.findall('.//w:t', ns))
                 for c in row.findall('w:tc', ns)] for row in table.findall('w:tr', ns)]
        if not rows or 'Passive TQ (Nm)' not in rows[0] or 'Position (rad)' not in rows[0]:
            continue
        pi, ti = rows[0].index('Position (rad)'), rows[0].index('Passive TQ (Nm)')
        for cells in rows[1:]:
            if len(cells) <= max(pi, ti):
                continue
            position, torque = float(cells[pi]), float(cells[ti])
            if not np.isfinite([position, torque]).all() or position in found:
                raise ValueError('Invalid or duplicate metadata calibration position')
            found[position] = torque
    if len(found) != 8:
        raise ValueError('IES passive table must contain eight distinct positions; review metadata')
    positions = sorted(found)
    return PassiveCalibration(positions, [found[p] for p in positions],
                              [{'source': source.name, 'sha256': file_hash(source),
                                'table': 'MVC and Passive TQ data from MVC trials',
                                'session': 'pre-test calibration', 'precision': 'recorded table values'}])


def fit_passive(trials, manifest, data_dir=None):
    override = manifest.get('passive_calibration')
    if override:
        calibration = PassiveCalibration(**override)
        if not calibration.evidence or any(e.get('session') not in ('test', 'pre-test calibration')
                                            for e in calibration.evidence):
            raise ValueError('Fixed passive override requires source-session evidence')
        for item in calibration.evidence:
            if not data_dir or file_hash(Path(data_dir) / item['source']) != item['sha256']:
                raise ValueError('Calibration evidence hash does not match source')
        return calibration
    if manifest['subject'] == 'IES' and data_dir:
        documented = _documented_ies(data_dir)
        if documented is not None:
            return documented
    points = []
    evidence = []
    for record, frame in zip(manifest['trials'], trials):
        if (record['kind'] != 'passive' or record['status'] != 'included'
                or record['session'] == 'retest'):
            continue
        fs = round(1 / record['sample_interval_s'])
        # Require contiguous stationary plateaus >=3 s; discard 0.5 s at each edge.
        for start, stop in record['valid_intervals']:
            bins = []
            for a in range(start, stop - fs + 1, fs):
                segment = frame.iloc[a:a + fs]
                bins.append((a, bool(segment.torque.std() <= .5
                                     and np.ptp(segment.position) <= .01)))
            runs = []
            first = None
            for a, accepted in bins + [(stop, False)]:
                if accepted and first is None:
                    first = a
                if not accepted and first is not None:
                    if a - first >= 3 * fs:
                        runs.append((first + fs // 2, a - fs // 2))
                    first = None
            # These recordings contain nonstationary tails after the initial hold.
            # Later quiet fragments are NOT independent passive measurements. A
            # reviewed valid_intervals override can explicitly select another hold.
            for a, b in runs[:1]:
                if a - start > 2.5 * fs:
                    continue
                segment = frame.iloc[a:b]
                # Check the entire plateau too; smooth drifts can pass individual bins.
                if segment.torque.std() > .5 or np.ptp(segment.position) > .02:
                    continue
                position = float(segment.position.median())
                torque = float(segment.torque.median())
                points.append((position, torque))
                evidence.append(dict(trial=record['trial'], session=record['session'],
                                     interval=[a, b], position=position, torque=torque,
                                     torque_sd_nm=float(segment.torque.std()),
                                     emg_rms={c: float(np.std(segment[c])) for c in ('gm', 'gl', 'sol', 'ta')},
                                     relaxation_evidence='passive acquisition label; EMG RMS retained for review'))
    # Each recorded plateau contributes once (longer recordings do not dominate).
    clusters = []
    for p, t in sorted(points):
        if clusters and abs(p - np.mean([v[0] for v in clusters[-1]])) <= .025:
            clusters[-1].append((p, t))
        else:
            clusters.append([(p, t)])
    return PassiveCalibration([float(np.mean([p for p, _ in g])) for g in clusters],
                              [float(np.median([t for _, t in g])) for g in clusters], evidence)
