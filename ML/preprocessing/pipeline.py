"""Audited offline pipeline: split raw records, filter independently, fit on train only."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import numpy as np
from scipy.signal import resample_poly

from ML.config import DATA_DIR
from ML.preprocessing.calibration import PassiveCalibration, fit_passive
from ML.preprocessing.emg_envelope import extract_envelope
from ML.preprocessing.trial_manifest import CHANNELS, load_inventory


@dataclass(frozen=True)
class PipelineConfig:
    sample_rate: int = 1000
    downsample: int = 10
    window: int = 20
    train_stride: int = 1
    train_fraction: float = .8
    gap_s: float = 1.0
    edge_trim_s: float = 1.0
    lp_cutoff: float = 2.0
    target_mode: str = 'active_torque'

    def __post_init__(self):
        if self.target_mode not in ('active_torque', 'measured_torque'):
            raise ValueError('Unknown target_mode')
        if min(self.sample_rate, self.downsample, self.window, self.train_stride) < 1:
            raise ValueError('Sampling and window parameters must be positive')
        if self.sample_rate % self.downsample or not 0 < self.train_fraction < 1:
            raise ValueError('Invalid downsampling or split fraction')
        if self.edge_trim_s < 0 or self.gap_s < 0 or not 0 < self.lp_cutoff < self.sample_rate / 2:
            raise ValueError('Invalid filter or boundary configuration')


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    measured: np.ndarray
    metadata: dict

    def select(self, mask):
        return SplitData(self.X[mask], self.y[mask], self.measured[mask],
                         {k: v[mask] for k, v in self.metadata.items()})


def join_splits(splits):
    if not splits:
        raise ValueError('No splits to concatenate')
    return SplitData(*(np.concatenate([getattr(s, k) for s in splits])
                       for k in ('X', 'y', 'measured')),
                     {k: np.concatenate([s.metadata[k] for s in splits]) for k in splits[0].metadata})


@dataclass
class DatasetBundle:
    splits: dict
    config: PipelineConfig
    preprocessing: dict
    manifest: dict
    coverage: list = field(default_factory=list)

    def __getitem__(self, split):
        return self.splits[split]


def raw_blocks(record, config):
    """[start, stop) support ranges; each is filtered in isolation."""
    blocks = []
    for start, stop in record['valid_intervals']:
        if record['session'] == 'retest':
            blocks.append(('test', start, stop))
        else:
            cut = start + int((stop - start) * config.train_fraction)
            half = int(np.ceil(config.gap_s * config.sample_rate / 2))
            blocks.extend([('train', start, cut - half), ('val', cut + half, stop)])
    return blocks


def process_block(frame, start, stop, config):
    if not 0 <= start < stop <= len(frame):
        raise ValueError('Invalid raw block bounds')
    required = ['position', 'torque', *CHANNELS]
    if not set(required).issubset(frame.columns):
        raise ValueError('Required signal channel missing')
    dt = float(frame.attrs.get('domainIncr', 1 / config.sample_rate))
    if not np.isclose(dt, 1 / config.sample_rate, rtol=1e-5):
        raise ValueError('Sampling rate mismatch: explicitly resample or change configuration')
    segment = frame.iloc[start:stop]
    if not np.isfinite(segment[required].to_numpy()).all():
        raise ValueError('Non-finite input')
    trim = int(np.ceil(config.edge_trim_s * config.sample_rate / config.downsample))
    minimum = (2 * trim + config.window) * config.downsample
    if len(segment) < max(minimum, 64):
        raise ValueError('Block too short after edge trimming and windowing')
    env = np.column_stack([extract_envelope(segment[c].to_numpy(), fs=config.sample_rate,
                                            lp_cutoff=config.lp_cutoff) for c in CHANNELS])
    signals = np.column_stack([env, segment.position, segment.torque])
    sampled = resample_poly(signals, 1, config.downsample, axis=0, padtype='line')
    index = np.arange(len(sampled)) * config.downsample + start
    if trim:
        sampled, index = sampled[trim:-trim], index[trim:-trim]
    sampled[:, :4] = np.maximum(sampled[:, :4], 0)
    return sampled.astype(np.float32), index


def empty_split(config):
    metadata = {k: np.array([], dtype=str if k in ('subject', 'position_id') else np.int64)
                for k in ('subject', 'trial', 'position_id', 'raw_start', 'raw_stop',
                          'window_start', 'sample_index')}
    metadata['time_s'] = np.array([], dtype=float)
    return SplitData(np.empty((0, config.window, 5), np.float32), np.array([], np.float32),
                     np.array([], np.float32), metadata)


def windows(signals, index, record, support, config, passive=None, stride=1):
    X = np.lib.stride_tricks.sliding_window_view(signals[:, :5], config.window, axis=0)
    X = np.transpose(X, (0, 2, 1))[::stride].copy()
    ends = np.arange(config.window - 1, len(signals), stride)
    measured = signals[ends, 5].copy()
    y = measured.copy()
    if config.target_mode == 'active_torque':
        if passive is None or len(passive.positions) < 2:
            return empty_split(config)
        y -= passive.predict(signals[ends, 4])
    n = len(y)
    meta = dict(subject=np.repeat(record['subject'], n), trial=np.full(n, record['trial']),
                position_id=np.repeat(record['position_id'], n),
                raw_start=np.full(n, support[0]), raw_stop=np.full(n, support[1]),
                window_start=index[ends - config.window + 1], sample_index=index[ends],
                time_s=index[ends] / config.sample_rate + record.get('domain_start_s', 0.))
    result = SplitData(X, y, measured, meta)
    return result.select(np.isfinite(y))


def build_bundle(trials, manifest, config=None, data_dir=None):
    config = config or PipelineConfig()
    passive = fit_passive(trials, manifest, data_dir) if config.target_mode == 'active_torque' else None
    processed = []
    coverage = []
    for record, frame in zip(manifest['trials'], trials):
        if record['kind'] != 'active' or record['status'] != 'included':
            coverage.append({'trial': record['trial'], 'status': record['status'],
                             'reason': record['reason'] or record['kind']})
            continue
        for split, start, stop in raw_blocks(record, config):
            try:
                signal, index = process_block(frame, start, stop, config)
            except ValueError as exc:
                coverage.append({'trial': record['trial'], 'split': split, 'status': 'unavailable',
                                 'reason': str(exc)})
                continue
            processed.append((split, signal, index, record, (start, stop)))
    train_signals = [s for split, s, *_ in processed if split == 'train']
    if not train_signals:
        raise ValueError(f"{manifest['subject']}: no eligible training blocks")
    scales = np.maximum(np.max(np.vstack([s[:, :4].max(axis=0) for s in train_signals]), axis=0), 1e-10)
    pieces = {s: [] for s in ('train', 'val', 'test')}
    for split, signal, index, record, support in processed:
        signal[:, :4] /= scales
        data = windows(signal, index, record, support, config, passive,
                       config.train_stride if split == 'train' else 1)
        pieces[split].append(data)
        coverage.append(dict(trial=record['trial'], position_id=record['position_id'], split=split,
                             windows=len(data.y), status='included' if len(data.y) else 'unavailable',
                             reason='' if len(data.y) else 'No supported passive calibration'))
    splits = {s: join_splits(p) if p else empty_split(config) for s, p in pieces.items()}
    metadata = dict(schema_version=1, config=asdict(config), channels=list(CHANNELS),
                    emg_scales=scales.tolist(), passive=passive.to_dict() if passive else None,
                    position_units='rad', target_units='Nm', normalization='training-block max',
                    mode='offline', edge_policy='independently filtered raw blocks; trimmed both ends',
                    source_sha256=manifest['source_sha256'],
                    manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
    return DatasetBundle(splits, config, metadata, manifest, coverage)


def load_bundle(subject, config=None, data_dir=DATA_DIR, overrides_path=None):
    trials, manifest = load_inventory(subject, data_dir, overrides_path)
    return build_bundle(trials, manifest, config, data_dir)


def source_normalize(bundles, target):
    """For transfer, undo per-subject scaling and fit one scale on source training only.

    Passive calibration is still subject-specific in active-torque mode; the
    protocol must disclose that calibration. Target EMG scales are NOT fitted.
    """
    scales = np.maximum(np.max(np.vstack([np.asarray(b.preprocessing['emg_scales'])
                                         for b in bundles]), axis=0), 1e-10)
    def convert(bundle, split):
        data = bundle[split]
        X = data.X.copy()
        X[:, :, :4] *= (np.asarray(bundle.preprocessing['emg_scales']) / scales).astype(np.float32)
        return SplitData(X, data.y.copy(), data.measured.copy(), dict(data.metadata))
    return ({s: join_splits([convert(b, s) for b in bundles]) for s in ('train', 'val')},
            {s: convert(target, s) for s in ('train', 'val', 'test')}, scales)
