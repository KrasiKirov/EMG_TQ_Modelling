"""Trial-aware metrics, strict JSON serialization, and timestamp-preserving plots."""
from pathlib import Path
import json
import numpy as np
from ML.evaluation.metrics import compute_metrics


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')


def evaluate(split, prediction):
    overall = compute_metrics(split.y, prediction)
    trials, positions = {}, {}
    m = split.metadata
    for subject in np.unique(m['subject']):
        for field, output in [('trial', trials), ('position_id', positions)]:
            for group in np.unique(m[field][m['subject'] == subject]):
                mask = (m['subject'] == subject) & (m[field] == group)
                row = compute_metrics(split.y[mask], prediction[mask])
                row.update(n=int(mask.sum()), trial_count=len(np.unique(m['trial'][mask])),
                           low_variance=bool(np.std(split.y[mask]) < .1),
                           angle_rad=float(np.mean(split.X[mask, -1, -1])))
                output[f'{subject}/{group}'] = row
    per_subject = {}
    for subject in np.unique(m['subject']):
        values = [v['rmse'] for k, v in positions.items() if k.startswith(subject + '/')]
        per_subject[str(subject)] = {'mean_position_rmse': float(np.mean(values)),
                                     'worst_position_rmse': float(np.max(values)),
                                     'position_count': len(values)}
    return dict(overall=overall, trials=trials, positions=positions, subjects=per_subject,
                mean_subject_position_rmse=float(np.mean([v['mean_position_rmse'] for v in per_subject.values()]))
                if per_subject else float('nan'))


def plot_predictions(split, prediction, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = list(dict.fromkeys(zip(split.metadata['subject'], split.metadata['trial'])))
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 2.2 * len(keys)), squeeze=False)
    for ax, (subject, trial) in zip(axes[:, 0], keys):
        mask = (split.metadata['subject'] == subject) & (split.metadata['trial'] == trial)
        ax.plot(split.metadata['time_s'][mask], split.y[mask], lw=.7, label='Measured target')
        ax.plot(split.metadata['time_s'][mask], prediction[mask], lw=.7, label='Prediction')
        ax.set(title=f'{subject} / trial {trial}', ylabel='Torque (Nm)', xlabel='Recorded time (s)')
        ax.legend(loc='upper right', fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


def aggregate_runs(rows):
    """Equal-subject means per seed; between-seed SD is not a subject confidence interval."""
    output = {}
    for kind in sorted({r['kind'] for r in rows}):
        matched = [r for r in rows if r['kind'] == kind]
        groups = {}
        for split in ('val', 'test'):
            seeds = {}
            for row in matched:
                if split not in row['splits']:
                    continue
                value = row['splits'][split]['mean_subject_position_rmse']
                if value is not None and np.isfinite(value):
                    seeds.setdefault(row['seed'], []).append(value)
            means = [float(np.mean(v)) for v in seeds.values()]
            if means:
                groups[split] = dict(mean_rmse=float(np.mean(means)),
                                     seed_sd=float(np.std(means)), seed_count=len(means),
                                     subjects_per_seed={s: len(v) for s, v in seeds.items()},
                                     interpretation='Between-seed variation, not population confidence')
        output[kind] = groups
    return output
