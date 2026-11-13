"""Sequential validation-only experiments with fixed adoption rules; no retest selection."""
import copy
import json
from pathlib import Path
import numpy as np

from ML.evaluation.report import write_json
from ML.training.benchmark import parser, run


def summarize(directory):
    rows = json.loads((Path(directory) / 'summary.json').read_text())
    by_seed = {}
    for row in rows:
        val = row['splits']['val']
        by_seed.setdefault(row['seed'], []).append(val['mean_subject_position_rmse'])
    return dict(mean=float(np.mean([np.mean(v) for v in by_seed.values()])),
                worst=float(max(p['rmse'] for r in rows for p in r['splits']['val']['positions'].values())),
                seconds=float(np.median([r['training_seconds'] for r in rows])),
                by_seed={str(k): float(np.mean(v)) for k, v in by_seed.items()})


def qualifies(candidate, reference, efficiency=False):
    if candidate['worst'] > reference['worst'] * 1.05:
        return False
    if efficiency:
        return candidate['mean'] <= reference['mean'] * 1.02 and candidate['seconds'] <= reference['seconds'] * .75
    consistent = sum(candidate['by_seed'][s] < value for s, value in reference['by_seed'].items())
    return (candidate['mean'] <= reference['mean'] * .95
            and consistent >= max(1, int(np.ceil(2 * len(reference['by_seed']) / 3))))


def main():
    p = parser()
    p.description = __doc__
    p.add_argument('--stages', nargs='+', choices=['history', 'dropout', 'loss', 'sensors', 'stride', 'batch'],
                   default=['history', 'dropout', 'loss', 'sensors', 'stride', 'batch'])
    args = p.parse_args()
    if args.evaluate_retest or args.protocol != 'within' or not args.output:
        p.error('Sweep requires --output, within protocol, and no --evaluate-retest')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    stages = args.stages
    del args.stages
    args.models = ['lstm']
    args.output = str(root / 'reference')
    current = copy.deepcopy(args)
    run(current)
    score = summarize(current.output)
    history = [{'stage': 'reference', 'config': vars(current), 'score': score}]
    candidates = {'history': ('window', [10, 20, 50]), 'dropout': ('dropout', [0., .1, .3]),
                  'loss': ('position_weights', [False, True]),
                  'sensors': ('omit_muscle', ['gm', 'gl', 'sol', 'ta']),
                  'stride': ('train_stride', [1, 5, 10]), 'batch': ('batch', [8, 32, 64])}
    for stage in stages:
        field, values = candidates[stage]
        choices = []
        for value in values:
            if value == getattr(current, field):
                continue
            candidate = copy.deepcopy(current)
            setattr(candidate, field, value)
            candidate.output = str(root / f'{stage}_{value}')
            run(candidate)
            result = summarize(candidate.output)
            accepted = qualifies(result, score, stage in ('stride', 'batch'))
            # Sensor ablations diagnose contribution; they never silently remove sensors.
            history.append(dict(stage=stage, config=vars(candidate), score=result,
                                meets_rule=accepted, adopted=False))
            if accepted and stage != 'sensors':
                choices.append((candidate, result, len(history)-1))
            write_json(root / 'selection.json', history)
        if choices:
            current, score, index = min(choices, key=lambda pair: pair[1]['seconds' if stage in ('stride', 'batch') else 'mean'])
            history[index]['adopted'] = True
    write_json(root / 'selection.json', history)
    write_json(root / 'selected_configuration.json', vars(current))


if __name__ == '__main__':
    main()
