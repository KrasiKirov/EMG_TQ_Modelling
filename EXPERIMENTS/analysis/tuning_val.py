"""Session-1 validation loss of the hyperparameter tuning, for the Results page's tuning study.

    python -m EXPERIMENTS.analysis.tuning_val [--out EXPERIMENTS/dashboard_react/src/data/tuning-val.json]

Two rounds, both judged by select_hparams.py's rule on validation loss only (session 2 is never read):
the screen (EXPERIMENTS/configs/HT_*.json) and the final round (HF_*.json), each against last semester's network
(u8, 2 layers, dropout 0.3, learning rate 0.003). Written: every final-round run's best validation loss (mean-squared
error in Nm², purged validation split), the rule's summary per network, the screen's size and the networks it sent to
the final round. Settings without local runs keep what the existing file has.
"""
import argparse
import json
from pathlib import Path

from EXPERIMENTS.analysis.select_hparams import PEOPLE, choose, finalists, load_table

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'EXPERIMENTS/results'
OUT = ROOT / 'EXPERIMENTS/dashboard_react/src/data/tuning-val.json'
BASE = 'u8_l2_d30_lr003'


def round_summary(prefix):
    table, coverage = load_table(pattern=f'{prefix}_*.json')
    best, chosen, summary = choose(table, f'{prefix}_{BASE}')
    return table, coverage, best, chosen, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    table, coverage, best, chosen, summary = round_summary('HF')
    networks = {}
    for cfg, s in summary.items():
        seeds = {}
        for p in PEOPLE:
            files = sorted((RESULTS / cfg / p / 'S1toS2').glob('seed*.json'))
            seeds[p] = {f.stem[4:]: json.loads(f.read_text())['train_info']['best_val_loss'] for f in files}
        networks[cfg] = dict(n_params=s['n_params'], mean_log_ratio=s['mean_log_ratio'], se=s['se'],
                             better_people=s['better_people'], n_people=s['n_people'],
                             median=table[cfg][1], seeds=seeds)
    s_table, s_cov, s_best, s_chosen, s_summary = round_summary('HT')
    out = dict(units='Nm²', baseline=f'HF_{BASE}', best=best, chosen=chosen, networks=networks,
               seeds_per_person=sorted(set(coverage.values())),
               screen=dict(configs=len(s_table), seeds_per_person=sorted(set(s_cov.values())), best=s_best, chosen=s_chosen,
                           finalists=finalists(s_summary, f'HT_{BASE}', k=4, also=(s_chosen,))))
    a.out.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'{a.out}: {len(networks)} final networks, chosen {chosen}; screen {len(s_table)} configurations')


if __name__ == '__main__':
    main()
