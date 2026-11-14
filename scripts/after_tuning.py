"""Unattended hand-off from the hyperparameter search to the window study.

    python scripts/after_tuning.py [--dry-run] [--log PATH]

1. waits until no tuning run (HT_*) is training;
2. screens the grid (150 epochs, 2 seeds) with EXPERIMENTS/analysis/select_hparams.py (validation loss);
3. finalist round: the 4 best configurations, the screen's one-standard-error pick and the previous network again with a 400-epoch limit and 3 seeds
   (HF_*), plus the grid's best with the original batch-8 training (HB8_*), 3 seeds;
4. picks the final hyperparameters among the finalists (validation loss, one-standard-error rule) and the epoch
   limit (400 if it lowers the chosen network's validation loss by at least 5 % against 150, paired by person and
   seed; otherwise 150), writes docs/HPARAM_SELECTION.md;
5. applies units / layers / dropout / learning rate / epoch limit to every current-round window-study spec, commits;
6. starts scripts/run_window_study.sh (output to --log).
Batch 8 is reported only: the study keeps batch 64 for run time.
--dry-run does 2 and shows the finalist specs it would write, without writing anything or starting runs.
Every step can be repeated: finished runs are skipped.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from EXPERIMENTS.analysis.select_hparams import choose, epoch_gain, finalists, load_table, val_losses  # noqa: E402

C = ROOT / 'EXPERIMENTS/configs'
EARLIER = ('WF1_lowpass2', 'W00b8_reference_batch8', 'W00n_reference_nogap', 'W00s_reference_scaled')
BASE = 'HT_u8_l2_d30_lr003'
N_FINAL = 4
FINAL_SEEDS = ['0', '1', '2']
GAIN_NEEDED = 0.05          # 400 epochs must lower validation loss by at least 5 % to be used in the study
SUBJECTS = ['YES', 'JM', 'EG', 'IES', 'CH', 'HM']


def tuning_running():
    r = subprocess.run(['pgrep', '-f', 'pipeline/run.py --exp HT_'], capture_output=True)
    q = subprocess.run(['pgrep', '-f', 'run_queue --exps HT_'], capture_output=True)
    return bool(r.stdout.strip() or q.stdout.strip())


def current_round():
    for p in sorted(C.glob('W*.json')):
        if p.stem.endswith('_lp4') or p.stem in EARLIER:
            continue
        yield p


def pct(log_ratio):
    return f'{100 * (np.exp(log_ratio) - 1):+.1f} %'


def finalist_specs(names):
    """{new id: spec dict}: HF_* = the grid configuration with up to 400 epochs; HB8_* = the first one with the
    original batch-8 training (same network and learning rate)."""
    out = {}
    for i, ht in enumerate(names):
        raw = json.loads((C / f'{ht}.json').read_text(encoding='utf-8'))
        ch = raw['changes']
        net = f"{ch['units']} units × {ch['layers']}, dropout {ch['dropout']}, learning rate {ch['lr']}"
        common = ('Same data and network as the tuning run ' + ht + '; chosen by session-1 validation loss only. '
                  'Training windows are fed in time order (no shuffling).')
        hf = 'HF_' + ht[3:]
        out[hf] = {'id': hf, 'title': f'Tuning finalist, 400 epochs: {net}',
                   'description': ('Finalist round of the hyperparameter search: at 150 epochs many runs were still '
                                   'improving, so the best grid configurations and the previous network are trained '
                                   'again with a 400-epoch limit (batch 64), 3 seeds. ' + common),
                   'tags': ['hyperparameter search', 'finalist'],
                   'changes': dict(ch, train_protocol='fast400')}
        if i == 0:
            hb = 'HB8_' + ht[3:]
            out[hb] = {'id': hb, 'title': f'Tuning check, batch 8: {net}',
                       'description': ("Batch-size check of the grid's best configuration with the original A0 "
                                       'training (batch 8, learning-rate factor 0.7 after 10 flat epochs, early stop '
                                       '25, at most 200 epochs). Reported only: the study keeps batch 64 for run '
                                       'time. ' + common),
                       'tags': ['hyperparameter search', 'batch size'],
                       'changes': dict(ch, train_protocol='orig')}
    return out


def table_lines(summary, best, chosen, label):
    lines = [f'| Configuration | Parameters | Validation loss vs previous network ({label}) | ± SE | People better |',
             '| --- | --- | --- | --- | --- |']
    for c in sorted(summary, key=lambda c: summary[c]['mean_log_ratio']):
        s = summary[c]
        tag = (' **best**' if c == best else '') + (' **chosen**' if c == chosen else '')
        lines.append(f"| `{c}`{tag} | {s['n_params']} | {pct(s['mean_log_ratio'])} | {100 * s['se']:.1f} % | "
                     f"{s['better_people']}/{s['n_people']} |")
    return lines


def apply(ch, protocol, dry):
    old_txt = 'LSTM 8×2 trained with batch 64'
    epochs = 400 if protocol == 'fast400' else 150
    new_txt = (f"LSTM {ch['units']} units × {ch['layers']} layer{'s' if ch['layers'] > 1 else ''}, dropout "
               f"{ch['dropout']}, learning rate {ch['lr']} (chosen by a validation-loss search), trained with batch 64 "
               f"for at most {epochs} epochs")
    n = 0
    for p in current_round():
        raw = json.loads(p.read_text(encoding='utf-8'))
        raw['changes'].update(units=ch['units'], layers=ch['layers'], dropout=ch['dropout'], lr=ch['lr'],
                              train_protocol=protocol)
        s = json.dumps(raw, indent=1, ensure_ascii=False).replace(old_txt, new_txt)
        n += 1
        if not dry:
            p.write_text(s + '\n', encoding='utf-8')
    return n, new_txt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--log', default=str(ROOT / 'EXPERIMENTS/logs/window_study_after_tuning.log'))
    ap.add_argument('--workers', default='6')
    a = ap.parse_args()
    while not a.dry_run and tuning_running():
        time.sleep(60)

    # 2. grid screen
    table, _ = load_table()
    if BASE not in table:
        sys.exit(f'baseline {BASE} incomplete; {len(table)} complete configurations - not applying anything')
    g_best, g_chosen, g_summary = choose(table, BASE)
    names = finalists(g_summary, BASE, k=N_FINAL, also=[g_chosen])
    specs = finalist_specs(names)
    print(f'grid: {len(table)} complete configurations; best {g_best}; finalists {names}')
    for sid, spec in specs.items():
        print(f"  {sid}: {spec['changes']['train_protocol']}")
    if a.dry_run:
        return

    # 3. finalist round
    for sid, spec in specs.items():
        (C / f'{sid}.json').write_text(json.dumps(spec, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    order = [s for s in specs if s.startswith('HB8_')] + [s for s in specs if s.startswith('HF_')]   # slowest first
    subprocess.run([sys.executable, '-m', 'EXPERIMENTS.pipeline.run_queue', '--exps', *order, '--subjects', *SUBJECTS,
                    '--directions', 'S1toS2', '--seeds', *FINAL_SEEDS, '--workers', a.workers], cwd=ROOT, check=False,
                   env={**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3'})

    # 4. final choice and epoch limit (a failed finalist run only drops that configuration or pair)
    f_table, _ = load_table(pattern='HF_*.json')
    f_base = 'HF_' + BASE[3:]
    if f_base not in f_table:
        sys.exit(f'finalist baseline {f_base} incomplete - not applying anything')
    f_best, f_chosen, f_summary = choose(f_table, f_base)
    ht_chosen = 'HT_' + f_chosen[3:]
    gain = epoch_gain(val_losses(f_chosen), val_losses(ht_chosen))
    protocol = 'fast400' if np.exp(gain['mean_log_ratio']) <= 1 - GAIN_NEEDED else 'fast'
    hb = [s for s in specs if s.startswith('HB8_')][0]
    hf_of_hb = 'HF_' + hb[4:]
    b8 = epoch_gain(val_losses(hb), val_losses(hf_of_hb))
    ch = json.loads((C / f'{f_chosen}.json').read_text(encoding='utf-8'))['changes']

    lines = ['# Hyperparameter selection', '',
             'All choices use session-1 validation loss (mean-squared error, purged validation split) only; '
             'session-2 test scores were not used. Every comparison is relative to the previous network '
             '(8 units × 2 layers, dropout 0.3, learning rate 0.003), person by person, so every person counts '
             'equally. Rule: the lowest mean log-ratio is the best; the chosen one is the smallest network within one '
             'standard error of it. All runs: window-study reference (200 ms, active torque, causal 2nd-order '
             'envelope, no shuffling, no per-angle scaling), batch 64.', '',
             '## 1. Grid screen (150 epochs, 2 seeds × 6 people)', '',
             f'{len(table)} complete configurations of 42: units 8/16 (1 or 2 layers) and 32 (1 layer), dropout '
             '0.1/0.3, learning rate 0.001/0.003/0.01, plus edge values for 8 and 16 units (dropout 0 with learning '
             'rate 0.01/0.03, dropout 0.1 with 0.03). Many runs were still improving at the 150-epoch limit, so this '
             f'step only picks the finalists: the {N_FINAL} best, the smallest network within one standard error of the best '
             '(marked), and the previous network.', '',
             *table_lines(g_summary, g_best, g_chosen, '150 epochs'), '',
             f'## 2. Finalists (400 epochs, {len(FINAL_SEEDS)} seeds × 6 people)', '',
             *table_lines(f_summary, f_best, f_chosen, '400 epochs'), '',
             f'**Chosen:** `{f_chosen}`: units {ch["units"]}, layers {ch["layers"]}, dropout {ch["dropout"]}, '
             f'learning rate {ch["lr"]}.', '',
             '## 3. Epoch limit', '',
             f'The chosen network with up to 400 epochs against the same network with 150 (paired by person and seed, '
             f"{gain['n_pairs']} pairs): validation loss {pct(gain['mean_log_ratio'])}, lower for "
             f"{gain['better_people']}/{gain['n_people']} people. Rule: use 400 if that is at least "
             f"{int(100 * GAIN_NEEDED)} % lower. **The study uses at most {400 if protocol == 'fast400' else 150} "
             'epochs.**', '',
             '## 4. Batch size (reported only)', '',
             f'`{hb}` (batch 8, original A0 training) against `{hf_of_hb}` (batch 64, 400 epochs), paired by person '
             f"and seed ({b8['n_pairs']} pairs): validation loss {pct(b8['mean_log_ratio'])}, lower for "
             f"{b8['better_people']}/{b8['n_people']} people. The study keeps batch 64: about 5x faster per epoch.", '']
    report = '\n'.join(lines)
    n, txt = apply(ch, protocol, dry=False)
    print(report)
    print(f'applied to {n} window-study specs: {txt}')
    (ROOT / 'docs/HPARAM_SELECTION.md').write_text(report, encoding='utf-8')
    subprocess.run(['git', 'add', 'docs/HPARAM_SELECTION.md', *[f'EXPERIMENTS/configs/{s}.json' for s in specs],
                    *[str(p.relative_to(ROOT)) for p in current_round()]], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', f'Window study uses the hyperparameters chosen on validation loss: '
                    f'{f_chosen[3:]}, at most {400 if protocol == "fast400" else 150} epochs'], cwd=ROOT, check=True)

    # 6. the study
    with open(a.log, 'w') as log:
        subprocess.run(['bash', 'scripts/run_window_study.sh'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


if __name__ == '__main__':
    main()
