"""One-off: move dashboard text for norm/* entries from the catalogue and presentation.ts into specs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'EXPERIMENTS/configs'
cat = json.loads((ROOT / 'EXPERIMENTS/dashboard_react/src/data/presentation-catalog.json').read_text(encoding='utf-8'))

PLACE = {   # dashboard id -> (spec id, group, sidebar label, order)
    'norm/A0':  ('A0_active_trainmax', 'Models', 'Balanced positions · active torque', 10),
    'norm/N0':  ('N0_total_trainmax', 'Normalization study', 'EMG ÷ session-1 maximum (control, total torque)', 10),
    'norm/N0cal': ('N0cal_total_trainmax', 'Calibration experiments', 'Total torque + 10 s calibration', 20),
    'norm/A0cal': ('A0cal_active_trainmax', 'Calibration experiments', 'Active torque + 10 s calibration', 30),
    'norm/N1': ('N1_total_trainp99', 'Normalization study', 'EMG ÷ session-1 99th percentile', 20),
    'norm/N2': ('N2_total_mvc', 'Normalization study', 'EMG ÷ maximum-effort (MVC) peak', 30),
    'norm/N3': ('N3_total_sessionp99', 'Normalization study', 'EMG ÷ each session’s 99th percentile', 40),
    'norm/NR0': ('NR0_total_trainmax', 'Normalization study', 'Ridge reference · session-1 maximum', 50),
    'norm/NR1': ('NR1_total_trainp99', 'Normalization study', 'Ridge reference · session-1 99th pct', 51),
    'norm/NR2': ('NR2_total_mvc', 'Normalization study', 'Ridge reference · MVC peak', 52),
    'norm/NR3': ('NR3_total_sessionp99', 'Normalization study', 'Ridge reference · session 99th pct', 53),
}
NAMES = {  # long display names, copied from src/lib/presentation.ts DISPLAY_NAMES
    'norm/A0': 'Balanced positions (predicts active torque)',
    'norm/A0cal': 'Balanced positions + 10 s calibration (predicts active torque)',
    'norm/N0': 'Balanced positions (predicts total torque)',
    'norm/N0cal': 'Balanced positions + 10 s calibration (predicts total torque)',
    'norm/N1': 'EMG ÷ session-1 99th percentile (instead of the maximum)',
    'norm/N2': 'EMG ÷ maximum-effort (MVC) peak',
    'norm/N3': 'EMG ÷ each session’s own 99th percentile',
    'norm/NR0': 'Ridge regression reference · EMG ÷ session-1 maximum',
    'norm/NR1': 'Ridge regression reference · EMG ÷ session-1 99th percentile',
    'norm/NR2': 'Ridge regression reference · EMG ÷ MVC peak',
    'norm/NR3': 'Ridge regression reference · EMG ÷ each session’s 99th percentile',
}
BLURBS = {  # copied from src/lib/presentation.ts BLURBS
    'norm/N0cal': 'The LSTM trained on session 1, unchanged; in session 2, the first 10 seconds of pushing at each angle fit a stretch and a shift. Scored from 30 s on.',
    'norm/N0': 'Trained on session 1, tested on session 2, no calibration. Predicts the total measured torque.',
    'norm/A0': 'Balanced positions; the network predicts active torque, with each session’s relaxed-hold passive torque subtracted.',
    'norm/A0cal': 'The active-torque LSTM plus the same 10-second calibration per angle. Scored from 30 s on.',
}
CAL = {'type': 'gain_offset', 'seconds': 10, 'score_from_s': 30}

for ident, (spec_id, group, label, order) in PLACE.items():
    e = cat['experiments'][ident]
    path = CFG / f'{spec_id}.json'
    if ident.endswith('cal'):
        base = json.loads((CFG / f"{spec_id.replace('cal_', '_')}.json").read_text(encoding='utf-8'))
        raw = dict(id=spec_id, title=e['title'], changes=base['changes'], description=e['purpose'],
                   results_from=base['id'], calibration=CAL, tags=['calibration', 'normalization'])
    else:
        raw = json.loads(path.read_text(encoding='utf-8'))
        raw.setdefault('tags', ['normalization'] + (['active torque', 'session passive'] if spec_id.startswith('A0') else []))
    raw['dashboard'] = dict(id=ident, group=group, label=label, name=NAMES[ident], order=order,
                            text={k: e[k] for k in ('purpose', 'target', 'protocol', 'overview', 'params') if k in e})
    if ident in BLURBS:
        raw['dashboard']['blurb'] = BLURBS[ident]
    path.write_text(json.dumps(raw, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(ident, '->', path.name)
