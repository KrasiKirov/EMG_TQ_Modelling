"""Remove every auto-gain experiment from src/data/results.json (rows, waveforms, curves, side tables).

Usage (from dashboard_react): python scripts/strip_autogain.py
The dashboard shows no auto-gain model, so their data are not shipped either.
"""
import json
import re
from pathlib import Path

PATH = Path(__file__).resolve().parents[1]/'src'/'data'/'results.json'
PAT = re.compile(r'auto-?gain|agc', re.I)

d = json.loads(PATH.read_text(encoding='utf-8'))
drop = {k for k, v in d['experiments'].items() if v.get('params', {}).get('emg') == 'autogain' or PAT.search(k)}
d['experiments'] = {k: v for k, v in d['experiments'].items() if k not in drop}
col = d['cols'].index('experiment')
d['rows'] = [r for r in d['rows'] if r[col] not in drop]
d['wave'] = {k: v for k, v in d['wave'].items() if k.split('|')[0] not in drop}
d['learning_curves'] = [c for c in d.get('learning_curves', []) if c['experiment'] not in drop]
d.pop('bias_variance', None)   # its "ours" rows are an auto-gain model
d.pop('round2', None)          # research screens, mostly auto-gain; not shown
left = len(PAT.findall(json.dumps({k: v for k, v in d.items()})))
PATH.write_text(json.dumps(d, separators=(',', ':')), encoding='utf-8')
print('dropped', sorted(drop), '| mentions left:', left)

# the presentation catalogue carries trial means and run times keyed by experiment id as well
CAT = PATH.parent/'presentation-catalog.json'
c = json.loads(CAT.read_text(encoding='utf-8'))
gone = lambda k: PAT.search(k.split('|')[0]) is not None
c['experiments'] = {k: v for k, v in c['experiments'].items() if not gone(k)}
c['rows'] = [r for r in c['rows'] if not gone(r['exp'])]
for field in ['positions', 'targets', 'predictions', 'means', 'times']:
    c[field] = {k: v for k, v in c[field].items() if not gone(k)}
used = set(c['targets'].values())
c['targetPool'] = {k: v for k, v in c['targetPool'].items() if k in used}
CAT.write_text(json.dumps(c, separators=(',', ':'), allow_nan=False), encoding='utf-8')
print('presentation catalogue mentions left:', len(PAT.findall(json.dumps({k: v for k, v in c.items() if k not in ('targetPool', 'predictions')}))))
