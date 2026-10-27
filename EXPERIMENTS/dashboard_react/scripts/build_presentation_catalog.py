from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
existing_path=root/'src/data/presentation-catalog.json'
existing=json.loads(existing_path.read_text(encoding='utf-8')) if existing_path.exists() else {}
data=json.loads((root/'src/data/catalog.json').read_text(encoding='utf-8'))
keep={'family/F_gru8','main/session-passive'}
data['experiments']={k:v for k,v in data['experiments'].items() if k in keep}
data['rows']=[r for r in data['rows'] if r['exp'] in keep]
for field in ['positions','targets','predictions']:
 data[field]={k:v for k,v in data[field].items() if any(k.startswith(e+'|') for e in keep)}
pool=set(data['targets'].values());data['targetPool']={k:v for k,v in data['targetPool'].items() if k in pool}
for field in ['means','times']:
 data[field]={k:v for k,v in data[field].items() if not k.startswith(('research/','family/')) or k.startswith('family/F_gru8|')}
# A catalogue refresh must not revert the imported 0.5 s + retest map runs to report-only data.
retest_map='main/session-passive'
if existing.get('experiments',{}).get(retest_map,{}).get('collection'):
 data['experiments'][retest_map]=existing['experiments'][retest_map]
 data['rows']=[r for r in data['rows'] if r['exp']!=retest_map]+[r for r in existing['rows'] if r['exp']==retest_map]
 for field in ['positions','targets','predictions','means','times']:
  data[field].update({k:v for k,v in existing[field].items() if k.startswith(retest_map+'|')})
 data['targetPool'].update({v:existing['targetPool'][v] for k,v in existing['targets'].items() if k.startswith(retest_map+'|')})
(root/'src/data/presentation-catalog.json').write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Focused catalogue saved. Earlier studies remain archived on disk.')
