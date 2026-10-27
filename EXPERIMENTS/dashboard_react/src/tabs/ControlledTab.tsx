import { Badge, Button, Group, Stack, Text, Title } from '@mantine/core';
import { DS } from '../dataset';
import { STORIES, STUDY_ORDER } from '../lib/experimentStory';
import { PARAM_KEYS, paramShort, shortId, resolve } from '../lib/params';
import { runsOf } from '../lib/data';
import { useView } from '../state';
import { DataTable, type Column } from '../components/common';
export function ControlledTab(){
 const {s,set}=useView();
 const rows=STUDY_ORDER.filter(id=>DS.experiments[id]).map(id=>({id,e:DS.experiments[id],subjects:DS.subjects.filter(subject=>resolve(DS,DS.experiments[id].params,subject,s.dir)!==null)}));
 const base=DS.experiments.X00_baseline.params;
 const open=(id:string)=>{set({experiment:id,tab:'overview',runA:'typical'});window.scrollTo({top:0});};
 const columns:Column<typeof rows[number]>[]=[
  {key:'id',label:'Experiment',render:r=><div><Text fw={600} size="sm">{STORIES[r.id]?.title??r.e.title}</Text><Text size="xs" c="dimmed">{shortId(r.id)}</Text></div>},
  {key:'change',label:'Changed from original',render:r=><Group gap={5}>{PARAM_KEYS.filter(k=>k!=='code'&&String(base[k])!==String(r.e.params[k])).map(k=><Badge key={k} variant="light" color="gray" fw={500}>{paramShort(k,r.e.params[k])}</Badge>)}{PARAM_KEYS.every(k=>k==='code'||String(base[k])===String(r.e.params[k]))&&<Text size="xs" c="dimmed">Reference settings</Text>}</Group>},
  {key:'subjects',label:'Subjects',render:r=>r.subjects.join(', ')||'Not run'},
  {key:'runs',label:'Seeds / subject',align:'right',render:r=>{const counts=r.subjects.map(subject=>{const res=resolve(DS,r.e.params,subject,s.dir);return new Set(res?runsOf(DS,res.exp,subject,s.dir).map(x=>x.seed):[]).size;});return counts.length?[...new Set(counts)].sort((a,b)=>a-b).join('–'):'—';}},
  {key:'open',label:'',render:r=><Button size="xs" variant="subtle" onClick={event=>{event.stopPropagation();open(r.id);}}>View results →</Button>}
 ];
 return <Stack gap="lg"><div className="page-heading"><Text size="xs" c="dimmed">Complete controlled series</Text><Title order={2}>Controlled experiments</Title><Text size="sm" c="dimmed">All 15 experiments remain available. Select a row to inspect its figures, metrics and method.</Text></div><DataTable columns={columns} rows={rows} rowKey={r=>r.id} onRowClick={r=>open(r.id)} rowClass={r=>r.id===s.experiment?'selected-row':''} minWidth={900} stickyFirst caption="All controlled experiments"/><Text size="xs" c="dimmed">The subject list follows the selected direction and includes equivalent passive-table runs where the project’s existing rule permits them. Earlier-study collections are excluded from this interface.</Text></Stack>;
}
