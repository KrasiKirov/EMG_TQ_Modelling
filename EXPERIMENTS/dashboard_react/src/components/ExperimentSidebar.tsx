import { useState } from 'react';
import { AppShell, Button, ScrollArea, SegmentedControl, Stack, Switch, Text, UnstyledButton } from '@mantine/core';
import { catalogTitle } from '../lib/catalog';
import { PRESENTATION_GROUPS } from '../lib/presentation';
import type { Dir } from '../lib/types';
import { useView } from '../state';
export function ExperimentSidebar({onNavigate}:{onNavigate:()=>void}){
 const {s,set}=useView();
 // a folded group still opens when it holds the page in view
 const [open,setOpen]=useState<Record<string,boolean>>(()=>Object.fromEntries(PRESENTATION_GROUPS.map(g=>[g.label,g.open||g.items.some(x=>x.id===s.experiment)])));
 const choose=(id:string)=>{set({experiment:id,tab:'overview',runA:'typical'});onNavigate();window.scrollTo({top:0});};
 return <><AppShell.Section grow component={ScrollArea} scrollbarSize={6}><Stack p="lg" gap="lg">
 <div><Text size="xs" c="dimmed">EMG → ankle torque</Text><Text fw={600} size="lg" mt={4}>Results</Text></div>
 <nav className="focus-collections" aria-label="Experiments">{PRESENTATION_GROUPS.map(g=><section key={g.label} data-tone={g.tone} data-open={open[g.label]||undefined}>
  <button type="button" className="collection-label" aria-expanded={!!open[g.label]} onClick={()=>setOpen(o=>({...o,[g.label]:!o[g.label]}))}><b>{g.label}</b><span className="collection-count">{g.items.length}</span><span className="collection-chevron" aria-hidden>{open[g.label]?'−':'+'}</span></button>
  {open[g.label]&&<><Text size="xs" c="dimmed" mb={6}>{g.caption}</Text>{g.items.map(x=><UnstyledButton key={x.id} className="focus-choice" data-active={s.experiment===x.id&&s.tab==='overview'||undefined} aria-current={s.experiment===x.id?'true':undefined} onClick={()=>choose(x.id)}><span className="focus-marker"/><span className="focus-name">{x.label||catalogTitle(x.id)}</span>{x.badge&&<span className="focus-badge">{x.badge}</span>}</UnstyledButton>)}</>}
 </section>)}</nav>
 <Button variant="default" size="xs" fullWidth onClick={()=>{set({tab:'experiments'});onNavigate();window.scrollTo({top:0});}}>All controlled experiments</Button>
 <div className="sidebar-section"><Text size="xs" fw={600} mb="xs">Train → test</Text><SegmentedControl fullWidth size="xs" value={s.dir} onChange={v=>set({dir:v as Dir,runA:'typical'})} data={[{value:'S1toS2',label:'Session 1 → 2'},{value:'S2toS1',label:'Session 2 → 1'}]}/></div>
 <details className="disclosure"><summary>Display settings</summary><Switch checked={s.floor} onChange={e=>set({floor:e.currentTarget.checked})} label="Display negative R² as 0" size="xs"/><Text size="xs" c="dimmed" mt="xs">Raw values remain in individual-run tables.</Text></details>
 <Text size="xs" c="dimmed">Retrospective, same-person evaluation.</Text>
 </Stack></AppShell.Section><AppShell.Section p="md" hiddenFrom="md"><Button fullWidth onClick={onNavigate}>Show results</Button></AppShell.Section></>;
}
