import { AppShell, Button, ScrollArea, SegmentedControl, Stack, Switch, Text, UnstyledButton } from '@mantine/core';
import { catalogTitle } from '../lib/catalog';
import { PRESENTATION_GROUPS } from '../lib/presentation';
import type { Dir } from '../lib/types';
import { useView } from '../state';
export function ExperimentSidebar({onNavigate}:{onNavigate:()=>void}){
 const {s,set}=useView();
 const choose=(id:string)=>{set({experiment:id,tab:'overview',runA:'typical'});onNavigate();window.scrollTo({top:0});};
 return <><AppShell.Section grow component={ScrollArea} scrollbarSize={6}><Stack p="lg" gap="xl">
 <div><Text size="xs" c="dimmed">EMG → ankle torque</Text><Text fw={600} size="lg" mt={4}>Research overview</Text></div>
 <nav className="focus-collections" aria-label="Presentation experiments">{PRESENTATION_GROUPS.map((g,index)=><section key={g.label} data-tone={g.tone}><div className="collection-label"><span>{String(index+1).padStart(2,'0')}</span><b>{g.label}</b></div><Text size="xs" c="dimmed" mb="sm">{g.caption}</Text>{g.ids.map(id=><UnstyledButton key={id} className="focus-choice" data-active={s.experiment===id&&s.tab==='overview'||undefined} aria-current={s.experiment===id?'true':undefined} onClick={()=>choose(id)}><span className="focus-marker"/><span>{id.startsWith('D01')?'Auto-gain + balanced positions':catalogTitle(id)}</span></UnstyledButton>)}</section>)}</nav>
 <Button variant="default" size="xs" fullWidth onClick={()=>{set({tab:'experiments'});onNavigate();window.scrollTo({top:0});}}>All 15 controlled experiments</Button>
 <div className="sidebar-section"><Text size="xs" fw={600} mb="xs">Train → test</Text><SegmentedControl fullWidth size="xs" value={s.dir} onChange={v=>set({dir:v as Dir,runA:'typical'})} data={[{value:'S1toS2',label:'Session 1 → 2'},{value:'S2toS1',label:'Session 2 → 1'}]}/></div>
 <details className="disclosure"><summary>Display settings</summary><Switch checked={s.floor} onChange={e=>set({floor:e.currentTarget.checked})} label="Display negative R² as 0" size="xs"/><Text size="xs" c="dimmed" mt="xs">Raw values remain in individual-run tables.</Text></details>
 <Text size="xs" c="dimmed">Retrospective, same-person evaluation.</Text>
 </Stack></AppShell.Section><AppShell.Section p="md" hiddenFrom="md"><Button fullWidth onClick={onNavigate}>Show results</Button></AppShell.Section></>;
}
