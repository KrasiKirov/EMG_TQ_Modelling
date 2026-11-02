import { Badge, Button, Group, Stack, Text, Title } from '@mantine/core';
import { ALL_IDS, EXTRA, catalogGroup, catalogTitle, catalogCode } from '../lib/catalog';
import { DS } from '../dataset';
import { STORIES, STUDY_ORDER } from '../lib/experimentStory';
import { shortId } from '../lib/params';
import { useView } from '../state';
export function ExperimentLibrary() {
  const {set} = useView();
  return <Stack gap="lg"><div className="page-heading"><Text size="sm" c="dimmed">The research process</Text><Title order={2}>What was tried, and why</Title><Text size="sm" c="dimmed">Start with the original model, test individual changes, then inspect the combinations.</Text></div>
    <div className="study-context"><b>How the candidate was chosen</b><p>The study targeted the two most toes-down ankle positions, where the original model struggled. YES and EG were used for screening, followed by combinations and tests on additional subjects. The final candidate was selected by judgment; none of the changes passed all predefined rules.</p><p>The later audit clarifies that JM and CH also influenced design decisions. These are retrospective results, not a fully independent validation of a best model.</p></div>
    {STUDY_ORDER.filter(id => DS.experiments[id]).map(id => {const x = STORIES[id]; return <article key={id} className="study-row"><div className="experiment-code">{shortId(id)}</div><Stack gap={7}><Group gap="sm"><Title order={3}>{x.title}</Title><Badge variant="light" color="gray">{x.stage}</Badge></Group><Text size="sm">{x.purpose}</Text><Text size="sm" c="dimmed">{x.finding}</Text><Text size="xs" fw={600}>{x.decision}</Text></Stack><Button variant="default" size="xs" onClick={() => {set({experiment:id,tab:'overview',runA:'typical'}); window.scrollTo({top:0});}}>Explore results</Button></article>;})}
    <div className="section-heading"><Title order={3}>Earlier models and teammate’s experiments</Title><Text size="sm" c="dimmed">Different targets and training protocols are labelled separately.</Text></div>{ALL_IDS.filter(id=>EXTRA[id]).map(id=><article key={id} className="study-row"><span className="experiment-code">{catalogCode(id)}</span><Stack gap={5}><Title order={3}>{catalogTitle(id)}</Title><Text size="xs" c="dimmed">{catalogGroup(id)}</Text><Text size="sm">{EXTRA[id].purpose}</Text></Stack><Button size="xs" variant="default" onClick={()=>{set({experiment:id,tab:'overview',runA:'typical'});window.scrollTo({top:0});}}>Explore results</Button></article>)}
    <Text size="xs" c="dimmed">Interpretations follow the experiment ledger, final report, and the corrections in the 24 October code audit. Numeric results come from the saved dashboard dataset.</Text>
  </Stack>;
}
