import { useMemo, useState } from 'react';
import { Badge, Button, Group, SegmentedControl, Select, SimpleGrid, Stack, Switch, Text, Title } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { DS } from '../dataset';
import { useView } from '../state';
import { near } from '../lib/data';
import { resolve } from '../lib/params';
import { STORIES } from '../lib/experimentStory';
import { median, summarize } from '../lib/stats';
import { fmt, posLabel, signed } from '../lib/format';
import { floorR2 } from '../lib/metrics';
import { pickRun, timeAxis } from '../lib/waves';
import type { Run } from '../lib/types';
import { DataTable, EmptyState, SectionCard, type Column } from '../components/common';
import { ALL_EXPERIMENTS, EXTRA, catalogCode as shortId, catalogTitle, catalogRuns, catalogPositions, catalogWave, fullMeans, runTiming, secondsLabel } from '../lib/catalog';
import { SupplementalMethod } from '../components/SupplementalMethod';
import { RetestMapMethod, RetestMapLearning } from '../components/RetestMapMethod';
import { SingleBoxplots } from '../components/SingleBoxplots';
import { ReportedExperimentTab } from './ReportedExperimentTab';
import { toneOf } from '../lib/presentation';
import { ExperimentMethod } from '../components/ExperimentMethod';
import { EChart } from '../components/EChart';
import { waveOption, type WaveLine } from '../charts/options';

const TIME=timeAxis(DS);
type Scope='all'|'toes'|'other';
const scopeRuns=(rs:Run[],scope:Scope)=>rs.filter(r=>scope==='all'||(scope==='toes'?r.toes===1:r.toes===0));
type Score='r2'|'mae'|'rmse'|'bias'|'r2_demeaned'|'mae_demeaned';
// Medians within each position first; each subject-position then gets one vote.
function score(rs:Run[],key:Score,floor:boolean):number {
  const cells=new Map<string,number[]>();
  for(const r of rs){const id=`${r.subj}|${r.pos}`;const a=cells.get(id)??[];a.push(key==='r2'||key==='r2_demeaned'?floorR2(r[key],floor):r[key]);cells.set(id,a);}
  return median([...cells.values()].map(a=>median(a)));
}
interface SubjectRow {subject:string;source:string|null;equivalent:boolean;rows:Run[];}
function csvDownload(rows:SubjectRow[],selected:string,dir:string){
  const header=['selected_experiment','source_experiment','equivalent_passive_table','subject','direction','seed','position_rad','toes_down','r2_raw','mae_Nm','rmse_Nm','bias_Nm','r2_demeaned_raw','mae_demeaned_Nm','peak_Nm'];
  const data=rows.flatMap(s=>s.rows.map(r=>[selected,r.exp,s.equivalent,r.subj,r.dir,r.seed,r.pos,r.toes,r.r2,r.mae,r.rmse,r.bias,r.r2_demeaned,r.mae_demeaned,r.peak]));
  const csv=[header,...data].map(r=>r.map(v=>`"${String(v).replace(/"/g,'""')}"`).join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=`${selected.replace(/\//g,'-')}_${dir}_scores.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

export function ExperimentTab(){const {s}=useView();const extra=EXTRA[s.experiment];return extra?.kind==='report'?<ReportedExperimentTab entry={extra}/>:<SavedExperimentTab key={s.experiment}/>;}
function SavedExperimentTab(){
  const {s,set,tokens}=useView();
  const e=ALL_EXPERIMENTS[s.experiment];
  const extra=EXTRA[s.experiment];
  const [demean,setDemean]=useState(false);
  const [pageView,setPageView]=useState('results');
  const [scope,setScope]=useState<Scope>('all');
  const [equivalents,setEquivalents]=useState(true);
  const mobile=!!useMediaQuery('(max-width: 48em)');
  const subjects=useMemo<SubjectRow[]>(()=>DS.subjects.map(subject=>{
    const exact=catalogRuns(e.id,subject,s.dir);
    const fallback=!EXTRA[e.id]&&!exact.length&&equivalents?resolve(DS,e.params,subject,s.dir):null;
    const source=exact.length?e.id:fallback?.exp??null;
    return {subject,source,equivalent:!!source&&source!==e.id,rows:source?catalogRuns(source,subject,s.dir):[]};
  }),[e,s.dir,equivalents]);
  const all=subjects.flatMap(x=>scopeRuns(x.rows,scope));
  const available=subjects.filter(x=>x.rows.length);
  const chosen=subjects.find(x=>x.subject===s.wsubj)??subjects[0];
  const ps=catalogPositions(e.id,chosen.subject,s.dir);
  const pos=!ps.length?0:s.pos===null?ps[0]:ps.reduce((a,b)=>Math.abs(b-s.pos!)<Math.abs(a-s.pos!)?b:a,ps[0]);
  const selectedRows=chosen.rows.filter(r=>near(r.pos,pos));
  const run=pickRun(selectedRows,s.runA);
  const key=run&&chosen.source?catalogWave(chosen.source,chosen.subject,s.dir,run.seed,pos,true):null;
  const pred=run&&chosen.source?catalogWave(chosen.source,chosen.subject,s.dir,run.seed,pos):null;
  const means=run&&chosen.source?fullMeans(chosen.source,chosen.subject,s.dir,run.seed,pos):null;
  const centred=(a:Float32Array,m:number)=>demean&&means?Float32Array.from(a,v=>v-m):a;
  const lines:WaveLine[]=[];
  if(key)lines.push({name:demean?'Target · mean removed':'Measured target',color:tokens.ink,width:2,data:centred(key,means?.[0]??0)});
  if(pred)lines.push({name:`Prediction · seed ${run!.seed}${demean?' · mean removed':''}`,color:tokens.a,width:1.6,data:centred(pred,means?.[1]??0)});
  const positionRows=ps.map(position=>({position,rows:chosen.rows.filter(r=>near(r.pos,position))}));
  const sc=(rs:Run[],k:Score)=>score(rs,k,s.floor);
  const summaryCols:Column<SubjectRow>[]=[
    {key:'subject',label:'Subject',render:r=><b>{r.subject}{r.subject===chosen.subject?' · selected':''}</b>},
    {key:'r2',label:'R²',align:'right',render:r=>fmt(sc(scopeRuns(r.rows,scope),'r2')),sort:r=>sc(scopeRuns(r.rows,scope),'r2')},
    {key:'mae',label:'MAE (Nm)',align:'right',render:r=>fmt(sc(scopeRuns(r.rows,scope),'mae')),sort:r=>sc(scopeRuns(r.rows,scope),'mae')},
    {key:'rmse',label:'RMSE (Nm)',align:'right',render:r=>fmt(sc(scopeRuns(r.rows,scope),'rmse'))},
    {key:'bias',label:'Bias (Nm)',align:'right',render:r=>signed(sc(scopeRuns(r.rows,scope),'bias'))},
    {key:'runs',label:'Runs',align:'right',render:r=>r.rows.length?new Set(r.rows.map(x=>x.seed)).size:'—'},
    {key:'positions',label:'Positions',align:'right',render:r=>new Set(r.rows.map(x=>x.pos)).size||'—'},
    {key:'fitTime',label:'Train time / seed',align:'right',render:r=>secondsLabel(median([...new Set(r.rows.map(x=>x.seed))].map(seed=>r.source?runTiming(r.source,r.subject,s.dir,seed)?.seconds??NaN:NaN)))},
    {key:'source',label:'Result source',render:r=>r.source?<span>{shortId(r.source)}{r.equivalent?' · equivalent table':''}</span>:<span className="missing-score" title={extra?.collection?.unavailable[r.subject]?.reason}>Unavailable</span>}
  ];
  const positionCols:Column<typeof positionRows[number]>[]=[
    {key:'pos',label:'Angle (rad)',render:r=><b>{posLabel(r.position)}{near(r.position,pos)?' · selected':''}</b>,sort:r=>r.position},
    {key:'r2',label:'R²',align:'right',render:r=>fmt(sc(r.rows,'r2'))},
    {key:'mae',label:'MAE (Nm)',align:'right',render:r=>fmt(sc(r.rows,'mae'))},
    {key:'rmse',label:'RMSE (Nm)',align:'right',render:r=>fmt(sc(r.rows,'rmse'))},
    {key:'bias',label:'Bias (Nm)',align:'right',render:r=>signed(sc(r.rows,'bias'))},
    {key:'r2d',label:'R² · shape only',align:'right',render:r=>fmt(sc(r.rows,'r2_demeaned'))},
    {key:'maed',label:'MAE · shape only (Nm)',align:'right',render:r=>fmt(sc(r.rows,'mae_demeaned'))},
    {key:'peak',label:'Peak target (Nm)',align:'right',render:r=>fmt(median(r.rows.map(x=>x.peak)),1)},
  ];
  const runCols:Column<Run>[]=[{key:'seed',label:'Seed',render:r=>`Seed ${r.seed}`,sort:r=>r.seed},
    {key:'r2',label:'Raw R²',align:'right',render:r=>fmt(r.r2)},
    {key:'mae',label:'MAE (Nm)',align:'right',render:r=>fmt(r.mae)},
    {key:'rmse',label:'RMSE (Nm)',align:'right',render:r=>fmt(r.rmse)},
    {key:'bias',label:'Bias (Nm)',align:'right',render:r=>signed(r.bias)},
    {key:'r2d',label:'Raw R² · shape only',align:'right',render:r=>fmt(r.r2_demeaned)},
    {key:'maed',label:'MAE · shape only (Nm)',align:'right',render:r=>fmt(r.mae_demeaned)},{key:'time',label:'Train time / seed',render:r=>secondsLabel(runTiming(r.exp,r.subj,r.dir,r.seed)?.seconds??NaN)}];
  const timesFor=(r:SubjectRow)=>[...new Set(r.rows.map(x=>x.seed))].map(seed=>r.source?runTiming(r.source,r.subject,s.dir,seed):null).filter((x):x is NonNullable<typeof x>=>x!==null);
  const allTimes=subjects.flatMap(timesFor);
  const chooseSubject=(subject:string)=>{set({wsubj:subject,pos:null,runA:'typical'});setPageView('waveforms');window.scrollTo({top:0});};
  return <Stack gap="lg" data-testid="single-experiment" className={`experiment-workspace mode-${pageView}`} data-tone={toneOf(e.id)}>
    <div className="page-heading"><Group gap="sm"><Text size="xs" c="dimmed">{shortId(e.id)} / {STORIES[e.id]?.stage??extra?.group??'Experiment'}</Text><Badge color="gray" variant="light">{s.dir==='S1toS2'?'Train session 1 → test session 2':'Train session 2 → test session 1'}</Badge></Group><Title order={2}>{catalogTitle(e.id)}</Title><Text size="sm" c="dimmed">{e.id.startsWith('D01')?'Position-balanced training: each angle scaled by its swing.':extra?.collection?`Session 1 → session 2 · ${Object.keys(extra.collection.coverage).length} people · ${Math.max(...Object.values(extra.collection.coverage).map(c=>c.seeds.length))} seeds each.`:extra?.kind==='normalization'?`Normalization study · ${e.id.startsWith('norm/A0')?'active':'total'} torque${e.id.endsWith('cal')?' · 10 s calibration, scored from 30 s':''} · session 1 → session 2 · ${new Set(catalogRuns(e.id,'EG','S1toS2').map(r=>r.seed)).size||3} seeds.`:extra?.kind==='family'?'Preliminary architecture experiment · not tested thoroughly.':'Controlled change to the original model.'}</Text><SegmentedControl className="workspace-switch" aria-label="Experiment view" value={pageView} onChange={v=>{setPageView(v);window.scrollTo({top:0});}} data={[{value:'results',label:'Results'},{value:'waveforms',label:'Waveforms'},{value:'method',label:'How it works'}]}/></div>
    {extra?.overview&&<div className="study-context experiment-overview" data-testid="experiment-overview"><b>Experiment summary</b><div className="settings-list">{extra.overview.settings.map(([k,v])=><div key={k}><span>{k}</span><b>{v}</b></div>)}</div><details className="disclosure"><summary>How each prediction is computed</summary><ol>{extra.overview.steps.map((x,i)=><li key={i}>{x}</li>)}</ol></details></div>}
    <section id="method" className="anchor-section workspace-method"><div className="section-heading"><Text size="xs" c="dimmed">01 / Understand the experiment</Text><Title order={3}>What changed, and how it works</Title></div>{extra?.kind==='retest-map'?<RetestMapMethod entry={extra}/>:extra?<SupplementalMethod entry={extra}/>:<ExperimentMethod experiment={e}/>}</section>
    <section id="performance" className="anchor-section workspace-results"><div className="section-heading"><Text size="xs" c="dimmed">02 / Read the results</Text><Title order={3}>Performance across all subjects</Title><Text size="sm" c="dimmed">{shortId(e.id)} · {s.dir==='S1toS2'?'train session 1 → test session 2':'train session 2 → test session 1'} · {extra?.kind==='retest-map'?'retest-session passive target':extra?.kind==='normalization'?(e.id.startsWith('norm/A0')?'active torque (clean passive table of each session subtracted)':'total measured torque (nothing subtracted)'):extra?.kind==='family'?'family-study target':'clean tested-session passive target'}</Text></div>
      <Stack gap="md"><Group justify="space-between"><SegmentedControl value={scope} onChange={v=>setScope(v as Scope)} size="xs" data={[{value:'all',label:'All positions'},{value:'toes',label:'Toes-down'},{value:'other',label:'Other positions'}]}/><Button variant="default" size="xs" onClick={()=>csvDownload(subjects,e.id,s.dir)}>Download raw scores</Button></Group>
      <SimpleGrid cols={{base:1,sm:3}} spacing="md"><div className="single-stat"><Text size="xs" c="dimmed">Average absolute error · MAE</Text><strong data-testid="single-mae">{fmt(sc(all,'mae'))}<small> Nm</small></strong><Text size="xs" c="dimmed">Lower is better · median summary</Text></div><div className="single-stat"><Text size="xs" c="dimmed">Fit to the target · R²</Text><strong data-testid="single-r2">{fmt(sc(all,'r2'))}</strong><Text size="xs" c="dimmed">1 = perfect · 0 = trial-mean reference</Text></div><div className="single-stat"><Text size="xs" c="dimmed">Subjects with results</Text><strong>{available.length}<small> / {DS.subjects.length}</small></strong><Text size="xs" c="dimmed">Missing results are never treated as zero</Text></div></SimpleGrid>
      {!available.length && <EmptyState title="No saved results in this direction" hint="Switch the train / test direction or choose another experiment. The original-code baseline was only run from session 1 to session 2." />}
      <details className="metric-guide metric-reference"><summary>Metric definitions and worked example</summary><p><b>MAE</b> is the average size of the prediction error in newton-metres (Nm). <b>RMSE</b> also measures error in Nm but gives larger misses more influence. Lower is better for both.</p><p><b>R²</b> compares squared error with predicting the trial’s average torque: 1 is perfect, 0 matches that reference, and negative values are worse. It is not percentage accuracy. <b>Bias</b> is prediction minus target, averaged over time: positive means the model predicts too much torque; negative means too little.</p><details className="disclosure"><summary>A worked example of MAE, RMSE, bias and R²</summary><div className="teaching-demo"><Text size="xs" c="dimmed">Illustrative three-sample example, not an experimental result.</Text><p>Target: [2, 4, 6] Nm. Prediction: [3, 3, 8] Nm. The signed errors are [+1, −1, +2] Nm.</p><p>MAE = (1 + 1 + 2) / 3 = <b>1.33 Nm</b>.<br/>RMSE = √((1² + (−1)² + 2²) / 3) = <b>1.41 Nm</b>.<br/>Bias = (1 − 1 + 2) / 3 = <b>+0.67 Nm</b>.<br/>The target mean is 4 Nm, so R² = 1 − 6 / ((2−4)² + (4−4)² + (6−4)²) = <b>0.25</b>.</p><p>A small Nm error can still give a poor R² when the target barely changes. Read the waveform and the error in Nm alongside R², especially at weak toes-down positions.</p></div></details><details className="disclosure"><summary>How summaries are calculated and what the ranges mean</summary><p>For each subject and position, take the median score across training seeds. Then take the median across the selected positions. The headline applies this to every available subject-position equally. These summaries are medians of per-trial scores, not scores recomputed from one pooled waveform. Run-level metrics appear further down.</p><p>Runs reuse the same recordings and change the training random seed. Their spread describes training variability, not uncertainty across new people. The detailed table keeps the sign of bias, so offsets in opposite directions can cancel in a median summary.</p></details></details>
      {s.floor&&<Text size="xs" c="dimmed">Display rule active: negative R² values are shown as 0 before aggregation. Raw scores remain available in the run table and CSV.</Text>}
      <DataTable columns={summaryCols} rows={subjects} rowKey={r=>r.subject} rowTestId={r=>`subject-summary-${r.subject}`} rowClass={r=>r.subject===chosen.subject?'selected-row':r.rows.length?'':'dim'} onRowClick={r=>chooseSubject(r.subject)} minWidth={730} stickyFirst caption="Selected experiment performance by subject" />
      <Text size="xs" c="dimmed">Select a subject row to inspect its positions and waveform.</Text>{extra?.collection&&s.dir==='S1toS2'&&<div className="coverage-note"><b>Coverage:</b> {Object.entries(extra.collection.coverage).map(([subject,c])=>`${subject} ${c.positions} positions`).join(' · ')}. IES has no passive recordings, so its documented pre-test passive table is used for both sessions. See How it works for the data fixes.</div>}
      {!extra&&<details className="disclosure provenance"><summary>Result sources and equivalent runs</summary><Switch checked={equivalents} onChange={ev=>setEquivalents(ev.currentTarget.checked)} label="Include runs with an equivalent passive table" size="xs"/><Text size="sm" mt="sm">Use the selected experiment’s exact runs first. If absent, the existing project rule can use an experiment differing only in the passive table for subjects other than JM, whose tables are identical. These sources are named in the table. Turn this off to show only the exact experiment’s saved runs.</Text></details>}
      </Stack>
    </section>
    <div className="workspace-results"><SingleBoxplots subjects={subjects} onSelect={(subject,pos)=>{set({wsubj:subject,pos,runA:'typical'});setPageView('waveforms');window.scrollTo({top:0});}}/></div>
    <details className="disclosure timing-panel workspace-results"><summary>Training time details · {secondsLabel(median(allTimes.map(t=>t.seconds)))} median per run</summary><SectionCard title="Recorded training time" description="Median wall-clock time for one seed’s training, covering all positions. This is not the total for all seeds."><Group gap="xl"><div><Text size="xs" c="dimmed">Median per run</Text><Text size="xl" fw={600}>{secondsLabel(median(allTimes.map(t=>t.seconds)))}</Text></div><div><Text size="xs" c="dimmed">Recorded runs</Text><Text size="xl" fw={600}>{allTimes.length}</Text></div><div><Text size="xs" c="dimmed">Observed range</Text><Text size="sm">{allTimes.length?`${secondsLabel(Math.min(...allTimes.map(t=>t.seconds)))} – ${secondsLabel(Math.max(...allTimes.map(t=>t.seconds)))}`:'Not recorded'}</Text></div></Group><Text size="xs" c="dimmed" mt="md">{[...new Set(allTimes.map(t=>t.definition))].join('; ')}. Preparation is excluded. 0.5 s + retest map timers include model construction, compilation and fit; controlled timers include prediction; family timers measure model.fit only. Different batch sizes, budgets and concurrent CPU load make these descriptive timings, not a controlled speed benchmark.</Text>{extra?.collection&&<Text size="xs" c="dimmed" mt="sm">Machine: {extra.collection.environment.cpu_count_visible} logical CPU cores · {extra.collection.environment.platform}. Execution: {extra.collection.environment.concurrency}. Threads per job: {extra.collection.environment.threads_per_job}.</Text>}</SectionCard></details>
    <section id="waveform" className="anchor-section workspace-waveforms"><SectionCard title="Measured target and predicted torque" description={`${chosen.subject} · ${ps.length?posLabel(pos)+' rad':'no available position'} · ${shortId(e.id)}${chosen.equivalent ? ` · source ${shortId(chosen.source!)}` : ''} · ${s.dir==='S1toS2'?'test session 2':'test session 1'}`}>
      <Stack gap="md"><Group justify="space-between"><SegmentedControl aria-label="Waveform torque centring" value={demean?'demeaned':'original'} onChange={v=>setDemean(v==='demeaned')} data={[{value:'original',label:'Original torque levels'},{value:'demeaned',label:'Demeaned · evaluation only'}]} size="xs"/><Text size="xs" c="dimmed">Training did not remove each trial’s mean.</Text></Group>{demean&&<Text size="sm" c="dimmed">Each curve’s full trial mean is removed independently for display. This is an evaluation diagnostic, not a new trained model. {means?'Full-resolution saved trial means are used. Demeaned RMSE is derived from the trial RMSE and bias; the removed offset is shown separately.':'No saved means are available for this selection.'}</Text>}<Group grow align="end"><Select label="Subject" aria-label="Detail subject" value={chosen.subject} data={DS.subjects} onChange={v=>v&&chooseSubject(v)}/><Select label="Ankle position (rad)" aria-label="Waveform ankle position" value={ps.length?String(pos):null} data={ps.map(p=>({value:String(p),label:posLabel(p)}))} onChange={v=>v&&set({pos:Number(v),runA:'typical'})}/><Select label="Training run" aria-label="Training run" value={s.runA} onChange={v=>v&&set({runA:v})} data={[{value:'typical',label:'Typical · middle R² run'},{value:'best',label:'Best R² run'},{value:'worst',label:'Worst R² run'},...selectedRows.map(r=>({value:`s${r.seed}`,label:`Seed ${r.seed}`}))]}/></Group>
      {run ? <><div className="wave-score-strip"><span>Seed <b>{run.seed}</b></span><span>Train time / seed <b>{chosen.source?secondsLabel(runTiming(chosen.source,chosen.subject,s.dir,run.seed)?.seconds??NaN):'Not recorded'}</b></span><span>{demean?'Demeaned R²':'Raw R²'} <b data-testid="wave-raw-r2">{fmt(demean?run.r2_demeaned:run.r2)}</b></span><span>{demean?'Demeaned MAE':'MAE'} <b>{fmt(demean?run.mae_demeaned:run.mae)} Nm</b></span><span>{demean?'Demeaned RMSE':'RMSE'} <b>{fmt(demean?Math.sqrt(Math.max(0,run.rmse**2-run.bias**2)):run.rmse)} Nm</b></span><span>{demean?'Removed offset (original bias)':'Bias'} <b>{signed(run.bias)} Nm</b></span></div>{pred&&key?<EChart key={`${e.id}|${s.dir}|${chosen.subject}|${pos}|${run.seed}|${demean}`} option={waveOption(TIME,lines,{start:0,end:100},tokens,mobile,demean)} height={mobile?350:390} label={`Single experiment ${shortId(e.id)}: measured and predicted torque for ${chosen.subject}, seed ${run.seed}`} />:<EmptyState title="Waveform not stored for this run" hint="The scores are available, but this exported dataset does not include its waveform. Try another seed."/>}<Text size="sm" c="dimmed">{extra?extra.target:'The target curve is the recording after subtracting the tested session’s passive-torque estimate; IES has no subtraction.'} The coloured line is this experiment’s prediction. Use the slider to zoom. Scores above use the full saved test prediction, not just the visible or downsampled chart segment.</Text><Text size="xs" c="dimmed">“Typical” selects the lower middle run after sorting by raw R² at this position. Best and worst also use raw R². These are different trainings on the same recording.</Text>
      <details className="disclosure"><summary>Every run at this position · raw metrics and variability</summary><Text size="sm" c="dimmed" mb="sm">R² range across runs (5th–95th percentile): {fmt(summarize(selectedRows.map(r=>r.r2)).p5)} to {fmt(summarize(selectedRows.map(r=>r.r2)).p95)}. Select a seed row to view its prediction.</Text><DataTable columns={runCols} rows={selectedRows} rowKey={r=>String(r.seed)} rowClass={r=>r.seed===run.seed?'selected-row':''} onRowClick={r=>set({runA:`s${r.seed}`})} minWidth={850} stickyFirst caption="Individual training runs and raw metrics"/></details></>:<EmptyState title="No prediction for this selection" hint="Choose a subject and direction with saved results."/>}
      </Stack></SectionCard></section>
    <section id="detail" className="workspace-waveforms">{chosen.source&&<details className="disclosure position-details"><summary>All position metrics</summary><DataTable columns={positionCols} rows={positionRows} rowKey={r=>String(r.position)} rowTestId={r=>`position-${r.position}`} rowClass={r=>near(r.position,pos)?'selected-row':''} onRowClick={r=>{set({pos:r.position,runA:'typical'});}} minWidth={990} stickyFirst caption="All performance metrics by ankle position"/><Text size="xs" c="dimmed">Each cell is the median across runs at that angle. Shape-only scores remove each curve’s own mean before scoring. Peak target is the largest absolute target torque, provided as context for the error size.</Text></details>}</section>
    <div className="workspace-waveforms">{extra?.collection?<RetestMapLearning key={`${e.id}|${chosen.subject}`} entry={extra} subject={chosen.subject} preferredSeed={run?.seed}/>:<SelectedLearning experiment={e.id} subject={chosen.subject} />}</div>
    <Text size="xs" c="dimmed">Data built {extra?.collection?.date??DS.built}. Same-person, cross-session evaluation; it does not demonstrate transfer to a new person. The project audit notes retrospective model selection and offline signal filtering.</Text>
  </Stack>;
}

function SelectedLearning({experiment,subject}:{experiment:string;subject:string}){
  const {s,tokens}=useView();
  const [open,setOpen]=useState(false);
  const c=DS.raw.learning_curves?.find(x=>x.experiment===experiment&&x.subject===subject&&x.direction===s.dir);
  const ix=(name:string)=>c?.cols.indexOf(name)??-1;
  const names=c?.cols??[];
  return <details className="disclosure detail-panel" onToggle={event=>setOpen(event.currentTarget.open)}><summary>Training history for this experiment and subject</summary>{c?<><Text size="sm" mb="md">{subject} · seed {c.seed} · {c.epochs} epochs · kept epoch {c.best_epoch}. This is a separately recorded diagnostic training run. Fit = training portion; validation = held-back portion of the training session; test = the other session, monitored but not used for early stopping.</Text>{open&&<EChart key={`${experiment}|${subject}|${s.dir}`} height={300} label="Selected experiment learning curve" option={{color:[tokens.a,tokens.val,tokens.b],tooltip:{trigger:'axis'},legend:{textStyle:{color:tokens.text}},grid:{left:60,right:20,bottom:40,top:40},xAxis:{type:'category',name:'Epoch',data:c.history.map((_,i)=>i+1),axisLabel:{color:tokens.textMuted}},yAxis:{type:'value',name:'MAE (Nm)',axisLabel:{color:tokens.textMuted},splitLine:{lineStyle:{color:tokens.grid}}},series:['mae_fit','mae_val','mae_test'].filter(n=>names.includes(n)).map(n=>({name:n.replace('mae_',''),type:'line',showSymbol:false,data:c.history.map(r=>r[ix(n)]),markLine:{symbol:'none',data:[{xAxis:String(c.best_epoch)}]}}))}}/>}</>:<Text size="sm" c="dimmed">No training curve was saved for this exact experiment, subject and direction. Test scores and waveforms, when stored, are shown above. Curves from a different model are not substituted here.</Text>}</details>;
}
