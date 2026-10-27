import { useState } from 'react';
import { Button, Group, SegmentedControl, Select, SimpleGrid, Stack, Text } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { DS } from '../dataset';
import { useView } from '../state';
import { ALL_IDS, EXTRA, catalogCode, catalogGroup, catalogRuns, catalogTitle, catalogWave, runTiming, secondsLabel } from '../lib/catalog';
import { ALL_EXPERIMENTS } from '../lib/catalog';
import { resolve } from '../lib/params';
import { median, summarize } from '../lib/stats';
import { fmt, posLabel } from '../lib/format';
import { pickRun, timeAxis } from '../lib/waves';
import { waveOption, type WaveLine } from '../charts/options';
import type { Run } from '../lib/types';
import { EChart } from '../components/EChart';
import { DataTable, EmptyState, SectionCard, type Column } from '../components/common';

const TIME=timeAxis(DS);
type Key='r2'|'mae'|'rmse';
// Resolve only the established controlled-passive-table equivalence. Extras never substitute runs.
function saved(id:string,subject:string,dir:'S1toS2'|'S2toS1'){
 const rows=catalogRuns(id,subject,dir);
 if(rows.length||EXTRA[id])return {id,rows};
 const fallback=resolve(DS,ALL_EXPERIMENTS[id].params,subject,dir);
 return {id:fallback?.exp??id,rows:fallback?.exp?catalogRuns(fallback.exp,subject,dir):[]};
}
function presetOptions(){
 const groups=new Map<string,{value:string;label:string}[]>();
 for(const id of ['main/session-passive',...ALL_IDS.filter(x=>x!=='main/session-passive')]){
  if(EXTRA[id]?.kind==='report')continue;
  const group=catalogGroup(id);const a=groups.get(group)??[];
  a.push({value:id,label:`${catalogCode(id)} · ${catalogTitle(id)}`});groups.set(group,a);
 }
 return [...groups].map(([group,items])=>({group,items}));
}

export function SavedComparisonTab(){
 const {s,set,tokens}=useView();const mobile=!!useMediaQuery('(max-width: 48em)');
 const [view,setView]=useState('summary');const [metric,setMetric]=useState<Key>('mae');
 const [runA,setRunA]=useState('typical'),[runB,setRunB]=useState('typical');
 const ids=[s.compareA,s.compareB];
 const value=(r:Run,k:Key)=>k==='r2'&&s.floor?Math.max(0,r.r2):r[k];
 const score=(rs:Run[],k:Key)=>median([...new Set(rs.map(r=>r.pos))].map(pos=>median(rs.filter(r=>r.pos===pos).map(r=>value(r,k)))));
 const pairs=DS.subjects.map(subject=>{
  const a=saved(ids[0],subject,s.dir),b=saved(ids[1],subject,s.dir);
  const pa=[...new Set(a.rows.map(r=>r.pos))].sort((x,y)=>x-y);
  const pb=[...new Set(b.rows.map(r=>r.pos))].sort((x,y)=>x-y);
  const matched=pa.flatMap(pos=>{const nearest=pb.reduce<number|null>((best,x)=>best===null||Math.abs(x-pos)<Math.abs(best-pos)?x:best,null);return nearest!==null&&Math.abs(nearest-pos)<=.02?[{a:pos,b:nearest}]:[];});
  const rowsA=a.rows.filter(r=>matched.some(p=>p.a===r.pos));
  const rowsB=b.rows.filter(r=>matched.some(p=>p.b===r.pos));
  return {subject,a,b,matched,rowsA,rowsB};
 });
 const eligible=pairs.filter(x=>x.matched.length);
 const chosen=eligible.find(x=>x.subject===s.wsubj)??eligible[0];
 const position=chosen?.matched.reduce((best,x)=>s.pos!==null&&Math.abs(x.a-s.pos)<Math.abs(best.a-s.pos)?x:best,chosen.matched[0]);
 const waveRuns=chosen&&position?[pickRun(chosen.rowsA.filter(r=>r.pos===position.a),runA),pickRun(chosen.rowsB.filter(r=>r.pos===position.b),runB)]:[];
 const training=(rs:Run[])=>secondsLabel(median([...new Set(rs.map(r=>r.seed))].map(seed=>runTiming(rs[0]?.exp??'',rs[0]?.subj??'',s.dir,seed)?.seconds??NaN)));
 const columns:Column<typeof pairs[number]>[]=[{key:'subject',label:'Subject',render:r=><b>{r.subject}</b>},
  {key:'pos',label:'Shared angles',render:r=>r.matched.length||'Unavailable'},
  {key:'r2a',label:'A · R²',render:r=>fmt(score(r.rowsA,'r2'))},{key:'r2b',label:'B · R²',render:r=>fmt(score(r.rowsB,'r2'))},
  {key:'maea',label:'A · MAE (Nm)',render:r=>fmt(score(r.rowsA,'mae'))},{key:'maeb',label:'B · MAE (Nm)',render:r=>fmt(score(r.rowsB,'mae'))},
  {key:'seeds',label:'Seeds A / B',render:r=>`${new Set(r.rowsA.map(x=>x.seed)).size} / ${new Set(r.rowsB.map(x=>x.seed)).size}`},
  {key:'timea',label:'A · train / seed',render:r=>training(r.rowsA)},{key:'timeb',label:'B · train / seed',render:r=>training(r.rowsB)}];
 const cells=eligible.flatMap(pair=>[pair.rowsA,pair.rowsB].map((rows,i)=>({
  label:`${pair.subject} · ${i===0?'A':'B'}`,color:i===0?tokens.a:tokens.b,
  values:[...new Set(rows.map(r=>r.seed))].map(seed=>median(rows.filter(r=>r.seed===seed).map(r=>value(r,metric))))
 })));
 const allValues=cells.flatMap(c=>c.values);const lo=Math.min(...allValues),hi=Math.max(...allValues);const pad=Math.max((hi-lo)*.15,metric==='r2'?.005:.02);
 const panels=chosen&&position?waveRuns.map((run,i)=>{
  if(!run)return null;
  const target=catalogWave(run.exp,chosen.subject,s.dir,run.seed,run.pos,true);
  const prediction=catalogWave(run.exp,chosen.subject,s.dir,run.seed,run.pos);
  return target&&prediction?{run,target,prediction,i}:null;
 }):[];
 const amplitudes=panels.flatMap(p=>p?[...p.target,...p.prediction].filter(Number.isFinite):[]);
 const bottom=Math.min(...amplitudes),top=Math.max(...amplitudes),yPad=Math.max((top-bottom)*.08,.1);
 const yMin=Math.floor((bottom-yPad)*10)/10,yMax=Math.ceil((top+yPad)*10)/10;
 return <Stack gap="lg">
  <SectionCard title="Choose saved experiments" description="The 0.5 s + retest map runs and controlled experiments, with explicit coverage and result sources.">
   <SimpleGrid cols={{base:1,md:2}}><Select searchable label="Model A" aria-label="Saved model A" value={ids[0]} data={presetOptions()} onChange={v=>v&&set({compareA:v,pos:null})}/><Select searchable label="Model B" aria-label="Saved model B" value={ids[1]} data={presetOptions()} onChange={v=>v&&set({compareB:v,pos:null})}/></SimpleGrid>
   <Group justify="space-between" mt="md"><Text size="sm" c="dimmed">{eligible.length} subjects have shared available angles · {s.dir==='S1toS2'?'session 1 → 2':'session 2 → 1'}</Text><Button size="xs" variant="default" onClick={()=>set({compareA:ids[1],compareB:ids[0]})}>Swap A / B</Button></Group>
  </SectionCard>
  <div className="coverage-note"><b>{ids.some(id=>!!EXTRA[id])?'Different protocols · descriptive comparison':'Available-run comparison'}</b><br/>Shared angles are matched within 0.02 rad. Metrics retain each run’s own target and valid evaluation interval; seeds are not paired. {ids.some(id=>EXTRA[id]?.kind==='retest-map')?'The 0.5 s + retest map model uses 0.5 s history, corrected trial labels and a passive map re-measured in the test session. D01 uses its controlled pipeline and training-position scaling. ':''}A side-by-side score does not isolate the model change or establish a matched improvement.</div>
  <SegmentedControl aria-label="Saved comparison view" value={view} onChange={setView} data={[{value:'summary',label:'Summary & box plots'},{value:'waveforms',label:'Waveforms'}]}/>
  {view==='summary'?<>
   <SectionCard title="Results on shared available angles" description="Median across seeds within each angle, then median across angles. Select a subject for its waveforms."><DataTable columns={columns} rows={pairs} rowKey={r=>r.subject} rowClass={r=>r.matched.length?'':'dim'} onRowClick={r=>{if(r.matched.length){set({wsubj:r.subject,pos:null});setView('waveforms');}}} minWidth={950} stickyFirst caption="Saved experiment comparison on shared angles"/><Text size="xs" c="dimmed" mt="sm">{pairs.filter(p=>p.a.id!==ids[0]||p.b.id!==ids[1]).map(p=>`${p.subject}: A source ${catalogCode(p.a.id)}, B source ${catalogCode(p.b.id)}`).join(' · ')}. Controlled passive-table equivalents are used only under the existing project rule. {s.floor?'Negative R² is displayed as 0; the original values remain in individual experiment downloads.':''}</Text></SectionCard>
   {eligible.length?<SectionCard title="Variation across training seeds" description="Each dot is one seed’s median score over the shared angles for that subject. Boxes show the middle 50%; whiskers show the 5th–95th percentiles." actions={<Select aria-label="Comparison box metric" value={metric} onChange={v=>v&&setMetric(v as Key)} data={[{value:'mae',label:'MAE (Nm)'},{value:'rmse',label:'RMSE (Nm)'},{value:'r2',label:'R²'}]} w={140}/>}>
    <EChart key={`${ids.join('|')}|${s.dir}|${metric}|${s.floor}`} height={350} label="Model A and B seed variability on shared angles" option={{animationDuration:150,tooltip:{trigger:'item',formatter:(p:{seriesType:string;dataIndex:number;value:number[]})=>{const k=p.seriesType==='boxplot'?p.dataIndex:Number(p.value[0]);const c=cells[k];return `${c.label}<br/>${c.values.length} seeds<br/>Median: ${fmt(median(c.values),3)}`;}},grid:{left:65,right:25,top:20,bottom:55},xAxis:{type:'category',data:cells.map(c=>c.label),axisLabel:{color:tokens.textMuted,fontSize:11,rotate:mobile?40:0},axisLine:{lineStyle:{color:tokens.axis}}},yAxis:{type:'value',min:metric==='r2'&&s.floor?Math.max(0,lo-pad):metric==='r2'?lo-pad:Math.max(0,lo-pad),max:metric==='r2'?Math.min(1,hi+pad):hi+pad,scale:true,axisLabel:{color:tokens.textMuted,formatter:(v:number)=>Number(v.toFixed(2)).toString()},splitLine:{lineStyle:{color:tokens.grid}}},series:[{type:'boxplot',name:'Seed distribution',data:cells.map(c=>{const q=summarize(c.values);return {value:[q.p5,q.p25,q.p50,q.p75,q.p95],itemStyle:{color:'transparent',borderColor:c.color,borderWidth:2}};}),boxWidth:[15,35]},{type:'scatter',name:'Individual seed',symbolSize:6,
     // Spread each box's dots sideways so seeds with near-equal scores stay countable.
     data:cells.flatMap((c,k)=>c.values.map((v,i)=>({value:[k,v],symbolOffset:[(i-(c.values.length-1)/2)*3,0],itemStyle:{color:c.color}})))}]}}/><Text size="xs" c="dimmed">Teal = A · amber = B. Seed percentile boxes describe these trainings; they are not confidence intervals across new subjects. This chart summarizes positions within each seed; the table summarizes seeds within each position.</Text>
   </SectionCard>:<EmptyState title="No shared results in this direction" hint="The 0.5 s + retest map runs are available only for session 1 → 2. Switch direction or choose other experiments."/>}
  </>:chosen&&position?<SectionCard title="Two predictions, each against its own target" description="Both panels use the same time and torque scales. Gaps are excluded samples; they are not filled with predictions.">
   <Stack gap="md"><SimpleGrid cols={{base:1,sm:2}}><Select label="Subject" aria-label="Comparison waveform subject" value={chosen.subject} data={eligible.map(p=>p.subject)} onChange={v=>v&&set({wsubj:v,pos:null})}/><Select label="Shared ankle angle" aria-label="Comparison waveform position" value={String(position.a)} data={chosen.matched.map(p=>({value:String(p.a),label:`A ${posLabel(p.a,3)} / B ${posLabel(p.b,3)} rad`}))} onChange={v=>v&&set({pos:Number(v)})}/></SimpleGrid>
    <SimpleGrid cols={{base:1,sm:2}}>{[chosen.rowsA.filter(r=>r.pos===position.a),chosen.rowsB.filter(r=>r.pos===position.b)].map((rows,i)=><Select key={i} label={`${i===0?'A':'B'} · training run`} aria-label={`Comparison run ${i===0?'A':'B'}`} value={i===0?runA:runB} onChange={v=>v&&(i===0?setRunA(v):setRunB(v))} data={[{value:'typical',label:'Typical · middle R²'},{value:'best',label:'Best R²'},{value:'worst',label:'Worst R²'},...[...new Set(rows.map(r=>r.seed))].map(seed=>({value:`s${seed}`,label:`Seed ${seed}`}))]}/>)}</SimpleGrid>
    {panels.map((p,i)=>{if(!p)return <EmptyState key={i} title="No waveform stored for this run"/>;const color=i===0?tokens.a:tokens.b;const lines:WaveLine[]=[{name:'Own evaluation target',color:tokens.ink,width:2,data:p.target},{name:`${i===0?'A':'B'} prediction · seed ${p.run.seed}`,color,width:1.6,data:p.prediction}];const option=waveOption(TIME,lines,{start:0,end:100},tokens,mobile);return <div key={`${p.run.exp}|${p.run.seed}`}><Group justify="space-between"><Text fw={600}>{i===0?'A':'B'} · {catalogTitle(ids[i])}</Text><Text size="sm">Raw R² {fmt(p.run.r2)} · MAE {fmt(p.run.mae)} Nm</Text></Group><Text size="xs" c="dimmed">{chosen.subject} · {posLabel(p.run.pos,3)} rad · source {catalogCode(p.run.exp)} · seed {p.run.seed}</Text><EChart height={mobile?330:310} label={`Comparison waveform ${i===0?'A':'B'}`} option={{...option,yAxis:{...(option.yAxis as object),min:yMin,max:yMax,axisLabel:{color:tokens.textMuted,fontSize:12,formatter:(v:number)=>Number(v.toFixed(1)).toString()}}}}/></div>;})}
    <Text size="xs" c="dimmed">Chart display is sampled every 0.2 s. Scores use each full-resolution saved trial, including its own valid interval. A “typical” seed is chosen separately for each model using raw R².</Text>
   </Stack>
  </SectionCard>:<EmptyState title="No shared waveform coverage" hint="Choose models with results in the selected direction."/>}
 </Stack>;
}
