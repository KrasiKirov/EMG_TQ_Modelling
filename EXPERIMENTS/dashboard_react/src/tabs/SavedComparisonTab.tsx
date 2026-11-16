import { useState } from 'react';
import { Button, Group, SegmentedControl, Select, SimpleGrid, Stack, Text } from '@mantine/core';
import { DS } from '../dataset';
import { useView } from '../state';
import { ALL_IDS, EXTRA, catalogCode, catalogGroup, catalogRuns, catalogTitle, catalogWave, runTiming, secondsLabel } from '../lib/catalog';
import { ALL_EXPERIMENTS } from '../lib/catalog';
import { resolve } from '../lib/params';
import { median } from '../lib/stats';
import { fmt, posLabel } from '../lib/format';
import { pickRun, timeAxis } from '../lib/waves';
import { waveOption, type WaveLine } from '../charts/options';
import { alpha } from '../theme';
import { RangeWindow, rangeDomain } from '../components/RangeWindow';
import type { Run } from '../lib/types';
import { EChart } from '../components/EChart';
import { SEED_KEY, seedSeries, seedTooltip, type SeedSide, type SpreadMode } from '../charts/seedbox';
import { MachineLine } from '../components/MachineInfo';
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
 const {s,set,tokens}=useView();
 const [view,setView]=useState('summary');const [metric,setMetric]=useState<Key>('mae');
 const [runA,setRunA]=useState('typical'),[runB,setRunB]=useState('typical');
 const [boxScale,setBoxScale]=useState('fit');const [spread,setSpread]=useState<SpreadMode>('box');const [custom,setCustom]=useState<[number,number]|null>(null);
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
 // Text shades of the chart colours: readable on the light table background, same hue as the A / B boxes.
 const inkA='#1d6f66',inkB='#94601f';
 const ink=(side:'A'|'B')=>side==='A'?inkA:inkB;
 // Better / worse only past the explorer's thresholds (0.02 R², 0.05 Nm); closer pairs count as about the same.
 const winner=(a:number,b:number,higherBetter:boolean,tie:number)=>!Number.isFinite(a)||!Number.isFinite(b)||Math.abs(a-b)<tie?null:(higherBetter?a>b:a<b)?'A':'B';
 const head=(side:'A'|'B',text:string)=><span style={{color:ink(side),fontWeight:700}}>{side} · {text}</span>;
 const scoreCell=(v:number,side:'A'|'B',win:'A'|'B'|null)=>{const best=win===side;
  return <span title={best?`Clearly better than ${side==='A'?'B':'A'}`:undefined} style={{color:ink(side),fontWeight:best?700:500,background:best?alpha(side==='A'?tokens.a:tokens.b,.18):undefined,borderRadius:6,padding:best?'2px 8px':undefined,whiteSpace:'nowrap'}}>{fmt(v)}{best?' ✓':''}</span>;};
 const pairCols=(key:Key,text:string,higherBetter:boolean,tie:number):Column<typeof pairs[number]>[]=>(['A','B'] as const).map(side=>({key:`${key}${side}`,label:head(side,text),align:'right' as const,
  render:r=>{const a=score(r.rowsA,key),b=score(r.rowsB,key);return scoreCell(side==='A'?a:b,side,winner(a,b,higherBetter,tie));}}));
 const columns:Column<typeof pairs[number]>[]=[{key:'subject',label:'Subject',render:r=><b>{r.subject}</b>},
  {key:'pos',label:'Shared angles',render:r=>r.matched.length||'Unavailable'},
  ...pairCols('r2','R²',true,.02),...pairCols('mae','MAE (Nm)',false,.05),
  {key:'seeds',label:<span><span style={{color:inkA,fontWeight:700}}>Seeds A</span> / <span style={{color:inkB,fontWeight:700}}>B</span></span>,render:r=><span><span style={{color:inkA}}>{new Set(r.rowsA.map(x=>x.seed)).size}</span> / <span style={{color:inkB}}>{new Set(r.rowsB.map(x=>x.seed)).size}</span></span>},
  {key:'timea',label:head('A','train / seed'),render:r=><span style={{color:inkA}}>{training(r.rowsA)}</span>},{key:'timeb',label:head('B','train / seed'),render:r=><span style={{color:inkB}}>{training(r.rowsB)}</span>}];
 const perSeed=(rows:Run[])=>[...new Set(rows.map(r=>r.seed))].sort((x,y)=>x-y).map(seed=>({seed,v:median(rows.filter(r=>r.seed===seed).map(r=>value(r,metric)))})).filter(p=>Number.isFinite(p.v));
 const sideA:SeedSide={name:`A · ${catalogCode(ids[0])}`,color:tokens.a,offset:-0.2,cells:eligible.map(pair=>perSeed(pair.rowsA))};
 const sideB:SeedSide={name:`B · ${catalogCode(ids[1])}`,color:tokens.b,offset:0.2,cells:eligible.map(pair=>perSeed(pair.rowsB))};
 const cells=[...sideA.cells,...sideB.cells].map(c=>({values:c.map(p=>p.v)}));
 const allValues=cells.flatMap(c=>c.values);const lo=Math.min(...allValues),hi=Math.max(...allValues);const pad=Math.max((hi-lo)*.15,metric==='r2'?.005:.02);
 // Custom range: a reader-chosen window (React state, so re-renders keep it), e.g. R² 0.95–1.
 const step=metric==='r2'?.005:.05,domain=rangeDomain(allValues,metric==='r2',s.floor,step);
 const win:[number,number]=custom&&custom[0]>=domain[0]&&custom[1]<=domain[1]?custom:metric==='r2'?[Math.max(domain[0],.9),domain[1]]:domain;
 const zoomed=boxScale==='custom',hiddenSeeds=zoomed?allValues.filter(v=>v<win[0]||v>win[1]).length:0;
 const yLo=zoomed?win[0]:metric==='r2'&&s.floor?Math.max(0,lo-pad):metric==='r2'?lo-pad:Math.max(0,lo-pad),yHi=zoomed?win[1]:metric==='r2'?Math.min(1,hi+pad):hi+pad;
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
  <SectionCard title="Choose saved experiments" description="Pick model A and model B.">
   <SimpleGrid cols={2}><Select searchable label={<span style={{color:inkA,fontWeight:700}}>● Model A</span>} aria-label="Saved model A" value={ids[0]} data={presetOptions()} onChange={v=>v&&set({compareA:v,pos:null})}/><Select searchable label={<span style={{color:inkB,fontWeight:700}}>● Model B</span>} aria-label="Saved model B" value={ids[1]} data={presetOptions()} onChange={v=>v&&set({compareB:v,pos:null})}/></SimpleGrid>
   <Group justify="space-between" mt="md"><Text size="sm" c="dimmed">{eligible.length} subjects have shared available angles · {s.dir==='S1toS2'?'session 1 → 2':'session 2 → 1'}</Text><Button size="xs" variant="default" onClick={()=>set({compareA:ids[1],compareB:ids[0]})}>Swap A / B</Button></Group>
  </SectionCard>
  <div className="coverage-note"><b>{ids.some(id=>!!EXTRA[id])?'Different protocols · descriptive comparison':'Available-run comparison'}</b><br/>Shared angles are matched within 0.02 rad. Metrics retain each run’s own target and valid evaluation interval; seeds are not paired. {ids.some(id=>EXTRA[id]?.kind==='retest-map')?'The 0.5 s + retest map model uses 0.5 s history, corrected trial labels and a passive map re-measured in the test session. ':''}A side-by-side score does not isolate the model change or establish a matched improvement.</div>
  <SegmentedControl aria-label="Saved comparison view" value={view} onChange={setView} data={[{value:'summary',label:'Summary & box plots'},{value:'waveforms',label:'Waveforms'}]}/>
  {view==='summary'?<>
   <SectionCard title="Results on shared available angles" description="Median over seeds, then angles. ✓ = clearly better (≥ 0.02 R² or 0.05 Nm)."><DataTable columns={columns} rows={pairs} rowKey={r=>r.subject} rowClass={r=>r.matched.length?'':'dim'} onRowClick={r=>{if(r.matched.length){set({wsubj:r.subject,pos:null});setView('waveforms');}}} minWidth={950} stickyFirst caption="Saved experiment comparison on shared angles"/><Stack gap={2} mt="sm"><Text size="xs" fw={600}>Computers behind the training times</Text><MachineLine id={ids[0]} side="A"/><MachineLine id={ids[1]} side="B"/></Stack><Text size="xs" c="dimmed" mt="sm">{(()=>{const subs=pairs.filter(p=>p.a.id!==ids[0]||p.b.id!==ids[1]).map(p=>`${p.subject}: A source ${catalogCode(p.a.id)}, B source ${catalogCode(p.b.id)}`).join(' · ');return subs?`${subs}. `:'';})()}{s.floor?'Negative R² shown as 0.':''}</Text></SectionCard>
   {eligible.length?<SectionCard title="Variation across training seeds" description="One dot per seed." actions={<Group gap="sm" align="end"><SegmentedControl size="xs" aria-label="Comparison spread over seeds" value={spread} onChange={v=>setSpread(v as SpreadMode)} data={[{value:'box',label:'Box plot'},{value:'ci',label:'Median ± 95 % CI'}]}/><SegmentedControl size="xs" aria-label="Comparison box scale" value={boxScale} onChange={setBoxScale} data={[{value:'fit',label:'Fit all subjects'},{value:'custom',label:'Custom range'}]}/><Select aria-label="Comparison box metric" value={metric} onChange={v=>{if(v){setMetric(v as Key);setCustom(null);}}} data={[{value:'mae',label:'MAE (Nm)'},{value:'rmse',label:'RMSE (Nm)'},{value:'r2',label:'R²'}]} w={140}/></Group>}>
    {zoomed&&<RangeWindow domain={domain} value={win} onChange={setCustom} step={step} digits={metric==='r2'?3:2} unit={metric==='r2'?'':' Nm'} hidden={hiddenSeeds} total={allValues.length}/>}
    <Text size="xs" c="dimmed">{SEED_KEY[spread]}</Text>
    <EChart key={`${ids.join('|')}|${s.dir}|${metric}|${s.floor}|${boxScale}|${spread}`} height={350} label="Model A and B seed variability on shared angles" option={{animationDuration:150,
     tooltip:{trigger:'item',confine:true,formatter:(p:{seriesType?:string;seriesIndex:number;dataIndex:number;value?:unknown})=>{const k=p.seriesType==='boxplot'?p.dataIndex:Array.isArray(p.value)?Number(p.value[3]):-1;if(!(k>=0))return '';const unit=metric==='r2'?'':' Nm';return `<b>${eligible[k].subject}</b>`+seedTooltip(sideA,k,3,unit)+seedTooltip(sideB,k,3,unit);}},
     grid:{left:65,right:25,top:20,bottom:40},
     xAxis:[{type:'category',data:eligible.map(p=>p.subject),axisLabel:{color:tokens.textMuted,fontSize:11},axisLine:{lineStyle:{color:tokens.axis}}},{type:'value',min:-0.5,max:eligible.length-0.5,show:false}],
     yAxis:{type:'value',min:yLo,max:yHi,scale:true,axisLabel:{color:tokens.textMuted,formatter:(v:number)=>Number(v.toFixed(3)).toString()},splitLine:{lineStyle:{color:tokens.grid}}},
     series:[sideA,sideB].flatMap(side=>seedSeries(side,tokens,{mode:spread,boxWidth:[10,28],clamp:zoomed?(v:number)=>Math.min(yHi,Math.max(yLo,v)):undefined,hidden:zoomed?(v:number)=>v<yLo||v>yHi:undefined}))}}/><Text size="xs" c="dimmed">Blue = A, orange = B. One dot per seed: its median over shared angles.</Text>
   </SectionCard>:<EmptyState title="No shared results in this direction" hint="The 0.5 s + retest map runs are available only for session 1 → 2. Switch direction or choose other experiments."/>}
  </>:chosen&&position?<SectionCard title="Two predictions, each against its own target" description="Same scales; gaps are excluded samples.">
   <Stack gap="md"><SimpleGrid cols={2}><Select label="Subject" aria-label="Comparison waveform subject" value={chosen.subject} data={eligible.map(p=>p.subject)} onChange={v=>v&&set({wsubj:v,pos:null})}/><Select label="Shared ankle angle" aria-label="Comparison waveform position" value={String(position.a)} data={chosen.matched.map(p=>({value:String(p.a),label:`A ${posLabel(p.a,3)} / B ${posLabel(p.b,3)} rad`}))} onChange={v=>v&&set({pos:Number(v)})}/></SimpleGrid>
    <SimpleGrid cols={2}>{[chosen.rowsA.filter(r=>r.pos===position.a),chosen.rowsB.filter(r=>r.pos===position.b)].map((rows,i)=><Select key={i} label={`${i===0?'A':'B'} · training run`} aria-label={`Comparison run ${i===0?'A':'B'}`} value={i===0?runA:runB} onChange={v=>v&&(i===0?setRunA(v):setRunB(v))} data={[{value:'typical',label:'Typical · middle R²'},{value:'best',label:'Best R²'},{value:'worst',label:'Worst R²'},...[...new Set(rows.map(r=>r.seed))].map(seed=>({value:`s${seed}`,label:`Seed ${seed}`}))]}/>)}</SimpleGrid>
    {panels.map((p,i)=>{if(!p)return <EmptyState key={i} title="No waveform stored for this run"/>;const color=i===0?tokens.a:tokens.b;const lines:WaveLine[]=[{name:'Own evaluation target',color:tokens.ink,width:2,data:p.target},{name:`${i===0?'A':'B'} prediction · seed ${p.run.seed}`,color,width:1.6,data:p.prediction}];const option=waveOption(TIME,lines,tokens);return <div key={`${p.run.exp}|${p.run.seed}`}><Group justify="space-between"><Text fw={600}>{i===0?'A':'B'} · {catalogTitle(ids[i])}</Text><Text size="sm">Raw R² {fmt(p.run.r2)} · MAE {fmt(p.run.mae)} Nm</Text></Group><Text size="xs" c="dimmed">{chosen.subject} · {posLabel(p.run.pos,3)} rad · source {catalogCode(p.run.exp)} · seed {p.run.seed}</Text><EChart height={310} label={`Comparison waveform ${i===0?'A':'B'}`} option={{...option,yAxis:{...(option.yAxis as object),min:yMin,max:yMax,axisLabel:{color:tokens.textMuted,fontSize:12,formatter:(v:number)=>Number(v.toFixed(1)).toString()}}}}/></div>;})}
    <Text size="xs" c="dimmed">Drawn every 0.2 s; typical seed chosen by raw R².</Text>
   </Stack>
  </SectionCard>:<EmptyState title="No shared waveform coverage" hint="Choose models with results in the selected direction."/>}
 </Stack>;
}
