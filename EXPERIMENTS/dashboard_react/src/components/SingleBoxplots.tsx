import { useState } from 'react';
import { Group, SegmentedControl, Select, SimpleGrid, Stack, Text } from '@mantine/core';
import { EChart } from './EChart';
import { SectionCard } from './common';
import { RangeWindow, rangeDomain, windowBox } from './RangeWindow';
import { summarize } from '../lib/stats';
import { fmt, posLabel } from '../lib/format';
import { useView } from '../state';
import type { Run } from '../lib/types';
type BoxMetric='r2'|'mae'|'rmse'|'bias'|'r2_demeaned';
export function SingleBoxplots({subjects,onSelect}:{subjects:{subject:string;rows:Run[]}[];onSelect:(subject:string,pos:number)=>void}){
 const {s,tokens}=useView();const [metric,setMetric]=useState<BoxMetric>('r2');const [scale,setScale]=useState('zoom');const [focus,setFocus]=useState('all');
 const [custom,setCustom]=useState<[number,number]|null>(null);
 const value=(r:Run)=>s.floor&&(metric==='r2'||metric==='r2_demeaned')?Math.max(r[metric],0):r[metric];
 const isR2=metric==='r2'||metric==='r2_demeaned',units=isR2?'':' (Nm)';
 const available=subjects.filter(x=>x.rows.length);const shared=available.flatMap(x=>x.rows.map(value)).filter(Number.isFinite);
 const plots=available.map(subject=>{
  const allPoses=[...new Set(subject.rows.map(r=>r.pos))].sort((a,b)=>a-b);const poses=focus==='all'?allPoses:allPoses.filter((_,i)=>i===Number(focus));const cells=poses.map(pos=>subject.rows.filter(r=>r.pos===pos));
  return {subject,poses,cells,summaries:cells.map(rs=>summarize(rs.map(value))),points:cells.flatMap(rs=>rs.map(value)).filter(Number.isFinite)};});
 // Custom range: the reader picks the visible window, e.g. 0.95–1 to compare positions that all sit near R² = 1.
 const step=isR2?.005:.05,domain=rangeDomain(shared,isR2,s.floor,step);
 const view:[number,number]=custom&&custom[0]>=domain[0]&&custom[1]<=domain[1]?custom:isR2?[Math.max(domain[0],.9),domain[1]]:domain;
 const drawn=plots.flatMap(p=>p.points);const hidden=scale==='custom'?drawn.filter(v=>v<view[0]||v>view[1]).length:0;
 return <SectionCard title="How stable are the results?" description="Boxes show the spread across training seeds at each ankle position.">
 <Stack gap="md"><Group justify="space-between" align="end"><Group gap="sm"><Select label="Metric" aria-label="Box plot metric" value={metric} onChange={v=>{if(v){setMetric(v as BoxMetric);setCustom(null);}}} data={[{value:'r2',label:'R²'},{value:'mae',label:'MAE (Nm)'},{value:'rmse',label:'RMSE (Nm)'},{value:'bias',label:'Bias (Nm)'},{value:'r2_demeaned',label:'VAF (shape only, 0–1)'}]} w={145}/><Select label="Position" aria-label="Box plot position focus" value={focus} onChange={v=>v&&setFocus(v)} data={[{value:'all',label:'All available positions'},...Array.from({length:8},(_,i)=>({value:String(i),label:`Available position ${i+1}`}))]} w={210}/></Group><div><Text size="xs" c="dimmed" mb={5}>Vertical scale</Text><SegmentedControl size="xs" value={scale} onChange={setScale} aria-label="Box plot scale" data={[{value:'zoom',label:'Fit each subject'},{value:'shared',label:'Same scale'},{value:'custom',label:'Custom range'}]}/></div></Group>
 <div className="plot-reading-key"><span><i className="quartile-key"/> middle 50%</span><span>━ median</span><span>┬ 5th–95th percentile whiskers</span><span>• individual runs</span></div>
 {scale==='custom'?<RangeWindow domain={domain} value={view} onChange={setCustom} step={step} digits={isR2?3:2} unit={units} hidden={hidden} total={drawn.length}/>
  :<Text size="xs" c="dimmed">{scale==='zoom'?'Each subject has its own labelled axis. All points stay included; no outliers are removed. ':''}To compare positions that all sit near R² = 1, choose Custom range and narrow it, for example to 0.95–1.</Text>}
 <SimpleGrid cols={{base:1,lg:2,xl:3}} spacing="lg">{plots.map(({subject,poses,cells,summaries,points})=>{
 if(!points.length)return <div key={subject.subject} className="subject-plot"><Text fw={600}>{subject.subject}</Text><Text size="sm" c="dimmed">This subject has no result at that available-position index. Choose another position.</Text></div>;
 const source=scale==='shared'?shared:points;
 let low=Math.min(...source),high=Math.max(...source);const minSpan=isR2?.004:.02;const span=Math.max(high-low,minSpan);const centre=(high+low)/2;low=centre-span*.65;high=centre+span*.65;
 if(isR2){high=Math.min(1,high);if(s.floor)low=Math.max(0,low);if(scale==='shared'&&s.floor){low=0;high=1;}}
 if(!isR2&&metric!=='bias')low=Math.max(0,low);
 if(scale==='custom'){low=view[0];high=view[1];}
 const inView=(v:number)=>scale!=='custom'||(v>=low&&v<=high);
 return <div key={subject.subject} className="subject-plot"><Group justify="space-between" mb="xs"><Text size="sm" fw={600}>{subject.subject}</Text><Text size="xs" c="dimmed">{new Set(subject.rows.map(r=>r.seed)).size} seeds</Text></Group><EChart key={`${subject.subject}|${metric}|${s.experiment}|${s.dir}|${scale}|${focus}`} height={265} label={`${subject.subject}: ${metric} box plots with percentile whiskers`} onClick={p=>{const k=p.seriesType==='boxplot'?p.dataIndex:Number(p.value?.[0]);if(Number.isInteger(k)&&poses[k]!==undefined)onSelect(subject.subject,poses[k]);}} option={{animationDuration:150,color:[tokens.a],tooltip:{trigger:'item',confine:true,formatter:(p:{seriesType:string;dataIndex:number;value:number[]})=>{const k=p.seriesType==='boxplot'?p.dataIndex:Number(p.value[0]);const q=summaries[k];return `${subject.subject} · ${posLabel(poses[k])} rad<br/>Median: ${fmt(q.p50,3)}${units}<br/>5–95%: ${fmt(q.p5,3)} to ${fmt(q.p95,3)}<br/>${q.n} training runs`;}},grid:{left:55,right:15,top:24,bottom:48},xAxis:{type:'category',data:poses.map(p=>posLabel(p)),axisLabel:{color:tokens.textMuted,fontSize:11,rotate:focus==='all'?30:0},axisLine:{lineStyle:{color:tokens.axis}}},yAxis:{type:'value',min:low,max:high,scale:true,axisLabel:{color:tokens.textMuted,fontSize:10,formatter:(v:number)=>Math.abs(v)<2?Number(v.toFixed(high-low<.05?3:2)).toString():Number(v.toFixed(1)).toString()},splitLine:{lineStyle:{color:tokens.grid}}},series:[{name:'Training runs',type:'boxplot',data:summaries.map(q=>scale==='custom'?windowBox(q,low,high):[q.p5,q.p25,q.p50,q.p75,q.p95]),itemStyle:{color:tokens.dark?'#163d3b':'#d9eeea',borderColor:tokens.a,borderWidth:1.5},boxWidth:[12,32]},{name:'Individual run',type:'scatter',symbolSize:5,itemStyle:{color:tokens.a,opacity:.65},data:cells.flatMap((rs,k)=>rs.map(r=>[k,value(r)])).filter(([,v])=>inView(v))}]}}/></div>;
 })}</SimpleGrid>{!available.length&&<Text c="dimmed" size="sm">No saved runs in this direction.</Text>}</Stack></SectionCard>;
}
