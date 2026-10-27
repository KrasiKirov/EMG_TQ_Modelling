import { useState } from 'react';
import { Group, SegmentedControl, Select, SimpleGrid, Stack, Text } from '@mantine/core';
import { EChart } from './EChart';
import { SectionCard } from './common';
import { summarize } from '../lib/stats';
import { fmt, posLabel } from '../lib/format';
import { useView } from '../state';
import { yZoom } from '../charts/options';
import type { Run } from '../lib/types';
type BoxMetric='r2'|'mae'|'rmse'|'bias'|'r2_demeaned';
export function SingleBoxplots({subjects,onSelect}:{subjects:{subject:string;rows:Run[]}[];onSelect:(subject:string,pos:number)=>void}){
 const {s,tokens}=useView();const [metric,setMetric]=useState<BoxMetric>('r2');const [scale,setScale]=useState('zoom');const [focus,setFocus]=useState('all');
 const value=(r:Run)=>s.floor&&(metric==='r2'||metric==='r2_demeaned')?Math.max(r[metric],0):r[metric];
 const isR2=metric==='r2'||metric==='r2_demeaned',units=isR2?'':' (Nm)';
 const available=subjects.filter(x=>x.rows.length);const shared=available.flatMap(x=>x.rows.map(value)).filter(Number.isFinite);
 return <SectionCard title="How stable are the results?" description="Boxes show the spread across training seeds at each ankle position.">
 <Stack gap="md"><Group justify="space-between" align="end"><Group gap="sm"><Select label="Metric" aria-label="Box plot metric" value={metric} onChange={v=>v&&setMetric(v as BoxMetric)} data={[{value:'r2',label:'R²'},{value:'mae',label:'MAE (Nm)'},{value:'rmse',label:'RMSE (Nm)'},{value:'bias',label:'Bias (Nm)'},{value:'r2_demeaned',label:'R² · demeaned'}]} w={145}/><Select label="Position" aria-label="Box plot position focus" value={focus} onChange={v=>v&&setFocus(v)} data={[{value:'all',label:'All available positions'},...Array.from({length:8},(_,i)=>({value:String(i),label:`Available position ${i+1}`}))]} w={210}/></Group><div><Text size="xs" c="dimmed" mb={5}>Vertical scale</Text><SegmentedControl size="xs" value={scale} onChange={setScale} aria-label="Box plot scale" data={[{value:'zoom',label:'Zoom each subject'},{value:'shared',label:'Same scale'}]}/></div></Group>
 <div className="plot-reading-key"><span><i className="quartile-key"/> middle 50%</span><span>━ median</span><span>┬ 5th–95th percentile whiskers</span><span>• individual runs</span></div>
 {scale==='zoom'&&<Text size="xs" c="dimmed">Each subject has its own labelled axis. Choose a single position to reveal small differences near R² = 1. All points stay included; no outliers are removed.</Text>}
 <Text size="xs" c="dimmed">Zoom in: drag the handles of the slider on the right of a plot, or hold Shift and scroll over it.</Text>
 <SimpleGrid cols={{base:1,lg:2,xl:3}} spacing="lg">{available.map(subject=>{
 const allPoses=[...new Set(subject.rows.map(r=>r.pos))].sort((a,b)=>a-b);const poses=focus==='all'?allPoses:allPoses.filter((_,i)=>i===Number(focus));const cells=poses.map(pos=>subject.rows.filter(r=>r.pos===pos));const summaries=cells.map(rs=>summarize(rs.map(value)));const points=cells.flatMap(rs=>rs.map(value)).filter(Number.isFinite);const source=scale==='shared'?shared:points;
 if(!points.length)return <div key={subject.subject} className="subject-plot"><Text fw={600}>{subject.subject}</Text><Text size="sm" c="dimmed">This subject has no result at that available-position index. Choose another position.</Text></div>;
 let low=Math.min(...source),high=Math.max(...source);const minSpan=isR2?.004:.02;const span=Math.max(high-low,minSpan);const centre=(high+low)/2;low=centre-span*.65;high=centre+span*.65;
 if(isR2){high=Math.min(1,high);if(s.floor)low=Math.max(0,low);if(scale==='shared'&&s.floor){low=0;high=1;}}
 if(!isR2&&metric!=='bias')low=Math.max(0,low);
 return <div key={subject.subject} className="subject-plot"><Group justify="space-between" mb="xs"><Text size="sm" fw={600}>{subject.subject}</Text><Text size="xs" c="dimmed">{new Set(subject.rows.map(r=>r.seed)).size} seeds</Text></Group><EChart key={`${subject.subject}|${metric}|${s.experiment}|${s.dir}|${scale}|${focus}`} height={265} label={`${subject.subject}: ${metric} box plots with percentile whiskers`} onClick={p=>{const k=p.seriesType==='boxplot'?p.dataIndex:Number(p.value?.[0]);if(Number.isInteger(k)&&poses[k]!==undefined)onSelect(subject.subject,poses[k]);}} option={{animationDuration:150,color:[tokens.a],tooltip:{trigger:'item',confine:true,formatter:(p:{seriesType:string;dataIndex:number;value:number[]})=>{const k=p.seriesType==='boxplot'?p.dataIndex:Number(p.value[0]);const q=summaries[k];return `${subject.subject} · ${posLabel(poses[k])} rad<br/>Median: ${fmt(q.p50,3)}${units}<br/>5–95%: ${fmt(q.p5,3)} to ${fmt(q.p95,3)}<br/>${q.n} training runs`;}},grid:{left:55,right:32,top:24,bottom:48},dataZoom:yZoom(tokens,24,48),xAxis:{type:'category',data:poses.map(p=>posLabel(p)),axisLabel:{color:tokens.textMuted,fontSize:11,rotate:focus==='all'?30:0},axisLine:{lineStyle:{color:tokens.axis}}},yAxis:{type:'value',min:low,max:high,scale:true,axisLabel:{color:tokens.textMuted,fontSize:10,formatter:(v:number)=>Math.abs(v)<2?Number(v.toFixed(3)).toString():Number(v.toFixed(1)).toString()},splitLine:{lineStyle:{color:tokens.grid}}},series:[{name:'Training runs',type:'boxplot',data:summaries.map(q=>[q.p5,q.p25,q.p50,q.p75,q.p95]),itemStyle:{color:tokens.dark?'#163d3b':'#d9eeea',borderColor:tokens.a,borderWidth:1.5},boxWidth:[12,32]},{name:'Individual run',type:'scatter',symbolSize:5,itemStyle:{color:tokens.a,opacity:.65},data:cells.flatMap((rs,k)=>rs.map(r=>[k,value(r)]))}]}}/></div>;
 })}</SimpleGrid>{!available.length&&<Text c="dimmed" size="sm">No saved runs in this direction.</Text>}</Stack></SectionCard>;
}
