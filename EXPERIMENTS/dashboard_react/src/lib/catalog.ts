import raw from '../data/presentation-catalog.json';
import { ROUND2_NAMES } from './aggregate';
import { DS } from '../dataset';
import { hiddenFromDisplay, STORIES, STUDY_ORDER } from './experimentStory';
import { friendly, shortId } from './params';
import { DISPLAY_NAMES } from './presentation';
import { positionsOf, runsOf } from './data';
import { keyWave, runWave } from './waves';
import type { Dir, Experiment, Run } from './types';
export interface ExtraExperiment extends Experiment {
  group:string; kind:'family'|'research'|'report'|'retest-map'|'normalization'; native:string; model:string; purpose:string; target:string; protocol:string;
  config?:Record<string,unknown>; revision?:string; sessionPassive?:boolean; reportText?:string; overview?:{settings:[string,string][];steps:string[]};
  reportRows?:{subject:string;rmse:number;worst?:number;sourceRmse?:number}[];
  collection?: {
    date:string; runs:number; rows:number; notes:string[];
    coverage:Record<string,{positions:number;seeds:number[];positionIds:string[];passiveMode:string}>;
    unavailable:Record<string,{reason:string}>;
    environment:{cpu_count_visible:number;platform:string;concurrency:string;threads_per_job:string;training_duration_definition:string};
    native:{subject:string;seed:number;rmse:number;worst:number}[];
    histories:Record<string,{loss:number[];val_loss:number[];loss_units:string;best_epoch_one_based:number;epochs_executed:number}>;
  };
}
export interface Timing {seconds:number;prep:number|null;definition:string}
interface CatalogData {experiments:Record<string,ExtraExperiment>;rows:Run[];positions:Record<string,number[]>;targets:Record<string,string>;targetPool:Record<string,string>;predictions:Record<string,string>;means:Record<string,[number,number]>;times:Record<string,Timing>}
const data=raw as unknown as CatalogData;
export const EXTRA=data.experiments;
export interface CalVariant {group:Record<'toes'|'other',{mae:number;r2:number;vaf:number}>;person:Record<'toes'|'other',Record<string,number>>;angle:Record<string,number>}
/** N0 / A0 with and without the 10 s calibration, all scored from 30 s on (import_normalization.py) */
export const CALIBRATION=(raw as unknown as {calibration?:{people:string[];variants:Record<string,CalVariant>}}).calibration??{people:[],variants:{}};
export const ALL_EXPERIMENTS:Record<string,Experiment>={...DS.experiments,...EXTRA};
export const ALL_IDS=[...STUDY_ORDER.filter(id=>DS.experiments[id]),...DS.expIds.filter(id=>!STUDY_ORDER.includes(id)),...Object.keys(EXTRA)]
  .filter(id=>!hiddenFromDisplay(id,`${ALL_EXPERIMENTS[id]?.title??''} ${EXTRA[id]?.purpose??''} ${EXTRA[id]?ROUND2_NAMES[EXTRA[id].native]??'':''}`));
export const catalogTitle=(id:string)=>DISPLAY_NAMES[id]??STORIES[id]?.title??(EXTRA[id]?.kind==='research'?(EXTRA[id].native==='E27_ridge'?'Ridge regression':EXTRA[id].native==='E28_ridge_agc'?'Ridge regression + auto-gain':ROUND2_NAMES[EXTRA[id].native]??EXTRA[id].title):EXTRA[id]?.title)??friendly(id);
const familyCodes:Record<string,string>={F_gru8:'GRU',F_lstm32:'LSTM32',F_mlp:'MLP',F_tcn:'TCN',H_pgain_posw:'LSTM8','main/window20':'M20','main/window50':'M50','main/session-passive':'M50P'};
export const catalogCode=(id:string)=>EXTRA[id]?(familyCodes[EXTRA[id].native]??EXTRA[id].native.split('/').pop()!.split('_')[0]):shortId(id);
export const catalogGroup=(id:string)=>(EXTRA[id]?.group??'Controlled experiments').replace(' · screen',' · screening').replace(' · confirm',' · confirmation').replace(' · final_check',' · final check').replace(' · passive',' · passive targets');
const index=new Map<string,Run[]>();
for(const r of data.rows){const k=`${r.exp}|${r.subj}|${r.dir}`;const a=index.get(k)??[];a.push(r);index.set(k,a);}
export const catalogRuns=(id:string,subject:string,dir:Dir)=>EXTRA[id]?(index.get(`${id}|${subject}|${dir}`)??[]):runsOf(DS,id,subject,dir);
export const catalogPositions=(id:string,subject:string,dir:Dir)=>EXTRA[id]?(data.positions[`${id}|${subject}|${dir}`]??[]):positionsOf(DS,subject,dir);
const cache=new Map<string,Float32Array>();
function block(b64:string|undefined,key:string,position:number,poses:number[]):Float32Array|null{
 if(!b64)return null;let a=cache.get(key);if(!a){const bin=atob(b64);const bytes=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);a=new Float32Array(bytes.buffer);cache.set(key,a);}
 const k=poses.findIndex(p=>Math.abs(p-position)<.005);return k<0?null:a.subarray(k*DS.grid.n,(k+1)*DS.grid.n);
}
export function catalogWave(id:string,subject:string,dir:Dir,seed:number,pos:number,target=false){
 if(!EXTRA[id])return target?keyWave(DS,subject,dir,pos):runWave(DS,id,subject,dir,seed,pos);
 const k=`${id}|${subject}|${dir}|${seed}`;return block((target?data.targetPool[data.targets[k]]:data.predictions[k]),`${target?'target':'pred'}|${k}`,pos,catalogPositions(id,subject,dir));
}
export function fullMeans(id:string,subject:string,dir:Dir,seed:number,pos:number){return data.means[`${id}|${subject}|${dir}|${seed}|${Number(pos.toFixed(3))}`]??null;}
export const runTiming=(id:string,subject:string,dir:Dir,seed:number)=>data.times[`${id}|${subject}|${dir}|${seed}`]??null;
export function secondsLabel(s:number){if(!Number.isFinite(s))return 'Not recorded';if(s<60)return `${s.toFixed(1)} s`;const m=Math.floor(s/60),r=Math.round(s%60);return r===60?`${m+1} min`:`${m} min ${r} s`;}
