/**
 * Sidebar groups, landing page, display names and blurbs.
 * Experiments tagged through `python -m EXPERIMENTS.pipeline.dashboard tag ...` arrive in the catalogue
 * (`dashboard` block and `experiments[id]` with kind 'spec'); nothing here needs editing to add one.
 * The LEGACY_* maps only name entries that predate experiment specs.
 */
import raw from '../data/presentation-catalog.json';

export const RETEST_MAP_ID='main/session-passive';
export interface SidebarItem { id:string; label:string; badge?:string }
export interface SidebarGroup { label:string; caption:string; tone:string; open:boolean; items:SidebarItem[] }
interface SpecEntry { name?:string; blurb?:string; kind?:string }
const data=raw as unknown as {dashboard?:{landing:string;groups:SidebarGroup[]};experiments:Record<string,SpecEntry>};
if(!data.dashboard) throw new Error('presentation-catalog.json has no dashboard block: run python -m EXPERIMENTS.pipeline.dashboard export');

export const PRESENTATION_GROUPS:SidebarGroup[]=data.dashboard.groups;
/** the model page the dashboard opens on */
export const NEW_MODEL_ID=data.dashboard.landing;

const LEGACY_NAMES:Record<string,string>={
 'main/session-passive':'LSTM, 0.5 s window, passive map re-measured in session 2',
 'X00_baseline':'Last semester’s LSTM',
 'family/F_gru8':'GRU 8×2 instead of the LSTM (network-type study)',
};
const LEGACY_BLURBS:Record<string,string>={
 'X00_baseline':'Last semester’s model, re-run unchanged with a plain mean-squared-error loss.',
};
const specEntries=Object.entries(data.experiments).filter(([,e])=>e.kind==='spec');
/** Names that say what each model is; used everywhere a model is named (catalogTitle). */
export const DISPLAY_NAMES:Record<string,string>={...LEGACY_NAMES,...Object.fromEntries(specEntries.filter(([,e])=>e.name).map(([id,e])=>[id,e.name!]))};
/** one plain sentence under the title of a model page */
export const BLURBS:Record<string,string>={...LEGACY_BLURBS,...Object.fromEntries(specEntries.filter(([,e])=>e.blurb).map(([id,e])=>[id,e.blurb!]))};
export const EXPLORATION_IDS=PRESENTATION_GROUPS.find(g=>g.label==='Exploration')?.items.map(i=>i.id)??[];
const TONE=new Map(PRESENTATION_GROUPS.flatMap(g=>g.items.map(i=>[i.id,g.tone] as const)));
export const toneOf=(id:string)=>TONE.get(id)??'teal';
