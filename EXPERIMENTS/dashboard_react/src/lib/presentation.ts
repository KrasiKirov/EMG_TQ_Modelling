export const RETEST_MAP_ID='main/session-passive';
/** the model page the dashboard opens on */
export const NEW_MODEL_ID='norm/A0cal';
/** Names that say what each model is; used everywhere a model is named (catalogTitle). */
export const DISPLAY_NAMES:Record<string,string>={
 'norm/A0':'Balanced positions (predicts active torque)',
 'norm/A0cal':'Balanced positions + 10 s calibration (predicts active torque)',
 'norm/N0':'Balanced positions (predicts total torque)',
 'norm/N0cal':'Balanced positions + 10 s calibration (predicts total torque)',
 'norm/N1':'EMG ÷ session-1 99th percentile (instead of the maximum)',
 'norm/N2':'EMG ÷ maximum-effort (MVC) peak',
 'norm/N3':'EMG ÷ each session’s own 99th percentile',
 'norm/NR0':'Ridge regression reference · EMG ÷ session-1 maximum',
 'norm/NR1':'Ridge regression reference · EMG ÷ session-1 99th percentile',
 'norm/NR2':'Ridge regression reference · EMG ÷ MVC peak',
 'norm/NR3':'Ridge regression reference · EMG ÷ each session’s 99th percentile',
 'main/session-passive':'LSTM, 0.5 s window, passive map re-measured in session 2',
 'X00_baseline':'Last semester’s LSTM',
 'family/F_gru8':'GRU 8×2 instead of the LSTM (network-type study)',
};
/** one plain sentence under the title of the pages people open most */
export const BLURBS:Record<string,string>={
 'norm/N0cal':'The LSTM trained on session 1, unchanged; in session 2, the first 10 seconds of pushing at each angle fit a stretch and a shift. Scored from 30 s on.',
 'norm/N0':'Trained on session 1, tested on session 2, no calibration. Predicts the total measured torque.',
 'norm/A0':'Balanced positions; the network predicts active torque, with each session’s relaxed-hold passive torque subtracted.',
 'norm/A0cal':'The active-torque LSTM plus the same 10-second calibration per angle. Scored from 30 s on.',
 'X00_baseline':'Last semester’s model, re-run unchanged with a plain mean-squared-error loss.',
};
export const EXPLORATION_IDS=['X02a_dropout_0.0','X02b_dropout_0.1','X02c_dropout_0.5','family/F_gru8'];
export interface SidebarItem { id:string; label:string; badge?:string }
export interface SidebarGroup { label:string; caption:string; tone:string; open:boolean; items:SidebarItem[] }
export const PRESENTATION_GROUPS:SidebarGroup[]=[
 {label:'Models',caption:'0.2 s window · LSTM 8 × 2 · 6 people',tone:'teal',open:true,items:[
  {id:'X00_baseline',label:'Last semester’s LSTM'},
  {id:'norm/A0',label:'Balanced positions'},
  {id:'norm/A0cal',label:'Balanced + 10 s calibration'},
  {id:'norm/N0',label:'Balanced · total torque'},
  {id:'norm/N0cal',label:'Balanced · total torque + 10 s calibration'}]},
 {label:'Earlier models',caption:'For comparison',tone:'amber',open:true,items:[
  {id:RETEST_MAP_ID,label:'LSTM, 0.5 s window + session-2 passive map'}]},
 {label:'Normalization study',caption:'Total torque · four ways to scale the EMG',tone:'blue',open:false,items:[
  {id:'norm/N0',label:'EMG ÷ session-1 maximum (control)'},
  {id:'norm/N1',label:'EMG ÷ session-1 99th percentile'},
  {id:'norm/N2',label:'EMG ÷ maximum-effort (MVC) peak'},
  {id:'norm/N3',label:'EMG ÷ each session’s 99th percentile'}]},
 {label:'Exploration',caption:'Preliminary · not a model ranking',tone:'violet',open:false,items:EXPLORATION_IDS.map(id=>({id,label:''}))},
];
export const toneOf=(id:string)=>id.startsWith('main/')||id==='X00_baseline'?'amber':['norm/N0cal','norm/N0','norm/A0','norm/A0cal'].includes(id)?'teal':id.startsWith('norm/')?'blue':EXPLORATION_IDS.includes(id)?'violet':'teal';
