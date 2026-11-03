export const RETEST_MAP_ID='main/session-passive';
/** the page the dashboard opens on: the final model */
export const FINAL_ID='norm/N0cal';
export const NEW_MODEL_ID=FINAL_ID;
/** one plain sentence under the title of the pages people open most */
export const BLURBS:Record<string,string>={
 'norm/N0cal':'The final model. The day-1 LSTM is kept as it is; on day 2, 10 seconds of pushing at each angle fit a stretch and a shift. Scored from 30 s on.',
 'norm/N0':'The final model before calibration: trained on day 1, tested on day 2, nothing corrected. Predicts the total measured torque.',
 'norm/A0':'Balanced positions without auto-gain: the same LSTM, but it predicts active torque, with each day’s relaxed-hold passive torque subtracted.',
 'norm/A0cal':'A0 plus the same 10-second calibration per angle. Scored from 30 s on.',
 'X00_baseline':'Last semester’s model, re-run unchanged: plain loss, so the angles with big torque swings dominate training.',
};
export const EXPLORATION_IDS=['X02a_dropout_0.0','X02b_dropout_0.1','X02c_dropout_0.5','family/F_gru8'];
export interface SidebarItem { id:string; label:string; badge?:string }
export interface SidebarGroup { label:string; caption:string; tone:string; open:boolean; items:SidebarItem[] }
export const PRESENTATION_GROUPS:SidebarGroup[]=[
 {label:'Final model',caption:'0.2 s · balanced positions · no auto-gain · 6 people × 3 seeds',tone:'teal',open:true,items:[
  {id:'norm/N0cal',label:'N0 + 10 s calibration',badge:'final'},
  {id:'norm/N0',label:'N0 · no calibration'},
  {id:'norm/A0',label:'A0 · balanced positions, active torque'},
  {id:'norm/A0cal',label:'A0 + 10 s calibration'}]},
 {label:'Earlier models',caption:'For comparison',tone:'amber',open:true,items:[
  {id:'X00_baseline',label:'Last semester (original code)'},
  {id:RETEST_MAP_ID,label:'0.5 s window + retest map'}]},
 {label:'Normalization study',caption:'Total torque · four ways to scale the EMG · report under “How it works”',tone:'blue',open:false,items:[
  {id:'norm/N0',label:'N0 · ÷ session-1 maximum (control)'},
  {id:'norm/N1',label:'N1 · ÷ session-1 99th percentile'},
  {id:'norm/N2',label:'N2 · ÷ maximum-effort trial'},
  {id:'norm/N3',label:'N3 · ÷ each session’s 99th percentile'}]},
 {label:'Exploration',caption:'Preliminary · not a model ranking',tone:'violet',open:false,items:EXPLORATION_IDS.map(id=>({id,label:''}))},
];
export const toneOf=(id:string)=>id.startsWith('main/')||id==='X00_baseline'?'amber':['norm/N0cal','norm/N0','norm/A0','norm/A0cal'].includes(id)?'teal':id.startsWith('norm/')?'blue':EXPLORATION_IDS.includes(id)?'violet':'teal';
