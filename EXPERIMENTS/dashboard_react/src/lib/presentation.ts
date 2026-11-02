export const RETEST_MAP_ID='main/session-passive';
export const NEW_MODEL_ID='D01_cleanpassive+autogain+equalweight';
export const NORMALIZATION_IDS=['norm/N0','norm/N1','norm/N2','norm/N3'];
export const EXPLORATION_IDS=['X02a_dropout_0.0','X02b_dropout_0.1','X02c_dropout_0.5','family/F_gru8'];
export const PRESENTATION_GROUPS=[
 {label:'0.5 s window + retest map',caption:'LSTM · 6 people · 10 seeds each',tone:'amber',ids:[RETEST_MAP_ID]},
 {label:'New model',caption:'Balanced positions · each angle scaled by its swing',tone:'teal',ids:[NEW_MODEL_ID]},
 {label:'Normalization study',caption:'Total torque · no auto-gain · 0.2 s · 3 seeds · report under “How it works”',tone:'blue',ids:NORMALIZATION_IDS},
 {label:'Exploration',caption:'Preliminary · not a model ranking',tone:'violet',ids:EXPLORATION_IDS}
];
export const toneOf=(id:string)=>id.startsWith('main/')?'amber':id.startsWith('norm/')?'blue':EXPLORATION_IDS.includes(id)?'violet':'teal';
