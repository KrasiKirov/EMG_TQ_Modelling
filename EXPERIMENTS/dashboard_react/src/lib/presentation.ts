export const TEAMMATE_ID='main/session-passive';
export const NEW_MODEL_ID='D01_cleanpassive+autogain+equalweight';
export const EXPLORATION_IDS=['X02a_dropout_0.0','X02b_dropout_0.1','X02c_dropout_0.5','family/F_gru8'];
export const PRESENTATION_GROUPS=[
 {label:'Teammate baseline',caption:'0.5 s + retest map · 12 supplied runs',tone:'amber',ids:[TEAMMATE_ID]},
 {label:'New model',caption:'Auto-gain + position scaling',tone:'teal',ids:[NEW_MODEL_ID]},
 {label:'Exploration',caption:'Preliminary · not a model ranking',tone:'violet',ids:EXPLORATION_IDS}
];
export const toneOf=(id:string)=>id.startsWith('main/')?'amber':EXPLORATION_IDS.includes(id)?'violet':'teal';
