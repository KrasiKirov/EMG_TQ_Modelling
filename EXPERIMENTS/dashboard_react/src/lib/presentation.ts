export const RETEST_MAP_ID='main/session-passive';
export const NEW_MODEL_ID='D01_cleanpassive+autogain+equalweight';
export const NORMALIZATION_IDS=['norm/N0','norm/N1','norm/N2','norm/N3'];
export const CALIBRATION_IDS=['norm/N0','norm/N0cal','norm/A0','norm/A0cal'];
export const EXPLORATION_IDS=['X02a_dropout_0.0','X02b_dropout_0.1','X02c_dropout_0.5','family/F_gru8'];
export const PRESENTATION_GROUPS=[
 {label:'0.5 s window + retest map',caption:'LSTM · 6 people · 10 seeds each',tone:'amber',ids:[RETEST_MAP_ID]},
 {label:'New model',caption:'Balanced positions · each angle scaled by its swing',tone:'teal',ids:[NEW_MODEL_ID]},
 {label:'Final model + 10 s calibration',caption:'0.2 s · balanced · no auto-gain · 6 people · 3 seeds · N0 total torque, A0 active torque · calibrated entries scored from 30 s on',tone:'blue',ids:CALIBRATION_IDS},
 {label:'Normalization study',caption:'Total torque · no auto-gain · 0.2 s · 3 seeds · N0, N3: 6 people; N1, N2: YES, JM, EG only · report under “How it works”',tone:'blue',ids:NORMALIZATION_IDS},
 {label:'Exploration',caption:'Preliminary · not a model ranking',tone:'violet',ids:EXPLORATION_IDS}
];
export const toneOf=(id:string)=>id.startsWith('main/')?'amber':id.startsWith('norm/')?'blue':EXPLORATION_IDS.includes(id)?'violet':'teal';
