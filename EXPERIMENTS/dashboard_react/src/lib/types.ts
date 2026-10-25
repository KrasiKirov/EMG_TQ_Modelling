// Shapes of results.json (written by EXPERIMENTS/code/build_dashboard.py) and of the derived objects.

export type Dir = 'S1toS2' | 'S2toS1';
export type MetricKey = 'r2' | 'mae' | 'r2_demeaned' | 'bias';
export type ParamKey = 'code' | 'passive' | 'emg' | 'weighting' | 'window' | 'dropout' | 'lr' | 'extra';
export type ParamValue = string | number;
export type Params = Record<ParamKey, ParamValue>;

export interface Experiment {
  id: string;
  title: string;
  hypothesis?: string;
  params: Params;
}

export interface Table {
  cols: string[];
  rows: (string | number)[][];
}

export interface LearningCurve {
  experiment: string;
  subject: string;
  direction: string;
  seed: number;
  best_epoch: number;
  epochs: number;
  cols: string[];
  history: number[][];
}

export interface RawData {
  built: string;
  subjects: string[];
  experiments: Record<string, Experiment>;
  cols: string[];
  rows: (string | number)[][];
  grid: [number, number, number];
  positions: Record<string, number[]>;
  key: Record<string, string>;
  wave: Record<string, string>;
  learning_curves?: LearningCurve[];
  round2?: Table;
  families?: Table;
  bias_variance?: Table;
}

/** one training run (seed) scored at one test position */
export interface Run {
  exp: string;
  subj: string;
  dir: Dir;
  seed: number;
  pos: number;
  r2: number;
  mae: number;
  rmse: number;
  bias: number;
  r2_demeaned: number;
  mae_demeaned: number;
  peak: number;
  toes: 0 | 1;
}

/** percentiles over runs (numpy default linear interpolation) */
export interface Summary {
  n: number;
  p1: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  p99: number;
}
