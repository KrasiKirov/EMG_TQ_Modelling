// Processing stages of one session-1 trial per person (written by EXPERIMENTS/analysis/pipeline_stages.py).
// The export checks that its stage chain reproduces the reference model's inputs exactly.
import raw from '../data/pipeline.json';

export interface PassivePoint { angle: number; passive: number; spread: number }
export interface PipelinePerson {
  subject: string; trial: number; session: string; angle: number; position: string;
  trainmax: number[]; raw_unit: string;
  /** 2 s at 1 kHz, per channel; rectified = |band| */
  excerpt: { t0: number; fs: number; raw: number[][]; demeaned_offset: number[]; band: number[][]; env: number[][]; env_zerophase4: number[][] };
  /** the same 2 s at 100 Hz: what the model reads (scaled EMG per channel, angle) and its label */
  inputs: { t0: number; fs: number; emg: number[][]; angle: number[]; label: number[] };
  /** the whole trial at 20 Hz */
  trial_view: { t0: number; fs: number; emg: number[][]; angle: number[]; measured: number[]; active: number[]; passive: number };
  psd: { f_step: number; db: number[][] };
  passive_by_session: Record<string, PassivePoint[]>;
  /** how the reference setting cuts this person's trials into windows (times in s, label = last sample of a window) */
  splits: Splits;
}
export interface SplitPart { n: number; first_input: number; first_label: number; last_label: number }
export interface SplitTrial { session: string; angle: number; start: number; end: number; samples: number; train?: SplitPart; val?: SplitPart; test?: SplitPart }
export interface Splits {
  span_samples: number; window: number; step: number; batch: number;
  n_train: number; n_val: number; n_test: number; updates_per_epoch: number; trials: SplitTrial[];
}
/** window layout of one window-study setting (same for every person) */
export interface WindowSetting {
  id: string; title: string; steps: number; step_ms: number; span_s: number; stride_ms: number; overlap_pct: number;
  train_windows: number; passes_per_epoch: number; val_gap_s: number; state: string; chunk_s: number | null; batch: number; shuffle: boolean;
  updates_per_epoch: number; epoch_limit: number; epochs_median: number | null; runs: number;
}
interface PipelineData {
  built: string; setting: string; channels: string[]; lp_order: number; trim_start_s: number; trim_end_s: number; window: number;
  filters: { f: number[]; band: number[]; lp2: number[]; lp4: number[] };
  people: Record<string, PipelinePerson>;
  settings: WindowSetting[];
}

export const PIPE = raw as unknown as PipelineData;
export const PIPE_PEOPLE = Object.keys(PIPE.people);
export const CHANNELS = PIPE.channels;

/** time axis (s) of a series sampled at fs starting at t0 */
export const times = (t0: number, fs: number, n: number) => Array.from({ length: n }, (_, i) => +(t0 + i / fs).toFixed(4));

/** [x, y] pairs for a line series */
export const pairs = (t: number[], y: number[]) => y.map((v, i) => [t[i], v]);

/** rows of the input window ending at 100 Hz sample `end` (inclusive): 20 steps × [GM, GL, SOL, TA, angle] */
export function windowAt(p: PipelinePerson, end: number, width = PIPE.window): number[][] {
  const out: number[][] = [];
  for (let i = end - width + 1; i <= end; i++) out.push([...p.inputs.emg.map((c) => c[i]), p.inputs.angle[i]]);
  return out;
}
