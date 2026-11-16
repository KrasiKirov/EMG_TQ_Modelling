// Window-study scores (written by EXPERIMENTS/analysis/study_scores.py): one row of metrics per run and position.
// Every metric is computed WITHIN one position (one trial at a fixed angle), so R² measures how well torque is
// tracked at that angle, not how well the torque level of each angle is guessed.
import raw from '../data/study.json';
import { median, mean } from './stats';

export type SMetric = 'mae' | 'rmse' | 'r2' | 'r2_demeaned';
export type Target = 'active' | 'total';

export interface StudySetting {
  id: string; title: string; question: string; pair: string | null;
  length_s: number; step_ms: number; overlap_pct: number; target: Target; state: string; chunk_s: number | null;
  shuffle: boolean; epochs: number;
}
interface StudyData {
  built: string; direction: string; metrics: string[]; people: string[]; positions: Record<string, number[]>;
  settings: Record<string, StudySetting>;
  /** exp -> person -> seed -> [position][metric] */
  runs: Record<string, Record<string, Record<string, number[][]>>>;
  epochs: Record<string, Record<string, Record<string, number | null>>>;
  windows: Record<string, Record<string, number | null>>;
}
export const STUDY = raw as unknown as StudyData;
export const PEOPLE = STUDY.people;
export const REF: Record<Target, string> = { active: 'W00_reference', total: 'WT1_total' };

export interface SMetricDef { key: SMetric; label: string; short: string; unit: string; error: boolean }
export const SMETRICS: Record<SMetric, SMetricDef> = {
  mae: { key: 'mae', label: 'MAE', short: 'MAE', unit: 'Nm', error: true },
  rmse: { key: 'rmse', label: 'RMSE', short: 'RMSE', unit: 'Nm', error: true },
  r2: { key: 'r2', label: 'R² per position', short: 'R²', unit: '', error: false },
  r2_demeaned: { key: 'r2_demeaned', label: 'Demeaned R² per position (shape)', short: 'R² demeaned', unit: '', error: false },
};
export const SMETRIC_KEYS: SMetric[] = ['mae', 'rmse', 'r2', 'r2_demeaned'];

/** half-width of the "no difference" band: 5 % for error ratios, 0.02 for R² differences */
export const band = (m: SMetric) => (SMETRICS[m].error ? 0.05 : 0.02);
/** the neutral value of a comparison: ratio 1 for errors, difference 0 for R² */
export const neutral = (m: SMetric) => (SMETRICS[m].error ? 1 : 0);

const col = (m: SMetric) => STUDY.metrics.indexOf(m);
export const has = (exp: string) => !!STUDY.settings[exp] && !!STUDY.runs[exp];
export const setting = (exp: string) => STUDY.settings[exp];
export const seedsOf = (exp: string, subj: string) => Object.keys(STUDY.runs[exp]?.[subj] ?? {});

/**
 * One run's score over its positions: errors (Nm) are averaged; R² values take the median, so one position with
 * almost no torque change (where a small offset sends R² far below 0) cannot dominate.
 */
export function runValue(rows: number[][], m: SMetric): number {
  const v = rows.map((r) => r[col(m)]);
  return SMETRICS[m].error ? mean(v) : median(v);
}

/** a person's score: median over the given seeds (default all) of the run score */
export function personValue(exp: string, subj: string, m: SMetric, seeds?: string[]): number {
  const runs = STUDY.runs[exp]?.[subj] ?? {};
  return median((seeds ?? Object.keys(runs)).filter((k) => runs[k]).map((k) => runValue(runs[k], m)));
}

/** a person × position cell: median over seeds */
export function cellValue(exp: string, subj: string, pos: number, m: SMetric, seeds?: string[]): number {
  const runs = STUDY.runs[exp]?.[subj] ?? {};
  return median((seeds ?? Object.keys(runs)).filter((k) => runs[k]?.[pos]).map((k) => runs[k][pos][col(m)]));
}

export interface PersonEffect { subj: string; effect: number; a: number; b: number; seeds: number }
export interface Comparison {
  exp: string; ref: string; metric: SMetric; people: PersonEffect[];
  /** median over people of the per-person effect (ratio for errors, difference for R²) */
  median: number; better: number; n: number; seeds: number;
  verdict: 'better' | 'worse' | 'same' | 'mixed';
}

/** effect of `exp` against `ref`, person by person on the seeds both have */
export function compare(exp: string, ref: string, m: SMetric): Comparison {
  const people: PersonEffect[] = [];
  for (const subj of PEOPLE) {
    const ra = new Set(seedsOf(ref, subj));
    const seeds = seedsOf(exp, subj).filter((k) => ra.has(k));
    if (!seeds.length) continue;
    const a = personValue(ref, subj, m, seeds), b = personValue(exp, subj, m, seeds);
    people.push({ subj, a, b, seeds: seeds.length, effect: SMETRICS[m].error ? b / a : b - a });
  }
  const med = median(people.map((p) => p.effect));
  const better = people.filter((p) => (SMETRICS[m].error ? p.effect < 1 : p.effect > 0)).length;
  return { exp, ref, metric: m, people, median: med, better, n: people.length,
           seeds: Math.max(0, ...people.map((p) => p.seeds)), verdict: verdictOf(med, better, people.length, m) };
}

/**
 * Clearly better / worse: the median effect is outside the no-difference band AND at least 5 of 6 people go
 * that way (with 6 people, 6/6 in one direction happens by chance 3 % of the time, 5/6 22 %).
 */
export function verdictOf(med: number, better: number, n: number, m: SMetric): Comparison['verdict'] {
  const d = med - neutral(m);
  if (!Number.isFinite(d) || !n) return 'same';
  if (Math.abs(d) <= band(m)) return 'same';
  const good = SMETRICS[m].error ? d < 0 : d > 0;
  const need = Math.max(1, n - 1);
  if (good && better >= need) return 'better';
  if (!good && n - better >= need) return 'worse';
  return 'mixed';
}

export function fmtEffect(v: number, m: SMetric): string {
  if (!Number.isFinite(v)) return '–';
  return SMETRICS[m].error ? `${v.toFixed(2)}×` : `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}`;
}

// ── the study's questions ────────────────────────────────────────────────────
const tot = (id: string, t: Target) => (t === 'total' ? `${id}_tot` : id);

export interface Question { key: string; title: string; ask: string; rows: string[]; ref: string; note?: string }

export function questions(t: Target): Question[] {
  const ref = REF[t];
  const qs: Question[] = [
    { key: 'length', title: 'Window length, full overlap', ask: 'Does more EMG history help when every window is kept?',
      rows: ['WL1_0500ms', 'WL2_1s', 'WL3_2s', 'WL4_5s', 'WL5_10s'].map((id) => tot(id, t)), ref,
      note: '500 ms and longer use 50 ms steps; a new training window every 10 ms (98–99.9 % overlap).' },
    { key: 'overlap', title: 'Overlap between training windows, 200 ms', ask: 'Does it matter how many overlapping windows the model trains on?',
      rows: ['WO1_90', 'WO2_75', 'WO3_50', 'WO4_00'].map((id) => tot(id, t)), ref,
      note: 'Less overlap also means fewer weight updates per epoch under the same epoch limit; the next question removes that difference.' },
    { key: 'overlap_eq', title: 'Overlap with equal weight updates, 200 ms', ask: 'Is low overlap worse because of fewer distinct windows, or only because of fewer updates?',
      rows: ['WO2_75_eq', 'WO3_50_eq', 'WO4_00_eq'].map((id) => tot(id, t)), ref,
      note: 'Each epoch reuses the training windows (5, 10 or 20 times) so it has about 880 weight updates, like the reference. Only the variety of windows differs.' },
    { key: 'step', title: 'Step size inside the window', ask: 'Is 1 s at 50 ms steps the same as 1 s at 10 ms steps?',
      rows: [tot('WL2c_1s_10ms', t)], ref: tot('WL2_1s', t),
      note: 'Compared with 1 s at 50 ms steps. The EMG envelope has a 2 Hz low-pass, so 50 ms steps already capture it.' },
    { key: 'carry', title: 'LSTM state carried through the trial', ask: 'Does keeping the LSTM state across a whole trial help?',
      rows: ['WC2_carry_500ms', 'WC3_carry_1s', 'WC4_carry_10s'].map((id) => tot(id, t)), ref,
      note: '10 s chunks: 1 seed, at most 400 epochs, and only 8 weight updates per epoch.' },
    { key: 'shuffle', title: 'Shuffled training windows', ask: 'Does feeding training windows in random order change anything?',
      rows: [tot('WS1_shuffled', t)], ref,
      note: 'Shuffled runs stop early (median 34 epochs). The hyperparameters were tuned without shuffling.' },
  ];
  if (t === 'total') qs.push({ key: 'epochs', title: 'Epoch limit 400 instead of 150', ask: 'Was total torque held back by the 150-epoch limit?',
    rows: ['WT1_total_400'], ref: 'WT1_total', note: '2 seeds. Validation loss is 6 % lower with 400 epochs.' });
  return qs.map((q) => ({ ...q, rows: q.rows.filter(has) })).filter((q) => q.rows.length && has(q.ref));
}

/** every total-torque setting against its own active-torque version */
export function totalVsActive(): string[] {
  return Object.values(STUDY.settings).filter((s) => s.target === 'total' && s.pair && has(s.pair) && s.question !== 'epochs')
    .sort((a, b) => order(a) - order(b)).map((s) => s.id);
}
const QORDER = ['reference', 'overlap', 'overlap_eq', 'length', 'step', 'shuffle', 'carry', 'grid'];
function order(s: StudySetting) { return QORDER.indexOf(s.question) * 1e5 + s.length_s * 1000 + (100 - s.overlap_pct); }

// ── the length × overlap grid ────────────────────────────────────────────────
export const GRID_LENGTHS = [['0500ms', '500 ms', 'WL1_0500ms'], ['1s', '1 s', 'WL2_1s'], ['2s', '2 s', 'WL3_2s'], ['5s', '5 s', 'WL4_5s'], ['10s', '10 s', 'WL5_10s']] as const;
export const GRID_OVERLAPS = [['o00', '0 %'], ['o50', '50 %'], ['o75', '75 %'], ['o90', '90 %'], ['full', 'full']] as const;

export function gridCell(L: (typeof GRID_LENGTHS)[number], o: (typeof GRID_OVERLAPS)[number][0], t: Target): string | null {
  const id = tot(o === 'full' ? L[2] : `WG_${L[0]}_${o}`, t);
  return has(id) ? id : null;
}

/** training windows of a setting, median over people (the grid's low-overlap long cells have very few) */
export function trainWindows(exp: string): number {
  return median(Object.values(STUDY.windows[exp] ?? {}).filter((v): v is number => typeof v === 'number'));
}
