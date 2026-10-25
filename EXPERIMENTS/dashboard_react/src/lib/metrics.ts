import type { MetricKey, Run } from './types';

export interface MetricDef {
  key: MetricKey;
  label: string; // full label, e.g. "R² demeaned"
  short: string; // for tight places
  unit: string; // '' or 'Nm'
  /** true = higher is better, false = lower is better, null = no better/worse direction (signed bias) */
  higherBetter: boolean | null;
  /** R²-type metrics: values below 0 are shown as 0 when the floor toggle is on */
  floorable: boolean;
}

export const METRICS: Record<MetricKey, MetricDef> = {
  r2: { key: 'r2', label: 'R²', short: 'R²', unit: '', higherBetter: true, floorable: true },
  mae: { key: 'mae', label: 'MAE', short: 'MAE', unit: 'Nm', higherBetter: false, floorable: false },
  r2_demeaned: { key: 'r2_demeaned', label: 'R² demeaned', short: 'R² dm', unit: '', higherBetter: true, floorable: true },
  bias: { key: 'bias', label: 'Bias', short: 'Bias', unit: 'Nm', higherBetter: null, floorable: false },
};

export const METRIC_KEYS: MetricKey[] = ['r2', 'mae', 'r2_demeaned', 'bias'];

export function metricLabel(m: MetricKey, withUnit = true): string {
  const d = METRICS[m];
  return withUnit && d.unit ? `${d.label} (${d.unit})` : d.label;
}

/**
 * R² display rule: each run's R² below 0 is set to 0 BEFORE medians / percentiles are taken
 * (same as the deck's box plots and headline numbers, deck_figures.py / presentation_metrics.py).
 */
export function metricValue(r: Run, m: MetricKey, floor: boolean): number {
  const v = r[m];
  return METRICS[m].floorable && floor ? Math.max(v, 0) : v;
}

export function floorR2(v: number, floor: boolean): number {
  return floor ? Math.max(v, 0) : v;
}

// ── comparisons (KPI cards, heat map, verdicts) ──────────────────────────────
// A signed bias has no better / worse direction, and positive and negative biases cancel when pooled.
// Comparisons therefore use the SIZE of the bias, |bias| (closer to 0 = better). Box plots keep the sign.

/** value used when two models are compared: |bias| for bias, else the displayed value */
export function compareValue(r: Run, m: MetricKey, floor: boolean): number {
  return m === 'bias' ? Math.abs(r.bias) : metricValue(r, m, floor);
}

/** label of the compared quantity, e.g. "R²" or "|bias|" */
export function compareLabel(m: MetricKey, withUnit = true): string {
  if (m === 'bias') return withUnit ? '|bias| (Nm)' : '|bias|';
  return metricLabel(m, withUnit);
}

export function compareHigherBetter(m: MetricKey): boolean {
  return m === 'r2' || m === 'r2_demeaned';
}

/**
 * Smallest difference that is called better / worse (smaller = "about the same"):
 * R²: 0.02. MAE and |bias|: 0.05 Nm or 2 % of the peak voluntary torque, whichever is larger.
 */
export function deltaThreshold(m: MetricKey, peak?: number): number {
  if (m === 'r2' || m === 'r2_demeaned') return 0.02;
  return Math.max(0.05, Number.isFinite(peak) ? 0.02 * (peak as number) : 0);
}

/** +1 when B − A = d is an improvement, −1 when worse, 0 when about the same / not comparable */
export function deltaVerdict(m: MetricKey, d: number, peak?: number): -1 | 0 | 1 {
  if (!Number.isFinite(d) || Math.abs(d) < deltaThreshold(m, peak) - 1e-9) return 0;
  return (compareHigherBetter(m) ? d > 0 : d < 0) ? 1 : -1;
}

/** round to the 2 decimals that are displayed, so a shown delta equals the difference of the shown values */
export function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

/** B − A of two medians as displayed (2 decimals) */
export function shownDelta(a: number, b: number): number {
  return round2(round2(b) - round2(a));
}
