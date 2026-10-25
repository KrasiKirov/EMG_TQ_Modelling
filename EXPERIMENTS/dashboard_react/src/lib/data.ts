// Data access: turns results.json into typed runs with fast lookups. No React in here.
import type { Dir, Experiment, RawData, Run, Table } from './types';

export const DIRS: Dir[] = ['S1toS2', 'S2toS1'];
export const POS_TOL = 0.005; // positions are rounded to 3 decimals; match within this

export function near(a: number, b: number): boolean {
  return Math.abs(a - b) < POS_TOL;
}

export interface Dataset {
  raw: RawData;
  built: string;
  subjects: string[];
  experiments: Record<string, Experiment>;
  expIds: string[];
  runs: Run[];
  /** "exp|subj|dir" -> runs (all positions, all seeds) */
  byExpSubjDir: Map<string, Run[]>;
  grid: { t0: number; dt: number; n: number };
}

function colIndex(cols: string[]): Record<string, number> {
  const ix: Record<string, number> = {};
  cols.forEach((c, i) => (ix[c] = i));
  return ix;
}

export function buildDataset(raw: RawData): Dataset {
  const ix = colIndex(raw.cols);
  const num = (r: (string | number)[], c: string) => {
    const v = r[ix[c]];
    return typeof v === 'number' ? v : Number(v);
  };
  const runs: Run[] = raw.rows.map((r) => ({
    exp: String(r[ix.experiment]),
    subj: String(r[ix.subject]),
    dir: String(r[ix.direction]) as Dir,
    seed: num(r, 'seed'),
    pos: num(r, 'position_rad'),
    r2: num(r, 'r2'),
    mae: num(r, 'mae'),
    rmse: num(r, 'rmse'),
    bias: num(r, 'bias'),
    r2_demeaned: num(r, 'r2_demeaned'),
    mae_demeaned: num(r, 'mae_demeaned'),
    peak: num(r, 'peak_torque'),
    toes: (num(r, 'toes_down') ? 1 : 0) as 0 | 1,
  }));
  const byExpSubjDir = new Map<string, Run[]>();
  for (const r of runs) {
    const k = `${r.exp}|${r.subj}|${r.dir}`;
    let a = byExpSubjDir.get(k);
    if (!a) byExpSubjDir.set(k, (a = []));
    a.push(r);
  }
  const [t0, dt, n] = raw.grid;
  return {
    raw,
    built: raw.built,
    subjects: raw.subjects,
    experiments: raw.experiments,
    expIds: Object.keys(raw.experiments),
    runs,
    byExpSubjDir,
    grid: { t0, dt, n },
  };
}

export function runsOf(ds: Dataset, exp: string, subj: string, dir: Dir): Run[] {
  return ds.byExpSubjDir.get(`${exp}|${subj}|${dir}`) ?? [];
}

export function hasRuns(ds: Dataset, exp: string, subj: string, dir: Dir): boolean {
  return ds.byExpSubjDir.has(`${exp}|${subj}|${dir}`);
}

export function seedsOf(ds: Dataset, exp: string, subj: string, dir: Dir): number[] {
  return Array.from(new Set(runsOf(ds, exp, subj, dir).map((r) => r.seed))).sort((a, b) => a - b);
}

/** sorted test positions (rad) of a person's tested session */
export function positionsOf(ds: Dataset, subj: string, dir: Dir): number[] {
  return ds.raw.positions[`${subj}|${dir}`] ?? [];
}

export function positionIndex(ds: Dataset, subj: string, dir: Dir, pos: number): number {
  return positionsOf(ds, subj, dir).findIndex((p) => near(p, pos));
}

/** the 2 most negative (plantarflexed) positions are "toes-down" */
export function isToesDownIndex(k: number): boolean {
  return k < 2;
}

/** experiments asked first for the peak torque (0.2 s window, every person and direction) */
const PEAK_SOURCES = ['X00f_baseline_framework', 'X00_baseline'];

/**
 * Peak voluntary torque (Nm) of each position: ONE value per person/session/position, used everywhere
 * (a 0.5 s window experiment trims the trial slightly and would give a marginally different peak).
 */
export function peakOf(ds: Dataset, subj: string, dir: Dir, pos: number): number {
  const order = [...PEAK_SOURCES.filter((e) => ds.experiments[e]), ...ds.expIds.filter((e) => !PEAK_SOURCES.includes(e))];
  for (const e of order) {
    const r = runsOf(ds, e, subj, dir).find((x) => near(x.pos, pos));
    if (r) return r.peak;
  }
  return NaN;
}

export function tableRows<T>(t: Table | undefined, map: (get: (c: string) => string | number) => T): T[] {
  if (!t) return [];
  const ix = colIndex(t.cols);
  return t.rows.map((r) => map((c) => r[ix[c]]));
}
