import { hiddenFromDisplay } from './experimentStory';
// Aggregations behind the KPI cards, box plots, heat map and tables. Pure functions, no React.
import { near, peakOf, positionsOf, runsOf, seedsOf, tableRows, type Dataset } from './data';
import { compareValue, deltaVerdict, floorR2, metricValue, shownDelta } from './metrics';
import { DEFAULT_A_ID, DEFAULT_B_ID, diffFromBaseline, friendly, passiveTwins, PRESET_GROUPS, REFERENCE_ID, resolve } from './params';
import { median, mean, summarize, EMPTY_SUMMARY } from './stats';
import type { Dir, MetricKey, ParamKey, Params, Run, Summary } from './types';

export function scopeSubjects(ds: Dataset, subj: string): string[] {
  return subj === 'All' ? ds.subjects : ds.subjects.includes(subj) ? [subj] : [];
}

/** runs of the experiment that a parameter set resolves to, for one person */
export function modelRuns(ds: Dataset, p: Params, subj: string, dir: Dir): Run[] {
  const r = resolve(ds, p, subj, dir);
  return r ? runsOf(ds, r.exp, subj, dir) : [];
}

export interface PeakRange {
  median: number;
  min: number;
  max: number;
}

function peakRange(v: number[]): PeakRange {
  return { median: median(v), min: v.length ? Math.min(...v) : NaN, max: v.length ? Math.max(...v) : NaN };
}

/**
 * Group value used everywhere (same convention as the deck): first the median over training runs at
 * each position, then the median across the positions of the group. Pooling runs × positions into one
 * median would land between two positions that behave differently, so it is not used for the centre.
 */
export function positionMedianCenter(items: { key: string; v: number }[]): number {
  const byKey = new Map<string, number[]>();
  items.forEach(({ key, v }) => {
    if (!Number.isFinite(v)) return;
    const a = byKey.get(key) ?? [];
    a.push(v);
    byKey.set(key, a);
  });
  return median(Array.from(byKey.values()).map((a) => median(a)));
}

/** summary whose centre (p50) is the per-position-median centre; the percentiles describe the spread over runs */
function groupSummary(items: { key: string; run: string; v: number }[]): Summary {
  const byRun = new Map<string, number[]>();
  items.forEach(({ run, v }) => {
    if (!Number.isFinite(v)) return;
    const a = byRun.get(run) ?? [];
    a.push(v);
    byRun.set(run, a);
  });
  // each run's value for the group = median across the group's positions in that run
  const s = summarize(Array.from(byRun.values()).map((a) => median(a)));
  return { ...s, p50: positionMedianCenter(items) };
}

// ── KPI cards ────────────────────────────────────────────────────────────────
export interface KpiSide {
  s: Summary;
  people: string[];
  /** test positions per person in this group (2 toes-down or 6 others) */
  positions: number;
  runsMin: number;
  runsMax: number;
}

export interface KpiGroup {
  a: KpiSide;
  b: KpiSide;
  /** shown median(B) − shown median(A) over the SAME people; NaN when no person has both models */
  delta: number;
  verdict: -1 | 0 | 1;
  /** peak voluntary torque (Nm) over the compared positions: median, min, max */
  peak: PeakRange;
}

export interface Kpis {
  toes: KpiGroup;
  other: KpiGroup;
  /** people in view with both models (the comparison is made on them only) */
  common: string[];
  /** people in view where only A / only B / neither is trained */
  onlyA: string[];
  onlyB: string[];
  neither: string[];
}

function kpiSide(ds: Dataset, p: Params, people: string[], dir: Dir, m: MetricKey, floor: boolean, toes: 0 | 1): KpiSide {
  const items: { key: string; run: string; v: number }[] = [];
  const runs: number[] = [];
  let positions = 0;
  for (const s of people) {
    const rs = modelRuns(ds, p, s, dir).filter((r) => r.toes === toes);
    if (!rs.length) continue;
    rs.forEach((r) => items.push({ key: `${s}|${r.pos.toFixed(3)}`, run: String(r.seed), v: compareValue(r, m, floor) }));
    runs.push(new Set(rs.map((r) => r.seed)).size);
    positions = Math.max(positions, new Set(rs.map((r) => r.pos.toFixed(3))).size);
  }
  return {
    // centre = median over positions of the per-position median over runs; 5–95 / 1–99 % = spread over runs
    s: items.length ? groupSummary(items) : EMPTY_SUMMARY,
    people: people.filter((s) => modelRuns(ds, p, s, dir).length > 0),
    positions,
    runsMin: runs.length ? Math.min(...runs) : 0,
    runsMax: runs.length ? Math.max(...runs) : 0,
  };
}

function peaksOf(ds: Dataset, people: string[], dir: Dir, toes: 0 | 1): PeakRange {
  const v: number[] = [];
  for (const s of people)
    positionsOf(ds, s, dir).forEach((p, k) => {
      if ((k < 2 ? 1 : 0) === toes) {
        const pk = peakOf(ds, s, dir, p);
        if (Number.isFinite(pk)) v.push(pk);
      }
    });
  return peakRange(v);
}

/**
 * KPI cards. When both models are trained for at least one person in view, A, B and B − A are all
 * pooled over those common people only, so the difference is like for like. Otherwise each side is
 * shown on its own people and there is no difference.
 */
export function computeKpis(ds: Dataset, A: Params, B: Params, subjects: string[], dir: Dir, m: MetricKey, floor: boolean): Kpis {
  const hasA = (s: string) => !!resolve(ds, A, s, dir);
  const hasB = (s: string) => !!resolve(ds, B, s, dir);
  const common = subjects.filter((s) => hasA(s) && hasB(s));
  const onlyA = subjects.filter((s) => hasA(s) && !hasB(s));
  const onlyB = subjects.filter((s) => !hasA(s) && hasB(s));
  const neither = subjects.filter((s) => !hasA(s) && !hasB(s));
  const group = (toes: 0 | 1): KpiGroup => {
    const peopleA = common.length ? common : subjects;
    const peopleB = common.length ? common : subjects;
    const a = kpiSide(ds, A, peopleA, dir, m, floor, toes);
    const b = kpiSide(ds, B, peopleB, dir, m, floor, toes);
    const peakPeople = common.length ? common : subjects.filter((s) => hasA(s) || hasB(s));
    const peak = peaksOf(ds, peakPeople.length ? peakPeople : subjects, dir, toes);
    const delta = common.length ? shownDelta(a.s.p50, b.s.p50) : NaN;
    return { a, b, delta, verdict: deltaVerdict(m, delta, peak.median), peak };
  };
  return { toes: group(1), other: group(0), common, onlyA, onlyB, neither };
}

// ── per-position box plots ───────────────────────────────────────────────────
export interface PositionStat {
  k: number;
  pos: number;
  toes: boolean;
  summary: Summary;
  points: { seed: number; v: number }[];
  peak: number;
}

/** box plots keep the SIGNED values (bias with its sign) */
export function positionStats(ds: Dataset, p: Params, subj: string, dir: Dir, m: MetricKey, floor: boolean): { exp: string | null; stats: PositionStat[] } {
  const res = resolve(ds, p, subj, dir);
  const positions = positionsOf(ds, subj, dir);
  if (!res) return { exp: null, stats: [] };
  const runs = runsOf(ds, res.exp, subj, dir);
  const stats = positions.map((pos, k) => {
    const at = runs.filter((r) => near(r.pos, pos)).sort((x, y) => x.seed - y.seed);
    const points = at.map((r) => ({ seed: r.seed, v: metricValue(r, m, floor) }));
    return {
      k,
      pos,
      toes: k < 2,
      summary: points.length ? summarize(points.map((x) => x.v)) : EMPTY_SUMMARY,
      points,
      peak: peakOf(ds, subj, dir, pos),
    };
  });
  return { exp: res.exp, stats };
}

// ── heat map B − A ───────────────────────────────────────────────────────────
export interface HeatCell {
  subj: string;
  k: number;
  pos: number;
  /** summaries of the compared value (|bias| for bias) over runs */
  sa: Summary;
  sb: Summary;
  a: number;
  b: number;
  d: number;
  verdict: -1 | 0 | 1;
  peak: number;
  /** signed bias medians (bias metric only), for the tooltip */
  biasA?: number;
  biasB?: number;
}

export function heatmap(ds: Dataset, A: Params, B: Params, dir: Dir, m: MetricKey, floor: boolean): HeatCell[] {
  const cells: HeatCell[] = [];
  for (const s of ds.subjects) {
    const ra = modelRuns(ds, A, s, dir);
    const rb = modelRuns(ds, B, s, dir);
    positionsOf(ds, s, dir).forEach((pos, k) => {
      const atA = ra.filter((r) => near(r.pos, pos));
      const atB = rb.filter((r) => near(r.pos, pos));
      const sa = summarize(atA.map((r) => compareValue(r, m, floor)));
      const sb = summarize(atB.map((r) => compareValue(r, m, floor)));
      const peak = peakOf(ds, s, dir, pos);
      const d = shownDelta(sa.p50, sb.p50);
      cells.push({
        subj: s,
        k,
        pos,
        sa,
        sb,
        a: sa.p50,
        b: sb.p50,
        d,
        verdict: deltaVerdict(m, d, peak),
        peak,
        biasA: m === 'bias' ? median(atA.map((r) => r.bias)) : undefined,
        biasB: m === 'bias' ? median(atB.map((r) => r.bias)) : undefined,
      });
    });
  }
  return cells;
}

// ── experiments table ────────────────────────────────────────────────────────
export const OURS_ALL_KEY = 'ours-all-people';

export interface ExperimentRow {
  /** row key: an experiment id, or OURS_ALL_KEY for the merged headline row */
  key: string;
  /** experiment id to load as model B when the row is clicked */
  id: string;
  synthetic: boolean;
  name: string;
  title: string;
  hypothesis: string;
  params: Params;
  diffs: ParamKey[];
  /** which experiment is used for whom (several for the merged row) */
  used: { exp: string; subjects: string[] }[];
  /** experiments with the same settings except the passive table */
  twins: string[];
  persons: string[];
  runsMin: number;
  runsMax: number;
  otherPersons: string[];
  otherRunsMin: number;
  otherRunsMax: number;
  matched: number; // number of (person, seed) pairs compared with the reference
  r2: Summary;
  refR2: Summary;
  mae: Summary;
  refMae: Summary;
  toesR2: number;
  refToesR2: number;
  dR2: number;
  toesMae: number;
  refToesMae: number;
  dMae: number;
  /** peak voluntary torque at the compared toes-down positions */
  peak: PeakRange;
  /** position of the row in the preset groups (for the default order) */
  order: number;
}

function runCountRange(ds: Dataset, exp: string, dir: Dir) {
  const persons = ds.subjects.filter((s) => runsOf(ds, exp, s, dir).length);
  const n = persons.map((s) => seedsOf(ds, exp, s, dir).length);
  return { persons, min: n.length ? Math.min(...n) : 0, max: n.length ? Math.max(...n) : 0 };
}

const PRESET_ORDER: string[] = PRESET_GROUPS.flatMap((g) => g.ids);

/** matched comparison with the reference for a person -> experiment map */
function matchedScores(ds: Dataset, perPerson: { subj: string; exp: string }[], dir: Dir, floor: boolean) {
  const xs: Run[] = [];
  const refs: Run[] = [];
  let matched = 0;
  const peakV: number[] = [];
  for (const { subj: s, exp } of perPerson) {
    const refSeeds = new Set(seedsOf(ds, REFERENCE_ID, s, dir));
    let any = false;
    for (const seed of seedsOf(ds, exp, s, dir)) {
      if (!refSeeds.has(seed)) continue;
      matched++;
      any = true;
      runsOf(ds, exp, s, dir).forEach((r) => r.seed === seed && r.toes && xs.push(r));
      runsOf(ds, REFERENCE_ID, s, dir).forEach((r) => r.seed === seed && r.toes && refs.push(r));
    }
    if (any) positionsOf(ds, s, dir).slice(0, 2).forEach((p) => peakV.push(peakOf(ds, s, dir, p)));
  }
  const item = (r: Run, v: number) => ({ key: `${r.subj}|${r.pos.toFixed(3)}`, run: `${r.subj}|${r.seed}`, v });
  const r2 = xs.length ? groupSummary(xs.map((r) => item(r, floorR2(r.r2, floor)))) : EMPTY_SUMMARY;
  const refR2 = refs.length ? groupSummary(refs.map((r) => item(r, floorR2(r.r2, floor)))) : EMPTY_SUMMARY;
  const mae = xs.length ? groupSummary(xs.map((r) => item(r, r.mae))) : EMPTY_SUMMARY;
  const refMae = refs.length ? groupSummary(refs.map((r) => item(r, r.mae))) : EMPTY_SUMMARY;
  return {
    matched,
    r2,
    refR2,
    mae,
    refMae,
    toesR2: r2.p50,
    refToesR2: refR2.p50,
    dR2: shownDelta(refR2.p50, r2.p50),
    toesMae: mae.p50,
    refToesMae: refMae.p50,
    dMae: shownDelta(refMae.p50, mae.p50),
    peak: peakRange(peakV.filter((v) => Number.isFinite(v))),
  };
}

/**
 * Median toes-down R² / MAE of each experiment vs the last-semester reference (X00f_baseline_framework),
 * computed on the SAME (person, seed) pairs for both (matched runs). Plus one merged row for "ours" on
 * every person, built with the resolution rule (D01, and C01 for YES / EG, as in the Overview).
 */
export function experimentsTable(ds: Dataset, dir: Dir, floor: boolean): ExperimentRow[] {
  const other: Dir = dir === 'S1toS2' ? 'S2toS1' : 'S1toS2';
  const orderOf = (id: string) => {
    const i = PRESET_ORDER.indexOf(id);
    return i < 0 ? PRESET_ORDER.length + ds.expIds.indexOf(id) : i;
  };
  const rows: ExperimentRow[] = ds.expIds.filter((id) => !hiddenFromDisplay(id, ds.experiments[id].title)).map((id) => {
    const e = ds.experiments[id];
    const here = runCountRange(ds, id, dir);
    const there = runCountRange(ds, id, other);
    const sc = matchedScores(
      ds,
      here.persons.map((s) => ({ subj: s, exp: id })),
      dir,
      floor,
    );
    return {
      key: id,
      id,
      synthetic: false,
      name: friendly(id),
      title: e.title,
      hypothesis: e.hypothesis ?? '',
      params: e.params,
      diffs: diffFromBaseline(ds, e.params),
      used: here.persons.length ? [{ exp: id, subjects: here.persons }] : [],
      twins: passiveTwins(ds, id),
      persons: here.persons,
      runsMin: here.min,
      runsMax: here.max,
      otherPersons: there.persons,
      otherRunsMin: there.min,
      otherRunsMax: there.max,
      ...sc,
      order: orderOf(id),
    };
  });

  // merged "ours" row: the parameter set of the default model B, resolved per person
  const oursP = ds.experiments[DEFAULT_B_ID]?.params;
  if (oursP) {
    const perPerson: { subj: string; exp: string }[] = [];
    const perPersonOther: { subj: string; exp: string }[] = [];
    for (const s of ds.subjects) {
      const r = resolve(ds, oursP, s, dir);
      if (r) perPerson.push({ subj: s, exp: r.exp });
      const ro = resolve(ds, oursP, s, other);
      if (ro) perPersonOther.push({ subj: s, exp: ro.exp });
    }
    const usedMap = new Map<string, string[]>();
    perPerson.forEach(({ subj, exp }) => usedMap.set(exp, [...(usedMap.get(exp) ?? []), subj]));
    // only worth a row when it adds people to the plain experiment row
    if (perPerson.length > runCountRange(ds, DEFAULT_B_ID, dir).persons.length) {
      const nSeeds = perPerson.map((x) => seedsOf(ds, x.exp, x.subj, dir).length);
      const nSeedsO = perPersonOther.map((x) => seedsOf(ds, x.exp, x.subj, other).length);
      rows.push({
        key: OURS_ALL_KEY,
        id: DEFAULT_B_ID,
        synthetic: true,
        name: `${friendly(DEFAULT_B_ID)} · every person`,
        title: ds.experiments[DEFAULT_B_ID].title,
        hypothesis: ds.experiments[DEFAULT_B_ID].hypothesis ?? '',
        params: oursP,
        diffs: diffFromBaseline(ds, oursP),
        used: Array.from(usedMap.entries()).map(([exp, subjects]) => ({ exp, subjects })),
        twins: [],
        persons: perPerson.map((x) => x.subj),
        runsMin: nSeeds.length ? Math.min(...nSeeds) : 0,
        runsMax: nSeeds.length ? Math.max(...nSeeds) : 0,
        otherPersons: perPersonOther.map((x) => x.subj),
        otherRunsMin: nSeedsO.length ? Math.min(...nSeedsO) : 0,
        otherRunsMax: nSeedsO.length ? Math.max(...nSeedsO) : 0,
        ...matchedScores(ds, perPerson, dir, floor),
        order: orderOf(DEFAULT_B_ID) - 0.5,
      });
    }
  }
  // default order: preset groups; experiments not trained in this direction last
  return rows.sort((a, b) => (a.persons.length ? 0 : 1) - (b.persons.length ? 0 : 1) || a.order - b.order);
}

// ── round 2 (earlier exploration) ────────────────────────────────────────────
export const ROUND2_NAMES: Record<string, string> = {
  E01_old: 'old pipeline',
  E02_minimal: 'passive α·P+β correction',
  E04_final_holdgain: 'passive from hold before trial',
  E06_final_drift: 'passive drift model',
  E10_final_mvcglobal: 'EMG ÷ MVC (whole leg)',
  E11_final_mvcangle: 'EMG ÷ MVC per position',
  E13_final_pf3: '×3 toes-down weight',
  E14_final_pf10: '×10 toes-down weight',
  E16_final_ga50: 'random EMG-size augmentation',
  E18_minimal_ga50: 'minimal + augmentation',
  E21_xsub_trainmax: 'pre-train on other 5 (train-max scaling)',
  E22_restanchor_final: 'in-trial passive line (report 2)',
  E23_restanchor_minimal: 'in-trial line, minimal',
  E24_final_norest: 'relaxed windows removed',
  E25_final_rest02: 'relaxed windows ×0.2',
  E26_final_sessp99: 'each session ÷ own 99th pct (not allowed: uses test session)',
  E27_ridge: 'ridge regression (no network)',
  E29_final_win40: '0.4 s window',
  E33_final_noscale: 'without divide-by-swing',
  E34_final_oldpassive: 'old passive table',
  E35_final_lstm8: 'report-2 pipeline with LSTM-8',
  E36_static_old: 'passive trials, old pipeline',
  E38_static_full_win40: 'E37 with 0.4 s window',
  FINAL: 'report-2 pipeline (bigger LSTM with experts)',
};

export const ROUND2_STUDY_ORDER = ['screen', 'confirm', 'ablation', 'passive', 'final_check'];

export interface GroupRow {
  key: string;
  study: string;
  id: string;
  name: string;
  persons: string[];
  runs: number;
  /** centre = median over positions of the per-position median over runs; percentiles = spread over runs (R² floored per run when the display rule is on) */
  toes: { r2: Summary; mae: Summary };
  other: { r2: Summary; mae: Summary };
  toesR2: number;
  otherR2: number;
  toesMae: number;
  otherMae: number;
  /** peak voluntary torque of the toes-down / other positions scored (NaN ranges when unknown) */
  toesPeak: PeakRange;
  otherPeak: PeakRange;
}

interface ScoreRow {
  group: string;
  study: string;
  id: string;
  subj: string;
  seed: number;
  pos: number;
  r2: number;
  mae: number;
  toes: number;
  peak: number;
}

function groupScores(rows: ScoreRow[], floor: boolean, name: (id: string) => string): GroupRow[] {
  const groups = new Map<string, ScoreRow[]>();
  rows.forEach((r) => {
    const a = groups.get(r.group) ?? [];
    a.push(r);
    groups.set(r.group, a);
  });
  return Array.from(groups.entries()).map(([key, rs]) => {
    const td = rs.filter((r) => r.toes);
    const ot = rs.filter((r) => !r.toes);
    const it = (r: ScoreRow, v: number) => ({ key: `${r.subj}|${r.pos.toFixed(3)}`, run: `${r.subj}|${r.seed}`, v });
    const sum = (x: ScoreRow[]) => ({
      r2: x.length ? groupSummary(x.map((r) => it(r, floorR2(r.r2, floor)))) : EMPTY_SUMMARY,
      mae: x.length ? groupSummary(x.map((r) => it(r, r.mae))) : EMPTY_SUMMARY,
    });
    const peaks = (x: ScoreRow[]) => {
      // one peak per person × position (the same for every run)
      const seen = new Map<string, number>();
      x.forEach((r) => Number.isFinite(r.peak) && seen.set(`${r.subj}|${r.peak}`, r.peak));
      return peakRange(Array.from(seen.values()));
    };
    const toes = sum(td);
    const other = sum(ot);
    return {
      key,
      study: rs[0].study,
      id: rs[0].id,
      name: name(rs[0].id),
      persons: Array.from(new Set(rs.map((r) => r.subj))),
      runs: new Set(rs.map((r) => `${r.subj}|${r.seed}`)).size,
      toes,
      other,
      toesR2: toes.r2.p50,
      otherR2: other.r2.p50,
      toesMae: toes.mae.p50,
      otherMae: other.mae.p50,
      toesPeak: peaks(td),
      otherPeak: peaks(ot),
    };
  });
}

export function round2Table(ds: Dataset, dir: Dir, floor: boolean): GroupRow[] {
  const rows = tableRows(ds.raw.round2, (g) => {
    const subj = String(g('subject'));
    const d = String(g('direction')) as Dir;
    return {
      group: `${g('study')}|${g('experiment')}`,
      study: String(g('study')),
      id: String(g('experiment')),
      subj,
      dir: d,
      seed: Number(g('seed')),
      r2: Number(g('r2')),
      mae: Number(g('mae')),
      toes: Number(g('toes_down')),
      pos: Number(g('position_rad')),
      peak: NaN,
    };
  }).filter((r) => r.dir === dir && !hiddenFromDisplay(r.id, ROUND2_NAMES[r.id] ?? ''));
  // round 2 was re-scored on the same answer key, so the peaks are the same as in the main data
  const peakCache = new Map<string, number>();
  rows.forEach((r) => {
    const k = `${r.subj}|${r.pos}`;
    if (!peakCache.has(k)) peakCache.set(k, peakOf(ds, r.subj, dir, r.pos));
    r.peak = peakCache.get(k) as number;
  });
  return groupScores(rows, floor, (id) => ROUND2_NAMES[id] ?? id).sort(
    (a, b) => ROUND2_STUDY_ORDER.indexOf(a.study) - ROUND2_STUDY_ORDER.indexOf(b.study) || a.id.localeCompare(b.id),
  );
}

export const FAMILY_NAMES: Record<string, string> = {
  H_pgain_posw: 'LSTM 8×2 (reference)',
  F_gru8: 'GRU 8×2',
  F_lstm32: 'LSTM 32×2',
  F_mlp: 'MLP',
  F_tcn: 'TCN',
};

export function familiesTable(ds: Dataset, floor: boolean): GroupRow[] {
  const rows = tableRows(ds.raw.families, (g) => ({
    group: String(g('model')),
    study: 'report 1',
    id: String(g('model')),
    subj: String(g('subject')),
    seed: Number(g('seed')),
    pos: Number(g('position_rad')),
    r2: Number(g('r2')),
    mae: Number(g('mae')),
    toes: Number(g('toes_down')),
    peak: NaN, // scored on each run's own target: no common peak torque
  }));
  const order = Object.keys(FAMILY_NAMES);
  return groupScores(rows, floor, (id) => FAMILY_NAMES[id] ?? id).sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
}

// ── learning curves ──────────────────────────────────────────────────────────
export const LC_OURS = DEFAULT_B_ID;
export const LC_LAST = REFERENCE_ID;

export interface CurveSeries {
  epochs: number[];
  fit: number[];
  val: number[];
  test: number[];
}

export interface LearningRow {
  subj: string;
  model: 'ours' | 'last';
  experiment: string;
  seed: number;
  epochs: number;
  best: number;
  maeFit: number;
  maeVal: number;
  maeTest: number;
  valFit: number;
  testVal: number;
  curve: CurveSeries;
}

export function learningRows(ds: Dataset): LearningRow[] {
  const out: LearningRow[] = [];
  for (const lc of ds.raw.learning_curves ?? []) {
    if (lc.experiment !== LC_OURS && lc.experiment !== LC_LAST) continue;
    const ix: Record<string, number> = {};
    lc.cols.forEach((c, i) => (ix[c] = i));
    const h = lc.history;
    const curve: CurveSeries = {
      epochs: h.map((r) => r[ix.epoch]),
      fit: h.map((r) => r[ix.mae_fit]),
      val: h.map((r) => r[ix.mae_val]),
      test: h.map((r) => r[ix.mae_test]),
    };
    const best = h.find((r) => r[ix.epoch] === lc.best_epoch) ?? h[h.length - 1];
    const maeFit = best[ix.mae_fit];
    const maeVal = best[ix.mae_val];
    const maeTest = best[ix.mae_test];
    out.push({
      subj: lc.subject,
      model: lc.experiment === LC_OURS ? 'ours' : 'last',
      experiment: lc.experiment,
      seed: lc.seed,
      epochs: lc.epochs,
      best: lc.best_epoch,
      maeFit,
      maeVal,
      maeTest,
      valFit: maeVal / maeFit,
      testVal: maeTest / maeVal,
      curve,
    });
  }
  const order = ds.subjects;
  return out.sort((a, b) => order.indexOf(a.subj) - order.indexOf(b.subj) || (a.model === 'last' ? -1 : 1) - (b.model === 'last' ? -1 : 1));
}

// ── bias / variance split ────────────────────────────────────────────────────
export interface BiasVarianceRow {
  subj: string;
  model: 'original' | 'ours';
  /** number of runs (seeds) in the split, from the data */
  runs: number;
  mse: number; // mean squared error of one run (Nm²), mean over the group's positions
  offsetShare: number;
  otherSysShare: number;
  runShare: number;
  ensR2: number; // median over positions of the run-average prediction's R²
  ensMae: number;
  runR2: number; // median over single runs (same runs) and positions
  runMae: number;
  exp: string;
}

export function biasVariance(ds: Dataset, toes: 0 | 1, floor: boolean): BiasVarianceRow[] {
  const rows = tableRows(ds.raw.bias_variance, (g) => ({
    model: String(g('model')) as 'original' | 'ours',
    subj: String(g('subject')),
    runs: Number(g('runs')),
    mse: Number(g('mse_runs')),
    sys: Number(g('systematic')),
    rr: Number(g('run_to_run')),
    off: Number(g('offset2')),
    r2e: Number(g('r2_ensemble')),
    maee: Number(g('mae_ensemble')),
    toes: Number(g('toes_down')),
  })).filter((r) => r.toes === toes);
  const out: BiasVarianceRow[] = [];
  const params: Record<string, Params | undefined> = {
    original: ds.experiments[DEFAULT_A_ID]?.params,
    ours: ds.experiments[DEFAULT_B_ID]?.params,
  };
  for (const s of ds.subjects)
    for (const m of ['original', 'ours'] as const) {
      const rs = rows.filter((r) => r.subj === s && r.model === m);
      if (!rs.length) continue;
      const mse = mean(rs.map((r) => r.mse));
      const sys = mean(rs.map((r) => r.sys));
      const rr = mean(rs.map((r) => r.rr));
      const off = mean(rs.map((r) => r.off));
      const nRuns = Math.max(...rs.map((r) => r.runs));
      // the same runs as the split: the first `runs` seeds of the shown experiment, learn 1 → test 2
      const p = params[m];
      const res = p ? resolve(ds, p, s, 'S1toS2') : null;
      const seeds = res ? seedsOf(ds, res.exp, s, 'S1toS2').slice(0, nRuns) : [];
      const single = res ? runsOf(ds, res.exp, s, 'S1toS2').filter((r) => r.toes === toes && seeds.includes(r.seed)) : [];
      out.push({
        subj: s,
        model: m,
        runs: nRuns,
        mse,
        offsetShare: off / mse,
        otherSysShare: (sys - off) / mse,
        runShare: rr / mse,
        ensR2: median(rs.map((r) => floorR2(r.r2e, floor))),
        ensMae: median(rs.map((r) => r.maee)),
        runR2: median(single.map((r) => floorR2(r.r2, floor))),
        runMae: median(single.map((r) => r.mae)),
        exp: res?.exp ?? '',
      });
    }
  return out;
}

/** "3" or "3–10": run counts in a set of bias / variance rows */
export function runsLabel(rows: { runs: number }[]): string {
  const n = Array.from(new Set(rows.map((r) => r.runs))).sort((a, b) => a - b);
  if (!n.length) return '–';
  return n.length === 1 ? String(n[0]) : `${n[0]}–${n[n.length - 1]}`;
}

