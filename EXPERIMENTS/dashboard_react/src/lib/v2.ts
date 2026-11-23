// Study v2 (pre-registered 18 Nov 2026): the page's only data. Everything comes from study_v2.json, written by
// EXPERIMENTS/analysis/study_v2_export.py: the final settings, their session-2 scores on confirm's own samples, the
// confirmatory decisions (confirm.json, copied in) and the session-1 validation scores that chose each setting.
import raw from '../data/study_v2.json';
import type { Machine, Versions } from './machine';

type Num = number | null;
interface Net { units: Num; layers: Num; dropout: Num; lr: Num; batch: Num }
interface Setting extends Net {
  title: string; kind: 'A0' | 'cell' | 'A1' | 'A2' | 'A3'; cell: string | null; model: 'lstm' | 'ridge_lagged';
  window_s: number; overlap: number | 'full' | null; state: 'reset' | 'carry' | null; chunk_s: number | null;
  target: 'active' | 'total';
}
interface FinalPerson { positions: number[]; seeds: number[]; mae: Num[][]; nmae: Num[][]; r2: Num[][]; peak: number[]; train_s: Record<string, Num> }
interface TuningConfig { id: string; seeds: number[]; val_nmae?: Record<string, number>; incomplete?: boolean; params: Net | null }
interface Compared { arm: string; comparator: string; effects: Record<string, number>; effect: number; decision: string }
interface Pooled { lengths?: string[]; per_length?: Record<string, Record<string, number>>; effects?: Record<string, number>; effect?: number; decision: string }
interface Choice { A0: string; cell_winners: Record<string, string>; common_nmae: Record<string, Record<string, number>> }
interface Confirm {
  families: { F1: Record<string, Compared | { decision: string }>; F2: Record<string, Pooled>; F3: Record<string, Compared> };
  limitations: string[]; choice: Choice; provenance_check: { commits: string[]; protocol_hash: string };
}
interface V2Data {
  tuning: Record<string, Record<string, Record<string, TuningConfig>>>; confirm: Confirm;
  settings: Record<string, Setting>; final: Record<string, Record<string, FinalPerson>>;
  score_from_s: Record<string, Record<string, number>>; versions: Versions[]; machine: Machine | null;
}
const D = raw as unknown as V2Data;

export const CONFIRM = D.confirm;
export const MACHINE = D.machine;
export const VERSIONS = D.versions;
export const PEOPLE = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES'].filter((p) => Object.values(D.final).some((f) => f[p]));
/** where scoring starts after each trial's start (s): the same for every setting */
export const SCORE_FROM = [...new Set(Object.values(D.score_from_s).flatMap((d) => Object.values(d)))];

export interface Param { key: string; label: string }
export interface Entry { id: string; group: 'v2'; title: string; params: Record<string, string>; target: 'active' | 'total' }

export const DEFAULT = 'FE_A0';
export const PARAMS: Param[] = [
  { key: 'model', label: 'Model' }, { key: 'window', label: 'Window' }, { key: 'overlap', label: 'Overlap' },
  { key: 'state', label: 'LSTM state' }, { key: 'torque', label: 'Torque' },
];
/** each filter's values in display order */
const ORDER: Record<string, string[]> = {
  model: ['tuned LSTM', "last semester's network (8×2)", 'lagged ridge'],
  window: ['0.2 s', '0.5 s', '1 s'],
  overlap: ['full', '75 %', '50 %', '0 %', 'n/a'],
  state: ['reset each window', 'carried, 2 s chunks', 'carried, 5 s chunks', 'carried, 10 s chunks', 'n/a'],
  torque: ['active', 'total'],
};

function paramsOf(s: Setting): Record<string, string> {
  const ridge = s.model === 'ridge_lagged';
  return {
    model: ridge ? 'lagged ridge' : s.kind === 'A2' ? "last semester's network (8×2)" : 'tuned LSTM',
    window: `${Math.round(s.window_s * 10) / 10} s`,
    // stride 13 of 50 samples is 74 %: shown as the grid's 75 %
    overlap: s.overlap === null ? 'n/a' : s.overlap === 'full' ? 'full' : `${Math.round((s.overlap * 100) / 25) * 25} %`,
    state: s.state === null ? 'n/a' : s.state === 'carry' ? `carried, ${s.chunk_s} s chunks` : 'reset each window',
    torque: s.target,
  };
}

export const SETTINGS: Entry[] = Object.entries(D.settings).map(([id, s]) => ({ id, group: 'v2', title: s.title, params: paramsOf(s), target: s.target }));
const BY_ID = new Map(SETTINGS.map((e) => [e.id, e]));
export const has = (id: string) => BY_ID.has(id);
export const entry = (id: string): Entry => BY_ID.get(id) ?? BY_ID.get(DEFAULT)!;
export const settingOf = (id: string): Setting => D.settings[id];

/** every value a filter takes, in display order */
export function valuesOf(key: string): string[] {
  const vs = new Set(SETTINGS.map((e) => e.params[key]));
  return ORDER[key].filter((v) => vs.has(v));
}

/** the setting with key = value that keeps as many of the current choices as possible */
export function pickFor(current: string, key: string, value: string): string {
  const now = entry(current);
  let best = current, score = -1;
  for (const e of SETTINGS) {
    if (e.params[key] !== value) continue;
    const sc = Object.keys(e.params).filter((k) => e.params[k] === now.params[k]).length;
    if (sc > score) { best = e.id; score = sc; }
  }
  return best;
}

/** the filters that differ between two settings, as "label: old → new" */
export function changedParams(from: string, to: string, except: string): string[] {
  const a = entry(from), b = entry(to);
  return PARAMS.filter((p) => p.key !== except && a.params[p.key] !== b.params[p.key])
    .map((p) => `${p.label}: ${a.params[p.key]} → ${b.params[p.key]}`);
}

/** one person's runs: values[position][k] for seed seeds[k]; peak |measured target| per position over the scored samples */
export interface PersonRuns { subj: string; positions: number[]; seeds: number[]; mae: number[][]; nmae: number[][]; r2: number[][]; peaks: number[] }
const nums = (g: Num[][]) => g.map((row) => row.map((v) => (v === null ? NaN : v)));

export function runsOf(id: string): PersonRuns[] {
  const f = D.final[id] ?? {};
  return PEOPLE.filter((p) => f[p]).map((subj) => {
    const x = f[subj];
    return { subj, positions: x.positions, seeds: x.seeds, mae: nums(x.mae), nmae: nums(x.nmae), r2: nums(x.r2), peaks: x.peak };
  });
}

/** one person's training times, seconds per seed */
export interface PersonTimes { subj: string; seeds: number[]; seconds: number[] }
export function timesOf(id: string): { people: PersonTimes[]; definition: string } {
  const people = runsOf(id).map((p) => {
    const t = Object.entries(D.final[id][p.subj].train_s).filter((x): x is [string, number] => typeof x[1] === 'number')
      .map(([sd, v]) => [Number(sd), v] as const).sort((a, b) => a[0] - b[0]);
    return { subj: p.subj, seeds: t.map((x) => x[0]), seconds: t.map((x) => x[1]) };
  }).filter((p) => p.seeds.length);
  return { people, definition: 'model training only, for one person and one seed' };
}

// ── the pre-registered comparisons ──────────────────────────────────────────
export interface ComparisonRow {
  family: 'F1' | 'F2' | 'F3'; label: string; role: 'arm' | 'comparator'; other: string;
  /** median over people of exp(log effect) − 1; F2: pooled over window lengths */
  effect: number; effects: Record<string, number>; decision: string;
  /** F2 only: the window length whose per-person effects are shown */
  length?: string;
}
const A0_CELL = Object.entries(CONFIRM.choice.cell_winners).find(([, v]) => v === CONFIRM.choice.A0)?.[0] ?? null;
/** confirm's run ids → the page's: A0's cell winner is A0 */
const pageId = (id: string) => (A0_CELL && id === `FE_cell_${A0_CELL}` ? DEFAULT : id);
const pct = (x: number) => Math.exp(x) - 1;
const pcts = (o: Record<string, number>) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, pct(v)]));
const nameOf = (id: string) => entry(id).title;

export const FAMILY: Record<ComparisonRow['family'], string> = {
  F1: 'F1 · history length (against 0.2 s, full overlap)',
  F2: 'F2 · window overlap (against full overlap)',
  F3: 'F3 · baselines and torque',
};

export function comparisonsOf(id: string): ComparisonRow[] {
  const out: ComparisonRow[] = [];
  const pair = (family: 'F1' | 'F3', c: Compared) => {
    const arm = pageId(c.arm), cmp = pageId(c.comparator);
    if (id !== arm && id !== cmp) return;
    out.push({ family, label: `${nameOf(arm)} vs ${nameOf(cmp)}`, role: id === arm ? 'arm' : 'comparator', other: id === arm ? cmp : arm,
      effect: pct(c.effect), effects: pcts(c.effects), decision: c.decision });
  };
  for (const c of Object.values(CONFIRM.families.F1)) if ('arm' in c) pair('F1', c);
  for (const [o, f] of Object.entries(CONFIRM.families.F2)) {
    if (!f.lengths || !f.per_length || f.effect === undefined) continue;
    for (const w of f.lengths) {
      const arm = `FE_cell_${w}_${o}`, cmp = `FE_cell_${w}_full`;
      if (id !== arm && id !== cmp) continue;
      out.push({ family: 'F2', label: `${nameOf(arm)} vs ${nameOf(cmp)}`, role: id === arm ? 'arm' : 'comparator', other: id === arm ? cmp : arm,
        effect: pct(f.effect), effects: pcts(f.per_length[w]), decision: f.decision, length: w });
    }
  }
  for (const c of Object.values(CONFIRM.families.F3)) pair('F3', c);
  return out;
}

// ── how the setting was chosen (session-1 validation only) ──────────────────
export interface RoundRow { round: string; config: string; id: string; params: Net | null; score: number; seeds: number[]; kept: boolean; winner: boolean }
export interface CellWinnerRow { cell: string; id: string; title: string; score: number; chosen: boolean }
export interface ChoiceInfo { rounds?: RoundRow[]; cellWinners?: CellWinnerRow[]; fixed?: string }
/** exp(mean over people of log nMAE): the selection's score, shown on the nMAE scale */
const geo = (v: Record<string, number>) => { const xs = Object.values(v); return Math.exp(xs.reduce((a, x) => a + Math.log(x), 0) / xs.length); };
const ROUND: Record<string, string> = { R1: 'coarse', R2: 'fine' };

function roundsOf(cell: string): RoundRow[] {
  const win = CONFIRM.choice.cell_winners[cell];
  return Object.entries(D.tuning[cell] ?? {}).flatMap(([r, cfgs]) => Object.entries(cfgs).map(([name, c]) => ({
    round: ROUND[r] ?? r, config: name, id: c.id, params: c.params, score: c.val_nmae ? geo(c.val_nmae) : NaN, seeds: c.seeds,
    kept: c.seeds.length > 1, winner: c.id === win,
  })));
}

export function choiceOf(id: string): ChoiceInfo {
  const s = D.settings[id];
  if (s.kind === 'A1') return { fixed: 'Lagged ridge regression: the EMG envelopes at 11 lags (0 to 0.5 s, every 50 ms); its α is picked in each run from 0.001 to 1000 on the session-1 validation block. Nothing else was tuned.' };
  if (s.kind === 'A2') return { fixed: "Last semester's network, fixed: 8 units × 2 layers, dropout 0.3, learning rate 0.003, batch 64, 0.2 s window at full overlap, trained with study v2's pipeline (the coarse round's anchor configuration). Not tuned." };
  if (s.kind === 'A3') return { fixed: "A0's configuration trained on total torque instead of active torque. Not tuned separately." };
  const info: ChoiceInfo = { rounds: s.cell ? roundsOf(s.cell) : [] };
  if (s.kind === 'A0') {
    const cn = CONFIRM.choice.common_nmae;
    info.cellWinners = Object.entries(CONFIRM.choice.cell_winners).filter(([, w]) => cn[w]).map(([cell, w]) => {
      const pid = has(`FE_cell_${cell}`) ? `FE_cell_${cell}` : pageId(`FE_cell_${cell}`);
      return { cell, id: w, title: has(pid) ? entry(pid).title.replace(/^A0 \(chosen\): /, '') : cell, score: geo(cn[w]), chosen: w === CONFIRM.choice.A0 };
    }).sort((a, b) => a.score - b.score);
  }
  return info;
}
