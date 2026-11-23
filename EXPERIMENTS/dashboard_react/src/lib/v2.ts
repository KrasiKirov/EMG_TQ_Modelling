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
interface TrainInfo { best_epoch: Num; epochs: Num; n_params: Num }
interface FinalPerson { positions: number[]; seeds: number[]; mae: Num[][]; nmae: Num[][]; r2: Num[][]; peak: number[]; train_s: Record<string, Num>; train?: Record<string, TrainInfo> }
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
  // in this order the filters narrow each other: a filter offers only the values that were run with the choices above it
  // memory comes before window length: a carried-state run has no window length (it trains on whole chunks)
  { key: 'model', label: 'Model' }, { key: 'state', label: 'Memory' }, { key: 'window', label: 'Window length' },
  { key: 'overlap', label: 'Overlap' }, { key: 'torque', label: 'Torque target' },
];
/** each filter's values in display order */
const ORDER: Record<string, string[]> = {
  model: ['tuned LSTM', "last semester's network (8×2)", 'lagged ridge'],
  window: ['0.2 s', '0.5 s', '1 s', 'n/a'],
  overlap: ['full', '75 %', '50 %', '0 %', 'n/a'],
  state: ['reset each window', 'carried, 2 s chunks', 'carried, 5 s chunks', 'carried, 10 s chunks', 'n/a'],
  torque: ['active', 'total'],
};

function paramsOf(s: Setting): Record<string, string> {
  const ridge = s.model === 'ridge_lagged';
  return {
    model: ridge ? 'lagged ridge' : s.kind === 'A2' ? "last semester's network (8×2)" : 'tuned LSTM',
    // a carried run's spec window only aligns the scored samples (run.py fit_carry): no window length
    window: s.state === 'carry' ? 'n/a' : `${Math.round(s.window_s * 10) / 10} s`,
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

/** settings that match every given filter value */
const matching = (fixed: Record<string, string>) => SETTINGS.filter((e) => Object.entries(fixed).every(([k, v]) => e.params[k] === v));
/** the current setting's values of the filters above `key` */
function above(current: string, key: string): Record<string, string> {
  const now = entry(current).params, out: Record<string, string> = {};
  for (const p of PARAMS) { if (p.key === key) break; out[p.key] = now[p.key]; }
  return out;
}

/** a filter's values in display order; a value is disabled when no run has it together with the filters above */
export function optionsOf(current: string, key: string): { value: string; label: string; disabled: boolean }[] {
  const all = new Set(SETTINGS.map((e) => e.params[key]));
  const fixed = above(current, key);
  return ORDER[key].filter((v) => all.has(v)).map((v) => ({ value: v, label: v, disabled: !matching({ ...fixed, [key]: v }).length }));
}

/** the setting after choosing key = value: the filters above stay, each filter below keeps its value if that was run
 * with the choices so far, otherwise takes its first value that was */
export function pickFor(current: string, key: string, value: string): string {
  const now = entry(current).params;
  const fixed = { ...above(current, key), [key]: value };
  if (!matching(fixed).length) return current;
  const below = PARAMS.slice(PARAMS.findIndex((p) => p.key === key) + 1);
  for (const p of below) {
    const ok = (v: string) => matching({ ...fixed, [p.key]: v }).length > 0;
    fixed[p.key] = ok(now[p.key]) ? now[p.key] : ORDER[p.key].find(ok)!;
  }
  return matching(fixed)[0].id;
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

// ── A0's cell winner is shown as A0 ──────────────────────────────────────────
const A0_CELL = Object.entries(CONFIRM.choice.cell_winners).find(([, v]) => v === CONFIRM.choice.A0)?.[0] ?? null;
/** confirm's run ids → the page's: A0's cell winner is A0 */
const pageId = (id: string) => (A0_CELL && id === `FE_cell_${A0_CELL}` ? DEFAULT : id);

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

// ── the tuning experiment and the overfitting check (session-1 validation vs session 2) ──
export interface TuningRun { cell: string; round: string; config: string; id: string; units: number; layers: number; dropout: number; lr: number; batch: number | null; score: number }
/** every tuning run with a complete validation score: 172 configurations over the 16 tuned window cells */
export function tuningRuns(): TuningRun[] {
  return Object.entries(D.tuning).flatMap(([cell, rounds]) => Object.entries(rounds).flatMap(([r, cfgs]) => Object.entries(cfgs)
    .filter(([, c]) => c.val_nmae && c.params && c.params.units !== null)
    .map(([config, c]) => ({ cell, round: ROUND[r] ?? r, config, id: c.id, units: c.params!.units as number, layers: c.params!.layers as number,
      dropout: c.params!.dropout as number, lr: c.params!.lr as number, batch: c.params!.batch, score: geo(c.val_nmae!) }))));
}
/** the tuning config a setting uses: its cell's winner (A0 and A3 use A0's cell winner); null for A1 and A2, which were not tuned */
export function chosenConfig(id: string): string | null {
  const s = D.settings[id];
  return s.kind === 'A1' || s.kind === 'A2' || !s.cell ? null : CONFIRM.choice.cell_winners[s.cell] ?? null;
}
export const isWinner = (configId: string) => Object.values(CONFIRM.choice.cell_winners).includes(configId);
export const kindOf = (id: string) => D.settings[id].kind;
export const cellOf = (id: string) => D.settings[id].cell;
/** the network of a setting, as "64×1" (null for ridge) */
export const sizeOf = (id: string) => { const s = D.settings[id]; return s.units ? `${s.units}×${s.layers}` : null; };

/** this setting's training, median over every final run: the epoch whose weights were kept, epochs run, weights */
export function trainOf(id: string): { bestEpoch: number; epochs: number; nParams: number; runs: number } | null {
  const xs = Object.values(D.final[id] ?? {}).flatMap((f) => Object.values(f.train ?? {}));
  const ok = xs.filter((x) => typeof x.best_epoch === 'number' && typeof x.epochs === 'number' && (x.epochs as number) > 0);
  if (!ok.length) return null;
  const med = (v: number[]) => { const a = [...v].sort((x, y) => x - y); const m = a.length >> 1; return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2; };
  return { bestEpoch: med(ok.map((x) => x.best_epoch as number)), epochs: med(ok.map((x) => x.epochs as number)),
    nParams: med(ok.map((x) => x.n_params as number)), runs: ok.length };
}

export interface OverfitPoint { id: string; title: string; size: string; val: number; test: number }
/** each tuned cell winner and last semester's network: its selection score on session-1 validation against its
 * session-2 nMAE (both exp of the mean over people of log nMAE; session 2 per person the median over seeds) */
export function overfitPoints(test: (id: string) => number): OverfitPoint[] {
  const cfg = new Map(tuningRuns().map((r) => [r.id, r]));
  const out: OverfitPoint[] = [];
  for (const [cell, w] of Object.entries(CONFIRM.choice.cell_winners)) {
    const pid = pageId(`FE_cell_${cell}`), r = cfg.get(w);
    if (!r || !has(pid)) continue;
    out.push({ id: pid, title: entry(pid).title, size: `${r.units}×${r.layers}`, val: r.score, test: test(pid) });
  }
  const anchor = Object.values(D.tuning.w20_full?.R1 ?? {}).find((c) => c.id.endsWith('_anchor'));
  const a2 = Object.keys(D.settings).find((k) => D.settings[k].kind === 'A2');
  if (anchor?.val_nmae && a2) out.push({ id: a2, title: entry(a2).title, size: '8×2', val: geo(anchor.val_nmae), test: test(a2) });
  return out.filter((p) => Number.isFinite(p.test));
}
