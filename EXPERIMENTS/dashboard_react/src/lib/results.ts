// Every experiment as one list of runs the Results page can show: the window study (study.json) and the earlier
// tuning finalists (the presentation catalog), all trained on session 1 and tested on session 2. Last semester's runs
// were removed from the page on 17 Nov 2026: they used the old pipeline.
import { catalogRuns, EXTRA, runTiming } from './catalog';
import { median } from './stats';
import { has as inStudy, paramsOf as studyParams, PARAMS as STUDY_PARAMS, STUDY } from './study';
import { TUNING_VAL } from './tuningVal';

export interface Param { key: string; label: string }
export interface Group { key: string; label: string; params: Param[]; first: string }
export interface Entry { id: string; group: string; title: string; params: Record<string, string>; target: 'active' | 'total' }

/** one person's runs: values[pos][k] for seed seeds[k] (NaN when missing), and the peak torque at each position */
export interface PersonRuns { subj: string; positions: number[]; seeds: number[]; mae: number[][]; r2: number[][]; peaks: number[] }

const DIR = 'S1toS2' as const;
const PEOPLE = STUDY.people;

// the tuning finalists
// one list of the 6 finalists (batch 64, up to 400 epochs); the batch-8 rerun was dropped 16 Nov 2026
const TUNING: Param[] = [{ key: 'network', label: 'Network' }];
const TUNING_BASELINE = TUNING_VAL.baseline;
const TUNING_NOTE: Record<string, string> = {
  [TUNING_VAL.chosen]: ' (chosen)', [TUNING_BASELINE]: " (last semester's network size, retrained on this pipeline)" };

// Experiments filters: the window study's, without training order, weight updates and epoch limit (dropped 16 Nov
// 2026 with the few runs that varied them). The 10 s carried-state run stays: it is the only one with that state.
// The step filter was dropped on 17 Nov 2026 with every setting that fed the network one sample every 50 ms: only
// 10 ms steps are shown.
const DROPPED = new Set(['order', 'updates', 'epochs', 'step']);
const EXP_PARAMS = STUDY_PARAMS.filter((p) => !DROPPED.has(p.key));
const shown = (id: string) => {
  const s = STUDY.settings[id];
  return !s.shuffle && !id.endsWith('_eq') && s.step_ms === 10 && (s.epochs === 150 || s.state === 'carry');
};
/**
 * Overlap as the study designed it, not the rounded ratio: the grid's 0/50/75/90 % (a stride must be a whole number of
 * 10 ms samples, so 75 % of 500 ms is really 76 %), and "full" for a new training window every 10 ms (95–99.9 %).
 */
export function overlapLabel(id: string): string {
  const s = STUDY.settings[id];
  if (s.state === 'carry') return 'n/a (state carried)';
  const m = /_o(\d\d)(_10ms)?(_tot)?$/.exec(id) ?? /^WO\d_(\d\d)/.exec(id);
  return m ? `${Number(m[1])} %` : 'full';
}
const dur = (sec: number) => (sec >= 1 ? `${sec} s` : `${Math.round(sec * 1000)} ms`);
const expParams = (id: string): Record<string, string> => {
  const p = Object.fromEntries(Object.entries(studyParams(id)).filter(([k]) => !DROPPED.has(k)));
  p.overlap = overlapLabel(id);
  if (STUDY.settings[id].state === 'carry') p.length = 'whole trial';
  return p;
};
/** one naming scheme for every setting, built from its parameters */
function studyTitle(id: string): string {
  const s = STUDY.settings[id];
  const torque = s.target === 'total' ? 'total torque' : 'active torque';
  if (s.state === 'carry') return `LSTM state carried through the trial, ${dur(s.chunk_s ?? 0)} training chunks · ${torque}`;
  const ov = overlapLabel(id);
  return `${dur(s.length_s)} window · ${ov === 'full' ? 'full overlap' : `${ov} overlap`} · ${torque}`;
}

export const GROUPS: Group[] = [
  { key: 'window', label: 'Experiments', params: EXP_PARAMS, first: 'W00_reference' },
  { key: 'tuning', label: 'Hyperparameter tuning', params: TUNING, first: 'HF_u16_l2_d00_lr01' },
];

function tuning(): Entry[] {
  const out: Entry[] = [];
  for (const id of Object.keys(EXTRA)) {
    const m = /u(\d+)_l(\d)_d(\d\d)_lr(\d+)/.exec(id);
    if (!m || !id.startsWith('HF_')) continue;
    const network = `${m[1]} units × ${m[2]} layer${m[2] === '1' ? '' : 's'}, dropout ${Number(m[3]) / 100}, learning rate 0.${m[4]}${TUNING_NOTE[id] ?? ''}`;
    out.push({ id, group: 'tuning', title: network, target: 'active', params: { network } });
  }
  return out;
}

/** every experiment with runs trained on session 1 and tested on session 2 */
export const ENTRIES: Entry[] = [
  ...Object.keys(STUDY.settings).filter((id) => inStudy(id) && shown(id)).map((id): Entry => ({
    id, group: 'window', title: studyTitle(id), target: STUDY.settings[id].target, params: expParams(id),
  })),
  ...tuning().filter((e) => PEOPLE.some((s) => catalogRuns(e.id, s, DIR).length)),
];
const BY_ID = new Map(ENTRIES.map((e) => [e.id, e]));
export const entry = (id: string) => BY_ID.get(id)!;
export const groupOf = (id: string) => GROUPS.find((g) => g.key === entry(id).group)!;

/** every value a parameter takes inside a group, numbers ascending */
export function valuesOf(group: string, key: string): string[] {
  const vs = [...new Set(ENTRIES.filter((e) => e.group === group).map((e) => e.params[key]))];
  const num = (v: string) => {
    const n = parseFloat(v);
    if (/chunks/.test(v)) return 1e6 + (/\d\s*ms/.test(v) ? n / 1000 : n);      // carried state after "reset"
    return /\d\s*ms/.test(v) ? n / 1000 : Number.isFinite(n) ? n : 1e9;      // words ("full", "whole trial") last
  };
  return vs.sort((a, b) => num(a) - num(b) || a.localeCompare(b));
}

/** the parameters that differ between two entries, as "label: old → new" (filters move to the nearest run setting) */
export function changedParams(from: string, to: string, except: string): string[] {
  const a = entry(from), b = entry(to), g = groupOf(to);
  if (a.group !== b.group) return [];
  return g.params.filter((p) => p.key !== except && a.params[p.key] !== b.params[p.key])
    .map((p) => `${p.label}: ${a.params[p.key]} → ${b.params[p.key]}`);
}

/** the setting every other one is compared with: the 200 ms reference of the same torque, or last semester's network */
export function referenceOf(id: string): string | null {
  const e = entry(id);
  const ref = e.group === 'tuning' ? TUNING_BASELINE : e.target === 'total' ? 'WT1_total' : 'W00_reference';
  return ref === id || !BY_ID.has(ref) ? null : ref;
}

/** warnings about a setting that the filters do not show */
export function notesOf(id: string): string[] {
  const out: string[] = [];
  const s = STUDY.settings[id];
  if (s?.epochs && s.epochs !== 150) out.push(`Trained for up to ${s.epochs} epochs (all other settings: 150).`);
  if (id === 'WC4_carry_10s') out.push('Also uses gradient clipping (1.0) and has 1 seed: not directly comparable with the other settings.');
  if (['WL2_1s', 'WC2_carry_500ms', 'WC3_carry_1s'].includes(id))
    out.push('Seeds 5–9 were added after seeds 0–4 had been seen (the run plan had 5 seeds); comparisons with the reference use matched seeds.');
  const n = new Set(runsOf(id).map((p) => p.seeds.length));
  if (n.size === 1 && [...n][0] === 1) out.push('Only 1 seed: no interval can be given, and the run shown is the only run.');
  return out;
}

/** largest |measured active torque| of a person over all positions (Nm): the scale for the normalised error */
export function activePeak(subj: string): number {
  const v = (STUDY.peaks?.active?.[subj] ?? []).filter(Number.isFinite);
  return v.length ? Math.max(...v) : NaN;
}

/** the experiment in the group with key = value that keeps as many of the current choices as possible */
export function pickFor(current: string, key: string, value: string): string {
  const now = entry(current);
  let best = current, score = -1;
  for (const e of ENTRIES) {
    if (e.group !== now.group || e.params[key] !== value) continue;
    const sc = Object.keys(e.params).filter((k) => e.params[k] === now.params[k]).length;
    if (sc > score) { best = e.id; score = sc; }
  }
  return best;
}

const r3 = (x: number) => Math.round(x * 1000) / 1000;

/** the runs of one experiment, per person (people without runs are left out) */
export function runsOf(id: string): PersonRuns[] {
  const e = entry(id);
  if (inStudy(id)) {
    const col = (m: string) => STUDY.metrics.indexOf(m);
    return PEOPLE.filter((s) => STUDY.runs[id]?.[s]).map((subj) => {
      const runs = STUDY.runs[id][subj];
      const seeds = Object.keys(runs).map(Number).sort((a, b) => a - b);
      const positions = STUDY.positions[subj] ?? [];
      const grid = (m: string) => positions.map((_, k) => seeds.map((sd) => runs[String(sd)]?.[k]?.[col(m)] ?? NaN));
      return { subj, positions, seeds, mae: grid('mae'), r2: grid('r2'), peaks: STUDY.peaks?.[e.target]?.[subj] ?? positions.map(() => NaN) };
    });
  }
  return PEOPLE.map((subj) => {
    const rows = catalogRuns(id, subj, DIR);
    const positions = [...new Set(rows.map((r) => r3(r.pos)))].sort((a, b) => a - b);
    const seeds = [...new Set(rows.map((r) => r.seed))].sort((a, b) => a - b);
    const at = (k: number, sd: number) => rows.find((r) => r3(r.pos) === positions[k] && r.seed === sd);
    const grid = (m: 'mae' | 'r2') => positions.map((_, k) => seeds.map((sd) => at(k, sd)?.[m] ?? NaN));
    const peaks = positions.map((p) => median(rows.filter((r) => r3(r.pos) === p).map((r) => r.peak).filter(Number.isFinite)));
    return { subj, positions, seeds, mae: grid('mae'), r2: grid('r2'), peaks };
  }).filter((p) => p.seeds.length);
}

/** a person's score: per seed, MAE averaged over positions and R² the median over positions; then the median over seeds */
export { maeRatio, personScore, scoreCI, worstR2 } from './score';

/** one person's training times, seconds per seed (only seeds with a saved time) */
export interface PersonTimes { subj: string; seeds: number[]; seconds: number[] }

/** saved training times of an experiment, and what each timer covered */
export function timesOf(id: string): { people: PersonTimes[]; definition: string } {
  if (inStudy(id)) {
    const people = PEOPLE.filter((s) => STUDY.times?.[id]?.[s]).map((subj) => {
      const t = Object.entries(STUDY.times![id][subj]).map(([sd, v]) => [Number(sd), v] as const).sort((a, b) => a[0] - b[0]);
      return { subj, seeds: t.map((x) => x[0]), seconds: t.map((x) => x[1]) };
    }).filter((p) => p.seeds.length);
    return { people, definition: 'model.fit only, for one person and one seed' };
  }
  const defs = new Set<string>();
  const people = runsOf(id).map((p) => {
    const t = p.seeds.map((sd) => [sd, runTiming(id, p.subj, DIR, sd)] as const).filter((x) => x[1] && Number.isFinite(x[1].seconds));
    t.forEach((x) => defs.add(x[1]!.definition));
    return { subj: p.subj, seeds: t.map((x) => x[0]), seconds: t.map((x) => x[1]!.seconds) };
  }).filter((p) => p.seeds.length);
  return { people, definition: [...defs].join('; ') };
}
