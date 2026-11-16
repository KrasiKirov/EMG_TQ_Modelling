// Every experiment as one list of runs the Results page can show: the window study (study.json) and the earlier
// experiments (results.json and the presentation catalog), all trained on session 1 and tested on session 2.
import { catalogRuns, catalogTitle, EXTRA, runTiming } from './catalog';
import { median } from './stats';
import { has as inStudy, paramsOf as studyParams, PARAMS as STUDY_PARAMS, STUDY } from './study';

export interface Param { key: string; label: string }
export interface Group { key: string; label: string; params: Param[]; first: string }
export interface Entry { id: string; group: string; title: string; params: Record<string, string>; target: 'active' | 'total' }

/** one person's runs: values[pos][k] for seed seeds[k] (NaN when missing), and the peak torque at each position */
export interface PersonRuns { subj: string; positions: number[]; seeds: number[]; mae: number[][]; r2: number[][]; peaks: number[] }

const DIR = 'S1toS2' as const;
const PEOPLE = STUDY.people;

// ── the earlier experiments: the tuning finalists, and last semester's LSTM as the starting point of the experiments ──
// one list of the 6 finalists (batch 64, up to 400 epochs); the batch-8 rerun was dropped 16 Nov 2026
const TUNING: Param[] = [{ key: 'network', label: 'Network' }];
const TUNING_NOTE: Record<string, string> = { HF_u16_l2_d00_lr01: ' (chosen)', HF_u8_l2_d30_lr003: " (last semester's network)" };
export const START = 'X00_baseline';

// Experiments filters: the window study's, without training order, weight updates and epoch limit (dropped 16 Nov
// 2026 with the few runs that varied them). The 10 s carried-state run stays: it is the only one with that state.
const DROPPED = new Set(['order', 'updates', 'epochs']);
const EXP_PARAMS = STUDY_PARAMS.filter((p) => !DROPPED.has(p.key));
const shown = (id: string) => {
  const s = STUDY.settings[id];
  return !s.shuffle && !id.endsWith('_eq') && (s.epochs === 150 || s.state === 'carry');
};
/** the runs titled "Reference" in the study, named by their settings instead */
const RENAMED: Record<string, string> = { W00_reference: '200 ms, 95 % overlap', WT1_total: '200 ms, 95 % overlap (total)' };
const expParams = (id: string) => Object.fromEntries(Object.entries(studyParams(id)).filter(([k]) => !DROPPED.has(k)));
const CURRENT = 'Current';
export const LAST_SEMESTER = 'Last semester';

export const GROUPS: Group[] = [
  { key: 'window', label: 'Experiments', params: [{ key: 'pipeline', label: 'Pipeline' }, ...EXP_PARAMS], first: 'W00_reference' },
  { key: 'tuning', label: 'Hyperparameter tuning', params: TUNING, first: 'HF_u16_l2_d00_lr01' },
];

function earlier(): Entry[] {
  // last semester's LSTM in the experiments' own terms (docs/BASELINE.md): 20 × 10 ms windows started every sample,
  // active torque, a fresh LSTM state per window, shuffled windows, at most 200 epochs
  const out: Entry[] = [{ id: START, group: 'window', title: catalogTitle(START), target: 'active', params: {
    pipeline: LAST_SEMESTER, length: '200 ms', overlap: '95 %', step: '10 ms', target: 'Active', state: 'Reset each window',
  } }];
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
    id, group: 'window', title: RENAMED[id] ?? STUDY.settings[id].title, target: STUDY.settings[id].target, params: { pipeline: CURRENT, ...expParams(id) },
  })),
  ...earlier().filter((e) => PEOPLE.some((s) => catalogRuns(e.id, s, DIR).length)),
];
const BY_ID = new Map(ENTRIES.map((e) => [e.id, e]));
export const entry = (id: string) => BY_ID.get(id)!;
export const groupOf = (id: string) => GROUPS.find((g) => g.key === entry(id).group)!;

/** every value a parameter takes inside a group, numbers ascending */
export function valuesOf(group: string, key: string): string[] {
  const vs = [...new Set(ENTRIES.filter((e) => e.group === group).map((e) => e.params[key]))];
  const num = (v: string) => { const n = parseFloat(v); return /\d\s*ms/.test(v) && !/chunks/.test(v) ? n / 1000 : n; };
  return vs.sort((a, b) => (Number.isFinite(num(a)) && Number.isFinite(num(b)) ? num(a) - num(b) : a.localeCompare(b)));
}

/** the experiment in the group with key = value that keeps as many of the current choices as possible */
export function pickFor(current: string, key: string, value: string): string {
  const now = entry(current);
  if (key === 'pipeline' && value === CURRENT) return GROUPS.find((g) => g.key === now.group)!.first; // back to the reference
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
export { personScore, scoreCI } from './score';

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
