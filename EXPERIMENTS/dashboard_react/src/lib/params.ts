// Model = parameter set. Labels, presets and the resolution rule (which experiment to show for a person).
import { hasRuns, seedsOf, type Dataset } from './data';
import type { Dir, ParamKey, ParamValue, Params } from './types';

export const PARAM_KEYS: ParamKey[] = ['code', 'passive', 'emg', 'weighting', 'window', 'dropout', 'lr', 'extra'];

export const PARAM_NAMES: Record<ParamKey, string> = {
  code: 'Code',
  passive: 'Passive table',
  emg: 'EMG scaling',
  weighting: 'Position weighting',
  window: 'Window',
  dropout: 'Dropout',
  lr: 'Learning rate',
  extra: 'Extra input',
};

const VALUE_LABELS: Partial<Record<ParamKey, Record<string, string>>> = {
  code: { original: 'original code (frozen copy)', framework: 'our framework (same model)' },
  passive: { original: 'original table', clean: 'clean table (static holds only)' },
  emg: { max: '÷ training max (last semester)', autogain: 'auto-gain (running per muscle)' },
  weighting: { none: 'none (plain Nm² loss)', equal: 'equal weight per position' },
  extra: { none: 'none', passive: 'passive value (rejected: circular)' },
};

/** short labels for badges */
const SHORT_LABELS: Partial<Record<ParamKey, Record<string, string>>> = {
  code: { original: 'original code', framework: 'our framework' },
  passive: { original: 'original table', clean: 'clean table' },
  emg: { max: 'EMG ÷ train max', autogain: 'EMG auto-gain' },
  weighting: { none: 'no weighting', equal: 'equal weight / position' },
  extra: { none: 'no extra input', passive: 'passive value input' },
};

/** compact labels for the select inputs (the dropdown shows the full paramLabel) */
const MEDIUM_LABELS: Partial<Record<ParamKey, Record<string, string>>> = {
  code: { original: 'original code', framework: 'our framework' },
  passive: { original: 'original table', clean: 'clean table' },
  emg: { max: '÷ training max', autogain: 'auto-gain' },
  weighting: { none: 'none', equal: 'equal per position' },
  extra: { none: 'none', passive: 'passive value' },
};

export function paramMedium(k: ParamKey, v: ParamValue): string {
  const m = MEDIUM_LABELS[k];
  if (m && m[String(v)]) return m[String(v)];
  if (k === 'window') return `${v} s`;
  return String(v);
}

export function paramLabel(k: ParamKey, v: ParamValue): string {
  const m = VALUE_LABELS[k];
  if (m && m[String(v)]) return m[String(v)];
  if (k === 'window') return `${v} s (${Math.round(Number(v) * 100)} samples)`;
  if (k === 'lr') return String(v);
  if (k === 'dropout') return String(v);
  return String(v);
}

export function paramShort(k: ParamKey, v: ParamValue): string {
  const m = SHORT_LABELS[k];
  if (m && m[String(v)]) return m[String(v)];
  if (k === 'window') return `window ${v} s`;
  if (k === 'lr') return `learning rate ${v}`;
  if (k === 'dropout') return `dropout ${v}`;
  return `${PARAM_NAMES[k]} ${v}`;
}

export const FRIENDLY: Record<string, string> = {
  X00_baseline: 'Last semester (original code)',
  X00f_baseline_framework: 'Last semester (re-run, our framework)',
  X01_clean_passive: 'Clean passive table',
  'X02a_dropout_0.0': 'Dropout 0',
  'X02b_dropout_0.1': 'Dropout 0.1',
  'X02c_dropout_0.5': 'Dropout 0.5',
  'X03_lr_0.001': 'Learning rate 0.001',
  X04_equal_angle_weight: 'Equal weight per position',
  X05_autogain: 'EMG auto-gain',
  'X06_window_0.4s': 'Window 0.4 s',
  T01_passive_value_input: 'Passive value as input (rejected: circular)',
  'C01_autogain+equalweight': 'Auto-gain + equal weight',
  'S2_cleanpassive+autogain': 'Clean table + auto-gain',
  'D01_cleanpassive+autogain+equalweight': 'Ours: clean table + auto-gain + equal weight',
  'D02_D01+window_0.5s': 'Ours + 0.5 s window',
};

export function friendly(id: string): string {
  return FRIENDLY[id] ?? id.replace(/_/g, ' ');
}

/** short code shown next to friendly names, e.g. "D01" */
export function shortId(id: string): string {
  return id.split('_')[0];
}

export const PRESET_GROUPS: { group: string; ids: string[] }[] = [
  { group: 'Reference', ids: ['X00_baseline', 'X00f_baseline_framework'] },
  { group: 'Ours', ids: ['D01_cleanpassive+autogain+equalweight', 'D02_D01+window_0.5s'] },
  {
    group: 'Single changes',
    ids: ['X01_clean_passive', 'X02a_dropout_0.0', 'X02b_dropout_0.1', 'X02c_dropout_0.5', 'X03_lr_0.001', 'X04_equal_angle_weight', 'X05_autogain', 'X06_window_0.4s', 'T01_passive_value_input'],
  },
  { group: 'Combinations', ids: ['C01_autogain+equalweight', 'S2_cleanpassive+autogain'] },
];

export const DEFAULT_A_ID = 'X00_baseline';
export const DEFAULT_B_ID = 'D01_cleanpassive+autogain+equalweight';
export const REFERENCE_ID = 'X00f_baseline_framework';

export function sameParams(p: Params, q: Params, skipPassive = false): boolean {
  return PARAM_KEYS.every((k) => (skipPassive && k === 'passive') || String(p[k]) === String(q[k]));
}

/** experiment whose parameters equal p exactly (regardless of data), or null */
export function exactExperiment(ds: Dataset, p: Params): string | null {
  return ds.expIds.find((e) => sameParams(ds.experiments[e].params, p)) ?? null;
}

/** all values a parameter takes over the experiments (sorted) */
export function paramOptions(ds: Dataset, k: ParamKey): ParamValue[] {
  const vals = Array.from(new Set(ds.expIds.map((e) => ds.experiments[e].params[k])));
  const order: Record<string, string[]> = {
    code: ['original', 'framework'],
    passive: ['original', 'clean'],
    emg: ['max', 'autogain'],
    weighting: ['none', 'equal'],
    extra: ['none', 'passive'],
  };
  return vals.sort((a, b) => {
    const o = order[k];
    if (o) return o.indexOf(String(a)) - o.indexOf(String(b));
    return Number(a) - Number(b);
  });
}

export interface Resolved {
  exp: string;
  /** true when the experiment was accepted although its passive table differs (identical tables except for JM) */
  viaPassive: boolean;
}

/**
 * Resolution rule: experiments with exactly these params that have rows for (subj, dir);
 * if none and subj !== 'JM', accept one whose params differ only in 'passive'.
 */
export function resolve(ds: Dataset, p: Params, subj: string, dir: Dir): Resolved | null {
  const exact = ds.expIds.filter((e) => sameParams(ds.experiments[e].params, p) && hasRuns(ds, e, subj, dir));
  if (exact.length) return { exp: exact[0], viaPassive: false };
  if (subj !== 'JM') {
    const loose = ds.expIds.filter((e) => sameParams(ds.experiments[e].params, p, true) && hasRuns(ds, e, subj, dir));
    if (loose.length) return { exp: loose[0], viaPassive: true };
  }
  return null;
}

export interface ModelStatus {
  /** experiment shown for each person in scope (null = not trained) */
  perSubject: { subj: string; res: Resolved | null; runs: number }[];
  /** experiment with exactly these parameters (whether or not it has runs in view), or null = custom combination */
  exact: string | null;
  /** experiment used for the most people in view (the exact one when it has runs), or null when nothing resolves */
  primary: string | null;
  found: string[];
  missing: string[];
  /**
   * experiments other than the exact one that are shown (they differ only in the passive table, which is
   * identical for everyone except JM): [expId, subjects]. With no exact experiment, every shown experiment is listed.
   */
  substitutes: { exp: string; subjects: string[] }[];
  runsMin: number;
  runsMax: number;
  /** hint when nothing matches */
  hint: string | null;
}

export function modelStatus(ds: Dataset, p: Params, subjects: string[], dir: Dir): ModelStatus {
  const perSubject = subjects.map((s) => {
    const res = resolve(ds, p, s, dir);
    return { subj: s, res, runs: res ? seedsOf(ds, res.exp, s, dir).length : 0 };
  });
  const found = perSubject.filter((x) => x.res).map((x) => x.subj);
  const missing = perSubject.filter((x) => !x.res).map((x) => x.subj);
  const exact = exactExperiment(ds, p);
  const counts = new Map<string, number>();
  perSubject.forEach((x) => x.res && counts.set(x.res.exp, (counts.get(x.res.exp) ?? 0) + 1));
  let primary: string | null = null;
  if (exact && counts.has(exact)) primary = exact;
  else if (counts.size) primary = Array.from(counts.entries()).sort((a, b) => b[1] - a[1])[0][0];
  const subsMap = new Map<string, string[]>();
  perSubject.forEach((x) => {
    if (x.res && x.res.exp !== exact) {
      const a = subsMap.get(x.res.exp) ?? [];
      a.push(x.subj);
      subsMap.set(x.res.exp, a);
    }
  });
  const runCounts = perSubject.filter((x) => x.res).map((x) => x.runs);
  let hint: string | null = null;
  if (!found.length || missing.length) {
    if (String(p.code) === 'original' && dir === 'S2toS1')
      hint = 'The original code only ran learn 1 → test 2. Set Code to "our framework" (same data and model).';
    else if (!found.length) hint = 'Pick a preset, or change one setting at a time from a trained experiment.';
  }
  return {
    perSubject,
    exact,
    primary,
    found,
    missing,
    substitutes: Array.from(subsMap.entries()).map(([exp, subjects]) => ({ exp, subjects })),
    runsMin: runCounts.length ? Math.min(...runCounts) : 0,
    runsMax: runCounts.length ? Math.max(...runCounts) : 0,
    hint,
  };
}

/** name of the model in view: the exact experiment's friendly name, or "Custom combination" */
export function modelTitle(st: ModelStatus): string {
  if (st.exact) return friendly(st.exact);
  return st.found.length ? 'Custom combination' : 'Not trained';
}

/**
 * Codes of the experiments actually shown, e.g. "D01; C01 for YES, EG" or, for one person, "C01 for YES".
 * Empty when nothing is trained in view.
 */
export function modelCodes(st: ModelStatus): string {
  const parts: string[] = [];
  const exactUsed = st.exact ? st.perSubject.some((x) => x.res?.exp === st.exact) : false;
  if (st.exact && exactUsed) parts.push(shortId(st.exact));
  st.substitutes.forEach((x) => parts.push(`${shortId(x.exp)} for ${x.subjects.join(', ')}`));
  return parts.join('; ');
}

/** "Last semester (original code) (X00)" or "Ours: … (D01; C01 for YES, EG)" */
export function modelName(st: ModelStatus): string {
  if (!st.found.length) return 'not trained for this view';
  const codes = modelCodes(st);
  return codes ? `${modelTitle(st)} (${codes})` : modelTitle(st);
}

/** experiments that differ from `id` only in the passive table (same model for everyone except JM) */
export function passiveTwins(ds: Dataset, id: string): string[] {
  const p = ds.experiments[id]?.params;
  if (!p) return [];
  return ds.expIds.filter((e) => e !== id && sameParams(ds.experiments[e].params, p, true));
}

/** parameters that differ from last semester (X00); 'code' only if nothing else differs */
export function diffFromBaseline(ds: Dataset, p: Params): ParamKey[] {
  const base = ds.experiments[DEFAULT_A_ID]?.params;
  if (!base) return [];
  const d = PARAM_KEYS.filter((k) => k !== 'code' && String(p[k]) !== String(base[k]));
  if (!d.length && String(p.code) !== String(base.code)) return ['code'];
  return d;
}

/** parse a parameter value string from a control / URL back into the stored type */
export function coerceParam(ds: Dataset, k: ParamKey, v: string): ParamValue {
  const o = paramOptions(ds, k).find((x) => String(x) === v);
  if (o !== undefined) return o;
  const n = Number(v);
  return Number.isFinite(n) && ['window', 'dropout', 'lr'].includes(k) ? n : v;
}
