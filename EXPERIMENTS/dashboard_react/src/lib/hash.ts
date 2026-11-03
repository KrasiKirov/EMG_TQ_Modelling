// URL hash <-> view state, e.g. #tab=waveforms&subj=JM&dir=S1toS2&metric=mae&floor=1&a=X00_baseline&b=D01_...&pos=-0.787&theme=dark
import { ALL_EXPERIMENTS } from './catalog';
import type { Dataset } from './data';
import { METRIC_KEYS } from './metrics';
import { coerceParam, exactExperiment, PARAM_KEYS } from './params';
import type { Dir, MetricKey, Params } from './types';

export const TABS = ['home', 'overview', 'comparison', 'waveforms', 'training', 'experiments', 'about'] as const;
export type TabKey = (typeof TABS)[number];

export interface ViewState {
  tab: TabKey;
  experiment: string;
  compareA: string;
  compareB: string;
  subj: string; // 'All' or a person
  dir: Dir;
  metric: MetricKey;
  floor: boolean;
  A: Params;
  B: Params;
  /** person shown in the waveform viewer */
  wsubj: string;
  /** position (rad) shown in the waveform viewer; null = first */
  pos: number | null;
  runA: string;
  runB: string;
}

/** split "a=1&b=2" without treating '+' as a space (experiment ids contain '+') */
export function parseHashParams(hash: string): Record<string, string> {
  const out: Record<string, string> = {};
  const h = hash.replace(/^#/, '');
  if (!h) return out;
  for (const part of h.split('&')) {
    if (!part) continue;
    const i = part.indexOf('=');
    const k = i < 0 ? part : part.slice(0, i);
    const v = i < 0 ? '' : part.slice(i + 1);
    try {
      out[decodeURIComponent(k)] = decodeURIComponent(v);
    } catch {
      out[k] = v;
    }
  }
  return out;
}

/** short names in the URL for long experiment ids */
const ALIASES: Record<string, string> = {
  'D01_cleanpassive+autogain+equalweight': 'D01',
  'C01_autogain+equalweight': 'C01',
};
const UNALIAS = Object.fromEntries(Object.entries(ALIASES).map(([id, a]) => [a, id]));
const alias = (id: string) => ALIASES[id] ?? id;
const unalias = (s: string | undefined) => (s === undefined ? s : UNALIAS[s] ?? s);

export function encodeParams(ds: Dataset, p: Params): string {
  const id = exactExperiment(ds, p);
  return id ? alias(id) : 'p:' + PARAM_KEYS.map((k) => String(p[k])).join(',');
}

export function decodeParams(ds: Dataset, s: string | undefined): Params | null {
  s = unalias(s);
  if (!s) return null;
  if (ds.experiments[s]) return { ...ds.experiments[s].params };
  if (s.startsWith('p:')) {
    const v = s.slice(2).split(',');
    if (v.length !== PARAM_KEYS.length) return null;
    const p = {} as Params;
    PARAM_KEYS.forEach((k, i) => (p[k] = coerceParam(ds, k, v[i])));
    return p;
  }
  // tolerate ids typed with spaces where '+' was meant
  const fixed = s.replace(/ /g, '+');
  if (ds.experiments[fixed]) return { ...ds.experiments[fixed].params };
  return null;
}

export function stateFromHash(ds: Dataset, hash: string, base: ViewState): ViewState {
  const q = parseHashParams(hash);
  (['exp', 'ca', 'cb'] as const).forEach((k) => q[k] !== undefined && (q[k] = unalias(q[k]) as string));
  const s: ViewState = { ...base };
  if (q.tab && (TABS as readonly string[]).includes(q.tab)) s.tab = q.tab as TabKey;
  if (q.exp && ALL_EXPERIMENTS[q.exp]) s.experiment = q.exp;
  if(q.ca&&ALL_EXPERIMENTS[q.ca])s.compareA=q.ca;
  if(q.cb&&ALL_EXPERIMENTS[q.cb])s.compareB=q.cb;
  if (q.subj && (q.subj === 'All' || ds.subjects.includes(q.subj))) s.subj = q.subj;
  if (q.dir === 'S1toS2' || q.dir === 'S2toS1') s.dir = q.dir;
  if (q.metric && (METRIC_KEYS as string[]).includes(q.metric)) s.metric = q.metric as MetricKey;
  if (q.floor !== undefined) s.floor = q.floor !== '0' && q.floor !== 'false';
  const a = decodeParams(ds, q.a);
  if (a) s.A = a;
  const b = decodeParams(ds, q.b);
  if (b) s.B = b;
  if (q.wsubj && ds.subjects.includes(q.wsubj)) s.wsubj = q.wsubj;
  else if (s.subj !== 'All') s.wsubj = s.subj;
  if (q.pos !== undefined && q.pos !== '' && Number.isFinite(Number(q.pos))) s.pos = Number(q.pos);
  if (q.ra) s.runA = q.ra;
  if (q.rb) s.runB = q.rb;
  return s;
}

export function hashFromState(ds: Dataset, s: ViewState, theme: string | null): string {
  const parts: [string, string][] = [
    ['tab', s.tab],
    ['exp', alias(s.experiment)],
    ['ca', alias(s.compareA)],
    ['cb', alias(s.compareB)],
    ['subj', s.subj],
    ['dir', s.dir],
    ['metric', s.metric],
    ['floor', s.floor ? '1' : '0'],
    ['a', encodeParams(ds, s.A)],
    ['b', encodeParams(ds, s.B)],
  ];
  if (s.subj === 'All') parts.push(['wsubj', s.wsubj]);
  if (s.pos !== null) parts.push(['pos', String(s.pos)]);
  if (s.runA !== 'typical') parts.push(['ra', s.runA]);
  if (s.runB !== 'typical') parts.push(['rb', s.runB]);
  if (theme) parts.push(['theme', theme]);
  return '#' + parts.map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
}

export function themeFromHash(hash: string): 'light' | 'dark' | null {
  const t = parseHashParams(hash).theme;
  return t === 'dark' || t === 'light' ? t : null;
}
