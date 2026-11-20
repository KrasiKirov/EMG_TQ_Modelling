// Measured torque against the best and the worst run at each person's worst positions (written by
// EXPERIMENTS/analysis/study_waves.py). Each curve keeps real 10 ms samples only: per 0.2 s slot, the lowest and the
// highest sample at their own times. Values: int16 in 0.01 Nm (`missing` = empty slot); times: uint8 offsets in
// 10 ms steps from the slot's start. Two points per slot, one block of 2 × slots points per worst position.
import raw from '../data/study-waves.json';

export interface RunScore { seed: number; mae: number; r2: number }
export interface WorstPosition { k: number; pos: number; median_r2: number; best: RunScore; worst: RunScore }
type Curve = 'measured' | 'best' | 'worst';
type PersonWaves = Record<Curve | `${Curve}_t`, string> & { seeds: number; positions: WorstPosition[] };
interface WaveData { slots: [number, number, number]; step: number; scale: number; missing: number; runs: Record<string, Record<string, PersonWaves>> }
const W = raw as unknown as WaveData;
const [T0, WIDTH, SLOTS] = W.slots;
const PTS = 2 * SLOTS;

/** first and last time the curves can cover (s) */
export const WAVE_SPAN: [number, number] = [T0, T0 + SLOTS * WIDTH];
export const worstPositions = (exp: string, subj: string): WorstPosition[] => W.runs[exp]?.[subj]?.positions ?? [];
export const seedCount = (exp: string, subj: string): number => W.runs[exp]?.[subj]?.seeds ?? 0;

const bytes = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
const cache = new Map<string, { t: Float64Array; v: Float32Array }>();
function decode(key: string, vals: string, offs: string) {
  let c = cache.get(key);
  if (c) return c;
  const vb = bytes(vals), ob = bytes(offs);
  const v = new Float32Array(vb.length >> 1), t = new Float64Array(ob.length);
  for (let i = 0; i < v.length; i++) {
    let x = vb[2 * i] | (vb[2 * i + 1] << 8);
    if (x >= 0x8000) x -= 0x10000;
    v[i] = x === W.missing ? NaN : x * W.scale;
    const slot = Math.floor((i % PTS) / 2);
    t[i] = Math.round((T0 + slot * WIDTH + ob[i] * W.step) * 100) / 100;
  }
  c = { t, v };
  cache.set(key, c);
  return c;
}

/** measured, best-run and worst-run torque at the i-th worst position of a person: times (s) and values (Nm) */
export function worstWaves(exp: string, subj: string, i: number): Record<Curve, { t: Float64Array; v: Float32Array }> | null {
  const r = W.runs[exp]?.[subj];
  if (!r || i >= r.positions.length) return null;
  const part = (name: Curve) => {
    const c = decode(`${name}:${exp}|${subj}`, r[name] as string, r[`${name}_t`] as string);
    return (i + 1) * PTS <= c.v.length ? { t: c.t.subarray(i * PTS, (i + 1) * PTS), v: c.v.subarray(i * PTS, (i + 1) * PTS) } : null;
  };
  const measured = part('measured'), best = part('best'), worst = part('worst');
  return measured && best && worst ? { measured, best, worst } : null;
}
