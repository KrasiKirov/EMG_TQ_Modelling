// Waveforms: base64 -> little-endian int16 -> /100 = Nm; 8 blocks of n samples (one per position). Decoded lazily, cached.
import { positionIndex, type Dataset } from './data';
import type { Dir, Run } from './types';

const cache = new Map<string, Float32Array>();

export function decodeInt16(b64: string): Float32Array {
  const bin = atob(b64);
  const n = bin.length >> 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const lo = bin.charCodeAt(2 * i);
    const hi = bin.charCodeAt(2 * i + 1);
    let v = lo | (hi << 8);
    if (v >= 0x8000) v -= 0x10000;
    out[i] = v / 100;
  }
  return out;
}

function decodeCached(cacheKey: string, b64: string | undefined): Float32Array | null {
  if (!b64) return null;
  let a = cache.get(cacheKey);
  if (!a) {
    a = decodeInt16(b64);
    cache.set(cacheKey, a);
  }
  return a;
}

function block(a: Float32Array | null, k: number, n: number): Float32Array | null {
  if (!a || k < 0 || (k + 1) * n > a.length) return null;
  return a.subarray(k * n, (k + 1) * n);
}

/** measured voluntary torque (answer key) at position pos */
export function keyWave(ds: Dataset, subj: string, dir: Dir, pos: number): Float32Array | null {
  const k = positionIndex(ds, subj, dir, pos);
  const id = `${subj}|${dir}`;
  return block(decodeCached('key:' + id, ds.raw.key[id]), k, ds.grid.n);
}

/** one run's predicted torque at position pos */
export function runWave(ds: Dataset, exp: string, subj: string, dir: Dir, seed: number, pos: number): Float32Array | null {
  const k = positionIndex(ds, subj, dir, pos);
  const id = `${exp}|${subj}|${dir}|${seed}`;
  return block(decodeCached('wave:' + id, ds.raw.wave[id]), k, ds.grid.n);
}

export function timeAxis(ds: Dataset): number[] {
  const { t0, dt, n } = ds.grid;
  return Array.from({ length: n }, (_, i) => Math.round((t0 + i * dt) * 100) / 100);
}

export type RunChoice = 'typical' | 'best' | 'worst' | `s${number}`;

/** runs at one position sorted by raw R² (ascending) */
export function sortedByR2(runs: Run[]): Run[] {
  return [...runs].sort((a, b) => a.r2 - b.r2 || a.seed - b.seed);
}

/** typical = median-R² run (index floor((n-1)/2) of runs sorted by raw R²), best, worst, or a seed */
export function pickRun(runs: Run[], choice: string): Run | null {
  if (!runs.length) return null;
  const s = sortedByR2(runs);
  if (choice === 'best') return s[s.length - 1];
  if (choice === 'worst') return s[0];
  if (choice.startsWith('s')) {
    const seed = Number(choice.slice(1));
    const r = s.find((x) => x.seed === seed);
    if (r) return r;
  }
  return s[Math.floor((s.length - 1) / 2)];
}
