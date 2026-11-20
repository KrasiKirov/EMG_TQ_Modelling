// One number per person: a score per seed, then the median over seeds and its confidence interval.
import { medianCI, type MedianCI } from './boxstats.ts';
import { mean, median } from './stats.ts';

/** a person's runs: values[position][seed index] */
export interface SeedGrid { seeds: number[]; mae: number[][]; r2: number[][] }

/** one score per seed: mean MAE over positions, median R² over positions (R² can be hugely negative at one position) */
export function seedScores(p: SeedGrid, m: 'mae' | 'r2'): number[] {
  const v = m === 'mae' ? p.mae : p.r2;
  return p.seeds.map((_, k) => {
    const xs = v.map((row) => row[k]).filter(Number.isFinite);
    return xs.length ? (m === 'mae' ? mean(xs) : median(xs)) : NaN;
  }).filter(Number.isFinite);
}

/** the person's score: median over seeds */
export const personScore = (p: SeedGrid, m: 'mae' | 'r2'): number => median(seedScores(p, m));

/** confidence interval of that median; null with fewer than 2 seeds (one run has no interval) */
export const scoreCI = (p: SeedGrid, m: 'mae' | 'r2'): MedianCI | null => {
  const v = seedScores(p, m);
  return v.length >= 2 ? medianCI(v) : null;
};

/** the worst position: per seed the lowest R² over positions, then the median over seeds */
export function worstR2(p: SeedGrid): number {
  const v = p.seeds.map((_, k) => Math.min(...p.r2.map((row) => row[k]).filter(Number.isFinite))).filter(Number.isFinite);
  return median(v);
}

/** per seed, the run's score (MAE: mean over positions; R²: median over positions) */
function bySeed(p: SeedGrid, m: 'mae' | 'r2'): Map<number, number> {
  const v = m === 'mae' ? p.mae : p.r2;
  const out = new Map<number, number>();
  p.seeds.forEach((sd, k) => {
    const xs = v.map((row) => row[k]).filter(Number.isFinite);
    if (xs.length) out.set(sd, m === 'mae' ? mean(xs) : median(xs));
  });
  return out;
}

/** MAE of `p` ÷ MAE of `ref` for the same person, each the median over the seeds both have */
export function maeRatio(p: SeedGrid, ref: SeedGrid): { ratio: number; seeds: number } {
  const a = bySeed(p, 'mae'), b = bySeed(ref, 'mae');
  const both = [...a.keys()].filter((sd) => b.has(sd));
  if (!both.length) return { ratio: NaN, seeds: 0 };
  return { ratio: median(both.map((sd) => a.get(sd)!)) / median(both.map((sd) => b.get(sd)!)), seeds: both.length };
}

/** what to call an interval over n runs: "95 % CI", or its real coverage when n is too small to reach 95 % */
export function ciName(ci: MedianCI, n: number): string {
  if (!Number.isFinite(ci.level)) return 'CI';
  return ci.level >= 0.95 ? '95 % CI' : `${Math.round(ci.level * 100)} % range (${n} runs)`;
}
