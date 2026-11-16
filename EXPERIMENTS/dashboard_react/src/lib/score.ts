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

/** confidence interval of that median */
export const scoreCI = (p: SeedGrid, m: 'mae' | 'r2'): MedianCI => medianCI(seedScores(p, m));

/** what to call an interval over n runs: "95 % CI", or its real coverage when n is too small to reach 95 % */
export function ciName(ci: MedianCI, n: number): string {
  if (!Number.isFinite(ci.level)) return 'CI';
  return ci.level >= 0.95 ? '95 % CI' : `${Math.round(ci.level * 100)} % range (${n} runs)`;
}
