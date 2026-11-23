// One number per person: a score per seed, then the median over seeds and its confidence interval.
import { medianCI, type MedianCI } from './boxstats.ts';
import { mean, median } from './stats.ts';

/** a person's runs: values[position][seed index] */
interface SeedGrid { seeds: number[]; mae: number[][]; nmae?: number[][]; r2: number[][] }
type ScoreMetric = 'mae' | 'nmae' | 'r2';
const gridOf = (p: SeedGrid, m: ScoreMetric): number[][] => (m === 'nmae' ? p.nmae ?? [] : m === 'mae' ? p.mae : p.r2);

/** one score per seed: MAE and nMAE the mean over positions (the pre-registered nMAE), R² the median over positions
 * (R² can be hugely negative at one position) */
function seedScores(p: SeedGrid, m: ScoreMetric): number[] {
  const v = gridOf(p, m);
  return p.seeds.map((_, k) => {
    const xs = v.map((row) => row[k]).filter(Number.isFinite);
    return xs.length ? (m === 'r2' ? median(xs) : mean(xs)) : NaN;
  }).filter(Number.isFinite);
}

/** the person's score: median over seeds */
export const personScore = (p: SeedGrid, m: ScoreMetric): number => median(seedScores(p, m));

/** confidence interval of that median; null with fewer than 2 seeds (one run has no interval) */
export const scoreCI = (p: SeedGrid, m: ScoreMetric): MedianCI | null => {
  const v = seedScores(p, m);
  return v.length >= 2 ? medianCI(v) : null;
};

/** the worst position: per seed the lowest R² over positions, then the median over seeds */
export function worstR2(p: SeedGrid): number {
  const v = p.seeds.map((_, k) => Math.min(...p.r2.map((row) => row[k]).filter(Number.isFinite))).filter(Number.isFinite);
  return median(v);
}
