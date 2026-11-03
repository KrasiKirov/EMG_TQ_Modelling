import { catalogRuns } from './catalog';
import { DS } from '../dataset';
import { median } from './stats';
import type { Dir } from './types';

/**
 * The dashboard's summary rule for one experiment page: median over (person, angle) cells of the median
 * over seeds, R² below 0 shown as 0 when `floor` is on. toes = 1: each person's two most toes-down angles.
 */
export function pageScore(id: string, toes: 0 | 1, key: 'mae' | 'r2' | 'r2_demeaned', floor: boolean, dir: Dir = 'S1toS2'): number {
  const cells: number[] = [];
  for (const subj of DS.subjects) {
    const rows = catalogRuns(id, subj, dir).filter((r) => r.toes === toes);
    for (const pos of [...new Set(rows.map((r) => r.pos))]) {
      const v = median(rows.filter((r) => r.pos === pos).map((r) => r[key]));
      cells.push(key !== 'mae' && floor ? Math.max(v, 0) : v);
    }
  }
  return median(cells);
}

/** people with results and the largest number of seeds any of them has */
export function coverage(id: string, dir: Dir = 'S1toS2'): { people: number; seeds: number } {
  let people = 0, seeds = 0;
  for (const subj of DS.subjects) {
    const rows = catalogRuns(id, subj, dir);
    if (!rows.length) continue;
    people++;
    seeds = Math.max(seeds, new Set(rows.map((r) => r.seed)).size);
  }
  return { people, seeds };
}
