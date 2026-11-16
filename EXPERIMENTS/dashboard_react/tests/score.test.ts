// Run with: npm test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ciName, personScore, scoreCI, seedScores } from '../src/lib/score.ts';
import { medianCI } from '../src/lib/boxstats.ts';

// 2 positions × 3 seeds
const p = { seeds: [0, 1, 2], mae: [[1, 2, 3], [3, 4, NaN]], r2: [[0.9, 0.8, 0.7], [0.5, 0.6, 0.4]] };

test('seed scores: MAE is the mean over positions, R² the median, per seed', () => {
  assert.deepEqual(seedScores(p, 'mae'), [2, 3, 3]);
  assert.deepEqual(seedScores(p, 'r2').map((v) => +v.toFixed(3)), [0.7, 0.7, 0.55]);
});

test('person score is the median of the seed scores', () => {
  assert.equal(personScore(p, 'mae'), 3);
});

test('score CI is the median CI over the seed scores', () => {
  assert.deepEqual(scoreCI(p, 'mae'), medianCI([2, 3, 3]));
});

test('a CI that reaches 95 % is named as such; a smaller one gives its real level and the run count', () => {
  assert.equal(ciName(medianCI([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 10), '95 % CI');
  assert.equal(ciName(medianCI([1, 2, 3]), 3), '75 % range (3 runs)');
});
