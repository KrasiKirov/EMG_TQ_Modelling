// Run with: npm test  (node's own test runner; node 23 strips the TypeScript types)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boxStats, medianCI } from '../src/lib/boxstats.ts';

test('box and median follow numpy linear percentiles', () => {
  const b = boxStats([1, 2, 3, 4, 5]);
  assert.deepEqual([b.q1, b.median, b.q3], [2, 3, 4]);
});

test('whiskers stop at the last run within 1.5 IQR, the rest are outliers', () => {
  // 10 seeds: q1 1.0225, q3 1.0675, IQR 0.045, upper fence 1.135
  const b = boxStats([1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.6]);
  assert.equal(b.hi, 1.08);
  assert.equal(b.lo, 1.0);
  assert.deepEqual(b.outliers, [1.6]);
});

test('without outliers the whiskers reach the smallest and largest run', () => {
  const b = boxStats([3, 1, 2]);
  assert.deepEqual([b.lo, b.hi, b.outliers.length], [1, 3, 0]);
});

test('10 seeds: the median CI runs from the 2nd to the 9th smallest run, 97.9 % coverage', () => {
  const ci = medianCI([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
  assert.deepEqual([ci.lo, ci.hi], [2, 9]);
  assert.equal(Math.round(ci.level * 1000) / 10, 97.9);
});

test('5 seeds: no narrower interval reaches 95 %, so the CI is the full range at 93.8 %', () => {
  const ci = medianCI([5, 4, 3, 2, 1]);
  assert.deepEqual([ci.lo, ci.hi], [1, 5]);
  assert.equal(Math.round(ci.level * 1000) / 10, 93.8);
});

test('non-finite values are ignored; empty input gives NaN', () => {
  assert.equal(boxStats([NaN, 2, 2]).n, 2);
  assert.ok(Number.isNaN(boxStats([]).median));
});
