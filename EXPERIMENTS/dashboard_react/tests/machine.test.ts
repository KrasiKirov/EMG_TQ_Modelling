// Run with: npm test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { environmentOf, machineFacts } from '../src/lib/machine.ts';

const M3 = {
  computer: 'MacBook Air', processor: 'Apple M3', cores: 8, performance_cores: 4, efficiency_cores: 4, ram_gb: 16,
  os: 'macOS 27.2.1', accelerator: 'none (CPU only)', parallel_runs: 6, threads_per_run: 1,
};
const V = [{ python: '3.13.2', tensorflow: '2.21.0', numpy: '2.4.3' }];

test('the machine block names processor, cores, memory, GPU, software and sharing', () => {
  assert.deepEqual(machineFacts(M3, V), [
    ['Computer', 'MacBook Air · macOS 27.2.1'],
    ['Processor', 'Apple M3 · 8 cores (4 performance + 4 efficiency)'],
    ['Memory (RAM)', '16 GB'],
    ['GPU', 'none (CPU only)'],
    ['Software', 'Python 3.13.2 · TensorFlow 2.21.0 · numpy 2.4.3'],
    ['Sharing', '6 runs trained at once, 1 TensorFlow thread each'],
  ]);
});

test('several software versions are all listed, and sharing can be left out', () => {
  const f = machineFacts(M3, [...V, { python: '3.11.1', tensorflow: '2.21.0', numpy: '2.4.4' }], { sharing: false });
  assert.equal(f.find(([k]) => k === 'Software')?.[1], 'Python 3.13.2 · TensorFlow 2.21.0 · numpy 2.4.3; Python 3.11.1 · TensorFlow 2.21.0 · numpy 2.4.4');
  assert.equal(f.some(([k]) => k === 'Sharing'), false);
});

test('an experiment shares the study machine only if every run reported the study software', () => {
  const old = { python: '3.11.1', tensorflow: '2.21.0', numpy: '2.4.4' };
  assert.equal(environmentOf(V, V), 'same');
  assert.equal(environmentOf([old], V), 'different');
  assert.equal(environmentOf([...V, old], V), 'different');
  assert.equal(environmentOf([], V), 'unknown');
  assert.equal(environmentOf(undefined, V), 'unknown');
});

test('one run at a time reads in the singular', () => {
  const f = machineFacts({ ...M3, parallel_runs: 1 }, V);
  assert.equal(f.at(-1)?.[1], '1 run trained at once, 1 TensorFlow thread each');
});
