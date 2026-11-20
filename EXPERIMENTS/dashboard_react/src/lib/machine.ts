// The computer the window study trained on (written by EXPERIMENTS/analysis/study_scores.py into study.json).
// The run files record software versions; the hardware is read on the same Mac when the scores are exported.
export interface Machine {
  computer: string | null; processor: string | null; cores: number; performance_cores: number; efficiency_cores: number;
  ram_gb: number; os: string; accelerator: string; parallel_runs: number; threads_per_run: number;
}
export type Versions = Record<string, string>;

const NAMES: Record<string, string> = { python: 'Python', tensorflow: 'TensorFlow', numpy: 'numpy' };

const same = (a: Versions, b: Versions) => ['python', 'tensorflow', 'numpy'].every((k) => a[k] === b[k]);

/**
 * Whether an experiment's runs trained in the window study's environment: 'same' only if every run reported software
 * the study's runs also reported; 'unknown' if its runs saved no versions.
 */
export function environmentOf(recorded: Versions[] | undefined, study: Versions[]): 'same' | 'different' | 'unknown' {
  if (!recorded?.length) return 'unknown';
  return recorded.every((v) => study.some((s) => same(v, s))) ? 'same' : 'different';
}

/** [label, value] rows for the machine block; `sharing: false` leaves out how many runs shared the CPU */
export function machineFacts(m: Machine, versions: Versions[], opts: { sharing?: boolean } = {}): [string, string][] {
  const sw = versions
    .map((v) => ['python', 'tensorflow', 'numpy'].filter((k) => v[k]).map((k) => `${NAMES[k]} ${v[k]}`).join(' · '))
    .join('; ');
  const rows: [string, string][] = [
    ['Computer', [m.computer, m.os].filter(Boolean).join(' · ')],
    ['Processor', `${m.processor ?? 'unknown'} · ${m.cores} cores (${m.performance_cores} performance + ${m.efficiency_cores} efficiency)`],
    ['Memory (RAM)', `${m.ram_gb} GB`],
    ['GPU', m.accelerator],
    ['Software', sw],
  ];
  if (opts.sharing !== false) {
    rows.push(['Sharing', `${m.parallel_runs} run${m.parallel_runs === 1 ? '' : 's'} at once, ${m.threads_per_run} TensorFlow thread${m.threads_per_run === 1 ? '' : 's'} each (the run script's settings when this page was built; the runs do not record it)`]);
  }
  return rows;
}
