// The computer study v2 trained on (written by EXPERIMENTS/analysis/study_v2_export.py into study_v2.json).
// The run files record software versions and threads; the hardware is read on the same Mac when the data is exported.
export interface Machine {
  computer: string | null; processor: string | null; cores: number; performance_cores: number; efficiency_cores: number;
  ram_gb: number; os: string; accelerator: string; parallel_runs: number; threads_per_run: number;
}
export type Versions = Record<string, string>;

const NAMES: Record<string, string> = { python: 'Python', tensorflow: 'TensorFlow', numpy: 'numpy' };

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
    rows.push(['Sharing', `${m.parallel_runs} run${m.parallel_runs === 1 ? '' : 's'} at once, ${m.threads_per_run} TensorFlow thread${m.threads_per_run === 1 ? '' : 's'} each (the protocol's worker count; each run records its threads)`]);
  }
  return rows;
}
