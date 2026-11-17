import { Text } from '@mantine/core';
import { STUDY } from '../lib/study';
import { EXTRA } from '../lib/catalog';
import { environmentOf, machineFacts, type Versions } from '../lib/machine';

/**
 * The computer the window study trained on: processor, cores, memory, GPU, software and how runs shared the CPU.
 * recorded: the software this experiment's runs reported (default: the window study's)
 */
export function MachineInfo({ sharing = true, recorded }: { sharing?: boolean; recorded?: Versions[] }) {
  const m = STUDY.machine;
  if (!m) return null;
  return (
    <div data-testid="machine-info" className="machine-info">
      <Text size="xs" fw={600} mb={4}>Computer the runs trained on</Text>
      <div className="settings-list">
        {machineFacts(m, recorded ?? STUDY.versions ?? [], { sharing }).map(([k, v]) => <div key={k}><span>{k}</span><b>{v}</b></div>)}
      </div>
      <Text size="xs" c="dimmed" mt={6}>
        {sharing ? '' : 'CPU sharing not recorded. '}Fanless laptop, CPU only: times are descriptive, not a benchmark.
      </Text>
    </div>
  );
}

/** The computer an experiment's runs trained on, when their recorded software shows they ran where the window study did. */
export function ExperimentMachine({id}:{id:string}){
  const folder=EXTRA[id]?.native??id;const recorded=STUDY.run_versions?.[folder];const env=environmentOf(recorded,STUDY.versions??[]);
  if(env==='same')return <MachineInfo sharing={!!STUDY.settings[folder]} recorded={recorded}/>;
  const sw=(recorded??[]).map(v=>`Python ${v.python} · TensorFlow ${v.tensorflow} · numpy ${v.numpy}`).join('; ');
  return <Text size="xs" c="dimmed" mt="sm" data-testid="timing-machine-note">Computer: not recorded{env==='different'?` (software: ${sw}).`:'.'}</Text>;
}

const folderOf = (id: string) => EXTRA[id]?.native ?? id;
const brief = () => {
  const x = STUDY.machine;
  return x ? `${x.processor} (${x.cores} cores: ${x.performance_cores} performance + ${x.efficiency_cores} efficiency), ${x.ram_gb} GB RAM, ${x.accelerator.startsWith('none') ? 'CPU only' : x.accelerator}` : '';
};

/** one line per compared experiment: the computer its training times come from, or why it is unknown */
export function MachineLine({ id, side }: { id: string; side: string }) {
  const recorded = STUDY.run_versions?.[folderOf(id)];
  const env = environmentOf(recorded, STUDY.versions ?? []);
  const sw = (recorded ?? []).map((v) => `Python ${v.python}, TensorFlow ${v.tensorflow}, numpy ${v.numpy}`).join('; ');
  return (
    <Text size="xs" c="dimmed" data-testid={`machine-line-${side}`}>
      <b>{side}</b> · {env === 'same' ? `${STUDY.machine?.computer ?? 'Mac'}: ${brief()}; ${sw}.`
        : env === 'different' ? `computer not recorded; its runs report ${sw}, a different environment from the window study.`
          : 'computer and software not recorded for these runs.'}
    </Text>
  );
}
