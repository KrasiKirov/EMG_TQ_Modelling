import { Text } from '@mantine/core';
import { MACHINE, VERSIONS } from '../lib/v2';
import { machineFacts } from '../lib/machine';

/** The computer study v2's runs trained on: processor, cores, memory, GPU, software and how runs shared the CPU. */
export function MachineInfo() {
  if (!MACHINE) return <Text size="xs" c="dimmed">Computer: not recorded.</Text>;
  return (
    <div data-testid="machine-info" className="machine-info">
      <Text size="xs" fw={600} mb={4}>Computer the runs trained on</Text>
      <div className="settings-list">
        {machineFacts(MACHINE, VERSIONS).map(([k, v]) => <div key={k}><span>{k}</span><b>{v}</b></div>)}
      </div>
      <Text size="xs" c="dimmed" mt={6}>Fanless laptop, CPU only: times are descriptive, not a benchmark.</Text>
    </div>
  );
}
