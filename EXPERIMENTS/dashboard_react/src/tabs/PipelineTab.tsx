import { useMemo, useState } from 'react';
import { Group, SegmentedControl, Select, SimpleGrid, Slider, Stack, Text, Title } from '@mantine/core';
import { EChart } from '../components/EChart';
import { Note, SectionCard } from '../components/common';
import { filterOption, passiveOption, spectrumOption, STAGE_TITLES, stageOption, trialPanelOption, windowOption } from '../charts/pipeline';
import { CHANNELS, PIPE, PIPE_PEOPLE, windowAt } from '../lib/pipeline';
import { useView } from '../state';
import { WindowStructure } from './PipelineWindows';

const STEPS: [string, string][] = [
  ['Raw EMG', '1000 samples per second from 4 muscles: GM, GL, SOL (calf) and TA (shin).'],
  ['Remove offset', 'Subtract the mean of the first second, so the signal sits around zero.'],
  ['Band-pass 30–300 Hz', '4th-order Butterworth, causal. Removes movement artefacts and drift below 30 Hz and noise above 300 Hz.'],
  ['Rectify', 'Absolute value: every swing counts as positive activity.'],
  ['Low-pass 2 Hz', '2nd-order Butterworth, causal (uses only past samples). The smooth result is the envelope; it lags by about 0.12 s.'],
  ['Trim and down-sample', 'Drop the first and last 1 s of the trial, then keep every 10th sample: 100 Hz.'],
  ['Scale', 'Divide each channel by its maximum over session 1. Session 2 uses the same divisor.'],
  ['Window', '20 steps (200 ms) of the 4 envelopes plus the ankle angle; the label is the active torque at the last step.'],
];

export function PipelineTab() {
  const { tokens } = useView();
  const [subj, setSubj] = useState(PIPE_PEOPLE.includes('JM') ? 'JM' : PIPE_PEOPLE[0]);
  const [ch, setCh] = useState(0);
  const p = PIPE.people[subj];
  const n = p.inputs.emg[0].length;
  const [end, setEnd] = useState(Math.min(n - 1, 100));
  const endIdx = Math.max(PIPE.window - 1, Math.min(n - 1, end));
  const rows = useMemo(() => windowAt(p, endIdx), [p, endIdx]);
  const endTime = p.inputs.t0 + endIdx / p.inputs.fs;

  return (
    <Stack gap="lg" data-testid="pipeline-page">
      <div className="page-heading">
        <Text size="sm" c="dimmed">Reference setting · one session-1 trial per person · built {PIPE.built}</Text>
        <Title order={2}>From raw EMG to model input</Title>
        <Text size="sm" c="dimmed">Session 1, middle position ({p.angle.toFixed(2)} rad). Matches the reference model's inputs exactly.</Text>
        <Group gap="sm" wrap="wrap" className="study-controls">
          <Select size="xs" w={120} aria-label="Person" value={subj} onChange={(v) => v && setSubj(v)} data={PIPE_PEOPLE} />
          <SegmentedControl size="xs" aria-label="EMG channel" value={String(ch)} onChange={(v) => setCh(Number(v))}
            data={CHANNELS.map((c, i) => ({ value: String(i), label: c }))} />
        </Group>
      </div>

      <SectionCard testId="pipeline-steps" title="The steps" description="Sensor to network.">
        <SimpleGrid cols={4} spacing="sm">
          {STEPS.map(([k, v], i) => (
            <div key={k} className="pipe-step">
              <Text size="xs" c="dimmed" fw={600}>{String(i + 1).padStart(2, '0')}</Text>
              <Text size="sm" fw={600}>{k}</Text>
              <Text size="xs" c="dimmed">{v}</Text>
            </div>
          ))}
        </SimpleGrid>
      </SectionCard>

      <SectionCard testId="pipeline-stages" title={`Each stage on ${subj}'s ${CHANNELS[ch]}`}
        description="The same 2 s excerpt at each stage.">
        <Stack gap="md">
          {STAGE_TITLES.map((title, i) => (
            <div key={title}>
              <Text size="sm" fw={600}>{title}</Text>
              <EChart label={`${subj} ${CHANNELS[ch]}: ${title}`} height={170} option={stageOption(p, ch, tokens, i)} />
            </div>
          ))}
        </Stack>
      </SectionCard>

      <SectionCard testId="pipeline-spectrum" title="What the filters keep"
        description="−20 dB = 10 % of the amplitude.">
        <SimpleGrid cols={3} spacing="lg">
          <div><Text size="sm" fw={600}>Raw {CHANNELS[ch]} power spectrum</Text>
            <EChart label={`Spectrum of raw ${CHANNELS[ch]}`} height={260} option={spectrumOption(p, ch, tokens)} /></div>
          <div><Text size="sm" fw={600}>Band-pass 30–300 Hz</Text>
            <EChart label="Band-pass filter response" height={260} option={filterOption('band', tokens)} /></div>
          <div><Text size="sm" fw={600}>Low-pass 2 Hz, 2nd order</Text>
            <EChart label="Low-pass filter response" height={260} option={filterOption('lp', tokens)} /></div>
        </SimpleGrid>
        <Note>2nd- vs 4th-order low-pass: lag 0.12 s vs 0.23 s; lower validation loss for all 6 people.</Note>
      </SectionCard>

      <SectionCard testId="pipeline-window" title="One input window"
        description="20 rows × 10 ms: 4 scaled envelopes and the ankle angle.">
        <Slider mb="md" min={PIPE.window - 1} max={n - 1} step={1} value={endIdx} onChange={setEnd} label={() => `${endTime.toFixed(2)} s`}
          aria-label="Window end" />
        <SimpleGrid cols={2} spacing="lg">
          <EChart label="One 20 × 5 input window" height={560} option={windowOption(rows, endTime, tokens)} />
          <Stack gap="xs">
            <Text size="sm">Window ends at <b>{endTime.toFixed(2)} s</b>.</Text>
            <Text size="sm">Label (what the network must output): active torque at the last row = <b>{p.inputs.label[endIdx].toFixed(2)} Nm</b>.</Text>
            <Text size="sm" c="dimmed">Bottom row = newest sample; the next window shares 19 of 20 rows.</Text>
            <Text size="sm" c="dimmed">EMG 1 = channel's session-1 maximum. Negative torque = plantarflexion.</Text>
          </Stack>
        </SimpleGrid>
      </SectionCard>

      <WindowStructure p={p} subj={subj} t={tokens} />

      <SectionCard testId="pipeline-trial" title="The whole trial"
        description={`${Math.round(p.trial_view.measured.length / p.trial_view.fs)} s after trimming · shaded: the excerpt above`}>
        <Stack gap="md">
          <div><Text size="sm" fw={600}>{CHANNELS[ch]} model input</Text>
            <EChart label={`Whole trial of ${subj}: ${CHANNELS[ch]} input`} height={180} option={trialPanelOption(p, tokens, 'emg', ch)} /></div>
          <div><Text size="sm" fw={600}>Active torque (the label) = measured − passive ({p.trial_view.passive.toFixed(2)} Nm)</Text>
            <EChart label={`Whole trial of ${subj}: active torque`} height={180} option={trialPanelOption(p, tokens, 'torque')} /></div>
          <div><Text size="sm" fw={600}>Ankle angle</Text>
            <EChart label={`Whole trial of ${subj}: ankle angle`} height={180} option={trialPanelOption(p, tokens, 'angle')} /></div>
        </Stack>
      </SectionCard>

      <SectionCard testId="pipeline-passive" title="Passive torque subtracted at each position"
        description="Relaxed holds, per session.">
        <EChart label={`Passive torque by angle for ${subj}`} height={300} option={passiveOption(p, tokens)} />
        {subj === 'IES' && <Note>IES: no relaxed holds; values from the experiment metadata.</Note>}
        {subj === 'YES' && <Note>YES session 2: hand-entered from the trial worksheet.</Note>}
      </SectionCard>
    </Stack>
  );
}
