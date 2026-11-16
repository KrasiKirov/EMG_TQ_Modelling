import { useMemo, useState } from 'react';
import { Group, SegmentedControl, Select, SimpleGrid, Slider, Stack, Text, Title } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { EChart } from '../components/EChart';
import { Note, SectionCard } from '../components/common';
import { passiveOption, spectrumOption, stagesOption, trialOption, windowOption } from '../charts/pipeline';
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
  const narrow = !!useMediaQuery('(max-width: 48em)');
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
        <Text size="sm" c="dimmed">
          Every stage of the processing, on real data. The trial shown is the middle position ({p.position} from most plantarflexed,
          {' '}{p.angle.toFixed(2)} rad) of session 1; the 2 s excerpt starts just before the strongest rise in muscle activity.
          The stages reproduce the reference model's inputs exactly (checked when the data are exported).
        </Text>
        <Group gap="sm" wrap="wrap" className="study-controls">
          <Select size="xs" w={120} aria-label="Person" value={subj} onChange={(v) => v && setSubj(v)} data={PIPE_PEOPLE} />
          <SegmentedControl size="xs" aria-label="EMG channel" value={String(ch)} onChange={(v) => setCh(Number(v))}
            data={CHANNELS.map((c, i) => ({ value: String(i), label: c }))} />
        </Group>
      </div>

      <SectionCard testId="pipeline-steps" title="The steps" description="In order, from the sensor to the network.">
        <SimpleGrid cols={{ base: 1, xs: 2, md: 4 }} spacing="sm">
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
        description={<>2 s of one trial. Pinch or Ctrl-scroll on the chart to zoom in time, drag to pan; all panels move together. In panel 4 the dashed line is the zero-phase
          envelope of earlier rounds: it starts rising <i>before</i> the burst because it looks at future samples, which a live device cannot do.</>}>
        <EChart label={`Processing stages of ${subj} ${CHANNELS[ch]}`} height={820} option={stagesOption(p, ch, tokens, 820)} />
        <Note>Panel 5: the faint line is the envelope divided by the session-1 maximum ({p.trainmax[ch].toPrecision(3)} {p.raw_unit}); the dots are the
          samples the model actually reads, one every 10 ms.</Note>
      </SectionCard>

      <SectionCard testId="pipeline-spectrum" title="What the filters keep"
        description={<>Left: how the raw {CHANNELS[ch]} signal's power is spread over frequency (whole trial). Right: how much of each frequency the filters let
          through. 0 dB passes unchanged; −20 dB is 10 % of the amplitude, −40 dB 1 %.</>}>
        <EChart label={`Spectrum of raw ${CHANNELS[ch]} and filter responses`} height={narrow ? 480 : 330} option={spectrumOption(p, ch, tokens, narrow)} />
        <Note>The 2nd-order low-pass cuts less steeply than the 4th-order one but lags about half as much (0.12 s against 0.23 s), and gave lower
          validation loss for all 6 people.</Note>
      </SectionCard>

      <SectionCard testId="pipeline-window" title="One input window"
        description="What the network reads for a single prediction: 20 rows, 10 ms apart, of the 4 scaled envelopes and the ankle angle. Move the slider to slide the window through the excerpt.">
        <Slider mb="md" min={PIPE.window - 1} max={n - 1} step={1} value={endIdx} onChange={setEnd} label={() => `${endTime.toFixed(2)} s`}
          aria-label="Window end" />
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
          <EChart label="One 20 × 5 input window" height={560} option={windowOption(rows, endTime, tokens)} />
          <Stack gap="xs">
            <Text size="sm">Window ends at <b>{endTime.toFixed(2)} s</b>.</Text>
            <Text size="sm">Label (what the network must output): active torque at the last row = <b>{p.inputs.label[endIdx].toFixed(2)} Nm</b>.</Text>
            <Text size="sm" c="dimmed">The bottom row is the newest sample. The next window starts one row later, so 19 of its 20 rows are shared with this one
              (95 % overlap). The angle column barely changes: within a trial the ankle is held still.</Text>
            <Text size="sm" c="dimmed">EMG values near 1 mean activity close to that channel's largest value in session 1. Torque is negative when the person
              pushes the foot down (plantarflexion), as recorded by the dynamometer.</Text>
          </Stack>
        </SimpleGrid>
      </SectionCard>

      <WindowStructure p={p} subj={subj} t={tokens} />

      <SectionCard testId="pipeline-trial" title="The whole trial"
        description={<>All {Math.round(p.trial_view.measured.length / p.trial_view.fs)} s that remain after trimming. The shaded band is the excerpt above. The active
          torque (the label) is the measured torque minus the passive torque at this position, {p.trial_view.passive.toFixed(2)} Nm.</>}>
        <EChart label={`Whole trial of ${subj}`} height={560} option={trialOption(p, tokens, 560)} />
      </SectionCard>

      <SectionCard testId="pipeline-passive" title="Passive torque subtracted at each position"
        description="Measured from relaxed holds before each trial, separately for each session. Positive at plantarflexed angles (stretched shin tissues pull the foot up), negative at dorsiflexed angles (the stretched calf pulls it down).">
        <EChart label={`Passive torque by angle for ${subj}`} height={300} option={passiveOption(p, tokens)} />
        {subj === 'IES' && <Note>IES recorded no relaxed holds: its passive values come from the experiment metadata, one table for both sessions.</Note>}
        {subj === 'YES' && <Note>YES's session-2 values are hand-entered from the trial worksheet.</Note>}
      </SectionCard>
    </Stack>
  );
}
