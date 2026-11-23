import { useMemo, type ReactNode } from 'react';
import { Accordion, List, SimpleGrid, Stack, Text } from '@mantine/core';
import { EChart } from '../components/EChart';
import { SectionCard } from '../components/common';
import { Takeaway } from '../components/Takeaway';
import { factorOption, overfitOption, sizeOption } from '../charts/tuning';
import { cellOf, isWinner, kindOf, overfitPoints, personScore, runsOf, tuningRuns } from '../lib/results';
import { overfitTakeaway, tuningTakeaway } from '../lib/takeaways';
import { useView } from '../state';

/** session-2 nMAE of a setting: exp of the mean over people of log (median over seeds) */
function combinedTest(id: string): number {
  const v = runsOf(id).map((p) => personScore(p, 'nmae')).filter((x) => Number.isFinite(x) && x > 0);
  return v.length ? Math.exp(v.reduce((a, x) => a + Math.log(x), 0) / v.length) : NaN;
}

/** how the networks were tuned, what the search found, and this setting's own search (details) */
export function TuningSection({ id, details }: { id: string; details: ReactNode }) {
  const { tokens } = useView();
  const runs = useMemo(() => tuningRuns(), []);
  const cell = cellOf(id);
  return (
    <SectionCard testId="study-tuning" title="Tuning experiment: validation (session 1)"
      description={`${runs.length} network configurations were trained and scored on session-1 validation only (the last 20 % of each session-1 trial, after a gap). Session 2 was not used for any choice. Score: the mean over people of log nMAE, shown on the nMAE scale; lower is better.`}>
      <Stack gap="lg">
        <div>
          <Text size="sm" fw={700} mb={4}>How it worked, for each window / memory cell</Text>
          <List type="ordered" size="sm" spacing={2}>
            <List.Item><b>Coarse:</b> the same 6 random configurations in every cell, plus the anchor (last semester's 8×2 network). One seed.</List.Item>
            <List.Item><b>Fine:</b> 4 configurations around the cell's best coarse one (learning rate up to ×/÷3, dropout ±0.1, units one step either way, same layers).</List.Item>
            <List.Item><b>Seed check:</b> the best 2 get a second seed; the winner is the one with the fewest weights within one standard error of the better one.</List.Item>
            <List.Item><b>A0:</b> the cell winner with the lowest validation error on the samples every cell predicts.</List.Item>
          </List>
          <Text size="sm" mt={6}>
            <b>Searched:</b> learning rate 0.0003–0.03 (log scale), dropout 0–0.3, units 8 / 16 / 32 / 64, layers 1 / 2, batch 32 / 64 / 128 (reset windows).{' '}
            <b>Fixed, not tuned:</b> Nadam optimizer, squared-error loss, up to 400 epochs, early stopping after 15 epochs without improvement, learning rate
            halved after 6, gradient clipping 1.0.
          </Text>
        </div>
        <div>
          <Text size="md" fw={600} mb={4}>Every tuning run by network size</Text>
          <EChart label="Validation nMAE of every tuning run by network size" height={380} option={sizeOption(runs, cell, isWinner, tokens)} />
        </div>
        <div>
          <Text size="md" fw={600} mb={4}>Every tuning run against dropout, learning rate and units</Text>
          <SimpleGrid cols={{ base: 1, md: 3 }} spacing="md">
            {(['dropout', 'lr', 'units'] as const).map((k) => (
              <EChart key={k} label={`Validation nMAE of every tuning run against ${k}`} height={280} option={factorOption(runs, k, tokens)} />
            ))}
          </SimpleGrid>
        </div>
        <Takeaway lines={tuningTakeaway(id)} testId="takeaway-tuning" />
        <Accordion variant="contained">
          <Accordion.Item value="search">
            <Accordion.Control><Text fw={600} size="sm">{kindOf(id) === 'A0' ? 'How A0 was chosen and its cell\'s search (tables)' : kindOf(id) === 'cell' ? 'This setting\'s search (tables)' : 'How this setting was set'}</Text></Accordion.Control>
            <Accordion.Panel>{details}</Accordion.Panel>
          </Accordion.Item>
        </Accordion>
      </Stack>
    </SectionCard>
  );
}

/** did the chosen networks overfit session 1? validation against session 2, and what guards against it */
export function OverfitSection({ id }: { id: string }) {
  const { tokens } = useView();
  const pts = useMemo(() => overfitPoints(combinedTest), []);
  return (
    <SectionCard testId="study-overfit" title="Did it overfit? Validation (session 1) against session 2"
      description="Guards in training: early stopping keeps the weights from the epoch with the lowest session-1 validation error (a held-out block after a gap), plus dropout and gradient clipping. The check below uses session 2, which no choice used: each point is a tuned cell winner or last semester's network, both axes combined over people.">
      <Stack gap="md">
        <EChart label="Validation nMAE against session-2 nMAE for each cell winner and last semester's network" height={400} option={overfitOption(pts, id, tokens)} />
        <Takeaway lines={overfitTakeaway(id, pts)} testId="takeaway-overfit" />
      </Stack>
    </SectionCard>
  );
}
