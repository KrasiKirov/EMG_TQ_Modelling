import { useMemo, type ReactNode } from 'react';
import { Accordion, List, SimpleGrid, Stack, Text } from '@mantine/core';
import { EChart } from '../components/EChart';
import { SectionCard } from '../components/common';
import { factorOption, overfitOption, sizeOption, FACTOR_LEVELS, type Factor } from '../charts/tuning';
import { DataTable, type Column } from '../components/common';
import { cellOf, chosenConfig, isWinner, kindOf, overfitPoints, personScore, runsOf, settingOf, tuningRuns, type TuningRun } from '../lib/results';
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
        <Hyperparameters id={id} runs={runs} />
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
      </Stack>
    </SectionCard>
  );
}

const FACTORS: { key: Factor; name: string; searched: string }[] = [
  { key: 'units', name: 'Units', searched: '8 / 16 / 32 / 64' },
  { key: 'layers', name: 'Layers', searched: '1 / 2' },
  { key: 'dropout', name: 'Dropout', searched: '0 – 0.3' },
  { key: 'lr', name: 'Learning rate', searched: '0.0003 – 0.03 (log scale)' },
  { key: 'batch', name: 'Batch size', searched: '32 / 64 / 128 (reset windows only)' },
];
const show = (k: Factor, v: number | null | undefined) => (v === null || v === undefined ? '–' : k === 'lr' ? String(+v.toPrecision(3)) : k === 'dropout' ? String(+v.toFixed(3)) : String(v));

/** each hyperparameter: what the search could pick, what this setting's cell tried, what it uses; one chart per hyperparameter */
function Hyperparameters({ id, runs }: { id: string; runs: TuningRun[] }) {
  const { tokens } = useView();
  const kind = kindOf(id), cell = cellOf(id), chosen = chosenConfig(id), s = settingOf(id);
  const mine = runs.filter((r) => r.cell === cell);
  // A2 is the 'anchor' tried in every cell; a tuned setting is its cell's runs, with its winner ringed
  const isMine = kind === 'A2' ? (r: TuningRun) => r.config === 'anchor' : (r: TuningRun) => r.cell === cell;
  const isPick = kind === 'A2' ? isMine : (r: TuningRun) => r.cell === cell && r.id === chosen;
  const tried = (k: Factor) => {
    const v = [...new Set(mine.map((r) => r[k]).filter((x): x is number => x !== null && x !== undefined))].sort((a, b) => a - b);
    if (!v.length) return kind === 'A1' || kind === 'A2' ? '–' : 'not used (whole chunks)';
    return FACTOR_LEVELS[k] ? v.join(' / ') : `${v.length} values, ${show(k, v[0])} – ${show(k, v[v.length - 1])}`;
  };
  const uses = (k: Factor) => (kind === 'A1' ? '–' : k === 'batch' && s.state === 'carry' ? 'none (whole chunks)' : show(k, s[k]));
  type Row = (typeof FACTORS)[number];
  const cols: Column<Row>[] = [
    { key: 'name', label: 'Hyperparameter', render: (r) => r.name },
    { key: 'searched', label: 'Search range', render: (r) => r.searched },
    ...(kind === 'A2' ? [] : [{ key: 'tried', label: kind === 'A3' ? 'Tried in A0\'s cell' : 'Tried in this setting\'s cell', render: (r: Row) => tried(r.key) }]),
    { key: 'uses', label: kind === 'A2' ? 'This setting (fixed, not tuned)' : 'This setting uses', render: (r) => <b>{uses(r.key)}</b> },
  ];
  const note = kind === 'A1' ? 'The lagged ridge has no network: its only setting, the regularisation strength, was chosen on session-1 validation (tables below).'
    : kind === 'A2' ? 'Last semester\'s network was not tuned: it keeps last semester\'s values. The same network is the "anchor" tried in every cell.'
    : kind === 'A3' ? 'A3 reuses A0\'s network unchanged (only the torque target differs), so its search is A0\'s.' : null;
  return (
    <div>
      <Text size="md" fw={600} mb={4}>Each hyperparameter: what was tried and what was chosen</Text>
      {note && <Text size="sm" mb={6}>{note}</Text>}
      {kind !== 'A1' && <DataTable columns={cols} rows={FACTORS} rowKey={(r) => r.key} dense minWidth={720} />}
      <Text size="sm" c="dimmed" mt="sm" mb={4}>Each dot is one tuning run (validation nMAE, session 1; lower is better). {kind === 'A1' ? 'Grey: every LSTM tuning run, for comparison.' : kind === 'A2' ? 'Grey: every other run. Ringed: this network, run as the anchor in each cell.' : `Grey: every cell. Blue: ${kind === 'A3' ? 'A0\'s' : 'this setting\'s'} cell. Ring: the configuration this setting uses.`}</Text>
      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
        {FACTORS.map((f) => (
          <div key={f.key}>
            <Text size="sm" fw={600}>{f.name}{kind !== 'A1' ? <> · uses <b>{uses(f.key)}</b></> : null}</Text>
            <EChart label={`Validation nMAE of every tuning run against ${f.name.toLowerCase()}`} height={240} option={factorOption(runs, f.key, isMine, isPick, tokens)} />
          </div>
        ))}
      </SimpleGrid>
    </div>
  );
}
