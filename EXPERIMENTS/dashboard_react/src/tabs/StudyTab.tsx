import { useMemo, useState } from 'react';
import { Group, Select, SimpleGrid, Stack, Text, Title } from '@mantine/core';
import { EChart } from '../components/EChart';
import { DataTable, Note, SectionCard, type Column } from '../components/common';
import { peakBarOption, R2_FLOOR, runtimeOption, studySeedOption } from '../charts/study';
import { SEED_KEY, type SeedSide } from '../charts/seedbox';
import { entry, GROUPS, groupOf, personScore, pickFor, runsOf, scoreCI, START, timesOf, valuesOf, type PersonRuns } from '../lib/results';
import { medianCI, type MedianCI } from '../lib/boxstats';
import { ciName } from '../lib/score';
import { ExperimentMachine } from '../components/MachineInfo';
import { SMETRICS } from '../lib/study';
import { median } from '../lib/stats';
import { useView } from '../state';

const f1 = (v: number) => (Number.isFinite(v) ? v.toFixed(1) : '–');
/** "median [lo, hi]" */
const withCI = (v: number, c: MedianCI | null, d: number) => (Number.isFinite(v) ? v.toFixed(d) : '–') + (c && Number.isFinite(c.lo) ? ` [${c.lo.toFixed(d)}, ${c.hi.toFixed(d)}]` : '');
/** name of the median's interval with n runs per person: "95 % CI", or e.g. "75 % range (3 runs)" */
const ciOf = (n: number) => ciName(medianCI(Array.from({ length: n }, (_, i) => i)), n);
const fewestSeeds = (people: { seeds: number[] }[]) => (people.length ? Math.min(...people.map((p) => p.seeds.length)) : 0);
/** the line under a seed chart: what the box shows and what the text under each box is */
const boxKey = (n: number) => `${SEED_KEY.box} Under each box: median [${ciOf(n)}].`;

/** One template: choose the study and its parameters at the top; every section below shows that experiment's runs. */
export function StudyTab() {
  const [pick, setPick] = useState<string>('W00_reference');
  const e = entry(pick);
  const g = groupOf(pick);
  const people = useMemo(() => runsOf(pick), [pick]);
  const seeds = [...new Set(people.map((p) => p.seeds.length))];
  return (
    <Stack gap="lg" data-testid="study-page">
      <div className="page-heading">
        <Title order={2}>Results</Title>
        <Text size="sm" c="dimmed">Trained on session 1, tested on session 2</Text>
      </div>
      <div className="filters" data-testid="filters">
        <Group gap="md" wrap="wrap" align="end">
          <Select label="Study" aria-label="Study" w={260} value={g.key} allowDeselect={false}
            data={GROUPS.map((x) => ({ value: x.key, label: x.label }))} onChange={(v) => v && setPick(GROUPS.find((x) => x.key === v)!.first)} />
          {g.params.map((x) => (
            <Select key={x.key} label={x.label} aria-label={x.label} w={x.key === 'network' ? 460 : 160} value={e.params[x.key]} allowDeselect={false}
              data={valuesOf(g.key, x.key)} onChange={(v) => v && setPick(pickFor(pick, x.key, v))} />
          ))}
        </Group>
        <Text size="sm" mt="sm" data-testid="setting-name">
          Showing: <b>{e.title}</b> · {people.length} people · {seeds.length === 1 ? seeds[0] : `${Math.min(...seeds)}–${Math.max(...seeds)}`} {seeds.length === 1 && seeds[0] === 1 ? 'seed' : 'seeds'} per person
        </Text>
      </div>
      {pick === START && (
        <Note>
          Last semester's code, unchanged. Same window as the current 200 ms, 95 % overlap run; different elsewhere: the EMG envelope
          is filtered forwards and backwards (4th order), so it uses future samples; LSTM 8 units × 2 layers, dropout 0.3; learning rate
          0.003, batch 8, up to 200 epochs; shuffled windows; no gap between training and validation data; last semester's passive table.
        </Note>
      )}
      {g.key === 'tuning' && (
        <Note>
          The chosen network (16 units × 2 layers, dropout 0, learning rate 0.01) had the lowest session-1 validation loss: 42 % below the previous
          network, lower for 6/6 people. Session-2 test scores were not used to choose it, so another finalist can score slightly better here.
        </Note>
      )}
      <Summary people={people} />
      <PerPosition people={people} m="mae" />
      <PerPosition people={people} m="r2" />
      <PeakTorque people={people} />
      <RunTime id={pick} />
      <SectionCard testId="study-machine" title="Processing capabilities">
        <ExperimentMachine id={pick} />
      </SectionCard>
    </Stack>
  );
}

interface Row { subj: string; mae: number; r2: number; peak: number; maeCI: MedianCI | null; r2CI: MedianCI | null }

/** one row per person: MAE, R² and peak torque */
function Summary({ people }: { people: PersonRuns[] }) {
  const rows: Row[] = people.map((p) => ({ subj: p.subj, mae: personScore(p, 'mae'), r2: personScore(p, 'r2'), peak: Math.max(...p.peaks.filter(Number.isFinite)),
    maeCI: scoreCI(p, 'mae'), r2CI: scoreCI(p, 'r2') }));
  rows.push({ subj: 'Median', mae: median(rows.map((r) => r.mae)), r2: median(rows.map((r) => r.r2)), peak: median(rows.map((r) => r.peak)), maeCI: null, r2CI: null });
  const ci = ciOf(fewestSeeds(people));
  const cols: Column<Row>[] = [
    { key: 'subj', label: 'Person', render: (r) => (r.subj === 'Median' ? <b>Median</b> : r.subj) },
    { key: 'mae', label: 'MAE (Nm)', align: 'right', render: (r) => withCI(r.mae, r.maeCI, 2) },
    { key: 'r2', label: 'R²', align: 'right', render: (r) => withCI(r.r2, r.r2CI, 2) },
    { key: 'peak', label: 'Peak torque (Nm)', align: 'right', render: (r) => f1(r.peak) },
  ];
  return (
    <SectionCard testId="study-summary" title="Summary" description={`Median over seeds [${ci}].`}>
      <DataTable columns={cols} rows={rows} rowKey={(r) => r.subj} dense minWidth={520} />
    </SectionCard>
  );
}

/** one chart per person: the metric at every position, a box over seeds */
function PerPosition({ people, m }: { people: PersonRuns[]; m: 'mae' | 'r2' }) {
  const { tokens } = useView();
  const side = (p: PersonRuns): SeedSide => ({
    name: 'Seeds', color: tokens.a, offset: 0,
    cells: (m === 'mae' ? p.mae : p.r2).map((row) => row.map((v, k) => ({ seed: p.seeds[k], v })).filter((x) => Number.isFinite(x.v))),
  });
  const unit = SMETRICS[m].unit ? ` (${SMETRICS[m].unit})` : '';
  return (
    <SectionCard testId={`study-${m}`} title={`${SMETRICS[m].short} per position${unit}`} description={`${boxKey(fewestSeeds(people))}${m === 'r2' ? ` ↓: below ${R2_FLOOR}, value in the label.` : ''}`}>
      <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="lg">
        {people.map((p) => (
          <div key={p.subj} className="subject-plot">
            <Text size="md" fw={600} mb={4}>{p.subj}</Text>
            <EChart label={`${p.subj}: ${SMETRICS[m].short} per position`} height={340}
              option={studySeedOption([side(p)], p.positions, m, 'box', tokens, false, p.subj)} />
          </div>
        ))}
      </SimpleGrid>
    </SectionCard>
  );
}

/** one chart per person: the peak measured torque at every position */
function PeakTorque({ people }: { people: PersonRuns[] }) {
  const { tokens } = useView();
  return (
    <SectionCard testId="study-peaks" title="Peak torque per position (Nm)" description="Largest measured |torque| of the predicted target in each trial.">
      <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="lg">
        {people.map((p) => (
          <div key={p.subj} className="subject-plot">
            <Text size="md" fw={600} mb={4}>{p.subj}</Text>
            <EChart label={`${p.subj}: peak torque per position`} height={300} option={peakBarOption(p.positions, p.peaks, tokens)} />
          </div>
        ))}
      </SimpleGrid>
    </SectionCard>
  );
}

/** training time of every run, median per person */
function RunTime({ id }: { id: string }) {
  const { tokens } = useView();
  const { people, definition } = useMemo(() => timesOf(id), [id]);
  return (
    <SectionCard testId="study-runtime" title="Training time per run (min)" description={people.length ? `${boxKey(fewestSeeds(people))} Timer: ${definition}.` : undefined}>
      {people.length
        ? <EChart label="Training time per run, every seed per person" height={320} option={runtimeOption(people, tokens)} />
        : <Text size="sm" c="dimmed">No training times were saved for this experiment.</Text>}
    </SectionCard>
  );
}
