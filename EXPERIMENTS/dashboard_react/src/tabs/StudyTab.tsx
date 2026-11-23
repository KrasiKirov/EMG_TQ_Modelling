import { useEffect, useMemo, useState } from 'react';
import { Accordion, Button, Code, CopyButton, Group, Select, SimpleGrid, Stack, Text, Title } from '@mantine/core';
import { EChart } from '../components/EChart';
import { DataTable, Note, SectionCard, type Column } from '../components/common';
import { allPeopleOption, peakBarOption, runtimeOption, SMETRICS, studySeedOption } from '../charts/study';
import { type SeedSide } from '../charts/seedbox';
import { changedParams, choiceOf, CONFIRM, DEFAULT, entry, has, PARAMS, personScore, pickFor, runsOf, SCORE_FROM, scoreCI,
  settingOf, timesOf, optionsOf, worstR2, type PersonRuns } from '../lib/results';
import { type MedianCI } from '../lib/boxstats';
import { MachineInfo } from '../components/MachineInfo';
import { median } from '../lib/stats';
import { useView } from '../state';
import { waveOption, type WaveLine } from '../charts/options';
import { seedCount, WAVE_SPAN, worstPositions, worstWaves } from '../lib/studyWaves';
import { fmt, posLabel } from '../lib/format';
import { HELP_SECTIONS, pageLimitation } from '../content/help';
import { Takeaway } from '../components/Takeaway';
import { headline } from '../lib/takeaways';
import { OverfitSection, TuningSection } from './Tuning';

const f1 = (v: number) => (Number.isFinite(v) ? v.toFixed(1) : '–');
const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? '' : 's'}`;
/** "median [lo, hi]"; no interval for a single seed */
const withCI = (v: number, c: MedianCI | null, d: number) => (Number.isFinite(v) ? v.toFixed(d) : '–') + (c && Number.isFinite(c.lo) ? ` [${c.lo.toFixed(d)}, ${c.hi.toFixed(d)}]` : '');
const seedRange = (s: number[]) => (s.length === 1 ? `seed ${s[0]}` : `seeds ${Math.min(...s)}–${Math.max(...s)}`);

/** the setting in the address bar (#setting=<id>); anything else (an old link, a typo) opens A0 */
function settingFromHash(): string {
  const id = new URLSearchParams(window.location.hash.replace(/^#/, '')).get('setting');
  return id && has(id) ? id : DEFAULT;
}

/** One template: the filters choose a study v2 setting; every section below shows that setting. */
export function StudyTab() {
  const [pick, setPickRaw] = useState<string>(settingFromHash);
  const [moved, setMoved] = useState<string[]>([]);
  useEffect(() => {
    const h = `#setting=${encodeURIComponent(pick)}`;
    if (window.location.hash !== h) window.history.replaceState(null, '', h);
  }, [pick]);
  useEffect(() => {
    const onHash = () => {
      const id = settingFromHash();
      const h = `#setting=${encodeURIComponent(id)}`;
      if (window.location.hash !== h) window.history.replaceState(null, '', h);   // an old link shows what it opened
      setPickRaw(id);
      setMoved([]);
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  const choose = (key: string, value: string) => {
    const next = pickFor(pick, key, value);
    setMoved(changedParams(pick, next, key));
    setPickRaw(next);
  };
  const e = entry(pick);
  const st = settingOf(pick);
  const people = useMemo(() => runsOf(pick), [pick]);
  const seeds = [...new Set(people.flatMap((p) => p.seeds))].sort((a, b) => a - b);
  const notes: string[] = [];
  if (seeds.length === 1) notes.push(`One run per person (${seedRange(seeds)}): no interval can be given, and the run shown is the only run.`);
  if (pick === DEFAULT) notes.push('Seeds 10–12 are the runs of its cell winner (the same configuration), so that cell is not listed separately.');
  if (st.target === 'total') notes.push("Predicts total torque. Its nMAE still divides by the person's active-torque scale (as pre-registered), so it is on the same scale as A0's.");
  return (
    <Stack gap="lg" data-testid="study-page">
      <div className="page-heading">
        <Title order={2}>Results: study v2</Title>
        <Text size="sm" c="dimmed">
          Pre-registered on 18 Nov 2026. Every model learns from session 1 and predicts session 2, which it never saw and which was not used to choose it.
          Open “How to use this page” for what each number means.
        </Text>
      </div>
      <Takeaway title="The study's answer (session 2, pre-registered decisions)" lines={headline()} testId="headline" />
      <Help />
      <div className="filters" data-testid="filters">
        <Group gap="md" wrap="wrap" align="end">
          {PARAMS.map((x) => (
            <Select key={x.key} label={x.label} aria-label={x.label} w={x.key === 'model' ? 240 : 170} value={e.params[x.key]} allowDeselect={false}
              data={optionsOf(pick, x.key)} onChange={(v) => v && choose(x.key, v)} />
          ))}
        </Group>
        <Text size="sm" mt={6} data-testid="setting-name">
          Showing: <b>{e.title}</b>{st.units ? ` · ${st.units} units × ${plural(st.layers ?? 0, 'layer')}` : ''} · {people.length} people · {seedRange(seeds)} per person
          {' '}· <Text span c="dimmed" size="sm">{e.id}</Text>
        </Text>
      </div>
      <Stack gap={4} mt={-8}>
        {moved.length > 0 && (
          <Text size="sm" data-testid="setting-moved">
            Only the run combinations can be chosen, so the filters below it changed: {moved.join('; ')}.
          </Text>
        )}
        {notes.length > 0 && (
          <Stack gap={4} data-testid="setting-notes">
            {notes.map((n) => <Note key={n}>{n}</Note>)}
          </Stack>
        )}
        <Text size="xs" c="dimmed">
          {e.target === 'total' ? 'Total torque: the measured torque.' : 'Active torque: measured torque minus the passive torque from the tested session\'s own relaxed holds (YES: lab worksheet; IES: experiment-sheet values, it has no relaxed holds).'}
        </Text>
      </Stack>
      <AllPeople people={people} m="nmae" />
      <AllPeople people={people} m="r2" />
      <WorstPositions id={pick} people={people} />
      <PerPosition people={people} m="mae" />
      <PerPosition people={people} m="r2" />
      <PeakTorque people={people} />
      <TuningSection id={pick} details={<HowChosen id={pick} />} />
      <OverfitSection id={pick} />
      <Details id={pick} people={people} />
    </Stack>
  );
}

/** setup details, collapsed: mounted only when open, so the charts inside get their real width */
function Details({ id, people }: { id: string; people: PersonRuns[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <Accordion variant="contained" value={open} onChange={setOpen} data-testid="study-details">
      <Accordion.Item value="details">
        <Accordion.Control><Text fw={600}>Reproduce and run details</Text></Accordion.Control>
        <Accordion.Panel>
          {open === 'details' && (
            <Stack gap="lg">
              <RunTime id={id} />
              <Reproduce id={id} people={people} />
              <SectionCard testId="study-machine" title="Processing capabilities">
                <MachineInfo />
              </SectionCard>
            </Stack>
          )}
        </Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  );
}

/** how to navigate and read the page */
function Help() {
  return (
    <Accordion variant="contained" data-testid="help">
      <Accordion.Item value="help">
        <Accordion.Control>How to use this page</Accordion.Control>
        <Accordion.Panel>
          <Stack gap="md">
            {HELP_SECTIONS.map((sec) => (
              <div key={sec.title}>
                <Text fw={600} size="sm" mb={4}>{sec.title}</Text>
                <Stack gap={4}>
                  {sec.items.map(([term, text]) => (
                    <Text key={term} size="sm"><b>{term}</b> · {text}</Text>
                  ))}
                </Stack>
              </div>
            ))}
            <div>
              <Text fw={600} size="sm" mb={4}>Limitations (pre-registered)</Text>
              <Stack gap={4}>{CONFIRM.limitations.map((l) => <Text key={l} size="sm">· {pageLimitation(l)}</Text>)}</Stack>
            </div>
          </Stack>
        </Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  );
}

interface Row { subj: string; nmae: number; mae: number; r2: number; worst: number; peak: number; nmaeCI: MedianCI | null; maeCI: MedianCI | null; r2CI: MedianCI | null }

/** every person on one chart against ankle position (median over seeds); the nMAE card also holds the table */
function AllPeople({ people, m }: { people: PersonRuns[]; m: 'nmae' | 'r2' }) {
  const { tokens } = useView();
  // colours follow the person's place in the study, so a filter never repaints anyone
  const order = useMemo(() => runsOf(DEFAULT).map((p) => p.subj), []);
  const lines = people.map((p) => ({ subj: p.subj, positions: p.positions, values: m === 'nmae' ? p.nmae : p.r2, ...(m === 'nmae' ? { mae: p.mae } : {}) }));
  const nmae = m === 'nmae';
  return (
    <SectionCard testId={`study-all-${m}`} title={nmae ? 'nMAE by position, all people (session 2)' : 'R² by position, all people (session 2)'}>
      <EChart label={`${nmae ? 'nMAE' : 'R²'} against ankle position for every person`} height={400} option={allPeopleOption(lines, order, m, tokens)} />
      {nmae && <Summary people={people} />}
    </SectionCard>
  );
}

/** one row per person: nMAE (the pre-registered metric), MAE, R² over positions and at the worst one, peak torque */
function Summary({ people }: { people: PersonRuns[] }) {
  const rows: Row[] = people.map((p) => ({
    subj: p.subj, nmae: personScore(p, 'nmae'), mae: personScore(p, 'mae'), r2: personScore(p, 'r2'), worst: worstR2(p),
    peak: Math.max(...p.peaks.filter(Number.isFinite)), nmaeCI: scoreCI(p, 'nmae'), maeCI: scoreCI(p, 'mae'), r2CI: scoreCI(p, 'r2'),
  }));
  rows.push({ subj: 'Median', nmae: median(rows.map((r) => r.nmae)), mae: median(rows.map((r) => r.mae)), r2: median(rows.map((r) => r.r2)),
    worst: median(rows.map((r) => r.worst)), peak: median(rows.map((r) => r.peak)), nmaeCI: null, maeCI: null, r2CI: null });
  const cols: Column<Row>[] = [
    { key: 'subj', label: 'Person', render: (r) => (r.subj === 'Median' ? <b>{r.subj}</b> : r.subj) },
    { key: 'nmae', label: 'nMAE', align: 'right', render: (r) => withCI(r.nmae, r.nmaeCI, 3) },
    { key: 'mae', label: 'MAE (Nm)', align: 'right', render: (r) => withCI(r.mae, r.maeCI, 2) },
    { key: 'r2', label: 'R², median of positions', align: 'right', render: (r) => withCI(r.r2, r.r2CI, 2) },
    { key: 'worst', label: 'R², worst position', align: 'right', render: (r) => fmt(r.worst) },
    { key: 'peak', label: 'Peak torque (Nm)', align: 'right', render: (r) => f1(r.peak) },
  ];
  return (
    <Accordion variant="contained" mt="md">
      <Accordion.Item value="table">
        <Accordion.Control><Text fw={600} size="sm">The numbers per person (table)</Text></Accordion.Control>
        <Accordion.Panel>
          <Text size="sm" mb="sm">Per person, the median over seeds. Every run is scored from {SCORE_FROM.join(', ')} s after each trial's start. nMAE (the pre-registered metric):
        the MAE at each position divided by the 99th percentile of the person's measured active torque on session 2, then the mean over positions.
        MAE (Nm): the same without the division. R²: within each position (one trial at one angle); the median of the positions hides the bad ones, so
        the worst position is shown too (per seed the lowest R², then the median over seeds). Peak torque: the largest |measured torque| of the
        predicted target over the scored samples. [ ]: interval for the median over seeds (about 95 %; none with 1 seed); it shows training
        randomness only. Median: the plain median of the 6 people.</Text>
          <DataTable columns={cols} rows={rows} rowKey={(r) => r.subj} dense minWidth={900} />
        </Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  );
}

/** session-1 validation: how this setting was chosen */
function HowChosen({ id }: { id: string }) {
  const c = useMemo(() => choiceOf(id), [id]);
  type R = NonNullable<typeof c.rounds>[number];
  type W = NonNullable<typeof c.cellWinners>[number];
  const n = (v: number | null | undefined, d?: number) => (v === null || v === undefined ? '–' : d === undefined ? String(v) : v.toFixed(d));
  const rcols: Column<R>[] = [
    { key: 'round', label: 'Round', render: (r) => r.round },
    { key: 'config', label: 'Config', render: (r) => (r.winner ? <b>{r.config}</b> : r.config) },
    { key: 'units', label: 'Units', align: 'right', render: (r) => n(r.params?.units) },
    { key: 'layers', label: 'Layers', align: 'right', render: (r) => n(r.params?.layers) },
    { key: 'dropout', label: 'Dropout', align: 'right', render: (r) => n(r.params?.dropout) },
    { key: 'lr', label: 'Learning rate', align: 'right', render: (r) => n(r.params?.lr) },
    { key: 'batch', label: 'Batch', align: 'right', render: (r) => n(r.params?.batch) },
    { key: 'score', label: 'Validation nMAE (session 1)', align: 'right', render: (r) => fmt(r.score, 4) },
    { key: 'seeds', label: 'Seeds', align: 'right', render: (r) => r.seeds.join(', ') },
    { key: 'mark', label: '', render: (r) => [r.kept ? 'seed check' : '', r.winner ? 'chosen' : ''].filter(Boolean).join(', ') },
  ];
  const wcols: Column<W>[] = [
    { key: 'title', label: 'Cell winner', render: (r) => (r.chosen ? <b>{r.title}</b> : r.title) },
    { key: 'id', label: 'Config', render: (r) => r.id },
    { key: 'score', label: 'Validation nMAE, common targets (session 1)', align: 'right', render: (r) => fmt(r.score, 4) },
    { key: 'chosen', label: '', render: (r) => (r.chosen ? 'A0' : '') },
  ];
  return (
    <div data-testid="study-choice">
      <Stack gap="md">
        {c.fixed && <Text size="sm">{c.fixed}</Text>}
        {c.cellWinners && (
          <div>
            <Text size="sm" fw={600} mb={4}>A0: the cell winner with the lowest validation nMAE on the targets every cell predicts (no 1-SE rule)</Text>
            <DataTable columns={wcols} rows={c.cellWinners} rowKey={(r) => r.cell} rowClass={(r) => (r.chosen ? 'selected-row' : undefined)} dense minWidth={720} />
          </div>
        )}
        {c.rounds && c.rounds.length > 0 && (
          <div>
            <Text size="sm" fw={600} mb={4}>Its cell's search</Text>
            <Text size="xs" c="dimmed" mb={4}>
              Coarse: 6 random draws shared by every cell plus the anchor (last semester's network). Fine: 4 draws around the cell's best coarse config.
              Seed check: the best 2 of coarse + fine get a second seed; the winner is the config with the fewest weights within one standard error of the
              better one (paired over people).
            </Text>
            <DataTable columns={rcols} rows={c.rounds} rowKey={(r) => r.id} rowClass={(r) => (r.winner ? 'selected-row' : undefined)} dense minWidth={960} />
          </div>
        )}
      </Stack>
    </div>
  );
}

/** the commands that re-run this setting, with what they need */
function Reproduce({ id, people }: { id: string; people: PersonRuns[] }) {
  const seeds = [...new Set(people.flatMap((p) => p.seeds))].sort((a, b) => a - b);
  const subjects = people.map((p) => p.subj);
  const cell = settingOf(id).cell;
  const shared = id === DEFAULT && cell ? seeds.filter((s) => s <= 12) : [];
  const own = seeds.filter((s) => !shared.includes(s));
  const queue = (exp: string, sd: number[]) => `.venv/bin/python -m EXPERIMENTS.pipeline.run_queue --exps ${exp} --subjects ${subjects.join(' ')} --seeds ${sd.join(' ')} --workers 6`;
  const one = `.venv/bin/python -m EXPERIMENTS.pipeline.run --exp ${id} --subject ${subjects[0] ?? 'JM'} --seed ${own[0] ?? 10} --threads 1`;
  const update = '.venv/bin/python -m EXPERIMENTS.analysis.study_v2_export && .venv/bin/python -m EXPERIMENTS.analysis.study_v2_waves && npm --prefix EXPERIMENTS/dashboard_react run build';
  const Cmd = ({ label, cmd }: { label: string; cmd: string }) => (
    <div>
      <Group justify="space-between" mb={4}>
        <Text size="sm" fw={600}>{label}</Text>
        <CopyButton value={cmd}>{({ copied, copy }) => <Button size="compact-xs" variant="light" onClick={copy}>{copied ? 'Copied' : 'Copy'}</Button>}</CopyButton>
      </Group>
      <Code block>{cmd}</Code>
    </div>
  );
  const prov = CONFIRM.provenance_check;
  return (
    <SectionCard testId="study-reproduce" title="Re-run this setting"
      description={`The setting is the spec file EXPERIMENTS/configs/${id}.json. Run the commands from the repository root. Every final run was made on commit ${prov.commits.map((c) => c.slice(0, 8)).join(', ')} with protocol ${prov.protocol_hash.slice(0, 8)} (EXPERIMENTS/protocol/study_v2.json).`}>
      <Stack gap="md">
        <Stack gap={4}>
          <Text size="sm"><b>1. Code and software</b> · the repository (fork KrasiKirov/EMG_TQ_Modelling) at that commit and a .venv with the versions these runs used (shown under Processing capabilities).</Text>
          <Text size="sm"><b>2. Recordings</b> · the 6 .flb files are participant data and are not in git. Get them from the lab, put them in ML/subjects/, or point EMG_DATA_DIR at their folder: <Code>export EMG_DATA_DIR=/path/to/recordings</Code></Text>
          <Text size="sm"><b>3. Run</b> · study v2 runs write to EXPERIMENTS/results_v2/{id} and refuse to start on uncommitted code. The queue skips runs whose results already exist with the same config; move that folder away first to train them again. With the same software and 1 thread, the same seed gives the same predictions.</Text>
        </Stack>
        {shared.length > 0 && cell && <Cmd label={`Seeds ${shared.join(', ')} (the runs of FE_cell_${cell}, the same configuration)`} cmd={queue(`FE_cell_${cell}`, shared)} />}
        <Cmd label={shared.length ? `Seeds ${own.join(', ')}` : 'Every run of this setting'} cmd={queue(id, own)} />
        <Cmd label="One run (one person, one seed)" cmd={one} />
        <Cmd label="Then update this page" cmd={update} />
      </Stack>
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
    <SectionCard testId={`study-${m}`} title={`${SMETRICS[m].short} per position${unit}`}>
      <SimpleGrid cols={1} spacing="lg">
        {people.map((p) => (
          <div key={p.subj} className="subject-plot">
            <Text size="md" fw={600} mb={4}>{p.subj}</Text>
            <EChart label={`${p.subj}: ${SMETRICS[m].short} per position`} height={320}
              option={studySeedOption([side(p)], p.positions, m, 'box', tokens, false, p.subj)} />
          </div>
        ))}
      </SimpleGrid>
    </SectionCard>
  );
}

/** per person, the 2 positions with the lowest median R²: measured torque against the best and the worst run there */
function WorstPositions({ id, people }: { id: string; people: PersonRuns[] }) {
  const { tokens } = useView();
  const target = entry(id).target === 'total' ? 'total torque (Nm)' : 'active torque (Nm)';
  const shown = people.filter((p) => worstPositions(id, p.subj).length);
  return (
    <SectionCard testId="study-waves" title="Measured and predicted torque at the worst positions"
      description={`Setting: ${entry(id).title}. For each person, the 2 positions where this setting's median R² over seeds is lowest, so they change with the setting (R² from the scored samples, as in the R² chart above). Best and worst run: the seeds with the highest and the lowest R² at that position. The curves show the whole trial.`}>
      {shown.length ? (
        <SimpleGrid cols={1} spacing="lg">
          {shown.map((p) => (
            <div key={p.subj} className="subject-plot">
              <Text size="md" fw={600} mb={4}>{p.subj} <Text span size="sm" c="dimmed" fw={400}>· {plural(seedCount(id, p.subj), 'seed')}</Text></Text>
              <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="md">
                {worstPositions(id, p.subj).map((b, i) => {
                  const w = worstWaves(id, p.subj, i);
                  if (!w) return null;
                  const single = seedCount(id, p.subj) === 1;
                  const lines: WaveLine[] = [
                    { name: 'measured', color: tokens.ink, width: 1.8, data: w.measured.v, time: w.measured.t },
                    { name: single ? `predicted · seed ${b.best.seed}` : `best · seed ${b.best.seed}`, color: tokens.a, width: 1.4, data: w.best.v, time: w.best.t },
                    ...(single ? [] : [{ name: `worst · seed ${b.worst.seed}`, color: tokens.b, width: 1.4, data: w.worst.v, time: w.worst.t }]),
                  ];
                  return (
                    <div key={b.k}>
                      <Text size="sm" fw={600}>{posLabel(b.pos, 3)} rad{b.k < 2 ? ' · one of the 2 most plantarflexed' : ''} · {single ? 'R²' : 'median R²'} {fmt(b.median_r2)}</Text>
                      <Text size="xs" c="dimmed" className="num" mb={4}>
                        {single ? `Only run: R² ${fmt(b.best.r2)}, MAE ${fmt(b.best.mae)} Nm`
                          : `Best: R² ${fmt(b.best.r2)}, MAE ${fmt(b.best.mae)} Nm · Worst: R² ${fmt(b.worst.r2)}, MAE ${fmt(b.worst.mae)} Nm`}
                      </Text>
                      <EChart label={`${p.subj} at ${posLabel(b.pos, 3)} rad: measured torque and the best and worst run`} height={300}
                        option={waveOption(WAVE_SPAN, lines, tokens, false, target)} />
                    </div>
                  );
                })}
              </SimpleGrid>
            </div>
          ))}
        </SimpleGrid>
      ) : (
        <Text size="sm" c="dimmed">No predictions were exported for this experiment.</Text>
      )}
    </SectionCard>
  );
}

/** one chart per person: the peak measured torque at every position */
function PeakTorque({ people }: { people: PersonRuns[] }) {
  const { tokens } = useView();
  return (
    <SectionCard testId="study-peaks" title="Peak torque per position (Nm)" description={`Largest absolute measured torque of the predicted target at each position, over the scored samples (from ${SCORE_FROM.join(', ')} s after the trial's start), the same samples the errors are scored on.`}>
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
  const { people } = useMemo(() => timesOf(id), [id]);
  return (
    <SectionCard testId="study-runtime" title="Training time per run (min)">
      {people.length
        ? <EChart label="Training time per run, every seed per person" height={320} option={runtimeOption(people, tokens)} />
        : <Text size="sm" c="dimmed">No training times were saved for this experiment.</Text>}
    </SectionCard>
  );
}
