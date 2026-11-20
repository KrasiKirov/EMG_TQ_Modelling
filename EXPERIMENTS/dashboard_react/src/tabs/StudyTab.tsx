import { useEffect, useMemo, useState } from 'react';
import { Accordion, Button, Code, CopyButton, Group, Select, SimpleGrid, Stack, Text, Title } from '@mantine/core';
import { EChart } from '../components/EChart';
import { DataTable, Note, SectionCard, type Column } from '../components/common';
import { peakBarOption, runtimeOption, studySeedOption } from '../charts/study';
import { type SeedSide } from '../charts/seedbox';
import { activePeak, changedParams, entry, ENTRIES, GROUPS, groupOf, maeRatio, notesOf, personScore, pickFor, referenceOf, runsOf, scoreCI, timesOf,
  valuesOf, worstR2, type PersonRuns } from '../lib/results';
import { type MedianCI } from '../lib/boxstats';
import { ExperimentMachine } from '../components/MachineInfo';
import { SMETRICS } from '../lib/study';
import { median } from '../lib/stats';
import { useView } from '../state';
import { waveOption, type WaveLine } from '../charts/options';
import { seedCount, WAVE_SPAN, worstPositions, worstWaves } from '../lib/studyWaves';
import { fmt, posLabel } from '../lib/format';
import { TUNING_VAL } from '../lib/tuningVal';
import { HELP_SECTIONS } from '../content/help';

const f1 = (v: number) => (Number.isFinite(v) ? v.toFixed(1) : '–');
const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? '' : 's'}`;
/** "median [lo, hi]"; no interval for a single seed */
const withCI = (v: number, c: MedianCI | null, d: number) => (Number.isFinite(v) ? v.toFixed(d) : '–') + (c && Number.isFinite(c.lo) ? ` [${c.lo.toFixed(d)}, ${c.hi.toFixed(d)}]` : '');

const DEFAULT = 'W00_reference';
/** the setting in the address bar (#setting=<id>), if it is one the page shows */
function settingFromHash(): string {
  const id = new URLSearchParams(window.location.hash.replace(/^#/, '')).get('setting');
  return id && ENTRIES.some((e) => e.id === id) ? id : DEFAULT;
}

/** One template: choose the study and its parameters at the top; every section below shows that experiment's runs. */
export function StudyTab() {
  const [pick, setPickRaw] = useState<string>(settingFromHash);
  const [moved, setMoved] = useState<string[]>([]);
  useEffect(() => {
    const h = `#setting=${encodeURIComponent(pick)}`;
    if (window.location.hash !== h) window.history.replaceState(null, '', h);
  }, [pick]);
  useEffect(() => {
    const onHash = () => { setPickRaw(settingFromHash()); setMoved([]); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  const choose = (key: string, value: string) => {
    const next = pickFor(pick, key, value);
    setMoved(changedParams(pick, next, key));
    setPickRaw(next);
  };
  const e = entry(pick);
  const g = groupOf(pick);
  const people = useMemo(() => runsOf(pick), [pick]);
  const seeds = [...new Set(people.map((p) => p.seeds.length))];
  const notes = notesOf(pick);
  return (
    <Stack gap="lg" data-testid="study-page">
      <div className="page-heading">
        <Title order={2}>Results</Title>
        <Text size="sm" c="dimmed">Every model here learns from session 1 and predicts session 2, which it never saw. Open “How to use this page” for what each number means.</Text>
      </div>
      <Help />
      <div className="filters" data-testid="filters">
        <Group gap="md" wrap="wrap" align="end">
          <Select label="Study" aria-label="Study" w={260} value={g.key} allowDeselect={false}
            data={GROUPS.map((x) => ({ value: x.key, label: x.label }))}
            onChange={(v) => { if (v) { setMoved([]); setPickRaw(GROUPS.find((x) => x.key === v)!.first); } }} />
          {g.params.map((x) => (
            <Select key={x.key} label={x.label} aria-label={x.label} w={x.key === 'network' ? 520 : 170} value={e.params[x.key]} allowDeselect={false}
              data={valuesOf(g.key, x.key)} onChange={(v) => v && choose(x.key, v)} />
          ))}
        </Group>
        <Text size="sm" mt="sm" data-testid="setting-name">
          Showing: <b>{e.title}</b> · {people.length} people · {seeds.length === 1 ? plural(seeds[0], 'seed') : `${Math.min(...seeds)}–${Math.max(...seeds)} seeds`} per person
          {' '}· <Text span c="dimmed" size="sm">{e.id}</Text>
        </Text>
        {moved.length > 0 && (
          <Text size="sm" c="dimmed" mt={4} data-testid="setting-moved">
            No run has exactly that combination, so the nearest one is shown. Also changed: {moved.join('; ')}.
          </Text>
        )}
        {notes.length > 0 && (
          <Stack gap={4} mt="sm" data-testid="setting-notes">
            {notes.map((n) => <Note key={n}>{n}</Note>)}
          </Stack>
        )}
        <Text size="xs" c="dimmed" mt="sm">
          {e.target === 'total' ? 'Total torque: the measured torque.' : 'Active torque: measured torque minus the passive torque from the tested session\'s own relaxed holds (YES: lab worksheet; IES: experiment-sheet values, it has no relaxed holds).'}
        </Text>
      </div>
      {g.key === 'tuning' && <ValidationLoss pick={pick} />}
      <Summary id={pick} people={people} title={g.key === 'tuning' ? 'Test on session 2 (not used to choose the network)' : 'Summary'} />
      {g.key !== 'tuning' && <WorstPositions id={pick} people={people} />}
      <PerPosition people={people} m="mae" />
      <PerPosition people={people} m="r2" />
      <PeakTorque people={people} />
      <RunTime id={pick} />
      <Reproduce id={pick} people={people} />
      <SectionCard testId="study-machine" title="Processing capabilities">
        <ExperimentMachine id={pick} />
      </SectionCard>
    </Stack>
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
          </Stack>
        </Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  );
}

/** the commands that re-run this setting, with what they need */
function Reproduce({ id, people }: { id: string; people: PersonRuns[] }) {
  const seeds = [...new Set(people.flatMap((p) => p.seeds))].sort((a, b) => a - b);
  const subjects = people.map((p) => p.subj);
  const all = `.venv/bin/python -m EXPERIMENTS.pipeline.run_queue --exps ${id} --subjects ${subjects.join(' ')} --directions S1toS2 --seeds ${seeds.join(' ')} --workers 6`;
  const one = `.venv/bin/python -m EXPERIMENTS.pipeline.run --exp ${id} --subject ${subjects[0] ?? 'JM'} --direction S1toS2 --seed ${seeds[0] ?? 0} --threads 1`;
  const update = entry(id).group === 'tuning'
    ? `.venv/bin/python -m EXPERIMENTS.pipeline.dashboard build ${id}`
    : '.venv/bin/python -m EXPERIMENTS.analysis.study_scores && .venv/bin/python -m EXPERIMENTS.analysis.study_waves && npm --prefix EXPERIMENTS/dashboard_react run build';
  const Cmd = ({ label, cmd }: { label: string; cmd: string }) => (
    <div>
      <Group justify="space-between" mb={4}>
        <Text size="sm" fw={600}>{label}</Text>
        <CopyButton value={cmd}>{({ copied, copy }) => <Button size="compact-xs" variant="light" onClick={copy}>{copied ? 'Copied' : 'Copy'}</Button>}</CopyButton>
      </Group>
      <Code block>{cmd}</Code>
    </div>
  );
  return (
    <SectionCard testId="study-reproduce" title="Re-run this setting"
      description={`The setting is the spec file EXPERIMENTS/configs/${id}.json. Run the commands from the repository root.`}>
      <Stack gap="md">
        <Stack gap={4}>
          <Text size="sm"><b>1. Code and software</b> · the repository (fork KrasiKirov/EMG_TQ_Modelling) and a .venv with the versions these runs used (shown under Processing capabilities).</Text>
          <Text size="sm"><b>2. Recordings</b> · the 6 .flb files are participant data and are not in git. Get them from the lab, put them in ML/subjects/, or point EMG_DATA_DIR at their folder: <Code>export EMG_DATA_DIR=/path/to/recordings</Code></Text>
          <Text size="sm"><b>3. Run</b> · the queue skips runs whose results already exist with the same config; move EXPERIMENTS/results/{id} away first to train them again. A single run overwrites its saved files. With the same software and 1 thread, the same seed gives the same predictions.</Text>
        </Stack>
        <Cmd label="Every run of this setting" cmd={all} />
        <Cmd label="One run (one person, one seed)" cmd={one} />
        <Cmd label="Then update this page" cmd={update} />
      </Stack>
    </SectionCard>
  );
}

interface NetRow { id: string; title: string; params: number; change: number; lo: number; hi: number; better: number; n: number; loss: number; mae: number; r2: number }
interface PersonValRow { subj: string; loss: number; lo: number; hi: number; base: number }

/** hyperparameter tuning: the session-1 validation loss that chose the network, every finalist and every person */
function ValidationLoss({ pick }: { pick: string }) {
  const V = TUNING_VAL;
  const nets: NetRow[] = useMemo(() => Object.entries(V.networks).map(([id, n]) => {
    const ppl = runsOf(id);
    return { id, title: entry(id)?.title ?? id, params: n.n_params, change: Math.exp(n.mean_log_ratio) - 1,
      lo: Math.exp(n.mean_log_ratio - n.se) - 1, hi: Math.exp(n.mean_log_ratio + n.se) - 1, better: n.better_people, n: n.n_people,
      loss: median(Object.values(n.median)),
      mae: median(ppl.map((p) => personScore(p, 'mae'))), r2: median(ppl.map((p) => personScore(p, 'r2'))) };
  }).sort((a, b) => a.change - b.change || a.params - b.params), [V]);
  const cur = V.networks[pick], base = V.networks[V.baseline];
  const people: PersonValRow[] = cur ? Object.keys(cur.median).map((subj) => {
    const s = Object.values(cur.seeds[subj] ?? {});
    return { subj, loss: cur.median[subj], lo: Math.min(...s), hi: Math.max(...s), base: base.median[subj] };
  }) : [];
  const pc = (v: number) => `${v > 0 ? '+' : ''}${(100 * v).toFixed(1)} %`;
  const mark = (r: NetRow) => [r.id === V.best ? 'best' : '', r.id === V.chosen ? 'chosen' : ''].filter(Boolean).join(', ');
  const netCols: Column<NetRow>[] = [
    { key: 'title', label: 'Network', render: (r) => (r.id === V.chosen ? <b>{r.title}</b> : r.title) },
    { key: 'params', label: 'Weights', align: 'right', render: (r) => r.params.toLocaleString('en-US') },
    { key: 'change', label: "Validation loss vs last semester's network", align: 'right',
      render: (r) => (r.id === V.baseline ? 'reference' : `${pc(r.change)} [${pc(r.lo)}, ${pc(r.hi)}]`) },
    { key: 'better', label: 'People lower', align: 'right', render: (r) => (r.id === V.baseline ? '–' : `${r.better}/${r.n}`) },
    { key: 'mark', label: 'Rule', render: (r) => mark(r) },
    { key: 'loss', label: `Median loss (${V.units})`, align: 'right', render: (r) => fmt(r.loss, 3) },
    { key: 'mae', label: 'Test MAE (Nm)', align: 'right', render: (r) => fmt(r.mae) },
    { key: 'r2', label: 'Test R²', align: 'right', render: (r) => fmt(r.r2, 3) },
  ];
  const personCols: Column<PersonValRow>[] = [
    { key: 'subj', label: 'Person', render: (r) => r.subj },
    { key: 'loss', label: `This network (${V.units})`, align: 'right', render: (r) => `${fmt(r.loss, 3)} [${fmt(r.lo, 3)}, ${fmt(r.hi, 3)}]` },
    { key: 'base', label: `Last semester's network size (${V.units})`, align: 'right', render: (r) => fmt(r.base, 3) },
  ];
  const seeds = V.seeds_per_person.join('–');
  const ch = nets.find((r) => r.id === V.chosen), bs = nets.find((r) => r.id === V.baseline);
  return (
    <SectionCard testId="study-validation" title="Validation loss (session 1), used to choose the network"
      description={`Mean-squared error on the session-1 validation split (the last 20 % of each session-1 trial, after a gap), at each run's best epoch, ${seeds} seeds per person. Rule: per person, the log of (this network's median loss ÷ last semester's network's); the mean over people ranks the networks (table order; shown as a % change, [mean ± one standard error]). Best = lowest; chosen = the network with the fewest weights within one standard error of the best. Session 2 was not used.`}>
      <Stack gap="md">
        <DataTable columns={netCols} rows={nets} rowKey={(r) => r.id} rowClass={(r) => (r.id === pick ? 'selected-row' : undefined)} dense minWidth={1040} />
        {ch && bs && (
          <Note>
            Validation loss of the chosen network: {pc(ch.change)} against last semester's network size. On session 2 (not used to choose), median test MAE
            {' '}{fmt(ch.mae)} vs {fmt(bs.mae)} Nm ({pc(ch.mae / bs.mae - 1)}) and test R² {fmt(ch.r2, 3)} vs {fmt(bs.r2, 3)}. Validation here is the end of each
            session-1 trial, so it does not measure the change between sessions.
          </Note>
        )}
        {cur && (
          <div>
            <Text size="sm" fw={600} mb={4}>Per person: {entry(pick)?.title}</Text>
            <DataTable columns={personCols} rows={people} rowKey={(r) => r.subj} dense minWidth={560} />
            <Text size="xs" c="dimmed" mt={4}>This network: median over seeds [lowest, highest seed].</Text>
          </div>
        )}
      </Stack>
    </SectionCard>
  );
}

interface Row { subj: string; mae: number; nmae: number; ratio: number; better?: string; r2: number; worst: number; peak: number; maeCI: MedianCI | null; r2CI: MedianCI | null }

/** one row per person: MAE (raw and ÷ peak), against the reference, R² over positions and at the worst one, peak torque */
function Summary({ id, people, title }: { id: string; people: PersonRuns[]; title: string }) {
  const ref = referenceOf(id);
  const refRuns = useMemo(() => (ref ? runsOf(ref) : []), [ref]);
  const rows: Row[] = people.map((p) => {
    const r = refRuns.find((x) => x.subj === p.subj);
    const mae = personScore(p, 'mae');
    return { subj: p.subj, mae, nmae: mae / activePeak(p.subj), ratio: r ? maeRatio(p, r).ratio : NaN, r2: personScore(p, 'r2'), worst: worstR2(p),
      peak: Math.max(...p.peaks.filter(Number.isFinite)), maeCI: scoreCI(p, 'mae'), r2CI: scoreCI(p, 'r2') };
  });
  const ratios = rows.map((r) => r.ratio).filter(Number.isFinite);
  rows.push({ subj: 'Median', mae: median(rows.map((r) => r.mae)), nmae: median(rows.map((r) => r.nmae)), ratio: median(ratios),
    better: ratios.length ? `${ratios.filter((v) => v < 1).length}/${ratios.length} lower` : undefined,
    r2: median(rows.map((r) => r.r2)), worst: median(rows.map((r) => r.worst)), peak: median(rows.map((r) => r.peak)), maeCI: null, r2CI: null });
  const refName = ref ? entry(ref).title : '';
  const cols: Column<Row>[] = [
    { key: 'subj', label: 'Person', render: (r) => (r.subj === 'Median' ? <b>Median</b> : r.subj) },
    { key: 'mae', label: 'MAE (Nm)', align: 'right', render: (r) => withCI(r.mae, r.maeCI, 2) },
    { key: 'nmae', label: 'MAE ÷ peak', align: 'right', render: (r) => (Number.isFinite(r.nmae) ? `${(100 * r.nmae).toFixed(1)} %` : '–') },
    { key: 'ratio', label: 'MAE vs reference', align: 'right',
      render: (r) => (!ref ? (r.subj === 'Median' ? '' : 'reference') : Number.isFinite(r.ratio) ? `${r.ratio.toFixed(2)}×${r.better ? ` (${r.better})` : ''}` : '–') },
    { key: 'r2', label: 'R², median of positions', align: 'right', render: (r) => withCI(r.r2, r.r2CI, 2) },
    { key: 'worst', label: 'R², worst position', align: 'right', render: (r) => fmt(r.worst) },
    { key: 'peak', label: 'Peak torque (Nm)', align: 'right', render: (r) => f1(r.peak) },
  ];
  return (
    <SectionCard testId="study-summary" title={title}
      description={<>
        Per person, the median over seeds. MAE: mean over the 8 positions. MAE ÷ peak: divided by the person's largest measured active torque
        (for every setting, also total torque), so people with small and large torques can be compared.
        {ref ? <> MAE vs reference: this setting's MAE ÷ that of <b>{refName}</b>, on the seeds both have; below 1 = lower error.</> : ' This is the reference the other settings are compared with.'}
        {' '}R²: within each position (one trial at one angle); the median of the 8 positions hides the bad ones, so the worst position is shown too
        (per seed the lowest R², then the median over seeds). [ ]: interval for the median over seeds (about 95 %; 94 % with 5 seeds; none with 1 seed);
        it shows training randomness only.
      </>}>
      <DataTable columns={cols} rows={rows} rowKey={(r) => r.subj} dense minWidth={900} />
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
      description={`Setting: ${entry(id).title}. For each person, the 2 positions where this setting's median R² over seeds is lowest, so they change with the setting. Best and worst run: the seeds with the highest and the lowest R² at that position. Negative angles are plantarflexed.`}>
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
    <SectionCard testId="study-peaks" title="Peak torque per position (Nm)" description="Largest absolute measured torque of the predicted target in each trial.">
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
