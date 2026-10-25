import { useMemo, useState } from 'react';
import { Alert, Group, SegmentedControl, SimpleGrid, Stack, Text } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { IconAlertTriangle, IconBulb } from '../icons';
import { DS } from '../dataset';
import { biasVariance, learningRows, LC_LAST, LC_OURS, runsLabel, scopeSubjects, type BiasVarianceRow, type LearningRow } from '../lib/aggregate';
import { dirLabel, fmt } from '../lib/format';
import { friendly, shortId } from '../lib/params';
import { useView } from '../state';
import { biasVarOption, BV_PARTS, learningOption } from '../charts/options';
import { EChart } from '../components/EChart';
import { DataTable, EmptyState, LineKey, Note, SectionCard, Swatch, type Column } from '../components/common';

export function TrainingTab() {
  const { s } = useView();
  return (
    <Stack gap="md">
      {s.dir !== 'S1toS2' && (
        <Alert variant="light" color="gray" radius="lg" icon={<IconAlertTriangle size={20} />} title="Shown for learn 1 → test 2 only" data-testid="training-dir-note">
          <Text size="sm">
            Learning curves and the error split were only recorded for learn session 1 → test session 2. The Direction switch ({dirLabel(s.dir)}) does
            not change this tab.
          </Text>
        </Alert>
      )}
      <Explainer />
      <LearningCurvesCard />
      <LearningTableCard />
      <BiasVarianceCard />
    </Stack>
  );
}

function Explainer() {
  return (
    <Alert variant="light" color="gray" radius="lg" icon={<IconBulb size={20} />} title="How to read the training curves" data-testid="training-explainer">
      <Stack gap={6}>
        <Text size="sm">
          Each model learns from the first 80 % of one session (<b>fit</b>), stops early when the last 20 % of that session (<b>validation</b>) stops
          improving, and is then scored on the other session (<b>test</b>, only watched, never used for training).
        </Text>
        <Text size="sm">
          <b>fit ≈ validation</b> → the network is not memorising within a session. <b>test ≫ validation</b> → the change between sessions (dataset
          shift: electrodes, skin, fatigue, EMG size) dominates the error, not over-fitting.
        </Text>
        <Text size="xs" c="dimmed">
          One run (seed 0) per person and model, learn session 1 → test session 2. Errors measured with dropout off. This tab shows fixed models
          (ours and last semester), not the Model A / B picks of the left panel.
        </Text>
      </Stack>
    </Alert>
  );
}

function LearningCurvesCard() {
  const { s, tokens } = useView();
  const rows = useMemo(() => learningRows(DS), []);
  const subjects = scopeSubjects(DS, s.subj);
  const compact = subjects.length > 1;
  if (!rows.length)
    return (
      <SectionCard title="Learning curves">
        <EmptyState title="No learning curves in this data file" hint="Run EXPERIMENTS/code/learning_curves.py, then rebuild the data." />
      </SectionCard>
    );
  return (
    <SectionCard
      title="Learning curves: MAE (Nm) per epoch"
      description={`ours = ${friendly(LC_OURS)} (${shortId(LC_OURS)}) · last semester = ${friendly(LC_LAST)} (${shortId(LC_LAST)}) · vertical line = epoch kept for ours`}
      testId="learning-curves"
    >
      <Stack gap="md">
        <div className="legend-row">
          <span>
            <LineKey color={tokens.fit} /> fit (first 80 % of training session)
          </span>
          <span>
            <LineKey color={tokens.val} /> validation (last 20 %)
          </span>
          <span>
            <LineKey color={tokens.test} /> test (other session)
          </span>
          <span>
            <LineKey color="var(--text-2)" /> ours
          </span>
          <span>
            <LineKey color="var(--text-2)" dashed /> last semester
          </span>
        </div>
        <SimpleGrid cols={compact ? { base: 1, md: 2, lg: 3 } : 1} spacing="lg" verticalSpacing="lg">
          {subjects.map((subj) => {
            const ours = rows.find((r) => r.subj === subj && r.model === 'ours') ?? null;
            const last = rows.find((r) => r.subj === subj && r.model === 'last') ?? null;
            return <CurveChart key={subj} subj={subj} ours={ours} last={last} compact={compact} />;
          })}
        </SimpleGrid>
      </Stack>
    </SectionCard>
  );
}

function CurveChart({ subj, ours, last, compact }: { subj: string; ours: LearningRow | null; last: LearningRow | null; compact: boolean }) {
  const { tokens } = useView();
  const option = useMemo(() => {
    // clip extreme early epochs so the interesting part is visible
    const vals = [ours, last].flatMap((r) => (r ? [...r.curve.fit, ...r.curve.val, ...r.curve.test] : [])).sort((a, b) => a - b);
    const p98 = vals.length ? vals[Math.floor(0.98 * (vals.length - 1))] : undefined;
    return learningOption(subj, ours?.curve ?? null, last?.curve ?? null, ours?.best ?? null, tokens, compact, p98 ? Math.ceil(p98 * 11) / 10 : undefined);
  }, [subj, ours, last, tokens, compact]);
  return (
    <Stack gap={6} data-testid={`lc-${subj}`}>
      <Group justify="space-between" wrap="nowrap">
        <Text fw={650}>{subj}</Text>
        <Text size="xs" c="dimmed" className="num">
          test MAE: last {fmt(last?.maeTest)} → ours {fmt(ours?.maeTest)} Nm
        </Text>
      </Group>
      {ours || last ? (
        <EChart option={option} height={compact ? 230 : 380} label={`Learning curves for ${subj}`} />
      ) : (
        <EmptyState compact title="No curve for this person" />
      )}
    </Stack>
  );
}

function LearningTableCard() {
  const { s } = useView();
  const subjects = scopeSubjects(DS, s.subj);
  const rows = useMemo(() => learningRows(DS), []).filter((r) => subjects.includes(r.subj));
  const columns: Column<LearningRow>[] = [
    { key: 'subj', label: 'Person', render: (r) => <Text fw={600} size="sm">{r.subj}</Text>, sort: (r) => r.subj },
    {
      key: 'model',
      label: 'Model',
      render: (r) => (
        <Group gap={6} wrap="nowrap">
          <LineKey color="var(--text-2)" dashed={r.model === 'last'} width={14} />
          <span style={{ whiteSpace: 'nowrap' }}>{r.model === 'ours' ? 'ours' : 'last semester'}</span>
        </Group>
      ),
      sort: (r) => r.model,
    },
    { key: 'epochs', label: 'Epochs (kept)', render: (r) => `${r.epochs} (${r.best})`, sort: (r) => r.best, align: 'right', title: 'epochs trained (epoch whose weights were kept)' },
    { key: 'fit', label: 'MAE fit', render: (r) => fmt(r.maeFit), sort: (r) => r.maeFit, align: 'right' },
    { key: 'val', label: 'MAE val', render: (r) => fmt(r.maeVal), sort: (r) => r.maeVal, align: 'right' },
    { key: 'test', label: 'MAE test', render: (r) => <b>{fmt(r.maeTest)}</b>, sort: (r) => r.maeTest, align: 'right' },
    { key: 'vf', label: 'val / fit', render: (r) => fmt(r.valFit), sort: (r) => r.valFit, align: 'right' },
    { key: 'tv', label: 'test / val', render: (r) => fmt(r.testVal), sort: (r) => r.testVal, align: 'right' },
  ];
  return (
    <SectionCard title="At the kept epoch" description="MAE in Nm at the epoch whose weights were kept (early stopping on validation)" testId="learning-table">
      <DataTable columns={columns} rows={rows} rowKey={(r) => `${r.subj}-${r.model}`} rowTestId={(r) => `lc-row-${r.subj}-${r.model}`} minWidth={640} dense caption="learning curve summary" />
      <Stack mt="sm">
        <Note>val / fit near 1 = no memorising within a session. test / val well above 1 = the session change costs more than anything learnt.</Note>
      </Stack>
    </SectionCard>
  );
}

function BiasVarianceCard() {
  const { s, tokens } = useView();
  const mobile = useMediaQuery('(max-width: 48em)');
  const [toes, setToes] = useState<'1' | '0'>('1');
  const [scale, setScale] = useState<'share' | 'abs'>('share');
  const subjects = scopeSubjects(DS, s.subj);
  const rows = useMemo(() => biasVariance(DS, toes === '1' ? 1 : 0, s.floor), [toes, s.floor]).filter((r) => subjects.includes(r.subj));
  const option = useMemo(() => biasVarOption(rows, tokens, !!mobile, scale === 'abs'), [rows, tokens, mobile, scale]);
  const nRuns = runsLabel(rows);
  const modelCell = (r: BiasVarianceRow) => <span style={{ whiteSpace: 'nowrap' }}>{r.model === 'ours' ? 'ours' : 'last semester'}</span>;
  const shareCols: Column<BiasVarianceRow>[] = [
    { key: 'subj', label: 'Person', render: (r) => <Text fw={600} size="sm">{r.subj}</Text>, sort: (r) => r.subj },
    { key: 'model', label: 'Model', render: modelCell, sort: (r) => r.model },
    { key: 'mse', label: 'MSE of one run (Nm²)', render: (r) => fmt(r.mse, 3), sort: (r) => r.mse, align: 'right' },
    { key: 'off', label: 'offset', render: (r) => `${fmt(r.offsetShare * 100, 0)} %`, sort: (r) => r.offsetShare, align: 'right' },
    { key: 'sys', label: 'other systematic', render: (r) => `${fmt(r.otherSysShare * 100, 0)} %`, sort: (r) => r.otherSysShare, align: 'right' },
    { key: 'rr', label: 'run-to-run', render: (r) => `${fmt(r.runShare * 100, 0)} %`, sort: (r) => r.runShare, align: 'right' },
  ];
  const ensCols: Column<BiasVarianceRow>[] = [
    { key: 'subj', label: 'Person', render: (r) => <Text fw={600} size="sm">{r.subj}</Text>, sort: (r) => r.subj },
    { key: 'model', label: 'Model', render: modelCell, sort: (r) => r.model },
    { key: 'maerun', label: 'MAE one run', render: (r) => fmt(r.runMae), sort: (r) => r.runMae, align: 'right', title: 'median over the runs and positions (Nm)' },
    { key: 'maeens', label: 'MAE average of runs', render: (r) => <b>{fmt(r.ensMae)}</b>, sort: (r) => r.ensMae, align: 'right', title: 'MAE of the mean prediction, median over positions (Nm)' },
    { key: 'r2run', label: 'R² one run', render: (r) => fmt(r.runR2), sort: (r) => r.runR2, align: 'right' },
    { key: 'r2ens', label: 'R² average of runs', render: (r) => <b>{fmt(r.ensR2)}</b>, sort: (r) => r.ensR2, align: 'right' },
  ];
  if (!DS.raw.bias_variance)
    return (
      <SectionCard title="Where the error comes from">
        <EmptyState title="No bias / variance table in this data file" />
      </SectionCard>
    );
  return (
    <SectionCard
      title="Where the error comes from"
      description={`one run's squared error split into parts · learn 1 → test 2 · ${nRuns} runs per person · last semester = original code (X00), ours = D01 (C01 for YES, EG)`}
      actions={
        <Group gap="xs" wrap="wrap">
          <SegmentedControl
            size="xs"
            value={toes}
            onChange={(v) => setToes(v as '1' | '0')}
            data={[
              { value: '1', label: 'toes-down positions' },
              { value: '0', label: 'other 6 positions' },
            ]}
            data-testid="bv-toggle"
          />
          <SegmentedControl
            size="xs"
            value={scale}
            onChange={(v) => setScale(v as 'share' | 'abs')}
            data={[
              { value: 'share', label: 'share (%)' },
              { value: 'abs', label: 'size (Nm²)' },
            ]}
            data-testid="bv-scale"
          />
        </Group>
      }
      testId="bias-variance"
    >
      <Stack gap="md">
        <div className="legend-row">
          {BV_PARTS.map((p) => (
            <span key={p.key}>
              <Swatch color={tokens[p.token]} /> {p.name}
            </span>
          ))}
        </div>
        <EChart option={option} height={Math.max(220, rows.length * 34 + 40)} label="Error of one run split into offset, other systematic and run-to-run" />
        <Note>
          {scale === 'share'
            ? 'Bars show shares of each model’s own error, so a small and a large error both fill the bar; switch to size (Nm²) to compare the amounts. '
            : 'Bars show the amount of squared error (Nm²); the whole bar is the mean squared error of one run. '}
          Offset = the whole predicted curve sits too high or too low. Other systematic = wrong shape that every run repeats. Run-to-run = the part
          that changes with the random seed, the only part that averaging runs can remove. If run-to-run is small, averaging (an ensemble) cannot help
          much.
        </Note>
        <SimpleGrid cols={1} spacing="lg">
          <Stack gap={6}>
            <Text size="sm" fw={600}>
              Share of the error
            </Text>
            <DataTable
              columns={shareCols}
              rows={rows}
              rowKey={(r) => `${r.subj}-${r.model}`}
              rowTestId={(r) => `bv-row-${r.subj}-${r.model}`}
              minWidth={520}
              dense
              caption="share of the error"
            />
          </Stack>
          <Stack gap={6}>
            <Text size="sm" fw={600}>
              One run vs the average of runs
            </Text>
            <DataTable
              columns={ensCols}
              rows={rows}
              rowKey={(r) => `${r.subj}-${r.model}`}
              rowTestId={(r) => `bv-ens-${r.subj}-${r.model}`}
              minWidth={520}
              dense
              caption="one run vs average of runs"
            />
          </Stack>
        </SimpleGrid>
        <Text size="xs" c="dimmed">
          Average of runs = score of the mean prediction of the same {nRuns} runs (median over this group&apos;s positions); one run = median over those
          runs and positions. MAE in Nm. R² below 0 follows the display rule.
        </Text>
      </Stack>
    </SectionCard>
  );
}
