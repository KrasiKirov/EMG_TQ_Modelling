import { useMemo } from 'react';
import { Accordion, Badge, Group, Stack, Text } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { DS } from '../dataset';
import { experimentsTable, familiesTable, round2Table, ROUND2_STUDY_ORDER, type ExperimentRow, type GroupRow, type PeakRange } from '../lib/aggregate';
import { dirLabel, fmt, listJoin, signed } from '../lib/format';
import { deltaVerdict } from '../lib/metrics';
import { DEFAULT_B_ID, exactExperiment, paramShort, REFERENCE_ID, sameParams, shortId } from '../lib/params';
import { useView } from '../state';
import { DataTable, Delta, EmptyState, MedianCell, Note, SectionCard, Swatch, type Column } from '../components/common';

export function ExperimentsTab() {
  return (
    <Stack gap="md">
      <ExperimentsCard />
      <Round2Card />
      <FamiliesCard />
    </Stack>
  );
}

function runsText(n: number, lo: number, hi: number): string {
  if (!n) return '–';
  return `${lo === hi ? lo : `${lo}–${hi}`} runs`;
}

function peopleText(persons: string[]): string {
  if (!persons.length) return 'not trained';
  return persons.length === DS.subjects.length ? `all ${persons.length}` : listJoin(persons);
}

function peakText(p: PeakRange): string {
  if (!Number.isFinite(p.median)) return '';
  return p.min === p.max ? `peak ${fmt(p.min, 1)} Nm` : `peak ${fmt(p.min, 1)}–${fmt(p.max, 1)} Nm`;
}

function ModelMark({ which }: { which: 'A' | 'B' }) {
  return (
    <Group gap={4} wrap="nowrap" style={{ flex: 'none' }} title={`current model ${which}`}>
      <Swatch which={which} size={9} />
      <Text size="xs" fw={700}>
        {which}
      </Text>
    </Group>
  );
}

function ExperimentsCard() {
  const { s, setModel } = useView();
  const mobile = !!useMediaQuery('(max-width: 48em)');
  const rows = useMemo(() => experimentsTable(DS, s.dir, s.floor), [s.dir, s.floor]);
  const hasMerged = rows.some((r) => r.synthetic);
  const other = s.dir === 'S1toS2' ? 'S2toS1' : 'S1toS2';
  // the merged "ours" row is what the Overview shows for the ours settings; mark it instead of the plain D01 row
  const marks = (r: ExperimentRow, p: typeof s.A) => {
    if (r.synthetic) return sameParams(r.params, p);
    if (hasMerged && r.id === DEFAULT_B_ID) return false;
    return exactExperiment(DS, p) === r.id;
  };
  const columns: Column<ExperimentRow>[] = [
    {
      key: 'name',
      label: mobile ? 'Experiment' : 'Experiment · changes vs last semester',
      render: (r) => (
        <Stack gap={2} style={{ minWidth: mobile ? 150 : 250, maxWidth: mobile ? 170 : 360, whiteSpace: 'normal' }}>
          <Group gap={6} wrap="nowrap" align="flex-start">
            <Text size="sm" fw={600} lh={1.35}>
              {r.name}
            </Text>
            {marks(r, s.A) && <ModelMark which="A" />}
            {marks(r, s.B) && <ModelMark which="B" />}
          </Group>
          <Text size="xs" c="dimmed" ff="monospace" style={{ overflowWrap: 'anywhere' }}>
            {r.synthetic ? r.used.map((u) => `${shortId(u.exp)}: ${listJoin(u.subjects)}`).join(' · ') : r.id}
          </Text>
          {!mobile && r.diffs.length > 0 && (
            <Group gap={4} mt={4}>
              {r.diffs.map((k) => (
                <Badge key={k} size="sm" variant="light" color="slate" tt="none" fw={500}>
                  {paramShort(k, r.params[k])}
                </Badge>
              ))}
            </Group>
          )}
          {r.synthetic && (
            <Text size="xs" c="dimmed">
              Same settings for everyone; C01 = D01 with the original passive table, which is identical except for JM. This is the Overview&apos;s “ours”.
            </Text>
          )}
          {!r.synthetic && r.twins.length > 0 && r.persons.length > 0 && (
            <Text size="xs" c="dimmed">
              Twin of {r.twins.map(shortId).join(', ')}: only the passive table differs (identical except for JM).
            </Text>
          )}
        </Stack>
      ),
      sort: (r) => r.order,
    },
    {
      key: 'people',
      label: 'People · runs',
      render: (r) => (
        <Stack gap={0} style={{ minWidth: 120 }}>
          <Text size="sm" style={{ whiteSpace: 'nowrap' }}>
            {peopleText(r.persons)}
          </Text>
          <Text size="xs" c="dimmed" className="num" style={{ whiteSpace: 'nowrap' }}>
            {r.persons.length ? `${runsText(r.persons.length, r.runsMin, r.runsMax)} · ${r.matched} matched` : ''}
          </Text>
          {r.persons.length > 0 && r.persons.length < DS.subjects.length && (
            <Text size="xs" c="dimmed" style={{ whiteSpace: 'nowrap' }}>
              {r.persons.length} of {DS.subjects.length} people
            </Text>
          )}
          <Text size="xs" c="dimmed" style={{ whiteSpace: 'nowrap' }}>
            {dirLabel(other)}: {r.otherPersons.length ? `${r.otherPersons.length} people` : 'not run'}
          </Text>
        </Stack>
      ),
      sort: (r) => r.persons.length * 100 + r.runsMax,
    },
    {
      key: 'r2',
      label: 'Toes-down R²',
      render: (r) => (
        <MedianCell
          s={r.r2}
          bold
          testId={`exp-r2-${r.key}`}
          sub={
            <Text size="xs" c="dimmed" className="num">
              ref {fmt(r.refToesR2)}
            </Text>
          }
        />
      ),
      sort: (r) => r.toesR2,
      align: 'right',
    },
    {
      key: 'dr2',
      label: 'Δ R²',
      render: (r) => <Delta value={signed(r.dR2)} verdict={r.id === REFERENCE_ID ? 0 : deltaVerdict('r2', r.dR2)} testId={`exp-dr2-${r.key}`} />,
      sort: (r) => r.dR2,
      align: 'right',
    },
    {
      key: 'mae',
      label: 'Toes-down MAE',
      render: (r) => (
        <MedianCell
          s={r.mae}
          bold
          unit=" Nm"
          testId={`exp-mae-${r.key}`}
          sub={
            <>
              <Text size="xs" c="dimmed" className="num">
                ref {fmt(r.refToesMae)}
              </Text>
              <Text size="xs" c="dimmed" className="num" style={{ whiteSpace: 'nowrap' }}>
                {peakText(r.peak)}
              </Text>
            </>
          }
        />
      ),
      sort: (r) => r.toesMae,
      align: 'right',
    },
    {
      key: 'dmae',
      label: 'Δ MAE',
      render: (r) => <Delta value={signed(r.dMae)} verdict={r.id === REFERENCE_ID ? 0 : deltaVerdict('mae', r.dMae, r.peak.median)} testId={`exp-dmae-${r.key}`} />,
      sort: (r) => r.dMae,
      align: 'right',
    },
  ];
  return (
    <SectionCard
      title="All experiments"
      description={`${dirLabel(s.dir, true)} · median over runs at each toes-down position, then over those positions, for the people each row was trained for · MAE in Nm · click a row to load it as model B`}
      testId="experiments"
    >
      <DataTable
        // phones: the numbers right after the (sticky) name, the people column last
        columns={mobile ? [...columns.filter((c) => c.key !== 'people'), ...columns.filter((c) => c.key === 'people')] : columns}
        rows={rows}
        rowKey={(r) => r.key}
        rowTestId={(r) => `exp-row-${r.key}`}
        onRowClick={(r) => setModel('B', DS.experiments[r.id].params)}
        rowClass={(r) => [marks(r, s.B) ? 'selected-row' : '', r.persons.length ? '' : 'dim'].filter(Boolean).join(' ') || undefined}
        minWidth={mobile ? 660 : 900}
        stickyFirst={mobile}
        caption="experiments"
      />
      <Stack gap={6} mt="sm">
        <Note>
          Each row is compared with last semester re-run in our framework ({shortId(REFERENCE_ID)}) on exactly the same people and seeds (ref, matched runs:
          e.g. seeds 0–2 when the reference has 3). Rows trained for fewer people are like for like within the row but not with each other: compare
          rows only when the People column matches. Δ = experiment − reference; green / red only when the difference is at least 0.02 R² or 0.05 Nm (2 %
          of the peak when larger). Hover a median for its 5–95 % and 1–99 % range.
          {s.floor ? ' R² below 0 counts as 0.' : ' R² shown raw.'}
        </Note>
      </Stack>
    </SectionCard>
  );
}

function groupColumns(withStudy: boolean, withPeak: boolean): Column<GroupRow>[] {
  const cols: Column<GroupRow>[] = [];
  cols.push(
    {
      key: 'id',
      label: withStudy ? 'Experiment' : 'Network',
      render: (r) => (
        <Stack gap={0} style={{ minWidth: 180, maxWidth: 380, whiteSpace: 'normal' }}>
          <Text size="sm" fw={600}>
            {withStudy ? r.id : r.name}
          </Text>
          <Text size="xs" c="dimmed">
            {withStudy ? r.name : r.id}
          </Text>
        </Stack>
      ),
      sort: (r) => r.id,
    },
    {
      key: 'persons',
      label: 'People',
      render: (r) => (
        <Text size="sm" style={{ whiteSpace: 'nowrap' }}>
          {peopleText(r.persons)}
        </Text>
      ),
      sort: (r) => r.persons.length,
    },
    { key: 'runs', label: 'Runs', render: (r) => r.runs, sort: (r) => r.runs, align: 'right', title: 'person × seed combinations' },
    { key: 'tr2', label: 'Toes-down R²', render: (r) => <MedianCell s={r.toes.r2} bold />, sort: (r) => r.toesR2, align: 'right' },
    { key: 'or2', label: 'Other R²', render: (r) => <MedianCell s={r.other.r2} />, sort: (r) => r.otherR2, align: 'right' },
    {
      key: 'tmae',
      label: 'Toes-down MAE',
      render: (r) => (
        <MedianCell
          s={r.toes.mae}
          bold
          unit=" Nm"
          sub={
            withPeak ? (
              <Text size="xs" c="dimmed" className="num" style={{ whiteSpace: 'nowrap' }}>
                {peakText(r.toesPeak)}
              </Text>
            ) : undefined
          }
        />
      ),
      sort: (r) => r.toesMae,
      align: 'right',
    },
    {
      key: 'omae',
      label: 'Other MAE',
      render: (r) => (
        <MedianCell
          s={r.other.mae}
          unit=" Nm"
          sub={
            withPeak ? (
              <Text size="xs" c="dimmed" className="num" style={{ whiteSpace: 'nowrap' }}>
                {peakText(r.otherPeak)}
              </Text>
            ) : undefined
          }
        />
      ),
      sort: (r) => r.otherMae,
      align: 'right',
    },
  );
  return cols;
}

const STUDY_TEXT: Record<string, string> = {
  screen: 'first screen of many ideas, mostly YES, JM, EG',
  confirm: 'the promising ideas on the other people',
  ablation: 'one part of the final pipeline removed at a time',
  passive: 'ways to estimate the passive torque',
  final_check: 'final pipeline on everyone',
};

function Round2Card() {
  const { s } = useView();
  const mobile = !!useMediaQuery('(max-width: 48em)');
  const rows = useMemo(() => round2Table(DS, s.dir, s.floor), [s.dir, s.floor]);
  const studies = ROUND2_STUDY_ORDER.filter((st) => rows.some((r) => r.study === st)).concat(
    Array.from(new Set(rows.map((r) => r.study))).filter((st) => !ROUND2_STUDY_ORDER.includes(st)),
  );
  const cols = useMemo(() => groupColumns(true, true), []);
  return (
    <SectionCard
      title="Earlier exploration (round 2)"
      description={`${dirLabel(s.dir, true)} · median over runs at each position, then over positions · MAE in Nm · grouped by study`}
      testId="round2"
    >
      {rows.length ? (
        <Accordion multiple defaultValue={studies.filter((st) => st !== 'screen')} variant="separated" radius="md" chevronPosition="left">
          {studies.map((st) => {
            const rs = rows.filter((r) => r.study === st);
            const people = Array.from(new Set(rs.flatMap((r) => r.persons)));
            return (
              <Accordion.Item key={st} value={st}>
                <Accordion.Control data-testid={`r2-study-${st}`}>
                  <Group gap="sm" wrap="wrap">
                    <Text size="sm" fw={600}>
                      {st.replace('_', ' ')}
                    </Text>
                    <Text size="xs" c="dimmed">
                      {rs.length} {rs.length === 1 ? 'experiment' : 'experiments'} · {peopleText(people)}
                      {STUDY_TEXT[st] ? ` · ${STUDY_TEXT[st]}` : ''}
                    </Text>
                  </Group>
                </Accordion.Control>
                <Accordion.Panel>
                  <DataTable
                    columns={cols}
                    rows={rs}
                    rowKey={(r) => r.key}
                    rowTestId={(r) => `r2-row-${r.key}`}
                    minWidth={mobile ? 680 : 820}
                    stickyFirst={mobile}
                    dense
                    caption={`round 2 ${st}`}
                  />
                </Accordion.Panel>
              </Accordion.Item>
            );
          })}
        </Accordion>
      ) : (
        <EmptyState compact title="No round-2 rows in this data file" />
      )}
      <Stack mt="sm">
        <Note>
          Caveat: an earlier exploration, mostly on 3 people (YES, JM, EG), with faster training settings (batch 64) and different pipelines; re-scored on
          the same answer key (so the peak torques are the same). Use it for direction, not for exact numbers, and do not compare it 1:1 with the table
          above. Hover a median for its 5–95 % and 1–99 % range.
        </Note>
      </Stack>
    </SectionCard>
  );
}

function FamiliesCard() {
  const { s } = useView();
  const mobile = !!useMediaQuery('(max-width: 48em)');
  const rows = useMemo(() => familiesTable(DS, s.floor), [s.floor]);
  const cols = useMemo(() => groupColumns(false, false), []);
  return (
    <SectionCard title="Network types (report 1)" description="LSTM vs GRU vs bigger LSTM vs MLP vs TCN · median over runs at each position, then over positions · MAE in Nm" testId="families">
      {rows.length ? (
        <DataTable columns={cols} rows={rows} rowKey={(r) => r.key} rowTestId={(r) => `fam-row-${r.key}`} minWidth={720} stickyFirst={mobile} dense caption="network types" />
      ) : (
        <EmptyState compact title="No network-type rows in this data file" />
      )}
      <Stack mt="sm">
        <Note>
          Caveat: each run is scored on its own target (the report-1 answer key), so compare these rows only with each other, not with the tables above;
          for the same reason there is no common peak torque to show next to the MAE. Hover a median for its 5–95 % and 1–99 % range.
        </Note>
      </Stack>
    </SectionCard>
  );
}


