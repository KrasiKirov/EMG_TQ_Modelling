import { useMemo, useState, type CSSProperties } from 'react';
import { AppShell, Badge, Button, Card, Collapse, Divider, Group, ScrollArea, SegmentedControl, Select, Stack, Switch, Text, Title, UnstyledButton } from '@mantine/core';
import { IconAlertTriangle, IconArrowsExchange, IconCheck, IconChevronRight, IconCircleX } from '../icons';
import { DS } from '../dataset';
import { scopeSubjects } from '../lib/aggregate';
import { METRIC_KEYS, METRICS } from '../lib/metrics';
import {
  coerceParam,
  exactExperiment,
  friendly,
  modelStatus,
  modelTitle,
  PARAM_KEYS,
  PARAM_NAMES,
  paramLabel,
  paramMedium,
  paramOptions,
  PRESET_GROUPS,
  shortId,
  DEFAULT_A_ID,
  type ModelStatus,
} from '../lib/params';
import { listJoin } from '../lib/format';
import type { Dir, MetricKey } from '../lib/types';
import { useView } from '../state';
import { Swatch } from './common';

const SEG_STYLES = { control: { flex: '1 1 auto' }, label: { paddingInline: 8 } };

export function Sidebar({ onNavigate }: { onNavigate: () => void }) {
  const { swap } = useView();
  return (
    <>
      <AppShell.Section grow component={ScrollArea} type="scroll" scrollbarSize={8}>
        <Stack gap="sm" p="md">
          <ViewControls />
          <Divider label="Models to compare" labelPosition="left" />
          <ModelCard which="A" />
          <ModelCard which="B" />
          <Button variant="subtle" color="gray" size="xs" fullWidth leftSection={<IconArrowsExchange size={14} />} onClick={swap} data-testid="swap">
            Swap A ⇄ B
          </Button>
<details className="disclosure"><summary>How model matching works</summary><Text size="xs" c="dimmed">Choose a preset or adjust its settings. Unavailable combinations are marked as not trained. Passive tables differ only for JM; equivalent runs may be used for other participants.</Text></details>
        </Stack>
      </AppShell.Section>
      <AppShell.Section p="md" hiddenFrom="md" style={{ borderTop: '1px solid var(--line)' }}>
        <Button fullWidth onClick={onNavigate}>
          Show results
        </Button>
      </AppShell.Section>
    </>
  );
}

function ViewControls() {
  const { s, setSubj, setDir, setMetric, set } = useView();
  return (
    <Stack gap="sm">
      <Title order={4} fz={14}>
        View
      </Title>
      <Stack gap={4}>
        <Text size="sm" fw={500}>
          Participant
        </Text>
        <Select size="sm" aria-label="Participant" value={s.subj} onChange={(v) => v && setSubj(v)} data={['All', ...DS.subjects].map((v) => ({ value: v, label: v === 'All' ? 'All participants' : v }))} data-testid="ctl-subj" />
      </Stack>
      <Stack gap={4}>
        <Text size="sm" fw={500}>
          Direction
        </Text>
        <SegmentedControl
          size="xs"
          fullWidth
          styles={SEG_STYLES}
          value={s.dir}
          onChange={(v) => setDir(v as Dir)}
          data={[
            { value: 'S1toS2', label: 'learn 1 → test 2' },
            { value: 'S2toS1', label: 'learn 2 → test 1' },
          ]}
          data-testid="ctl-dir"
        />
      </Stack>
      <Stack gap={4}>
        <Text size="sm" fw={500}>
          Metric
        </Text>
        <SegmentedControl
          size="xs"
          fullWidth
          styles={SEG_STYLES}
          value={s.metric}
          onChange={(v) => setMetric(v as MetricKey)}
          data={METRIC_KEYS.map((k) => ({ value: k, label: METRICS[k].label }))}
          data-testid="ctl-metric"
        />
      </Stack>
<details className="disclosure"><summary>Display options</summary>      <Switch
        checked={s.floor}
        onChange={(e) => set({ floor: e.currentTarget.checked })}
        label="Show R² below 0 as 0"
        description="Supervisor's display rule. Off = raw values (can be far below 0)."
        size="sm"
        data-testid="ctl-floor"
      /></details>
    </Stack>
  );
}

function presetData() {
  return PRESET_GROUPS.map((g) => ({
    group: g.group,
    items: g.ids.filter((id) => DS.experiments[id]).map((id) => ({ value: id, label: friendly(id) })),
  })).concat(
    // any experiment not listed above (future data) still shows up
    (() => {
      const listed = new Set(PRESET_GROUPS.flatMap((g) => g.ids));
      const rest = DS.expIds.filter((id) => !listed.has(id));
      return rest.length ? [{ group: 'Other', items: rest.map((id) => ({ value: id, label: friendly(id) })) }] : [];
    })(),
  );
}

function ModelCard({ which }: { which: 'A' | 'B' }) {
  const { s, setModel } = useView();
  const p = s[which];
  const st = useMemo(() => modelStatus(DS, p, scopeSubjects(DS, s.subj), s.dir), [p, s.subj, s.dir]);
  const exact = exactExperiment(DS, p);
  const base = DS.experiments[DEFAULT_A_ID]?.params;
  const data = useMemo(presetData, []);
  const accent = which === 'A' ? 'var(--c-a)' : 'var(--c-b)';
  const nChanged = base ? PARAM_KEYS.filter((k) => String(base[k]) !== String(p[k])).length : 0;
  // the single settings are folded away while a preset is picked; a custom combination shows them
  const [openState, setOpen] = useState<boolean | null>(null);
  const open = openState ?? !exact;
  return (
    <Card padding="md" className="model-card" style={{ '--accent': accent } as CSSProperties} data-testid={`model-${which}`}>
      <Stack gap="xs">
        <Group justify="space-between" wrap="nowrap">
          <Group gap={8} wrap="nowrap">
            <Swatch which={which} size={12} />
            <Title order={3}>Model {which}</Title>
          </Group>
          <StatusBadge st={st} />
        </Group>
        <Select
          label="Preset"
          size="sm"
          data={data}
          value={exact}
          placeholder="Custom combination"
          title={exact ? friendly(exact) : 'Custom combination'}
          onChange={(id) => id && DS.experiments[id] && setModel(which, DS.experiments[id].params)}
          renderOption={({ option, checked }) => (
            <Group gap={8} wrap="nowrap" justify="space-between" style={{ width: '100%' }}>
              <Stack gap={0} style={{ minWidth: 0 }}>
                <Text size="sm">{option.label}</Text>
                <Text size="xs" c="dimmed">
                  {option.value}
                </Text>
              </Stack>
              {checked && <IconCheck size={15} style={{ flex: 'none', color: accent }} />}
            </Group>
          )}
          maxDropdownHeight={360}
          comboboxProps={{ width: 320, position: 'bottom-start' }}
          data-testid={`preset-${which}`}
        />
        <UnstyledButton onClick={() => setOpen(!open)} aria-expanded={open} className="settings-toggle" data-testid={`settings-toggle-${which}`}>
          <Group gap={6} wrap="nowrap">
            <IconChevronRight size={14} style={{ transform: open ? 'rotate(90deg)' : undefined, transition: 'transform 120ms', flex: 'none' }} />
            <Text size="sm" fw={500}>
              Adjust settings
            </Text>
            <Text size="xs" c="dimmed">
              {nChanged === 0 ? 'same as last semester' : `${nChanged} differ${nChanged === 1 ? 's' : ''} from last semester`}
            </Text>
          </Group>
        </UnstyledButton>
        <Collapse in={open}>
          <div className="param-grid">
            {PARAM_KEYS.map((k) => {
              const changed = base && String(base[k]) !== String(p[k]);
              return (
                <ParamRow
                  key={k}
                  label={PARAM_NAMES[k]}
                  changed={!!changed}
                  value={String(p[k])}
                  data={paramOptions(DS, k).map((v) => ({ value: String(v), label: paramMedium(k, v) }))}
                  long={Object.fromEntries(paramOptions(DS, k).map((v) => [String(v), paramLabel(k, v)]))}
                  baseValue={base ? String(base[k]) : undefined}
                  onChange={(v) => setModel(which, { ...p, [k]: coerceParam(DS, k, v) })}
                  testId={`param-${which}-${k}`}
                  accent={accent}
                />
              );
            })}
          </div>
          <Text size="xs" c="dimmed" mt={8}>
            <span className="changed-dot inline" /> = differs from last semester (original code)
          </Text>
        </Collapse>
        <details className="disclosure" open={st.missing.length > 0 || undefined}><summary>Run details{st.missing.length > 0 ? ' · missing results' : ''}</summary><StatusBlock which={which} st={st} onFramework={String(p.code) === 'original' && st.missing.length > 0 ? () => setModel(which, { ...p, code: 'framework' }) : undefined} /></details>
      </Stack>
    </Card>
  );
}

function ParamRow({
  label,
  changed,
  value,
  data,
  long,
  baseValue,
  onChange,
  testId,
  accent,
}: {
  label: string;
  changed: boolean;
  value: string;
  data: { value: string; label: string }[];
  long: Record<string, string>;
  baseValue?: string;
  onChange: (v: string) => void;
  testId: string;
  accent: string;
}) {
  return (
    <>
      <label className="param-label" title={changed ? 'differs from last semester' : undefined}>
        {label}
        {changed && <span className="changed-dot" aria-label="differs from last semester" />}
      </label>
      <Select
        size="xs"
        data={data}
        value={value}
        onChange={(v) => v !== null && onChange(v)}
        aria-label={label}
        title={long[value]}
        data-testid={testId}
        comboboxProps={{ width: 290, position: 'bottom-end' }}
        styles={{ input: { fontSize: 13 } }}
        renderOption={({ option, checked }) => (
          <Group gap={8} wrap="nowrap" justify="space-between" style={{ width: '100%' }}>
            <Stack gap={0} style={{ minWidth: 0 }}>
              <Text size="sm">{long[option.value] ?? option.label}</Text>
              {option.value === baseValue && (
                <Text size="xs" c="dimmed">
                  last semester
                </Text>
              )}
            </Stack>
            {checked && <IconCheck size={15} style={{ flex: 'none', color: accent }} />}
          </Group>
        )}
      />
    </>
  );
}

function StatusBadge({ st }: { st: ModelStatus }) {
  if (!st.found.length)
    return (
      <Badge color="red" variant="light" leftSection={<IconCircleX size={12} />}>
        not trained
      </Badge>
    );
  if (st.missing.length)
    return (
      <Badge color="gray" variant="light" leftSection={<IconAlertTriangle size={12} />}>
        partly
      </Badge>
    );
  return (
    <Badge color="green" variant="light" leftSection={<IconCheck size={12} />}>
      trained
    </Badge>
  );
}

function StatusBlock({ which, st, onFramework }: { which: 'A' | 'B'; st: ModelStatus; onFramework?: () => void }) {
  const [open, setOpen] = useState(false);
  const hypId = st.exact ?? st.primary;
  const hyp = hypId ? DS.experiments[hypId]?.hypothesis : '';
  const runs = st.runsMin === st.runsMax ? `${st.runsMin}` : `${st.runsMin}–${st.runsMax}`;
  return (
    <Stack gap={6} data-testid={`status-${which}`}>
      {st.found.length ? (
        <>
          <Text size="sm" fw={600} lh={1.35}>
            {modelTitle(st)}
          </Text>
          <Text size="xs" c="dimmed" ff="monospace" style={{ overflowWrap: 'anywhere' }}>
            {st.exact ?? 'no experiment has exactly these settings'}
          </Text>
          <Text size="xs" c="dimmed">
            {runs} runs per person · {st.found.length === 1 ? st.found[0] : `${st.found.length} people`} in view
          </Text>
        </>
      ) : (
        <Text size="sm" fw={600} c="var(--c-bad)">
          Not trained for this person/direction.
        </Text>
      )}
      {st.substitutes.map((x) => (
        <Text key={x.exp} size="xs" c="dimmed" data-testid={`substitute-${which}`}>
          {st.exact ? (
            <>
              Shown with <b>{shortId(x.exp)}</b> ({friendly(x.exp)}) for {listJoin(x.subjects)}: the same model for {x.subjects.length === 1 ? 'this person' : 'them'}, because the
              original and clean passive tables are identical (they differ only for JM).
            </>
          ) : (
            <>
              Uses <b>{shortId(x.exp)}</b> ({friendly(x.exp)}) for {listJoin(x.subjects)}: same settings except the passive table, which is identical for
              everyone except JM.
            </>
          )}
        </Text>
      ))}
      {st.found.length > 0 && st.missing.length > 0 && (
        <Group gap={6} wrap="nowrap" align="flex-start">
          <IconAlertTriangle size={14} style={{ flex: 'none', marginTop: 2, color: 'var(--text-3)' }} />
          <Text size="xs">Not trained for {listJoin(st.missing)}.</Text>
        </Group>
      )}
      {st.hint && (
        <Text size="xs" c="dimmed">
          {st.hint}
        </Text>
      )}
      {onFramework && (
        <Button size="xs" variant="default" onClick={onFramework} style={{ alignSelf: 'flex-start' }} data-testid={`use-framework-${which}`}>
          Switch Code to our framework
        </Button>
      )}
      {hyp && (
        <>
          <UnstyledButton onClick={() => setOpen((o) => !o)} aria-expanded={open} style={{ fontSize: 13, color: 'var(--text-2)' }}>
            <Group gap={4} wrap="nowrap">
              <IconChevronRight size={14} style={{ transform: open ? 'rotate(90deg)' : undefined, transition: 'transform 120ms' }} />
              Why {hypId && hypId !== st.exact ? `${shortId(hypId)} was` : 'this was'} tried
            </Group>
          </UnstyledButton>
          <Collapse in={open}>
            <Text size="sm" c="dimmed" pl={18}>
              {hyp}
            </Text>
          </Collapse>
        </>
      )}
    </Stack>
  );
}
