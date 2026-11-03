import { Menu, Select, SegmentedControl, Switch, UnstyledButton } from '@mantine/core';
import { ALL_IDS, catalogTitle } from '../lib/catalog';
import { DS } from '../dataset';
import { PRESENTATION_GROUPS } from '../lib/presentation';
import type { Dir } from '../lib/types';
import { useView } from '../state';

/** Model switcher at the top of a model page: the final-model family as pills, everything else in one search box. */
export function ModelBar() {
  const { s, set } = useView();
  const choose = (id: string) => { set({ experiment: id, tab: 'overview', runA: 'typical' }); window.scrollTo({ top: 0 }); };
  const [main, ...rest] = PRESENTATION_GROUPS;
  const seen = new Set<string>();
  const groups = [...rest.map((g) => ({ group: g.label, items: g.items.map((x) => ({ value: x.id, label: x.label || catalogTitle(x.id) })) })),
    { group: 'Controlled experiments', items: ALL_IDS.filter((id) => DS.experiments[id]).map((id) => ({ value: id, label: catalogTitle(id) })) }]
    .map((g) => ({ ...g, items: g.items.filter((x) => !seen.has(x.value) && !main.items.some((m) => m.id === x.value) && seen.add(x.value)) }))
    .filter((g) => g.items.length);
  const other = !main.items.some((x) => x.id === s.experiment);
  return (
    <div className="model-bar" role="navigation" aria-label="Choose a model">
      <div className="model-pills">
        {main.items.map((x) => <UnstyledButton key={x.id} className="model-pill" data-active={s.experiment === x.id || undefined} aria-current={s.experiment === x.id ? 'true' : undefined} onClick={() => choose(x.id)}>
          {x.label}{x.badge && <span className="focus-badge">{x.badge}</span>}
        </UnstyledButton>)}
      </div>
      <Select className="model-search" size="sm" searchable placeholder="Other models…" aria-label="Other models" value={other ? s.experiment : null} data={groups} onChange={(v) => v && choose(v)} comboboxProps={{ width: 360, position: 'bottom-end' }} maxDropdownHeight={420} />
      <Menu position="bottom-end" withinPortal>
        <Menu.Target><UnstyledButton className="model-settings" aria-label="Evaluation settings">{s.dir === 'S1toS2' ? 'Session 1 → 2' : 'Session 2 → 1'} ▾</UnstyledButton></Menu.Target>
        <Menu.Dropdown p="sm">
          <Menu.Label>Train → test</Menu.Label>
          <SegmentedControl fullWidth size="xs" value={s.dir} onChange={(v) => set({ dir: v as Dir, runA: 'typical' })} data={[{ value: 'S1toS2', label: 'Session 1 → 2' }, { value: 'S2toS1', label: 'Session 2 → 1' }]} />
          <Menu.Label mt="sm">Display</Menu.Label>
          <Switch checked={s.floor} onChange={(e) => set({ floor: e.currentTarget.checked })} label="Show negative R² as 0" size="xs" />
        </Menu.Dropdown>
      </Menu>
    </div>
  );
}
