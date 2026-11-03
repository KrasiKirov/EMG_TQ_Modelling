import { useState } from 'react';
import { ActionIcon, AppShell, Burger, Group, SegmentedControl, Text, Title, Tooltip, UnstyledButton } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { IconMoon, IconSun } from './icons';
import type { TabKey } from './lib/hash';
import { ViewProvider, useView } from './state';
import { useScrollFade } from './components/useScrollFade';
import { Sidebar } from './components/Sidebar';
import { ModelBar } from './components/ModelBar';
import { HomeTab } from './tabs/HomeTab';
import { ExperimentTab } from './tabs/ExperimentTab';
import { AllExperimentsTab } from './tabs/AllExperimentsTab';
import { OverviewTab } from './tabs/OverviewTab';
import { WaveformsTab } from './tabs/WaveformsTab';
import { SavedComparisonTab } from './tabs/SavedComparisonTab';

export function App() {
  return (
    <ViewProvider>
      <Shell />
    </ViewProvider>
  );
}

/** top-level pages; 'comparison' also covers its waveform / training sub-views */
const NAV: { key: TabKey; label: string }[] = [
  { key: 'home', label: 'Overview' },
  { key: 'overview', label: 'Models' },
  { key: 'comparison', label: 'Compare' },
  { key: 'experiments', label: 'Experiments' },
];

function Shell() {
  const [opened, { toggle, close }] = useDisclosure(false);
  const { s, setTab, scheme, toggleScheme } = useView();
  const comparing = ['comparison', 'waveforms', 'training'].includes(s.tab);
  const [comparisonMode, setComparisonMode] = useState(s.tab === 'waveforms' || s.tab === 'training' ? 'settings' : 'saved');
  const withPanel = comparing && comparisonMode === 'settings';
  const navRef = useScrollFade<HTMLDivElement>();
  const current = comparing ? 'comparison' : s.tab;
  const go = (k: TabKey) => { setTab(k); close(); window.scrollTo({ top: 0 }); };

  return (
    <><a className="skip-link" href="#main-content" onClick={e => { e.preventDefault(); document.getElementById('main-content')?.focus(); }}>Skip to results</a>
    <AppShell header={{ height: 64 }} navbar={withPanel ? { width: { base: 288, lg: 300 }, breakpoint: 'md', collapsed: { mobile: !opened } } : undefined} padding={0}>
      <AppShell.Header className="topbar">
        <Group h="100%" px={{ base: 'sm', sm: 'lg' }} gap="md" wrap="nowrap" justify="space-between">
          <Group gap="sm" wrap="nowrap" style={{ minWidth: 0, flex: 'none' }}>
            {withPanel && <Burger opened={opened} onClick={toggle} hiddenFrom="md" size="sm" aria-label="Show comparison settings" />}
            <UnstyledButton className="brand" onClick={() => go('home')} aria-label="Overview">
              <span className="brand-mark" aria-hidden>τ</span>
              <span className="brand-text"><Title order={1} fz={16} lh={1.2}>Ankle torque</Title><Text size="xs" c="dimmed" data-testid="built">EMG → torque</Text></span>
            </UnstyledButton>
          </Group>
          <nav className="topnav scroll-x" ref={navRef} aria-label="Pages">
            {NAV.map(n => <UnstyledButton key={n.key} className="topnav-item" data-active={current === n.key || undefined} aria-current={current === n.key ? 'page' : undefined} onClick={() => go(n.key)} data-testid={`tab-${n.key}`}>{n.label}</UnstyledButton>)}
          </nav>
          <Tooltip label={scheme === 'dark' ? 'Light mode' : 'Dark mode'}>
            <ActionIcon variant="default" size="lg" radius="md" onClick={toggleScheme} aria-label="Toggle light / dark mode" data-testid="theme-toggle" style={{ flex: 'none' }}>
              {scheme === 'dark' ? <IconSun size={18} stroke={1.7} /> : <IconMoon size={18} stroke={1.7} />}
            </ActionIcon>
          </Tooltip>
        </Group>
      </AppShell.Header>

      {withPanel && <AppShell.Navbar><Sidebar onNavigate={close} /></AppShell.Navbar>}

      <AppShell.Main className="app-main" id="main-content" tabIndex={-1}>
        <div className={`tab-body page-${current}`}>
          {current === 'home' && <HomeTab />}
          {current === 'overview' && <><ModelBar /><ExperimentTab /></>}
          {current === 'comparison' && <>
            <div className="page-heading"><Text size="sm" c="dimmed">Side-by-side analysis</Text><Title order={2}>Compare two models</Title><Text size="sm" c="dimmed">Inspect saved predictions, coverage and seed variability. Use controlled settings for the original parameter-by-parameter comparison.</Text></div>
            <SegmentedControl mb="lg" aria-label="Comparison workspace" value={comparisonMode} onChange={v => { setComparisonMode(v); setTab('comparison'); }} data={[{ value: 'saved', label: 'Saved experiments' }, { value: 'settings', label: 'Controlled settings' }]} />
            {comparisonMode === 'saved' ? <SavedComparisonTab /> : <><Group className="comparison-subnav" mb="lg">{([['comparison', 'Summary'], ['waveforms', 'Waveforms']] as const).map(([tab, label]) => <button key={tab} data-active={s.tab === tab || undefined} onClick={() => setTab(tab)}>{label}</button>)}</Group>
              {s.tab === 'waveforms' ? <WaveformsTab /> : <OverviewTab />}</>}
          </>}
          {current === 'experiments' && <AllExperimentsTab />}
        </div>
      </AppShell.Main>
    </AppShell></>
  );
}
