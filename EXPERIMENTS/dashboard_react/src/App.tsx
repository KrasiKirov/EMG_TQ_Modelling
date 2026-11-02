import { useState } from 'react';
import { ActionIcon, AppShell, Burger, Group, SegmentedControl, Stack, Tabs, Text, Title, Tooltip } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { IconBook2, IconLayoutDashboard, IconFlask, IconMoon, IconSun, IconTimeline, IconWaveSine } from './icons';
import { TABS, type TabKey } from './lib/hash';
import { ViewProvider, useView } from './state';
import { useScrollFade } from './components/useScrollFade';
import { Sidebar } from './components/Sidebar';
import { ExperimentSidebar } from './components/ExperimentSidebar';
import { ExperimentTab } from './tabs/ExperimentTab';
import { ControlledTab } from './tabs/ControlledTab';
import { OverviewTab } from './tabs/OverviewTab';
import { WaveformsTab } from './tabs/WaveformsTab';
import { TrainingTab } from './tabs/TrainingTab';
import { SavedComparisonTab } from './tabs/SavedComparisonTab';

import { AboutTab } from './tabs/AboutTab';

export function App() {
  return (
    <ViewProvider>
      <Shell />
    </ViewProvider>
  );
}

const TAB_META: Record<TabKey, { label: string; icon: typeof IconLayoutDashboard }> = {
  overview: { label: 'Results', icon: IconLayoutDashboard },
  comparison: { label: 'Comparison', icon: IconLayoutDashboard },
  waveforms: { label: 'Waveforms', icon: IconWaveSine },
  training: { label: 'Training', icon: IconTimeline },
  experiments: { label: 'Controlled experiments', icon: IconFlask },
  about: { label: 'Guide', icon: IconBook2 },
};

function Shell() {
  const [opened, { toggle, close }] = useDisclosure(false);
  const { s, setTab, scheme, toggleScheme } = useView();
  const comparing = ['comparison', 'waveforms', 'training'].includes(s.tab);
  const [comparisonMode,setComparisonMode]=useState(s.tab==='waveforms'||s.tab==='training'?'settings':'saved');
  const tabsRef = useScrollFade<HTMLDivElement>();

  return (
    <><a className="skip-link" href="#main-content" onClick={e => {e.preventDefault();document.getElementById('main-content')?.focus();}}>Skip to results</a><AppShell header={{ height: 64 }} navbar={{ width: { base: 288, lg: 300 }, breakpoint: 'md', collapsed: { mobile: !opened } }} padding={0}>
      <AppShell.Header>
        <Group h="100%" px="md" gap="sm" wrap="nowrap" justify="space-between">
          <Group gap="sm" wrap="nowrap" style={{ minWidth: 0 }}>
            <Burger opened={opened} onClick={toggle} hiddenFrom="md" size="sm" aria-label={comparing ? "Show comparison settings" : "Choose an experiment"} />
            <Stack gap={0} style={{ minWidth: 0 }}>
              <Title order={1} fz={{ base: 15, sm: 18 }} lh={1.25} style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                Ankle torque
              </Title>
              <Text size="xs" c="dimmed" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} data-testid="built">
                Muscle activity → torque prediction
              </Text>
            </Stack>
          </Group>
          <Group gap="xs" wrap="nowrap">
            <Tooltip label={scheme === 'dark' ? 'Light mode' : 'Dark mode'}>
              <ActionIcon variant="default" size="lg" radius="md" onClick={toggleScheme} aria-label="Toggle light / dark mode" data-testid="theme-toggle">
                {scheme === 'dark' ? <IconSun size={18} stroke={1.7} /> : <IconMoon size={18} stroke={1.7} />}
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar>
        {comparing&&comparisonMode==='settings' ? <Sidebar onNavigate={close} /> : <ExperimentSidebar onNavigate={close} />}
      </AppShell.Navbar>

      <AppShell.Main className="app-main" id="main-content" tabIndex={-1}>
        <Tabs value={comparing ? 'comparison' : s.tab} onChange={(v) => v && setTab(v as TabKey)} className="main-tabs" keepMounted={false}>
          <Tabs.List px={{ base: 'md', sm: 'lg' }} pt="xs" ref={tabsRef} className="scroll-x">
            {TABS.filter(k => k !== 'waveforms' && k !== 'training' && k !== 'about').map((k) => {
              const Icon = TAB_META[k].icon;
              return (
                <Tabs.Tab key={k} value={k} leftSection={<Icon size={16} stroke={1.7} />} data-testid={`tab-${k}`}>
                  {TAB_META[k].label}
                </Tabs.Tab>
              );
            })}
          </Tabs.List>
          <div  className="tab-body">
            <Tabs.Panel value="overview"><ExperimentTab /></Tabs.Panel>
            <Tabs.Panel value="comparison">
              <div className="page-heading"><Text size="sm" c="dimmed">Side-by-side analysis</Text><Title order={2}>Compare two models</Title><Text size="sm" c="dimmed">Inspect saved predictions, coverage and seed variability. Use controlled settings for the original parameter-by-parameter comparison.</Text></div>
              <SegmentedControl mb="lg" aria-label="Comparison workspace" value={comparisonMode} onChange={v=>{setComparisonMode(v);setTab('comparison');}} data={[{value:'saved',label:'Saved experiments'},{value:'settings',label:'Controlled settings'}]}/>
              {comparisonMode==='saved'?<SavedComparisonTab/>:<><Group className="comparison-subnav" mb="lg">{([['comparison','Summary'],['waveforms','Waveforms'],['training','Training audit']] as const).map(([tab,label])=><button key={tab} data-active={s.tab===tab||undefined} onClick={()=>setTab(tab)}>{label}</button>)}</Group>
              {s.tab==='waveforms' ? <WaveformsTab /> : s.tab==='training' ? <TrainingTab /> : <OverviewTab />}</>}
            </Tabs.Panel>
            <Tabs.Panel value="experiments"><ControlledTab /></Tabs.Panel>
            <Tabs.Panel value="about">
              <AboutTab />
            </Tabs.Panel>
          </div>
        </Tabs>
      </AppShell.Main>
    </AppShell></>
  );
}
