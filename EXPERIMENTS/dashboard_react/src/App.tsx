import { ActionIcon, AppShell, Burger, Group, Stack, Tabs, Text, Title, Tooltip } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { IconBook2, IconLayoutDashboard, IconFlask, IconMoon, IconSun, IconTimeline, IconWaveSine } from './icons';
import { DS } from './dataset';
import { TABS, type TabKey } from './lib/hash';
import { ViewProvider, useView } from './state';
import { useScrollFade } from './components/useScrollFade';
import { Sidebar } from './components/Sidebar';
import { OverviewTab } from './tabs/OverviewTab';
import { WaveformsTab } from './tabs/WaveformsTab';
import { TrainingTab } from './tabs/TrainingTab';
import { ExperimentsTab } from './tabs/ExperimentsTab';
import { AboutTab } from './tabs/AboutTab';

export function App() {
  return (
    <ViewProvider>
      <Shell />
    </ViewProvider>
  );
}

const TAB_META: Record<TabKey, { label: string; icon: typeof IconLayoutDashboard }> = {
  overview: { label: 'Overview', icon: IconLayoutDashboard },
  waveforms: { label: 'Waveforms', icon: IconWaveSine },
  training: { label: 'Training', icon: IconTimeline },
  experiments: { label: 'Experiments', icon: IconFlask },
  about: { label: 'About', icon: IconBook2 },
};

function Shell() {
  const [opened, { toggle, close }] = useDisclosure(false);
  const { s, setTab, scheme, toggleScheme } = useView();
  const tabsRef = useScrollFade<HTMLDivElement>();

  return (
    <AppShell header={{ height: 64 }} navbar={{ width: { base: 320, lg: 360 }, breakpoint: 'md', collapsed: { mobile: !opened } }} padding={0}>
      <AppShell.Header>
        <Group h="100%" px="md" gap="sm" wrap="nowrap" justify="space-between">
          <Group gap="sm" wrap="nowrap" style={{ minWidth: 0 }}>
            <Burger opened={opened} onClick={toggle} hiddenFrom="md" size="sm" aria-label="Show settings" />
            <Stack gap={0} style={{ minWidth: 0 }}>
              <Title order={1} fz={{ base: 15, sm: 18 }} lh={1.25} style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                EMG → ankle torque · results explorer
              </Title>
              <Text size="xs" c="dimmed" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} data-testid="built">
                Data built {DS.built} · answer key: measured torque − passive torque from the passive trials of the tested session
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
        <Sidebar onNavigate={close} />
      </AppShell.Navbar>

      <AppShell.Main className="app-main">
        <Tabs value={s.tab} onChange={(v) => v && setTab(v as TabKey)} className="main-tabs" keepMounted={false}>
          <Tabs.List px={{ base: 'md', sm: 'lg' }} pt="xs" ref={tabsRef} className="scroll-x">
            {TABS.map((k) => {
              const Icon = TAB_META[k].icon;
              return (
                <Tabs.Tab key={k} value={k} leftSection={<Icon size={16} stroke={1.7} />} data-testid={`tab-${k}`}>
                  {TAB_META[k].label}
                </Tabs.Tab>
              );
            })}
          </Tabs.List>
          <div style={{ padding: 'var(--mantine-spacing-md)', maxWidth: 1600 }} className="tab-body">
            <Tabs.Panel value="overview">
              <OverviewTab />
            </Tabs.Panel>
            <Tabs.Panel value="waveforms">
              <WaveformsTab />
            </Tabs.Panel>
            <Tabs.Panel value="training">
              <TrainingTab />
            </Tabs.Panel>
            <Tabs.Panel value="experiments">
              <ExperimentsTab />
            </Tabs.Panel>
            <Tabs.Panel value="about">
              <AboutTab />
            </Tabs.Panel>
          </div>
        </Tabs>
      </AppShell.Main>
    </AppShell>
  );
}
