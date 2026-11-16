import { AppShell, Group, Text } from '@mantine/core';
import { ViewProvider } from './state';
import { StudyTab } from './tabs/StudyTab';
export function App() {
  return (
    <ViewProvider>
      <Shell />
    </ViewProvider>
  );
}

/** the dashboard is one template page: parameter filters on top, the same results below */
function Shell() {
  return (
    <AppShell header={{ height: 56 }} padding={0}>
      <AppShell.Header className="topbar">
        <Group h="100%" px="lg" wrap="nowrap">
          <Text fw={600} size="md" className="brand" data-testid="built">EMG → ankle torque</Text>
        </Group>
      </AppShell.Header>
      <AppShell.Main className="app-main" id="main-content" tabIndex={-1}>
        <div className="tab-body page-study"><StudyTab /></div>
      </AppShell.Main>
    </AppShell>
  );
}
