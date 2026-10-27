import type { ReactNode } from 'react';
import { Code, List, SimpleGrid, Stack, Text, Title } from '@mantine/core';
import { DS } from '../dataset';
import { SectionCard } from '../components/common';

function Term({ name, children }: { name: string; children: ReactNode }) {
  return (
    <Stack gap={2}>
      <Text fw={650} size="sm">
        {name}
      </Text>
      <Text size="sm" c="dimmed">
        {children}
      </Text>
    </Stack>
  );
}

export function AboutTab() {
  const nRows = DS.runs.length;
  const nWaves = Object.keys(DS.raw.wave).length;
  return (
    <Stack gap="md">
      <SectionCard title="What this is" description={`data built ${DS.built} · ${nRows} run × position scores · ${nWaves} stored predictions`}>
        <Text size="sm" maw={900}>
          A small LSTM network (1,001 parameters, 0.2 s input window) predicts the <b>voluntary</b> ankle torque (measured torque minus passive torque)
          from 4 surface-EMG signals and the ankle angle, for 6 people (YES, EG, JM, IES, CH, HM). Each person has 2 sessions; one model per person is
          trained on one session and tested on the other. Training is repeated with different random seeds (runs) and the spread over runs is shown
          everywhere. Choose one experiment to inspect its method, all-subject scores and waveforms. Use the separate Comparison tab to compare two models.
        </Text>
      </SectionCard>

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
        <SectionCard title="Definitions">
          <Stack gap="sm">
            <Term name="Answer key">
              Measured torque minus the passive torque from the passive trials of the tested session. IES has no passive trials, so nothing is subtracted.
            </Term>
            <Term name="R²">1 = perfect; 0 = as good as predicting the trial&apos;s average torque; below 0 = worse than that. Computed per position.</Term>
            <Term name="MAE (Nm)">
              Mean absolute error, in newton-metres. Read it next to the peak voluntary torque of that position: 1 Nm is a lot at a 3 Nm position and little
              at a 20 Nm one.
            </Term>
            <Term name="Demeaned">Each trial&apos;s mean is removed from both curves before scoring, so only the shape counts (not an offset).</Term>
            <Term name="Bias (Nm)">Mean of prediction − measured. Positive = the model predicts too much torque on average.</Term>
            <Term name="Toes-down">The 2 most negative (plantarflexed) test positions of each person. Negative angle = toes down.</Term>
            <Term name="Runs / seeds">
              The same model trained again with a different random seed. The test trials never change, so the spread over runs is the training
              randomness.
            </Term>
            <Term name="Directions">learn 1 → test 2 (primary) and learn 2 → test 1 (reverse check).</Term>
          </Stack>
        </SectionCard>

        <Stack gap="md">
          <SectionCard title="Comparison display rules">
            <List size="sm" spacing={6}>
              <List.Item>
                R² below 0 is shown as 0 by default (switch in the left panel to see raw values). Each run&apos;s R² is set to 0 first, then medians and
                percentiles are taken, as in the deck&apos;s box plots and headline numbers. The appendix tables (presentation_tables.md) take the percentile
                of the raw values first, so a few tail percentiles (5 % / 95 %) can differ slightly; medians agree.
              </List.Item>
              <List.Item>
                Medians, not means. A value for a group of positions (e.g. &quot;toes-down&quot; or &quot;other 6 positions&quot;, one person or all) is the median over runs at
                each position first, then the median across the positions (for one person&apos;s 2 toes-down positions: the average of the two), as in the deck.
              </List.Item>
              <List.Item>
                Spread = 5th–95th percentile (whiskers, KPI cards) and 1st–99th percentile (tooltips, KPI cards) over the training runs, not the standard deviation.
                In the KPI cards and tables each run&apos;s value for the group is the median across the group&apos;s positions in that run.
              </List.Item>
              <List.Item>MAE (Nm) instead of RMSE, always next to the peak voluntary torque of the position.</List.Item>
              <List.Item>
                B − A compares the two models on the same people only: when one model was not trained for some people, the KPI cards say which people
                were used.
              </List.Item>
              <List.Item>
                Green / red (better / worse) only when the difference is at least 0.02 for R², or at least 0.05 Nm and 2 % of the peak torque for MAE;
                smaller differences are &quot;about the same&quot; (well inside the run-to-run spread).
              </List.Item>
              <List.Item>
                Bias is compared by its size, |bias|, so positive and negative offsets of different people do not cancel; the box plots keep the sign.
              </List.Item>
              <List.Item>
                Percentiles use numpy&apos;s default linear interpolation: sort the n values, idx = (n − 1)·p, interpolate between the neighbours.
              </List.Item>
            </List>
          </SectionCard>
          <SectionCard title="Reading a box plot">
            <Text size="sm">
              The box covers the middle half of the runs (25th to 75th percentile), the line inside is the median, and the whiskers reach the 5th and 95th
              percentile. Dots are the individual runs. A short box = runs agree; a box high up = good for R² (low = good for MAE).
            </Text>
          </SectionCard>
        </Stack>
      </SimpleGrid>

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
        <SectionCard title="Refresh the data">
          <Stack gap="xs">
            <Text size="sm">From the repository root, after the training jobs have finished, in this order:</Text>
            <Code block>
              {[
                'python EXPERIMENTS/code/presentation_metrics.py',
                'python EXPERIMENTS/code/deck_figures.py',
                'python EXPERIMENTS/code/build_dashboard.py',
                'cd EXPERIMENTS/dashboard_react',
                'npm run build',
              ].join('\n')}
            </Code>
            <List size="sm" spacing={4} c="dimmed" type="ordered">
              <List.Item>scores every saved run → results/presentation_runs.csv</List.Item>
              <List.Item>bias / variance split → results/bias_variance_runs.csv</List.Item>
              <List.Item>
                writes <Code>src/data/results.json</Code> (never edit it by hand)
              </List.Item>
              <List.Item>
                builds one self-contained <Code>dist/index.html</Code> that opens by double-click, with no server and no internet
              </List.Item>
            </List>
            <Text size="sm" c="dimmed">
              Each step reads the previous step&apos;s output.
            </Text>
          </Stack>
        </SectionCard>
        <SectionCard title="Hardware">
          <Stack gap={4}>
            <Title order={4} fw={600}>
              AMD Ryzen 7 3700X (8 cores / 16 threads), 32 GB RAM
            </Title>
            <Text size="sm" c="dimmed">
              CPU-only TensorFlow 2.21. No GPU was used for any run shown here.
            </Text>
          </Stack>
        </SectionCard>
      </SimpleGrid>
    </Stack>
  );
}
