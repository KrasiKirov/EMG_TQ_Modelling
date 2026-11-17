import { Text } from '@mantine/core';
import { allSeeds, STUDY } from '../lib/study';

const span = (s: number[]) => (s.length ? (s.length > 2 && s[s.length - 1] - s[0] === s.length - 1 ? `${s[0]}–${s[s.length - 1]}` : s.join(', ')) : '–');

/** How the training seeds were set up: shown under every chart that spreads results over seeds. */
export function SeedSetup({ open = false }: { open?: boolean }) {
  const ten = Object.values(STUDY.settings).filter((s) => allSeeds(s.id).length >= 10).map((s) => s.title);
  const few = Object.values(STUDY.settings).filter((s) => allSeeds(s.id).length < 5).map((s) => `${s.title} (seed ${span(allSeeds(s.id))})`);
  return (
    <details className="disclosure" open={open} data-testid="seed-setup">
      <summary>How the training seeds were set up</summary>
      <Text size="sm" mt="xs">
        A run is one person trained with one seed (TensorFlow <code>set_random_seed</code>). Seeds change only the starting weights,
        shuffle order and dropout; the same seed number gives the same starting weights in every setting, so comparisons use matched seeds.
      </Text>
      <Text size="sm" mt="xs">
        Seeds 0–4 for every setting; 0–9 for {ten.length ? ten.join(', ') : 'none'}{few.length ? `; fewer for ${few.join(', ')}` : ''}.
      </Text>
    </details>
  );
}
