import { Stack, Text } from '@mantine/core';

/** "What this shows": plain-language sentences under a section (lib/takeaways.ts builds them from the page's numbers) */
export function Takeaway({ lines, title = 'What this shows', testId }: { lines: string[]; title?: string; testId?: string }) {
  if (!lines.length) return null;
  return (
    <div className="takeaway" data-testid={testId}>
      <Text size="sm" fw={700} mb={4}>{title}</Text>
      <Stack gap={4}>
        {lines.map((l) => <Text key={l} size="sm">{l}</Text>)}
      </Stack>
    </div>
  );
}
