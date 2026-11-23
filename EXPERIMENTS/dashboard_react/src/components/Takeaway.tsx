import { Stack, Text } from '@mantine/core';

/** a titled box of short sentences (the study's answer under the page title) */
export function Takeaway({ lines, title, testId }: { lines: string[]; title: string; testId?: string }) {
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
