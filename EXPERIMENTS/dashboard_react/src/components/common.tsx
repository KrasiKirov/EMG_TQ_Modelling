import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Card, Group, Stack, Table, Text, Title } from '@mantine/core';
import { IconArrowDown, IconArrowUp, IconArrowsSort, IconInfoCircle } from '../icons';
import { useScrollFade } from './useScrollFade';

export function SectionCard({
  title,
  description,
  actions,
  children,
  testId,
  style,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  testId?: string;
  style?: CSSProperties;
}) {
  return (
    <Card data-testid={testId} style={style}>
      <Group justify="space-between" align="flex-start" gap="sm" mb="md" wrap="wrap">
        <Stack gap={4} style={{ minWidth: 0, flex: '1 1 260px' }}>
          <Title order={3}>{title}</Title>
          {description && (
            <Text size="sm" c="dimmed">
              {description}
            </Text>
          )}
        </Stack>
        {actions && <Group gap="xs">{actions}</Group>}
      </Group>
      {children}
    </Card>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <Group gap={8} align="flex-start" wrap="nowrap">
      <IconInfoCircle size={16} stroke={1.7} style={{ flex: 'none', marginTop: 2, color: 'var(--text-3)' }} />
      <Text size="sm" c="dimmed">
        {children}
      </Text>
    </Group>
  );
}

// ── sortable table ───────────────────────────────────────────────────────────
export interface Column<T> {
  key: string;
  label: ReactNode;
  render: (row: T) => ReactNode;
  sort?: (row: T) => number | string;
  align?: 'left' | 'right';
  width?: number | string;
  title?: string;
}

function cmp(a: number | string, b: number | string): number {
  const an = typeof a === 'number' && !Number.isFinite(a);
  const bn = typeof b === 'number' && !Number.isFinite(b);
  if (an && bn) return 0;
  if (an) return 1; // missing values last
  if (bn) return -1;
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  return String(a).localeCompare(String(b));
}

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  rowTestId,
  onRowClick,
  rowClass,
  minWidth = 640,
  initialSort,
  dense = false,
  caption,
  stickyFirst = false,
}: {
  columns: Column<T>[];
  rows: T[];
  rowKey: (r: T) => string;
  rowTestId?: (r: T) => string;
  onRowClick?: (r: T) => void;
  /** 'selected-row' and / or 'dim' */
  rowClass?: (r: T) => string | undefined;
  minWidth?: number;
  initialSort?: { key: string; desc?: boolean };
  dense?: boolean;
  caption?: string;
  /** keep the first column in view while the table scrolls sideways */
  stickyFirst?: boolean;
}) {
  const scrollRef = useScrollFade<HTMLDivElement>();
  const [sort, setSort] = useState<{ key: string; desc: boolean } | null>(initialSort ? { key: initialSort.key, desc: !!initialSort.desc } : null);
  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col?.sort) return rows;
    const f = col.sort;
    const out = [...rows].sort((a, b) => {
      const va = f(a);
      const vb = f(b);
      const missA = typeof va === 'number' && !Number.isFinite(va);
      const missB = typeof vb === 'number' && !Number.isFinite(vb);
      if (missA || missB) return cmp(va, vb); // missing last in both directions
      return sort.desc ? cmp(vb, va) : cmp(va, vb);
    });
    return out;
  }, [rows, sort, columns]);

  return (
    <div className="scroll-x" ref={scrollRef}>
      <Table
        className={`data-table${stickyFirst ? ' sticky-first' : ''}`}
        style={{ minWidth }}
        highlightOnHover={!!onRowClick}
        verticalSpacing={dense ? 6 : 'xs'}
        horizontalSpacing="sm"
        fz="sm"
        aria-label={caption}
      >
        <Table.Thead>
          <Table.Tr>
            {columns.map((c) => {
              const active = sort?.key === c.key;
              return (
                <Table.Th key={c.key} className={c.align === 'right' ? 'r' : undefined} style={{ width: c.width }} title={c.title}>
                  {c.sort ? (
                    <button
                      type="button"
                      className="sort-btn"
                      onClick={() => setSort((p) => (p?.key === c.key ? { key: c.key, desc: !p.desc } : { key: c.key, desc: c.align === 'right' }))}
                      aria-label={`sort by ${typeof c.label === 'string' ? c.label : c.key}`}
                    >
                      {c.label}
                      {active ? (
                        sort!.desc ? (
                          <IconArrowDown size={13} />
                        ) : (
                          <IconArrowUp size={13} />
                        )
                      ) : (
                        <IconArrowsSort size={13} style={{ opacity: 0.35 }} />
                      )}
                    </button>
                  ) : (
                    c.label
                  )}
                </Table.Th>
              );
            })}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {sorted.map((r) => (
            <Table.Tr
              key={rowKey(r)}
              data-testid={rowTestId?.(r)}
              className={[onRowClick ? 'clickable' : '', rowClass?.(r) ?? ''].join(' ').trim() || undefined}
              tabIndex={onRowClick ? 0 : undefined}
              onKeyDown={onRowClick ? (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onRowClick(r); } } : undefined}
              onClick={onRowClick ? () => onRowClick(r) : undefined}
            >
              {columns.map((c) => (
                <Table.Td key={c.key} className={c.align === 'right' ? 'r' : undefined}>
                  {c.render(r)}
                </Table.Td>
              ))}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </div>
  );
}
