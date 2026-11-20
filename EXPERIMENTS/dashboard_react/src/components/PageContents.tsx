import { useEffect, useRef, useState } from 'react';

interface Entry { id: string; label: string }

/** the visible section cards of the current page, in page order */
function scan(main: HTMLElement): Entry[] {
  const out: Entry[] = [];
  main.querySelectorAll<HTMLElement>('.tab-body [data-section]').forEach((el, i) => {
    if (!el.offsetParent) return;                       // hidden view or a closed <details>
    if (el.parentElement?.closest('[data-section]')) return; // a card inside another card
    if (!el.id) el.id = `section-${i + 1}`;
    const label = el.querySelector('[data-section-title]')?.textContent?.trim();
    if (label) out.push({ id: el.id, label });
  });
  return out;
}

/**
 * "On this page": one link per section card, kept up to date as the page changes. Sits under the top bar and stays
 * there while scrolling, so any section of a long page is one click away. Hidden when a page has fewer than 3 sections.
 */
export function PageContents({ page }: { page: string }) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const last = useRef('');
  useEffect(() => {
    // watch the main area, which stays mounted, not the page body, which is replaced when the page changes
    const root = document.getElementById('main-content');
    if (!root) return;
    let timer = 0;
    const update = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        const next = scan(root);
        const key = JSON.stringify(next);
        if (key !== last.current) { last.current = key; setEntries(next); }
      }, 50);
    };
    update();
    const obs = new MutationObserver(update);
    obs.observe(root, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'open'] });
    return () => { obs.disconnect(); window.clearTimeout(timer); };
  }, [page]);
  if (entries.length < 3) return null;
  return (
    <nav className="page-contents" aria-label="On this page" data-testid="page-contents">
      <span className="page-contents-label">On this page:</span>
      {entries.map((e) => (
        <button key={e.id} type="button" onClick={() => document.getElementById(e.id)?.scrollIntoView({ block: 'start' })}>
          {e.label}
        </button>
      ))}
    </nav>
  );
}
