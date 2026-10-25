import { useEffect, useRef, type RefObject } from 'react';

/**
 * Sets data-fade="left" | "right" | "both" on a horizontally scrolling element while content is hidden
 * at that edge (styles.css turns it into a soft fade), so a sideways scroll is visible even where the
 * system hides scrollbars (macOS, phones).
 */
export function useScrollFade<T extends HTMLElement>(): RefObject<T> {
  const ref = useRef<T>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => {
      const max = el.scrollWidth - el.clientWidth;
      const left = el.scrollLeft > 2;
      const right = el.scrollLeft < max - 2;
      const v = max <= 2 ? '' : left && right ? 'both' : right ? 'right' : left ? 'left' : '';
      if (v) el.setAttribute('data-fade', v);
      else el.removeAttribute('data-fade');
    };
    update();
    el.addEventListener('scroll', update, { passive: true });
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(update);
      ro.observe(el);
      if (el.firstElementChild) ro.observe(el.firstElementChild);
    } else window.addEventListener('resize', update);
    return () => {
      el.removeEventListener('scroll', update);
      if (ro) ro.disconnect();
      else window.removeEventListener('resize', update);
    };
  }, []);
  return ref;
}
