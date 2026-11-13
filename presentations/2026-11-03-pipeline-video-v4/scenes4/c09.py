import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, FadeIn, GrowFromEdge, LaggedStart, VGroup

from common4 import place
from style4 import (C_BAD, C_FIT, C_GOOD, C_HI, C_PRED, C_TQ, C_VAL, DIM, FAINT, INK, BeatScene, Plot, T, bars, card,
                    facts, load)

PH_C = {0: FAINT, 1: C_BAD, 2: C_FIT, 3: C_VAL}


class Ch09(BeatScene):
    def construct(self):
        z, f = load('w01'), facts()['w01']
        t, y, p, lab = (z[k] for k in ('ex_t', 'ex_y', 'ex_p', 'ex_lab'))
        t, y, p = t.astype(float), y.astype(float), p.astype(float)
        # two pushes around the middle of the trial
        d = np.diff((lab > 0).astype(int))
        starts = np.where(d == 1)[0]
        s0 = starts[len(starts) // 2] if len(starts) else len(t) // 2
        a, b = max(0.0, t[s0] - 1.5), min(t[-1], t[s0] + 8.5)
        m = (t >= a) & (t <= b)

        with self.beat('c09_b01'):
            tag = self.intro('c09')
            lo, hi = float(min(y[m].min(), p[m].min())) - 2, float(max(y[m].max(), p[m].max())) + 2
            P = place(Plot((a, b), (lo, hi), 11.5, 3.2, xticks=[round(a + 1), round((a + b) / 2), round(b - 1)],
                           yticks=[], xfmt=lambda v: f'{v:g} s'), -5.4, -1.2)
            bands = VGroup()
            idx = np.where(m)[0]
            seg_start = idx[0]
            for i in list(idx[1:]) + [None]:
                if i is None or lab[i] != lab[seg_start]:
                    end = (i - 1) if i is not None else idx[-1]
                    if lab[seg_start] > 0:
                        bands.add(P.band(t[seg_start], t[end], PH_C[int(lab[seg_start])], 0.22, y0=lo, y1=hi))
                    seg_start = i if i is not None else seg_start
            cy, cp = P.curve(t[m], y[m], C_TQ, 2.4), P.curve(t[m], p[m], C_PRED, 2.4)
            key = VGroup(*[T(n, 20, c) for n, c in (('onset', C_BAD), ('plateau', C_FIT), ('release', C_VAL),
                                                     ('measured', C_TQ), ('prediction', C_PRED))]).arrange(RIGHT, buff=0.45)\
                .next_to(P, UP, buff=0.15)
            self.play(FadeIn(P), FadeIn(bands), Create(cy), Create(cp), FadeIn(key), run_time=1.6)
            self.until(self.cue('So we measured both'))
            meter = card([f'best alignment of prediction and torque: median lag {f["lag_s"]["median"]:.2f} s',
                          f'{round(f["lag_s"]["share_within_50ms"] * 100)} % of {f["trials"]} trials within 50 ms'],
                         width=8.6, size=24, color=C_GOOD).move_to([0, -3.0, 0])
            self.play(FadeIn(meter), run_time=0.6)
            self.wait(self.left())

        with self.beat('c09_b02'):
            self.clear_all(keep=[tag])
            sm = f['shape_mae_median']
            names = ['rest', 'onset', 'plateau', 'release']
            g = bars([sm[n] for n in names], names, [FAINT, C_BAD, C_FIT, C_VAL], width=8, height=3.6,
                     value_fmt=lambda v: f'{v:.2f} Nm', label_size=24, value_size=24).move_to([0, -0.2, 0])
            head = T('error in each phase of a push (each trial’s offset removed) · 6 people × 3 seeds', 22, DIM)\
                .move_to([0, 2.6, 0])
            self.play(FadeIn(head), FadeIn(g.axis), FadeIn(g.labels), run_time=0.5)
            self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in g.bars], lag_ratio=0.25),
                      LaggedStart(*[FadeIn(v) for v in g.vals], lag_ratio=0.25), run_time=1.6)
            self.wait(self.left())

        with self.beat('c09_b03'):
            self.clear_all(keep=[tag])
            cav = card(['Before the windows are cut, the EMG envelope is smoothed with a zero-phase filter.',
                        'It spreads each moment over about ±0.4 s: past and future.',
                        'Fine offline. A live device would need a filter that only looks back.'], width=11, size=26,
                       title='One honest caveat', color=C_VAL, title_color=C_VAL)
            self.play(FadeIn(cav, shift=UP * 0.2), run_time=0.8)
            self.wait(self.left())
