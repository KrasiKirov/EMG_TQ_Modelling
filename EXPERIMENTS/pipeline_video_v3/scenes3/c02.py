import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Create, Dot, FadeIn, FadeOut, LaggedStart, ValueTracker, VGroup, always_redraw,
                   rush_into)

from common3 import Rows, push_bounds, tq_at
from style3 import C_HI, C_VAL, DIM, INK, BeatScene, T, card, facts, load


class Ch02(BeatScene):
    def construct(self):
        z, f = load('v01'), facts()['v01']['eg_trial']
        ps, pe = push_bounds(z, 23.5)
        s0, s1 = ps - 1.6, ps + 4.6
        t = z['eg_t']
        i = int(np.searchsorted(t, ps))
        while i > 0 and z['eg_tq'][i] < -4.5:
            i -= 1
        onset = float(t[i])
        te0 = round(ps + 0.6, 2)

        with self.beat('c02_b01'):
            self.intro('c02')
            rows = Rows(z, s0, s1, top=2.45)
            self.play(FadeIn(rows.static), LaggedStart(*[Create(c) for c in rows.curves], lag_ratio=0.1), run_time=2.0)
            self.until(self.cue('One window is'))
            box = rows.window_box(te0)
            lab1 = T('input: 20 samples × 5 signals = 0.2 s', 22, C_HI).next_to(box, UP, buff=0.12)
            self.play(FadeIn(box), FadeIn(lab1), run_time=0.8)
            self.until(self.cue('Its answer is'))
            dot = Dot(rows.tq.p(te0, tq_at(z, te0)), radius=0.08, color=C_HI)
            lab2 = T('answer: the torque at the last sample', 22, C_HI).move_to(rows.tq.p(te0 + 1.9, -14))
            self.play(FadeIn(dot, scale=2), FadeIn(lab2), run_time=0.8)
            self.wait(self.left())

        with self.beat('c02_b02'):
            self.play(FadeOut(lab1), FadeOut(lab2), run_time=0.4)
            te = ValueTracker(te0)
            self.remove(box)
            box = always_redraw(lambda: rows.window_box(te.get_value()))
            counter = always_redraw(lambda: T(f'window {int(round(te.get_value() * 100)) - 18:,} of {f["windows"]:,}', 24,
                                              INK).to_edge(UP, buff=0.32).shift(RIGHT * 3.6))
            trail = VGroup(dot)
            last = [te0]

            def grow(m):
                v = te.get_value()
                while last[0] + 0.03 <= v:
                    last[0] += 0.03
                    m.add(Dot(rows.tq.p(last[0], tq_at(z, last[0])), radius=0.035, color=C_HI))
            trail.add_updater(grow)
            self.add(box, counter, trail)
            note = T('neighbouring windows share 19 of their 20 samples', 22, DIM).next_to(rows, DOWN, buff=0.18)
            for _ in range(3):                                  # three single steps, slowly
                self.play(te.animate.set_value(te.get_value() + 0.01), run_time=0.7)
            self.play(FadeIn(note), run_time=0.5)
            self.until(self.cue('Slide all the way'))
            self.play(te.animate.set_value(s1 - 0.05), run_time=max(1.5, self.left() - 2.2), rate_func=rush_into)
            trail.clear_updaters()
            box.clear_updaters()
            counter.clear_updaters()
            total = card([f'Slide through the whole {f["seconds"]:g}-second trial → {f["windows"]:,} windows'],
                         width=8.2, size=26).move_to([0, 0.2, 0])
            self.play(FadeIn(total), run_time=0.6)
            self.wait(self.left())

        with self.beat('c02_b03'):
            self.play(FadeOut(total), FadeOut(counter), FadeOut(note), FadeOut(box), FadeOut(trail), run_time=0.5)
            band = rows.tq.band(onset - 0.1, pe + 0.2, C_VAL, 0.10, y0=-35, y1=2)
            moments = [('onset', onset + 0.08), ('rise', ps + 0.05), ('plateau', (ps + pe) / 2), ('release', pe + 0.12)]
            boxes, labels, dots = VGroup(), VGroup(), VGroup()
            for name, tm in moments:
                b = rows.window_box(tm, opacity=0.25)
                boxes.add(b)
                labels.add(T(name, 20, C_HI).next_to(b, UP, buff=0.1 + 0.32 * (len(labels) % 2)))
                dots.add(Dot(rows.tq.p(tm, tq_at(z, tm)), radius=0.07, color=C_HI))
            self.play(FadeIn(band), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(VGroup(b, l, d)) for b, l, d in zip(boxes, labels, dots)], lag_ratio=0.35),
                      run_time=2.4)
            self.until(self.cue('And it is also'))
            msg = T('each one is a full example: the last 0.2 s → the torque now', 22, INK).next_to(rows, DOWN, buff=0.18)
            self.play(FadeIn(msg), run_time=0.6)
            self.wait(self.left())
