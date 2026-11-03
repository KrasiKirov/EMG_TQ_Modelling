import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Dot, FadeIn, FadeOut, LaggedStart, Line, Rectangle, RoundedRectangle,
                   VGroup)

from common4 import place
from style4 import (BG, C_GOOD, C_HI, C_PRED, C_TQ, C_VAL, DIM, FAINT, INK, MUSCLE_C, PANEL, BeatScene, Plot, T, W,
                    card, facts, load)


class Ch06(BeatScene):
    def construct(self):
        z, f = load('w02'), facts()['w02']
        tt, yy = z['trial_t'].astype(float), z['trial_y'].astype(float)
        te = f['window_time_s']
        win = z['win'].astype(float)                    # (20, 5)
        c1 = z['l1_c'].astype(float)                    # (20, 8)
        fg = z['l1_f'].astype(float)

        with self.beat('c06_b01'):
            tag = self.intro('c06')
            a, b = te - 2.5, te + 2.5
            m = (tt >= a) & (tt <= b)
            lo, hi = float(yy[m].min()) - 2, float(yy[m].max()) + 2
            P = place(Plot((a, b), (lo, hi), 11.0, 1.9, yticks=[v for v in (-25, 0) if lo <= v <= hi],
                           xticks=[a + 0.5, te, b - 0.5], xfmt=lambda v: f'{v:.1f} s', ylabel='Nm'), -5.3, 0.65)
            cur = P.curve(tt[m], yy[m], C_TQ, 2.4)
            band = P.band(te - 0.19, te, C_HI, 0.25)
            dot = Dot(P.p(te, f['window_torque_Nm']), radius=0.08, color=C_HI)
            tl = T(f'EG · session 2 · end of a push · torque at the last step: {f["window_torque_Nm"]:.1f} Nm'.replace('-', '−'), 22, C_HI)\
                .next_to(P, UP, buff=0.15)
            self.play(FadeIn(P), FadeIn(cur), run_time=0.8)
            self.play(FadeIn(band), FadeIn(dot), FadeIn(tl), run_time=0.6)
            self.until(self.cue('Twenty time steps'))
            cell, grid = 0.36, VGroup()
            colors = MUSCLE_C + ['#C9D1D9']
            for r in range(5):
                col = win[:, r]
                rng = (col.max() - col.min()) or 1
                for c in range(20):
                    v = 0.15 + 0.85 * (col[c] - col.min()) / rng if r < 4 else 0.5
                    grid.add(Rectangle(width=cell, height=cell, stroke_color=BG, stroke_width=1.2, fill_color=colors[r],
                                       fill_opacity=float(v)).move_to([(c - 9.5) * cell, (2 - r) * cell - 1.6, 0]))
            rl = VGroup(*[T(n, 18, colors[r]).next_to(grid[r * 20], LEFT, buff=0.15)
                          for r, n in enumerate(['GM', 'GL', 'soleus', 'TA', 'angle'])])
            ol = T('oldest  →  newest', 20, DIM).next_to(grid, DOWN, buff=0.12)
            self.play(FadeIn(grid), FadeIn(rl), FadeIn(ol), run_time=0.8)
            self.wait(self.left())
            self.top = VGroup(P, cur, band, dot, tl)

        with self.beat('c06_b02'):
            self.play(FadeOut(self.top), VGroup(grid, rl, ol).animate.scale(0.8).move_to([-3.4, 1.5, 0]), run_time=0.8)
            body = RoundedRectangle(corner_radius=0.2, width=5.6, height=3.6, fill_color=PANEL, fill_opacity=1,
                                    stroke_color=C_PRED, stroke_width=2).move_to([3.2, -0.6, 0])
            bt = T('LSTM layer 1 · memory (8 numbers)', 22, C_PRED).next_to(body, UP, buff=0.12)
            base = Line(body.get_left() + RIGHT * 0.4, body.get_right() + LEFT * 0.4, stroke_color=FAINT).set_y(body.get_y())
            xs = np.linspace(body.get_left()[0] + 0.8, body.get_right()[0] - 0.8, 8)
            cmax = max(1e-6, float(np.abs(c1).max()))

            def bars(k):
                g = VGroup()
                for u, x in enumerate(xs):
                    h = 1.4 * c1[k, u] / cmax
                    r = Rectangle(width=0.4, height=max(abs(h), 0.02), stroke_width=0, fill_color=C_PRED if h >= 0 else C_VAL,
                                  fill_opacity=0.9)
                    r.move_to([x, body.get_y() + h / 2, 0])
                    g.add(r)
                return g
            gates = VGroup(T('forget: keep how much of the old memory?', 20, INK),
                           T('write: add what from this step?', 20, INK),
                           T('read: pass on what?', 20, INK)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)\
                .next_to(body, LEFT, buff=0.4).shift(DOWN * 0.9)
            self.play(FadeIn(body), FadeIn(bt), FadeIn(base), run_time=0.6)
            self.until(self.cue('At every step'))
            self.play(LaggedStart(*[FadeIn(g) for g in gates], lag_ratio=0.3), run_time=1.2)
            self.until(self.cue('These bars are'))
            hi = Rectangle(width=0.29, height=1.5, stroke_color=C_HI, stroke_width=3).move_to(
                [grid[0].get_x(), grid[40].get_y(), 0])
            cur_bars = bars(0)
            step = T('step 1 of 20', 22, C_HI).next_to(body, DOWN, buff=0.15)
            fgl = T(f'forget gate (average): {fg[0].mean():.2f}', 20, DIM).next_to(step, DOWN, buff=0.08)
            self.add(hi, cur_bars, step, fgl)
            dt = max(0.12, (self.left() - 0.4) / 20)
            for k in range(1, 20):
                nb = bars(k)
                self.play(hi.animate.set_x(grid[k].get_x()), cur_bars.animate.become(nb),
                          step.animate.become(T(f'step {k + 1} of 20', 22, C_HI).move_to(step)),
                          fgl.animate.become(T(f'forget gate (average): {fg[k].mean():.2f}', 20, DIM).move_to(fgl)),
                          run_time=dt)
            self.wait(self.left())

        with self.beat('c06_b03'):
            out = card(['→ layer 2 → × the angle’s swing',
                        f'prediction: {f["window_prediction_Nm"]:.1f} Nm'.replace('-', '−'),
                        f'measured: {f["window_torque_Nm"]:.1f} Nm'.replace('-', '−')], width=5.0, size=24,
                       title='after step 20', color=C_HI, title_color=C_HI).move_to([-3.7, -1.7, 0])
            self.play(FadeOut(gates), FadeIn(out, shift=UP * 0.2), run_time=0.8)
            self.until(self.cue('Then the memory is wiped'))
            flat = VGroup(*[Rectangle(width=0.4, height=0.02, stroke_width=0, fill_color=FAINT, fill_opacity=0.9)
                            .move_to([x, body.get_y(), 0]) for x in xs])
            self.play(cur_bars.animate.become(flat), FadeOut(hi), run_time=0.8)
            nxt = T('next window: memory starts empty', 24, C_PRED).move_to(step)
            self.play(FadeOut(step), FadeOut(fgl), FadeIn(nxt), run_time=0.5)
            self.wait(self.left())

        with self.beat('c06_b04'):
            self.clear_all(keep=[tag])
            chk = card([f'recomputed by hand from the saved weights, for {f["hand_checked_windows"]} windows',
                        f'largest difference from the model’s own output: {f["hand_vs_model_max_abs"]:.1e} (rounding)',
                        'retrained with the same seed: identical to the saved run'], width=11, size=26,
                       title='Nothing hidden', color=C_GOOD, title_color=C_GOOD)
            self.play(FadeIn(chk, shift=UP * 0.2), run_time=0.8)
            self.wait(self.left())
