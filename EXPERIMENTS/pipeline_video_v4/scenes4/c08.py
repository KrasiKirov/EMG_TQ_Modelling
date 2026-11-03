import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Create, Dot, FadeIn, FadeOut, LaggedStart, Rectangle, RoundedRectangle,
                   Transform, VGroup)

from common4 import place
from style4 import (ANGLE_C, C_BAD, C_FIT, C_GOOD, C_HI, C_PRED, C_TQ, DIM, INK, PANEL, BeatScene, Plot, T, W, card,
                    facts, load)


def strip(colors, y, label, lcolor):
    cells = VGroup()
    for b in range(7):                         # 7 batches of 8 windows
        batch = VGroup(*[Rectangle(width=0.13, height=0.4, stroke_width=0, fill_color=colors[b * 8 + k], fill_opacity=0.95)
                         for k in range(8)]).arrange(RIGHT, buff=0.02)
        cells.add(batch)
    cells.arrange(RIGHT, buff=0.18).move_to([1.3, y, 0])
    lab = T(label, 24, lcolor).next_to(cells, LEFT, buff=0.35)
    return VGroup(lab, cells)



class Ch08(BeatScene):
    def construct(self):
        z, f = load('w02'), facts()['w02']
        tt, yy = z['trial_t'].astype(float), z['trial_y'].astype(float)
        pt, ps = z['p_time'].astype(float), z['p_shuf'].astype(float)
        rng = np.random.default_rng(5)
        with self.beat('c08_b01'):
            tag = self.intro('c08')
            rng = np.random.default_rng(2)
            rows = VGroup()
            for k in range(6):
                cells = VGroup(*[Rectangle(width=0.22, height=0.32, stroke_width=0, fill_color=C_FIT,
                                           fill_opacity=float(rng.uniform(0.2, 0.9))) for _ in range(20)]).arrange(RIGHT, buff=0.03)
                lbl = T(f'samples t{k}−19 … t{k}', 18, DIM).next_to(cells, LEFT, buff=0.25)
                ans = T(f'→ torque(t{k})', 20, C_HI).next_to(cells, RIGHT, buff=0.25)
                rows.add(VGroup(lbl, cells, ans))
            rows.arrange(DOWN, buff=0.18).move_to([0, 0.6, 0])
            head = T('one row = 20 past samples of the 5 signals → the torque at that moment', 24, INK).next_to(rows, UP, buff=0.35)
            self.play(FadeIn(head), LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.15), run_time=1.6)
            self.until(self.cue('Shuffle the rows'))
            order = [3, 0, 5, 1, 4, 2]
            ys = [r.get_y() for r in rows]
            self.play(*[rows[k].animate.set_y(ys[order.index(k)]) for k in range(6)], run_time=1.2)
            note = card(['least squares gives exactly the same answer in any row order:',
                         'the history lives inside each row'], width=10, size=26, color=C_GOOD).move_to([0, -2.4, 0])
            self.play(FadeIn(note), run_time=0.6)
            self.until(self.cue('Our training examples'))
            self.play(Transform(head, T('our training examples are exactly these rows', 26, C_FIT).move_to(head)), run_time=0.6)
            self.wait(self.left())

        with self.beat('c08_b02'):
            self.clear_all(keep=[tag])
            lo, hi = float(min(yy.min(), pt.min())) - 2, float(max(yy.max(), pt.max())) + 2
            P = place(Plot((0, tt[-1]), (lo, hi), 12.0, 3.4, xticks=[0, 30, 60, 89], yticks=[-30, 0],
                           xfmt=lambda v: f'{v:g} s', ylabel='torque (Nm)'), -5.6, -1.4)
            cy, cp = P.curve(tt, yy, C_TQ, 2.0), P.curve(tt, pt, C_PRED, 2.0)
            leg = T('measured (white) · predicted window by window in time order (green)', 22, DIM).next_to(P, DOWN, buff=0.15)
            self.play(FadeIn(P), Create(cy), run_time=1.2)
            self.play(Create(cp), FadeIn(leg), run_time=1.4)
            self.until(self.cue('Then we shuffled'))
            rng = np.random.default_rng(4)
            deck = VGroup(*[RoundedRectangle(corner_radius=0.05, width=0.5, height=0.34, fill_color=PANEL, fill_opacity=1,
                                             stroke_color=C_FIT, stroke_width=1.6) for _ in range(14)])
            deck.arrange(RIGHT, buff=0.08).move_to([0, 2.6, 0])
            self.play(FadeIn(deck), run_time=0.4)
            for _ in range(2):
                perm = rng.permutation(14)
                pos = [c.get_center() for c in deck]
                self.play(*[deck[k].animate.move_to(pos[perm[k]]) for k in range(14)], run_time=0.5)
            idx = rng.choice(len(tt), 160, replace=False)
            dots = VGroup(*[Dot(P.p(tt[i], ps[i]), radius=0.04, color=C_HI) for i in idx])
            self.play(FadeOut(deck), LaggedStart(*[FadeIn(d, shift=DOWN * 0.3) for d in dots], lag_ratio=0.01),
                      run_time=2.0)
            self.until(self.cue('The two curves are identical'))
            badge = card([f'largest difference over {f["order_test_windows"]:,} windows: '
                          f'{f["order_test_max_abs_Nm"]:.1e} Nm (rounding)'], width=9, size=26, color=C_GOOD)\
                .move_to([0, 2.6, 0])
            self.play(FadeIn(badge), run_time=0.6)
            self.wait(self.left())

        with self.beat('c08_b03'):
            self.clear_all(keep=[tag])
            mixed = [ANGLE_C[i] for i in rng.integers(0, 8, 56)]
            runs = [ANGLE_C[min(7, k // 7)] for k in range(56)]
            s1 = strip(mixed, 1.2, 'shuffled', C_GOOD)
            s2 = strip(runs, -0.4, 'not shuffled', C_BAD)
            head = T('7 batches of 8 windows · colour = ankle angle', 22, DIM).move_to([0.6, 2.3, 0])
            self.play(FadeIn(head), run_time=0.4)
            self.until(self.cue('Shuffled, every batch'))
            self.play(FadeIn(s1), run_time=0.8)
            self.until(self.cue('Not shuffled'))
            self.play(FadeIn(s2), run_time=0.8)
            drift = Arrow([-3.2, -1.3, 0], [5.6, -1.3, 0], buff=0, color=C_BAD, stroke_width=4)
            dl = T('the model keeps leaning towards the last trial it saw', 22, C_BAD).next_to(drift, DOWN, buff=0.1)
            self.play(FadeIn(drift), FadeIn(dl), run_time=0.7)
            self.until(self.cue('Shuffling is about balanced'))
            note = T('about balanced learning, not about time', 28, C_HI).move_to([0, -2.9, 0])
            self.play(FadeIn(note), run_time=0.5)
            self.until(self.cue('And it cannot leak'))
            leak = T('the deck holds session 1 only · session 2 is never in it', 22, DIM).move_to([0, -3.4, 0])
            self.play(FadeIn(leak), run_time=0.5)
            self.wait(self.left())
