import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Create, Cross, FadeIn, FadeOut, LaggedStart, Line, Rectangle, Transform,
                   VGroup)

from common4 import place
from style4 import (C_BAD, C_FIT, C_GOOD, C_HI, C_TEST, C_TQ, C_VAL, DIM, INK, BeatScene, Plot, T, W, card, facts,
                    load)


def split_bar(width, height, frac=0.8, c1=C_FIT, c2=C_VAL):
    a = Rectangle(width=width * frac, height=height, fill_color=c1, fill_opacity=0.85, stroke_width=0)
    b = Rectangle(width=width * (1 - frac), height=height, fill_color=c2, fill_opacity=0.85, stroke_width=0)
    b.next_to(a, RIGHT, buff=0)
    return VGroup(a, b)


class Ch04(BeatScene):
    def construct(self):
        F = facts()['v01']
        f, nw = F['eg_trial'], F['n_windows']
        z = load('v01')
        val_used = nw['val'] // 8

        with self.beat('c04_b01'):
            self.intro('c04')
            head = T(f'EG · one session-1 trial · {f["windows"]:,} windows in time order', 24, DIM).move_to([0, 2.3, 0])
            bar = split_bar(12, 0.8).move_to([0, 1.35, 0])
            arrow = T('time →', 20, DIM).next_to(bar, UP, buff=0.08).align_to(bar, RIGHT)
            self.play(FadeIn(head), FadeIn(bar[0]), FadeIn(bar[1]), FadeIn(arrow), run_time=0.8)
            self.until(self.cue('The first 80 percent'))
            l1 = W(f'fit · {f["fit_windows"]:,} windows · first {f["fit_seconds"]:.0f} s · the model learns from these',
                   22, C_FIT, width=44).next_to(bar[0], DOWN, buff=0.2)
            self.play(FadeIn(l1), run_time=0.6)
            self.until(self.cue('The last 20 percent'))
            l2 = W(f'validation · {val_used:,} windows · last {f["val_seconds"]:.0f} s', 22, C_VAL, width=22)\
                .next_to(bar[1], DOWN, buff=0.2)
            self.play(FadeIn(l2), run_time=0.6)
            self.until(self.cue('It is one block'))
            stripes = VGroup(*[Rectangle(width=0.25, height=0.5, stroke_width=0, fill_opacity=0.85,
                                         fill_color=C_VAL if k % 5 == 4 else C_FIT) for k in range(40)])
            stripes.arrange(RIGHT, buff=0.04).move_to([0, -1.5, 0])
            no = Cross(stripes, stroke_color=C_BAD, stroke_width=6)
            nolab = T('not every 5th window', 22, C_BAD).next_to(stripes, DOWN, buff=0.15)
            yes = T('✓ one block at the end of the trial', 22, C_GOOD).next_to(l2, DOWN, buff=0.25).align_to(l2, RIGHT)
            self.play(FadeIn(stripes), run_time=0.5)
            self.play(Create(no), FadeIn(nolab), FadeIn(yes), run_time=0.8)
            self.wait(self.left())

        with self.beat('c04_b02'):
            self.play(*[FadeOut(m) for m in (stripes, no, nolab, yes, l1, l2, head, arrow)], run_time=0.5)
            angles = f['session1_angles']
            s1 = VGroup(*[split_bar(5.2, 0.32) for _ in angles]).arrange(DOWN, buff=0.14).move_to([-2.6, 0.55, 0])
            labs = VGroup(*[T(f'{a:+.2f} rad'.replace('-', '−'), 17, DIM).next_to(b, LEFT, buff=0.15)
                            for a, b in zip(angles, s1)])
            t1 = T('session 1 · 8 trials, one per angle', 22, INK).next_to(s1, UP, buff=0.25)
            self.play(Transform(bar, s1[5]), FadeIn(t1), run_time=0.8)
            self.play(LaggedStart(*[FadeIn(b) for b in s1], lag_ratio=0.1), FadeIn(labs), run_time=1.2)
            self.remove(bar)
            self.until(self.cue('For one person'))
            tot = VGroup(T(f'fit · {nw["train"]:,} windows', 24, C_FIT), T(f'validation · {nw["val"]:,} windows', 24, C_VAL))\
                .arrange(DOWN, buff=0.12, aligned_edge=LEFT).next_to(s1, DOWN, buff=0.3).align_to(s1, LEFT)
            self.play(FadeIn(tot), run_time=0.6)
            self.until(self.cue('Session 2 is kept'))
            s2 = VGroup(*[Rectangle(width=3.6, height=0.32, fill_color=C_TEST, fill_opacity=0.85, stroke_width=0)
                          for _ in angles]).arrange(DOWN, buff=0.14).move_to([4.6, 0.55, 0])
            t2 = T('session 2', 22, INK).next_to(s2, UP, buff=0.25)
            tot2 = W(f'test · {nw["test"]:,} windows · never used for training', 24, C_TEST, width=26)\
                .next_to(s2, DOWN, buff=0.3)
            wall = Line([1.9, 2.4, 0], [1.9, -2.4, 0], stroke_color=DIM, stroke_width=2)
            self.play(Create(wall), FadeIn(t2), LaggedStart(*[FadeIn(b) for b in s2], lag_ratio=0.08), run_time=1.2)
            self.play(FadeIn(tot2), run_time=0.6)
            self.wait(self.left())

        with self.beat('c04_b03'):
            self.clear_all(keep=[self.tag])
            cards = VGroup(
                card(['the model’s numbers are adjusted on these'], width=4.1, title='Fit', title_color=C_FIT, color=C_FIT),
                card(['only measured, never learned from', 'decides how long to train, and which version to keep'],
                     width=4.1, title='Validation', title_color=C_VAL, color=C_VAL),
                card(['session 2', 'scored at the very end'], width=4.1, title='Test', title_color=C_TEST, color=C_TEST),
            ).arrange(RIGHT, buff=0.35, aligned_edge=UP).move_to([0, 0.3, 0])
            big = T('validation ≠ fine-tuning', 30, C_VAL, weight='BOLD').next_to(cards, DOWN, buff=0.5)
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.3), run_time=1.4)
            self.play(FadeIn(big), run_time=0.6)
            self.wait(self.left())

    def intro(self, cid, hold=1.3):
        self.tag = super().intro(cid, hold)
        return self.tag
