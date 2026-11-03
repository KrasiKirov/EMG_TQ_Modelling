from manim import DOWN, UP, Create, FadeIn, FadeOut, LaggedStart, VGroup

from common3 import Rows
from style3 import DIM, INK, BeatScene, T, chapter_tag, facts, load


class Ch01(BeatScene):
    def construct(self):
        z, f = load('v01'), facts()['v01']['eg_trial']
        with self.beat('c01_b01'):
            title = T('How the model is fed,', 46, INK, weight='BOLD')
            title2 = T('and how the calibration works', 46, INK, weight='BOLD')
            sub = T('the final model · real recordings · session 1 → session 2', 26, DIM)
            card = VGroup(title, title2, sub).arrange(DOWN, buff=0.25)
            self.play(FadeIn(card, shift=UP * 0.2), run_time=0.9)
            self.until(self.cue('Here is one real'))
            self.play(FadeOut(card), run_time=0.5)
            tag = chapter_tag('c01')
            rows = Rows(z, 0, f['seconds'], top=2.85)
            cap = T(f"EG · session 1 · one angle ({f['angle_rad']:+.3f} rad) · {f['seconds']:g} s · 100 samples per second"
                    .replace('-', '−'), 22, DIM).next_to(rows, DOWN, buff=0.25)
            self.play(FadeIn(tag), FadeIn(rows.static), run_time=0.6)
            self.play(LaggedStart(*[Create(c) for c in rows.curves], lag_ratio=0.12), run_time=3.2)
            self.play(FadeIn(cap), run_time=0.5)
            self.wait(self.left())
