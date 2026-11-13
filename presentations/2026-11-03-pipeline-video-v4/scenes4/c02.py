from manim import DOWN, Create, FadeIn, LaggedStart

from common4 import Rows
from style4 import DIM, BeatScene, T, facts, load


class Ch02(BeatScene):
    def construct(self):
        z, f = load('v01'), facts()['v01']['eg_trial']
        with self.beat('c02_b01'):
            self.intro('c02')
            rows = Rows(z, 0, f['seconds'], top=2.85)
            cap = T(f"EG · session 1 · one angle ({f['angle_rad']:+.3f} rad) · {f['seconds']:g} s · 100 samples per second"
                    .replace('-', '−'), 22, DIM).next_to(rows, DOWN, buff=0.25)
            self.play(FadeIn(rows.static), run_time=0.6)
            self.play(LaggedStart(*[Create(c) for c in rows.curves], lag_ratio=0.12), run_time=3.2)
            self.play(FadeIn(cap), run_time=0.5)
            self.wait(self.left())
