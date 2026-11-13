from manim import DOWN, RIGHT, UP, FadeIn, LaggedStart, VGroup

from style4 import C_FIT, C_GOOD, C_VAL, BeatScene, card


class Ch10(BeatScene):
    def construct(self):
        with self.beat('c10_b01'):
            self.intro('c10')
            settled = card(['the history is inside each window', 'window order does not change the predictions',
                            'shuffling only balances the training', 'the predictions are not late'], width=5.8, size=24,
                           title='Settled', color=C_GOOD, title_color=C_GOOD)
            open_ = card(['is 0.2 s the best look-back?',
                          'hand-wrist study: the error kept falling up to about 0.3–0.4 s of history (Wang et al. 2024)',
                          'next: the same model with 0.5 s and 1 s windows'], width=6.2, size=24,
                         title='Open', color=C_VAL, title_color=C_VAL)
            VGroup(settled, open_).arrange(RIGHT, buff=0.5, aligned_edge=UP).move_to([0, -0.1, 0])
            self.play(FadeIn(settled, shift=UP * 0.2), run_time=0.8)
            self.until(self.cue('What is open'))
            self.play(FadeIn(open_, shift=UP * 0.2), run_time=0.8)
            self.wait(self.left())
