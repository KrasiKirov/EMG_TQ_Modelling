from manim import DOWN, LEFT, RIGHT, UP, FadeIn, VGroup

from style4 import C_FIT, C_HI, C_PRED, C_VAL, INK, BeatScene, T, W


class Ch11(BeatScene):
    def construct(self):
        with self.beat('c11_b01'):
            self.intro('c11')
            lines = [('■', 'Each window is a complete example: the last 0.2 s of muscle activity → the torque at its end',
                      'Each window is', C_HI),
                     ('■', 'Windows are dealt in shuffled batches of 8, and each one is read on its own', 'The windows are dealt', C_FIT),
                     ('■', 'Inside, the memory builds up over the 20 steps, and starts empty for every window', 'Inside, the', C_PRED),
                     ('■', 'Shuffling changes the order of the examples, not their contents: it balances training',
                      'So shuffling changes', C_FIT),
                     ('■', 'Open question: the length of the look-back (test 0.5 s and 1 s windows)', 'And the open question', C_VAL)]
            rows = VGroup(*[VGroup(T(i, 24, c), W(txt, 24, INK, width=70)).arrange(RIGHT, buff=0.3, aligned_edge=UP)
                            for i, txt, _, c in lines])
            rows.arrange(DOWN, buff=0.42, aligned_edge=LEFT).move_to([0, -0.2, 0])
            for r, (_, _, cue, _) in zip(rows, lines):
                self.until(self.cue(cue))
                self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.6)
            self.wait(self.left())
