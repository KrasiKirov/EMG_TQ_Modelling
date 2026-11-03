from manim import DOWN, LEFT, UP, FadeIn, FadeOut, LaggedStart, VGroup

from style4 import C_FIT, C_HI, C_VAL, DIM, INK, BeatScene, T, W, card, chapter_tag


class Ch01(BeatScene):
    def construct(self):
        with self.beat('c01_b01'):
            title = VGroup(T('Inside the LSTM', 54, INK, weight='BOLD'),
                           T('how the windows are prepared, fed and read · real data', 28, DIM)).arrange(DOWN, buff=0.3)
            self.play(FadeIn(title, shift=UP * 0.2), run_time=0.9)
            self.until(self.cue('The torque at any'))
            self.play(FadeOut(title), run_time=0.5)
            tag = chapter_tag('c01')
            q = card(['“Torque at time t needs information from earlier moments.',
                      'So why cut the recording into windows, and shuffle them?”'], width=10.5, size=30,
                     title='The question', color=C_VAL)
            self.play(FadeIn(tag), FadeIn(q, shift=UP * 0.2), run_time=0.8)
            self.wait(self.left())

        with self.beat('c01_b02'):
            self.play(q.animate.scale(0.6).to_edge(UP, buff=0.75), run_time=0.6)
            items = [('1', 'How the inputs are prepared', 'First, how the inputs'),
                     ('2', 'How they are fed to the model', 'Then, how they are fed'),
                     ('3', 'What happens inside the LSTM', 'Then, what happens'),
                     ('4', 'Why the order of the windows does not matter', 'And finally')]
            rows = VGroup(*[VGroup(T(n, 34, C_HI, weight='BOLD'), W(txt, 30, INK, width=50)).arrange(LEFT * -1, buff=0.4)
                            for n, txt, _ in items]).arrange(DOWN, buff=0.38, aligned_edge=LEFT).move_to([0, -0.7, 0])
            for r, (_, _, cue) in zip(rows, items):
                self.until(self.cue(cue))
                self.play(FadeIn(r, shift=UP * 0.15), run_time=0.6)
            self.until(self.cue('At the end'))
            foot = T('then: the end of a push, and what is still open', 28, C_FIT).next_to(rows, DOWN, buff=0.45)
            self.play(FadeIn(foot), run_time=0.6)
            self.wait(self.left())
