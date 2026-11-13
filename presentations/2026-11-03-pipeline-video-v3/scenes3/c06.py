from manim import DOWN, LEFT, RIGHT, FadeIn, Line, Rectangle, RoundedRectangle, VGroup

from style3 import BG, C_FIT, C_GM, C_HI, C_VAL, INK, PANEL, BeatScene, T

C_CAL = C_GM


def icon_window():
    bg = Rectangle(width=0.9, height=0.55, stroke_color=INK, stroke_width=1.5)
    box = Rectangle(width=0.22, height=0.55, stroke_width=0, fill_color=C_HI, fill_opacity=0.6).align_to(bg, LEFT).shift(RIGHT * 0.3)
    return VGroup(bg, box)


def icon_split():
    a = Rectangle(width=0.72, height=0.3, fill_color=C_FIT, fill_opacity=0.9, stroke_width=0)
    b = Rectangle(width=0.18, height=0.3, fill_color=C_VAL, fill_opacity=0.9, stroke_width=0).next_to(a, RIGHT, buff=0)
    return VGroup(a, b)


def icon_deck():
    return VGroup(*[RoundedRectangle(corner_radius=0.04, width=0.6, height=0.4, fill_color=PANEL, fill_opacity=1,
                                     stroke_color=C_FIT, stroke_width=1.5).shift(RIGHT * 0.06 * k + DOWN * 0.05 * k)
                    for k in range(3)])


def icon_line():
    return VGroup(Line([-0.45, -0.25, 0], [0.45, 0.25, 0], stroke_color=C_CAL, stroke_width=4))


class Ch06(BeatScene):
    def construct(self):
        with self.beat('c06_b01'):
            self.intro('c06')
            lines = [
                (icon_window(), 'Session 1 is cut into overlapping 0.2 s windows (one sample apart)', 'Overlapping'),
                (icon_split(), 'In each trial: first 80 % fit (learn), last 20 % validation (decide when to stop)',
                 'The first 80'),
                (icon_deck(), 'Shuffled batches of 8 windows · each window stays in time order', 'Batches of eight'),
                (icon_line(), 'Session 2: 10 s per angle → a stretch and a shift for the predictions', 'And in session'),
            ]
            rows = VGroup(*[VGroup(ic, T(txt, 26, INK)).arrange(RIGHT, buff=0.4) for ic, txt, _ in lines])
            rows.arrange(DOWN, buff=0.55, aligned_edge=LEFT).move_to([0, -0.2, 0])
            for row, (_, _, cue) in zip(rows, lines):
                self.until(self.cue(cue))
                self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.6)
            self.wait(self.left())
