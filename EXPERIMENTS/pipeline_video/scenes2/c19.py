"""Chapter 19: Summary and next steps (no code on screen)."""
from manim import *
from style2 import *
import textwrap

F = facts()['y01']['B_group_summaries']['X00_baseline']
R2_OTHER = F['other_36']['r2_floor0']          # 0.97
R2_TD = F['toes_down_12']['r2_floor0']         # 0.77


def step_list(heading, sub, items, size=26, chars=70, top=2.0, gap=0.36):
    """items: list of (number, text-or-Mobject). Returns (head, rows) with rows[i] = VGroup(number, text)."""
    head = VGroup(T(heading, 24, C_HI, weight='BOLD'), T(sub, 38, INK, weight='BOLD')) \
        .arrange(RIGHT, buff=0.3, aligned_edge=DOWN).move_to([0, 2.75, 0]).align_to([-6.7, 0, 0], LEFT)
    rows = VGroup()
    for n, txt in items:
        num = T(str(n), size + 6, C_HI, weight='BOLD')
        if isinstance(txt, str):       # a backslash-n in the text forces a line break; parts wrap separately
            txt = T(chr(10).join(chr(10).join(textwrap.wrap(part, chars)) for part in txt.split(chr(10))), size, INK,
                    line_spacing=0.8)
        body = txt
        row = VGroup(num, body)
        body.next_to(num, RIGHT, buff=0.35).align_to(num, UP)
        rows.add(row)
    rows.arrange(DOWN, buff=gap, aligned_edge=LEFT).move_to([0, 0, 0]).align_to([-6.7, 0, 0], LEFT)
    rows.shift(UP * (top - rows.get_top()[1]))
    for row in rows:                      # right-align the numbers' column
        row[0].align_to([-6.0, 0, 0], RIGHT)
        row[1].align_to([-5.75, 0, 0], LEFT)
        row[1].align_to(row[0], UP)
    return head, rows


def icon_curve(plot_w, plot_h, x, ys, colors, widths=None):
    """Small stacked curves as an icon: ys is a list of arrays scaled to [0,1]."""
    g = VGroup()
    n = len(ys)
    for i, (y, c) in enumerate(zip(ys, colors)):
        y = np.asarray(y, float)
        xs = np.linspace(0, plot_w, len(y))
        base = -i * plot_h / n
        pts = [[a, base + b * plot_h / n * 0.8, 0] for a, b in zip(xs, y)]
        g.add(VMobject(stroke_color=c, stroke_width=2.2).set_points_as_corners(pts))
    return g.move_to(ORIGIN)


class Ch19(BeatScene):
    def construct(self):
        # ── b01: the whole pipeline ──────────────────────────────────────────
        with self.beat('c19_b01'):
            tag = self.intro('c19')
            d = load('y02')
            sl = slice(430, 620)
            tt = d['s1_t'][sl]

            def nrm(a):
                a = np.asarray(a, float)[sl]
                a = a[np.linspace(0, len(a) - 1, 60).astype(int)]
                return (a - a.min()) / max(a.max() - a.min(), 1e-9)
            emg = icon_curve(2.2, 1.7, tt, [nrm(d['s1_gm']), nrm(d['s1_gl']), nrm(d['s1_sol']), nrm(d['s1_ta']),
                                             nrm(d['s1_angle']) * 0.3 + 0.35],
                             MUSCLE_C + [C_ANGLE])
            ic1 = VGroup(emg, W('4 EMG envelopes + angle', 22, DIM, width=16, align=ORIGIN)).arrange(DOWN, buff=0.3)
            # windows: one curve with a bracket (a 0.2 s window)
            wy = nrm(d['s1_sol'])
            wc = icon_curve(2.2, 0.9, tt, [wy], [C_SOL])
            wbox = Rectangle(width=2.2 * 0.2 / float(tt[-1] - tt[0]), height=1.05, stroke_color=C_HI, stroke_width=3, fill_color=C_HI,
                             fill_opacity=0.15).move_to(wc.get_center())
            ic2 = VGroup(VGroup(wc, wbox), W('windows 0.2 s long', 22, DIM, width=20, align=ORIGIN)) \
                .arrange(DOWN, buff=0.3)
            # LSTM: two columns of 8 cells
            cols = VGroup()
            for _ in range(2):
                cols.add(VGroup(*[Circle(radius=0.075, stroke_width=0, fill_color=C_PRED, fill_opacity=0.95)
                                  for _ in range(8)]).arrange(DOWN, buff=0.07))
            cols.arrange(RIGHT, buff=0.45)
            lbox = RoundedRectangle(corner_radius=0.12, width=cols.width + 0.5, height=cols.height + 0.4,
                                    stroke_color=C_PRED, stroke_width=2.5, fill_color=PANEL, fill_opacity=1)
            lstm = VGroup(lbox, cols)
            cols.move_to(lbox)
            ic3 = VGroup(lstm, VGroup(T('small LSTM', 22, DIM), T('2 layers × 8 cells', 22, DIM)).arrange(DOWN, buff=0.1)) \
                .arrange(DOWN, buff=0.3)
            # active torque
            ty = nrm(d['s1_target'])
            tc = icon_curve(2.2, 1.1, tt, [ty], [C_TQ])
            ic4 = VGroup(tc, VGroup(T('active torque (Nm)', 22, DIM),
                                    T('= measured − passive table', 18, DIM)).arrange(DOWN, buff=0.08)) \
                .arrange(DOWN, buff=0.3)
            icons = [ic1, ic2, ic3, ic4]
            xs = [-5.0, -1.65, 1.65, 5.0]
            for ic, x in zip(icons, xs):
                ic.move_to([x, 0.7, 0])
            for ic in icons:       # tops aligned
                ic.shift(UP * (1.1 - ic[0].get_center()[1]))
            arrows = VGroup(*[Arrow([a + 1.4, 1.1, 0], [b - 1.4, 1.1, 0], buff=0.0, stroke_width=4,
                                    color=DIM, max_tip_length_to_length_ratio=0.35)
                              for a, b in zip(xs[:-1], xs[1:])])
            self.until(self.cue('The baseline turns'))
            self.play(FadeIn(ic1, shift=UP * 0.15), run_time=0.8)
            self.until(self.cue('It cuts them'))
            self.play(GrowArrow(arrows[0]), FadeIn(ic2, shift=UP * 0.15), run_time=0.8)
            self.until(self.cue('trains a very small LSTM'))
            self.play(GrowArrow(arrows[1]), FadeIn(ic3, shift=UP * 0.15), run_time=0.8)
            self.until(self.cue('to predict the active torque'))
            self.play(GrowArrow(arrows[2]), FadeIn(ic4, shift=UP * 0.15), run_time=0.8)
            s1 = VGroup(T('learn', 28, INK), T('session 1', 28, C_S1, weight='BOLD')).arrange(RIGHT, buff=0.2)
            s2 = VGroup(T('test', 28, INK), T('session 2', 28, C_S2, weight='BOLD')).arrange(RIGHT, buff=0.2)
            line = VGroup(s1, T('→', 32, DIM), s2).arrange(RIGHT, buff=0.5).move_to([0, -2.1, 0])
            self.until(self.cue('Then it is tested'))
            self.play(FadeIn(s1), FadeIn(line[1]), run_time=0.5)
            self.play(FadeIn(s2), run_time=0.5)

        # ── b02: summary bullets ─────────────────────────────────────────────
        with self.beat('c19_b02'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            hd = T('Summary', 40, INK, weight='BOLD').move_to([0, 2.7, 0]).align_to([-6.7, 0, 0], LEFT)
            items = [
                M(f'other positions: R² <b><span foreground="{C_GOOD}">{fmt(R2_OTHER)}</span></b>', 28),
                M(f'toes-down: R² <b><span foreground="{C_BAD}">{fmt(R2_TD)}</span></b>, though the error in Nm is '
                  'smaller there', 28),
                M('little to predict → offsets are over half of the lost score', 28),
                M('offsets come from the answer key (YES) and from the\nmodel reacting to changed EMG (EG, IES)', 28),
                M('not overfitting session 1: a session shift that\nvalidation cannot see', 28),
                M('at toes-down, the R² depends heavily on the seed', 28),
            ]
            rows = bullets(items, 28, buff=0.32).move_to([0, -0.55, 0]).align_to([-6.7, 0, 0], LEFT)
            self.play(FadeIn(hd), run_time=0.4)
            cues = ['At the thirty-six cells', 'At toes-down it drops', 'because there is so little',
                    'Those offsets come', 'The model is not overfitting', 'And at toes-down']
            for row, c in zip(rows, cues):
                self.until(self.cue(c))
                self.play(FadeIn(row, shift=RIGHT * 0.25), run_time=0.6)

        # ── b03: next steps 1-3 ──────────────────────────────────────────────
        with self.beat('c19_b03'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            head, rows = step_list('Next steps', 'how we measure', [
                (1, 'report MAE, offset and shape-only R² (VAF, already computed) next to R²; flag negative '
                    'values instead of hiding them'),
                (2, "compare against a guess the model could make: session 1's average per position"),
                (3, 'many seeds: at least 5 (as the report says), ideally 10; fully repeatable runs')],
                size=28, chars=62, gap=0.6, top=1.5)
            self.play(FadeIn(head), run_time=0.5)
            for row, c in zip(rows, ['Step one', 'Step two', 'Step three']):
                self.until(self.cue(c))
                self.play(FadeIn(row, shift=RIGHT * 0.25), run_time=0.7)

        # ── b04: next steps 4-7 ──────────────────────────────────────────────
        with self.beat('c19_b04'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            head, rows = step_list('Next steps', 'the answer key and the data', [
                (4, "confirm which code produced last semester's numbers"),
                (5, 'a short relaxed hold right after each active trial'),
                (6, 'try predicting the total torque, as the three modelling papers do'),
                (7, 'a third session on a different day · fix the comparisons in advance · keep some people only '
                    'for confirmation')], size=28, chars=72, gap=0.5, top=1.8)
            self.play(FadeIn(head), run_time=0.5)
            for row, c in zip(rows, ['Step four', 'Step five', 'Step six', 'Step seven']):
                self.until(self.cue(c))
                self.play(FadeIn(row, shift=RIGHT * 0.25), run_time=0.7)

        # ── b05: next steps 8-10 ─────────────────────────────────────────────
        with self.beat('c19_b05'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            head, rows = step_list('Next steps', 'the model (1/2)', [
                (8, 'a clean within-session ceiling: a held-back block with a gap, never used for stopping'),
                (9, 'fine-tune on a little labelled data from the new session' + chr(10) +
                    "(Zhang 2022's idea), e.g. the first 15–20 s of each trial" + chr(10) +
                    '(our choice); score only the rest'),
                (10, 'sweep the envelope smoothing from below 1 Hz up to 6 Hz' + chr(10) + '(Zhang 2019, Zhang 2022)')],
                size=28, chars=68, gap=0.55, top=1.7)
            self.play(FadeIn(head), run_time=0.5)
            for row, c in zip(rows, ['Step eight', 'Step nine', 'Step ten']):
                self.until(self.cue(c))
                self.play(FadeIn(row, shift=RIGHT * 0.25), run_time=0.7)

        # ── b06: next steps 11-13 ────────────────────────────────────────────
        with self.beat('c19_b06'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            head, rows = step_list('Next steps', 'the model (2/2)', [
                (11, VGroup(W('more memory cells (both LSTM papers used far more than 8) · longer context (Kim hints; '
                              'Zhang 2022 used shorter slices)', 26, INK, width=70),
                             W('so far mixed: 0.4 s worse on the baseline, 0.5 s on D01 slightly better', 24, C_HI,
                               width=84)).arrange(DOWN, buff=0.15, aligned_edge=LEFT)),
                (12, 'a simple per-position model as a standing reference (Zhang 2019; an earlier ridge version '
                     'exists)'),
                (13, 'auto-gain only for clearly active muscles')],
                size=26, chars=70, gap=0.5, top=1.8)
            self.play(FadeIn(head), run_time=0.5)
            for row, c in zip(rows, ['Step eleven', 'Step twelve', 'step thirteen']):
                self.until(self.cue(c))
                self.play(FadeIn(row, shift=RIGHT * 0.25), run_time=0.7)

        # ── b07: already tried ───────────────────────────────────────────────
        with self.beat('c19_b07'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            ttl = VGroup(T('Already tried', 34, INK, weight='BOLD'),
                         T('in an earlier round (3 people, faster training settings)', 22, DIM)) \
                .arrange(RIGHT, buff=0.3, aligned_edge=DOWN).move_to([0, 2.85, 0]).align_to([-6.7, 0, 0], LEFT)

            def tried(lead, rest, col=INK):
                return None
            lines = [
                ('a straight-line correction from the first seconds of each session-2 trial',
                 'old pipeline 0.57 → 0.90 with 30 s; newer pipelines +0.01 to +0.02, worse with 5–10 s'),
                ('ridge regression', 'well behind without auto-gain, ≈ 0.05 behind with it'),
                ('pre-training on the other 5 people', 'worse'),
                ('extra weight on toes-down (×3, ×10)', 'no help'),
                ('random EMG-size augmentation', 'helps without auto-gain, adds nothing with it'),
            ]
            rows = VGroup()
            for lead, rest in lines:
                body = VGroup(W(lead, 23, INK, width=chars_for(11.2, 23), weight='BOLD'),
                              W(rest, 23, C_PASSIVE, width=chars_for(11.2, 23))).arrange(DOWN, buff=0.08,
                                                                                         aligned_edge=LEFT)
                rows.add(body)
            rows.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            panel = RoundedRectangle(corner_radius=0.15, width=13.1, height=rows.height + 0.5, fill_color=PANEL,
                                     fill_opacity=1, stroke_color=FAINT, stroke_width=2).move_to([0, -0.5, 0])
            rows.move_to(panel).align_to(panel, LEFT).shift(RIGHT * 0.4)
            self.play(FadeIn(ttl), FadeIn(panel), run_time=0.6)
            for row, c in zip(rows, ['A straight-line correction', 'A ridge regression', 'Pre-training on the other',
                                     'Extra weight on toes-down', 'And random EMG-size']):
                self.until(self.cue(c))
                self.play(FadeIn(row, shift=RIGHT * 0.25), run_time=0.7)

        # ── b08: closing card ────────────────────────────────────────────────
        with self.beat('c19_b08'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            head = T('When asked about our method:', 34, INK, weight='BOLD')
            lines = [('what the code does', C_S1), ('what we tested', C_GOOD), ('what was a choice', C_S2),
                     ("what we don't know yet", C_HI)]
            rows = VGroup(*[T(t, 36, c, weight='BOLD') for t, c in lines]).arrange(DOWN, buff=0.4,
                                                                                    aligned_edge=LEFT)
            grp = VGroup(head, rows).arrange(DOWN, buff=0.55, aligned_edge=LEFT).move_to([0, 0, 0])
            self.play(FadeIn(head), run_time=0.5)
            for row, c in zip(rows, ['here is what the code does', 'here is what we tested',
                                     'here is what was a choice', 'here is what we do not know']):
                self.until(self.cue(c))
                self.play(FadeIn(row, shift=RIGHT * 0.25), run_time=0.5)
            self.until(self.cue('Thanks for watching') - 0.04)
            title = VGroup(T('EMG → ankle torque', 52, INK, weight='BOLD'),
                           T('the baseline, in plain English', 32, DIM)).arrange(DOWN, buff=0.3)
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)
            self.play(FadeIn(title), run_time=0.5)
