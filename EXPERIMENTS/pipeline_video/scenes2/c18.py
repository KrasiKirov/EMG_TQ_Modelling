"""Chapter 18: What the papers did differently (no code on screen)."""
from manim import *
from style2 import *

C_OURS, C_Z22, C_KIM, C_Z19, C_RAN = C_PRED, '#6FC3FF', '#C792EA', '#FFB86C', '#FF9AA2'
PAPER = {'ours': ('Ours', C_OURS), 'z22': ('Zhang 2022', C_Z22), 'kim': ('Kim 2020', C_KIM),
         'z19': ('Zhang 2019', C_Z19), 'ran': ('Ranaldi 2022', C_RAN)}
DIFFS = ['sessions', 'target', 'score', 'network', 'envelope', 'scaling']
LABEL_X, TEXT_X, RIGHT_X = -6.7, -4.45, 6.7


def make_table(rows, top=1.95, size=24, gap=0.4):
    """rows: list of (key, text, note-or-None). Returns a Group with .row[key] = VGroup(label, text[, note]).
    Parts are positioned for the full content (notes included) but only label+text are meant to be added first."""
    chars = chars_for(RIGHT_X - TEXT_X, size)
    g = VGroup()
    g.row, g.parts = {}, {}
    y = top
    for i, (key, txt, note) in enumerate(rows):
        name, col = PAPER[key]
        label = T(name, size, col, weight='BOLD')
        body = W(txt, size, INK, width=chars)
        note_m = W(note, 22, INK, width=chars_for(RIGHT_X - TEXT_X, 22)) if note else None
        text_block = VGroup(body, note_m).arrange(DOWN, buff=0.12, aligned_edge=LEFT) if note_m else body
        h = max(label.height, text_block.height)
        label.move_to([0, y - h / 2, 0]).align_to([LABEL_X, 0, 0], LEFT)
        text_block.move_to([0, y - h / 2, 0]).align_to([TEXT_X, 0, 0], LEFT)
        label.align_to(text_block, UP)
        sep = Line([LABEL_X, y - h - gap / 2, 0], [RIGHT_X, y - h - gap / 2, 0], stroke_color=FAINT, stroke_width=1)
        main = VGroup(label, body)
        g.parts[key] = dict(label=label, body=body, note=note_m, main=main, sep=sep,
                            box=VGroup(label, text_block))
        y -= h + gap
    return g


def chips(active):
    """Bottom strip of the six differences; ``active`` index is lit, earlier ones are dim, later ones faint."""
    items = VGroup()
    for i, n in enumerate(DIFFS):
        col = C_HI if i == active else DIM
        items.add(T(f'{i + 1} {n}', 20, col, weight='BOLD' if i == active else 'NORMAL'))
        if i < active:
            items[-1].set_opacity(0.55)
    items.arrange(RIGHT, buff=0.55).move_to([0, -3.45, 0])
    return items


def header(n, name):
    lab, word = T(f'Difference {n}', 24, C_HI, weight='BOLD'), T(name, 40, INK, weight='BOLD')
    # descender depth of the big word, so that the two baselines (not the bounding boxes) line up
    nodesc = name.translate(str.maketrans('gpqyj', 'oovvi'))
    depth = word.height - T(nodesc, 40, INK, weight='BOLD').height
    g = VGroup(lab, word).arrange(RIGHT, buff=0.35, aligned_edge=DOWN)
    word.shift(DOWN * depth)
    return g.move_to([0, 2.75, 0]).align_to([LABEL_X, 0, 0], LEFT)


def glow(mob, color=C_HI, pad=0.07):
    return RoundedRectangle(corner_radius=0.1, width=mob.width + 2 * pad, height=mob.height + 2 * pad,
                            stroke_color=color, stroke_width=3, fill_color=color, fill_opacity=0.08).move_to(mob)


class Ch18(BeatScene):
    def go(self, tag, n, name):
        """Fade everything except the tag and the chip strip, move the chip highlight, return the new header."""
        hd, newchips = header(n, name), chips(n - 1)
        gone = [m for m in self.mobjects if m is not tag and m is not self.chip]
        self.play(*[FadeOut(m) for m in gone], Transform(self.chip, newchips), run_time=0.6)
        return hd

    def construct(self):
        # ── b01: the four papers ─────────────────────────────────────────────
        with self.beat('c18_b01'):
            tag = self.intro('c18')
            specs = [
                ('z22', 'Zhang et al. 2022', 'LSTM', 'hip, knee, ankle torque in walking, jumps, squats',
                 'Zhang and colleagues, 2022'),
                ('kim', 'Kim et al. 2020', 'LSTM', 'wrist angle and torque', 'Kim and colleagues'),
                ('z19', 'Zhang et al. 2019', 'high-density EMG', 'elbow force', 'Zhang and colleagues, 2019'),
                ('ran', 'Ranaldi et al. 2022', 'EMG processing', 'how to compute the EMG envelope',
                 'Ranaldi and colleagues'),
            ]
            cards = []
            for key, name, kind, what, _ in specs:
                col = PAPER[key][1]
                c = card([T(f'{kind}', 22, col, weight='BOLD'), W(what, 26, INK, width=34)], width=6.2,
                         title=name, color=col, size=24, title_color=INK)
                cards.append(c)
            grid = VGroup(*cards)
            hh = max(c.box.height for c in cards)
            for c in cards:
                top = c.box.get_top()
                c.box.stretch_to_fit_height(hh)
                c.box.align_to(top, UP)
            cards[0].move_to([-3.35, 1.45, 0]); cards[1].move_to([3.35, 1.45, 0])
            cards[2].move_to([-3.35, -1.5, 0]); cards[3].move_to([3.35, -1.5, 0])
            note = T('the file name says "Renaldi"', 18, DIM).next_to(cards[3], DOWN, buff=0.15)
            for (key, name, kind, what, phrase), c in zip(specs, cards):
                self.until(self.cue(phrase))
                self.play(FadeIn(c, shift=UP * 0.2), run_time=0.7)
            self.play(FadeIn(note), run_time=0.5)

        # ── b02: difference 1, sessions ──────────────────────────────────────
        with self.beat('c18_b02'):
            self.play(FadeOut(grid), FadeOut(note), run_time=0.5)
            hd, self.chip = header(1, 'sessions'), chips(0)
            tb = make_table([
                ('ours', '2 sessions: learn one, test the other', None),
                ('z22', 'held-back part of the same recordings, new people, new movements', None),
                ('kim', 'a new 80 s phase after a 2-min break', None),
                ('z19', 'other repetitions of the same trial', None),
                ('ran', 'no model', None)])
            P = tb.parts
            self.play(FadeIn(hd), FadeIn(self.chip), FadeIn(P['ours']['box']), run_time=0.7)
            self.until(self.cue('Kim comes closest'))
            self.play(FadeIn(P['kim']['box']), run_time=0.6)
            self.until(self.cue('The others test'))
            self.play(FadeIn(P['z22']['box']), run_time=0.6)
            self.play(FadeIn(P['z19']['box']), FadeIn(P['ran']['box']), run_time=0.6)
            self.until(self.cue('Our two sessions'))
            gl = glow(P['ours']['box'])
            self.play(FadeIn(gl), run_time=0.6)
            self.until(self.cue('So we should not expect'))
            msg = T('so: do not expect to match their scores', 26, C_HI).move_to([0, -2.75, 0])
            self.play(FadeIn(msg, shift=UP * 0.1), run_time=0.6)

        # ── b03: difference 2, target ────────────────────────────────────────
        with self.beat('c18_b03'):
            hd = self.go(tag, 2, 'target')
            tb = make_table([
                ('ours', 'active = measured − passive table', None),
                ('z22', 'total joint torque (motion capture + force plates)', None),
                ('kim', 'measured net torque', None),
                ('z19', 'total force', None),
                ('ran', 'no model', None)])
            P = tb.parts
            self.play(FadeIn(hd), FadeIn(P['ours']['box']), run_time=0.6)
            self.until(self.cue('The three modelling papers'))
            self.play(FadeIn(P['z22']['box']), run_time=0.5)
            self.play(FadeIn(P['kim']['box']), run_time=0.5)
            self.play(FadeIn(P['z19']['box']), FadeIn(P['ran']['box']), run_time=0.5)
            self.until(self.cue('Our subtraction'))
            gl = glow(P['ours']['box'])
            self.play(FadeIn(gl), run_time=0.5)

        # ── b04: difference 3, score ─────────────────────────────────────────
        with self.beat('c18_b04'):
            hd = self.go(tag, 3, 'score')
            rows3 = [
                ('ours', 'R² per position (punishes offsets)', None),
                ('z22', 'error per kg of body mass · error ÷ torque range', None),
                ('kim', 'variance accounted for (formula not given; usually ignores offsets)',
                 'torque not scored in the zero-resistance condition: mostly noise'),
                ('ran', 'correlation at the best lag (ignores offset and scale) + error after normalising', None),
                ('z19', 'R² per force level: best 0.93 at 60 % of max effort → 0.84 at 20 %', None)]
            # b04 shows the rows packed (no empty slot); b05 opens the slot for Kim's note and pushes the rows
            # below it down (tb3 = the final layout, notes included)
            tb = make_table([(k, t, None) for k, t, _ in rows3])
            self.tb3 = make_table(rows3)
            self.tb3a = tb
            P = tb.parts
            self.play(FadeIn(hd), FadeIn(P['ours']['box'][0]), FadeIn(P['ours']['body']), run_time=0.6)
            self.until(self.cue('Zhang 2022 reports'))
            self.play(FadeIn(P['z22']['box']), run_time=0.5)
            self.until(self.cue('Kim reports'))
            self.play(FadeIn(P['kim']['main']), run_time=0.5)
            self.until(self.cue('Ranaldi mainly'))
            self.play(FadeIn(P['ran']['box']), run_time=0.5)

        # ── b05: score notes ─────────────────────────────────────────────────
        with self.beat('c18_b05'):
            P, Pa = self.tb3.parts, self.tb3a.parts
            self.until(self.cue('Kim did not score'))
            self.play(FadeIn(P['kim']['note']), Pa['ran']['box'].animate.move_to(P['ran']['box']), run_time=0.7)
            self.until(self.cue('Our toes-down positions'))
            self.play(Indicate(P['kim']['note'], color=C_HI, scale_factor=1.05), run_time=0.9)
            self.until(self.cue('And Zhang 2019'))
            self.play(FadeIn(P['z19']['box']), run_time=0.7)
            self.until(self.cue('their best went'))
            self.play(Indicate(P['z19']['body'], color=C_HI, scale_factor=1.03), run_time=0.9)

        # ── b06: difference 4, network ───────────────────────────────────────
        with self.beat('c18_b06'):
            hd = self.go(tag, 4, 'network')
            tb = make_table([
                ('ours', '2 layers × 8 cells · dropout 0.3 · batch 8 · 0.2 s windows', None),
                ('z22', '2–4 layers × 50 cells · dropout 0.4 · batch 50 · input slices of only 5 time steps', None),
                ('kim', '1 layer × 200 cells · dropout 0.5 · for one participant, shuffling the training data in '
                        'short chunks hurt, mostly for the angle', None),
                ('z19', 'a cubic polynomial: 4 numbers, fitted per person and force level', None)], top=1.95, gap=0.45)
            P = tb.parts
            self.play(FadeIn(hd), run_time=0.5)
            self.until(self.cue('Zhang 2022 used two to four'))
            self.play(FadeIn(P['z22']['box']), run_time=0.6)
            self.until(self.cue('Kim one layer'))
            self.play(FadeIn(P['kim']['main']), run_time=0.6)
            self.until(self.cue('ours has two layers'))
            self.play(FadeIn(P['ours']['box']), run_time=0.6)
            self.until(self.cue('Zhang 2022 fed even shorter'))
            self.play(Indicate(P['z22']['body'], color=C_HI, scale_factor=1.03), run_time=0.9)
            self.until(self.cue('Kim found'))
            self.play(Indicate(P['kim']['body'], color=C_HI, scale_factor=1.03), run_time=0.9)
            self.until(self.cue('And Zhang 2019'))
            self.play(FadeIn(P['z19']['box']), run_time=0.6)

        # ── b07: difference 5, envelope ──────────────────────────────────────
        with self.beat('c18_b07'):
            hd = self.go(tag, 5, 'envelope')
            tb = make_table([
                ('ours', '30–300 Hz · rectify · 2 Hz', None),
                ('z22', '30–300 Hz · rectify · 6 Hz', None),
                ('z19', '20–400 Hz · cut-off tuned per person and force level: 1–3 Hz, most often 1 Hz, the lowest '
                        'allowed', None),
                ('ran', '25–450 Hz + 50 Hz notch · slow smoothing (200–500 ms) scored highest on their correlation '
                        'measure, which favours smooth envelopes; an adaptive envelope was comparable', None),
                ('kim', 'size (RMS) + frequency-shape features over 100 ms', None),],
                top=1.95, size=22, gap=0.3)
            P = tb.parts
            self.play(FadeIn(hd), run_time=0.5)
            self.until(self.cue('Our band-pass'))
            self.play(FadeIn(P['ours']['box']), FadeIn(P['z22']['box']), run_time=0.7)
            self.until(self.cue('Zhang 2019 tuned'))
            P['ran']['body'].set_color(DIM)
            self.play(FadeIn(P['z19']['box']), FadeIn(P['ran']['box']), run_time=0.6)
            self.until(self.cue('Kim used different'))
            self.play(FadeIn(P['kim']['box']), run_time=0.6)
            self.until(self.cue('Unlike the size'))
            gl = glow(P['kim']['box'])
            self.play(FadeIn(gl), run_time=0.6)

        # ── b08: difference 6, scaling ───────────────────────────────────────
        with self.beat('c18_b08'):
            hd = self.go(tag, 6, 'scaling')
            tb = make_table([
                ('ours', '÷ session-1 maximum, also applied to session 2', None),
                ('z22', '÷ maximum over all its trials (same recording as the test)', None),
                ('z19', 'maximum effort (MVC) only sets the target force levels', None),
                ('kim', 'none described', None),
                ('ran', 'not described', None)], top=1.95, gap=0.4)
            P = tb.parts
            self.play(FadeIn(hd), run_time=0.5)
            self.until(self.cue('A textbook answer'))
            self.play(FadeIn(P['ours']['box']), run_time=0.6)
            self.until(self.cue('Zhang 2022 divides'))
            self.play(FadeIn(P['z22']['box']), run_time=0.6)
            self.until(self.cue('Zhang 2019 uses'))
            self.play(FadeIn(P['z19']['box']), run_time=0.6)
            self.until(self.cue('Kim describes'))
            self.play(FadeIn(P['kim']['box']), FadeIn(P['ran']['box']), run_time=0.6)
            self.until(self.cue('We did try'))
            side = card([W('tried here earlier: ÷ MVC did worse than auto-gain', 24, INK, width=60)], width=8.8,
                        color=C_HI, size=24).move_to([0, -2.6, 0])
            self.play(FadeIn(side, shift=UP * 0.15), run_time=0.7)

        # ── b09: verdict ─────────────────────────────────────────────────────
        with self.beat('c18_b09'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.6)
            left = card([bullets(['LSTM on EMG envelopes + joint angle', 'same band-pass', 'squared-error loss',
                                  'early stopping'], 28, chars=26, bcolor=C_GOOD)], width=6.4,
                        title='Follows Zhang 2022 closely', color=C_GOOD, size=28, title_color=C_GOOD)
            right = card([bullets(['passive subtraction', '8 cells', "applying session 1's scale to session 2 "
                                   '(a problem none of the papers faced)'], 28, chars=26, bcolor=C_S2)], width=6.4,
                         title='Our own choices', color=C_S2, size=28, title_color=C_S2)
            left.move_to([-3.4, 0.2, 0]); right.move_to([3.4, 0.2, 0])
            left.align_to(right, UP)
            self.play(FadeIn(left, shift=UP * 0.2), run_time=0.8)
            self.until(self.cue('But the passive subtraction'))
            self.play(FadeIn(right, shift=UP * 0.2), run_time=0.8)

        # ── b10: transfer learning ───────────────────────────────────────────
        with self.beat('c18_b10'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            title = VGroup(T('Zhang 2022: worst-case error', 28, INK), T('(% of the torque range)', 22, DIM)).arrange(DOWN, buff=0.1).move_to([-3.4, 2.5, 0])
            ch = bars([20, 10], ['new person, no calibration', 'pre-trained network, retrained on that person'],
                      [C_BAD, C_GOOD], width=5.6, height=3.0, vmax=24, value_fmt=lambda v: f'≈ {v:.0f} %',
                      label_size=20, value_size=36, gap=1.2)
            ch.move_to([-3.4, -0.85, 0])
            desc = card([W('Take the network trained on other people, replace its last layers, retrain it on '
                           "recordings from the new person (all layers gently adjusted).", 22, INK, width=40)],
                        width=5.6, color=FAINT, size=22).move_to([3.9, 1.5, 0])
            self.play(FadeIn(title), run_time=0.5)
            self.until(self.cue('For a new person'))
            self.play(FadeIn(ch.axis), FadeIn(ch.labels[0]), GrowFromEdge(ch.bars[0], DOWN), run_time=0.9)
            self.play(FadeIn(ch.vals[0]), run_time=0.4)
            self.until(self.cue('roughly double'))
            base = DashedVMobject(Line([-3.4 - 3.1, ch.axis.get_y() + 8.7 / 24 * 3.0, 0],
                                       [-3.4 + 4.3, ch.axis.get_y() + 8.7 / 24 * 3.0, 0], stroke_color=C_HI,
                                       stroke_width=2), num_dashes=40)
            bl = T('within-person worst case: 8.7 %', 18, C_HI)
            bl.next_to(base, RIGHT, buff=0.15)
            self.play(Create(base), FadeIn(bl), run_time=0.7)
            self.until(self.cue('They then took'))
            self.play(FadeIn(desc, shift=LEFT * 0.2), run_time=0.7)
            self.until(self.cue('That brought'))
            self.play(FadeIn(ch.labels[1]), GrowFromEdge(ch.bars[1], DOWN), run_time=0.9)
            self.play(FadeIn(ch.vals[1]), run_time=0.4)
