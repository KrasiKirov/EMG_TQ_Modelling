from manim import *
from style2 import *

Y2 = load('y02')
SIG_C = [C_GM, C_GL, C_SOL, C_TA, C_ANGLE]
CY = 0.7                      # vertical centre of the network diagram
MEM = C_HI                    # colour of "memory"


# ── real input windows (EG, normalised envelopes) ───────────────────────────
def real_window(prefix, start=None, absolute=False):
    """(20, 5) opacities: four muscle envelopes (scaled by their own max) + a constant angle row."""
    arr = np.stack([Y2[f'{prefix}_{k}'] for k in ('gm', 'gl', 'sol', 'ta')], 1)
    arr = arr / arr.max(0)
    if start is None:                                  # the window whose columns change most
        starts = np.arange(0, len(arr) - 20, 5)
        start = int(starts[np.argmax([arr[s:s + 20].std(0).sum() for s in starts])])
    w = arr[start:start + 20]
    if absolute:                                       # brightness relative to the whole trial: a quiet window stays dim
        op = 0.15 + 0.85 * w
    else:
        lo, hi = w.min(0), w.max(0)
        op = 0.15 + 0.85 * (w - lo) / np.maximum(hi - lo, 1e-9)
    return np.concatenate([op, np.full((20, 1), 0.5)], 1)


WIN_A, WIN_B, WIN_C = real_window('s1'), real_window('s2', 3000, absolute=True), real_window('s1', 5200)


def window_grid(win, cw=0.1, ch=0.2, gap=0.014):
    cols = VGroup()
    for t in range(20):
        col = VGroup(*[Rectangle(width=cw, height=ch, stroke_width=0, fill_color=SIG_C[r], fill_opacity=float(win[t, r]))
                       for r in range(5)]).arrange(DOWN, buff=gap)
        cols.add(col)
    cols.arrange(RIGHT, buff=gap)
    frame = SurroundingRectangle(cols, buff=0.05, stroke_color=FAINT, stroke_width=1.5, corner_radius=0.05)
    g = VGroup(cols, frame)
    g.cols, g.frame = cols, frame
    return g


def lstm_layer(name='LSTM layer'):
    box = RoundedRectangle(corner_radius=0.14, width=2.1, height=1.8, stroke_color=INK, stroke_width=2,
                           fill_color=PANEL, fill_opacity=1)
    title = T(name, 22, INK, weight='BOLD').move_to(box.get_top() + DOWN * 0.38)
    sq = VGroup(*[Square(0.18, stroke_color=MEM, stroke_width=1.2, fill_color=MEM, fill_opacity=0.1)
                  for _ in range(8)]).arrange(RIGHT, buff=0.03).move_to(box.get_center() + DOWN * 0.08)
    lab = T('8 memory cells', 18, DIM).move_to(box.get_bottom() + UP * 0.37)
    g = VGroup(box, title, sq, lab)
    g.box, g.sq, g.lab = box, sq, lab
    return g


def output_box(w=1.8):
    box = RoundedRectangle(corner_radius=0.14, width=w, height=1.8, stroke_color=C_PRED, stroke_width=2,
                           fill_color=PANEL, fill_opacity=1)
    txt = VGroup(T('output', 20, DIM), T('1 number', 26, C_PRED, weight='BOLD'), T('active torque', 20, INK),
                 T('in Nm', 20, DIM)).arrange(DOWN, buff=0.08).move_to(box)
    g = VGroup(box, txt)
    g.box = box
    return g


def centers(widths, gap):
    total = sum(widths) + gap * (len(widths) - 1)
    x, out = -total / 2, []
    for w in widths:
        out.append(x + w / 2)
        x += w + gap
    return out


def arrow(a, b):
    return Arrow(a, b, buff=0.06, color=DIM, stroke_width=3, tip_length=0.16, max_stroke_width_to_length_ratio=30,
                 max_tip_length_to_length_ratio=0.6)


class Ch08(BeatScene):
    def construct(self):
        # ============================================================ b01 : the stack
        with self.beat('c08_b01'):
            tag = self.intro('c08')
            head = M('<b>LSTM</b> = long short-term memory, a type built for sequences', 28).move_to([0, 2.55, 0])
            win = window_grid(WIN_A)
            inp_lab = W('input window: 20 time steps × 5 signals', 18, DIM, width=22, align=ORIGIN)
            l1, l2, out = lstm_layer(), lstm_layer(), output_box()
            xs = centers([2.3, 2.1, 2.1, 1.8], 0.8)
            win.move_to([xs[0], CY, 0])
            inp_lab.next_to(win, DOWN, buff=0.2)
            l1.move_to([xs[1], CY, 0])
            l2.move_to([xs[2], CY, 0])
            out.move_to([xs[3], CY, 0])
            ars = VGroup(arrow(win.get_right(), l1.get_left()), arrow(l1.get_right(), l2.get_left()),
                         arrow(l2.get_right(), out.get_left()))
            foot = VGroup(T('1,001', 54, C_HI, weight='BOLD'), T('adjustable numbers (weights)', 30, INK)).arrange(
                RIGHT, buff=0.3, aligned_edge=DOWN).move_to([0, -1.9, 0])

            self.play(FadeIn(head, shift=DOWN * 0.15), run_time=0.8)
            self.until(self.cue('It reads the window'))
            self.play(FadeIn(win), FadeIn(inp_lab), run_time=0.9)
            self.until(self.cue('and keeps a small memory'))
            self.play(FadeIn(l1, shift=RIGHT * 0.2), GrowArrow(ars[0]), run_time=0.9)
            self.until(self.cue('The baseline stacks two'))
            self.play(FadeIn(l2, shift=RIGHT * 0.2), GrowArrow(ars[1]), run_time=0.9)
            self.until(self.cue('Each has a memory of just eight numbers'))
            self.play(*[s.animate.set_fill(opacity=o) for s, o in zip(l1.sq, np.linspace(0.25, 0.9, 8))],
                      *[s.animate.set_fill(opacity=o) for s, o in zip(l2.sq, np.linspace(0.9, 0.25, 8))], run_time=1.0)
            self.play(Indicate(l1.sq, color=MEM, scale_factor=1.12), Indicate(l2.sq, color=MEM, scale_factor=1.12),
                      run_time=1.2)
            self.until(self.cue('After them comes a single output'))
            self.play(FadeIn(out, shift=RIGHT * 0.2), GrowArrow(ars[2]), run_time=0.9)
            self.until(self.cue('In total, the network has'))
            self.play(FadeIn(foot, shift=UP * 0.15), run_time=0.9)
        self.play(FadeOut(head), FadeOut(foot), run_time=0.5)

        # ============================================================ b02 : read the window step by step
        with self.beat('c08_b02'):
            rng = np.random.default_rng(8)
            cap = T('like reading a sentence, word by word', 26, DIM).move_to([0, -2.1, 0])
            hi = SurroundingRectangle(win.cols[0], buff=0.03, stroke_color=INK, stroke_width=3, corner_radius=0.04)
            self.play(FadeIn(cap), FadeIn(hi), run_time=0.6)
            f_end = self.cue('After the twentieth time step')
            n_steps = 20
            t0b, durb = self._beat
            step_t = max(0.2, (t0b + f_end * durb - self.renderer.time - 0.1) / n_steps)
            s1v, s2v = rng.random(8), rng.random(8)
            lab_prev = None
            for k in range(n_steps):
                s1v = 0.55 * s1v + 0.45 * rng.random(8)
                s2v = 0.65 * s2v + 0.35 * s1v
                lab = T(f'time step {k + 1} of 20', 20, INK).next_to(win, UP, buff=0.18)
                if lab_prev is not None:
                    self.remove(lab_prev)
                self.add(lab)
                lab_prev = lab
                self.play(hi.animate.move_to(win.cols[k]),
                          *[s.animate.set_fill(opacity=float(0.08 + 0.92 * v)) for s, v in zip(l1.sq, s1v)],
                          *[s.animate.set_fill(opacity=float(0.08 + 0.92 * v)) for s, v in zip(l2.sq, s2v)],
                          run_time=step_t, rate_func=linear)
            self.until(f_end)
            pred = T('prediction for this window', 22, C_PRED).next_to(out, DOWN, buff=0.25)
            self.play(out.box.animate.set_fill(C_PRED, opacity=0.15).set_stroke(C_PRED, width=5), FadeIn(pred, shift=UP * 0.1),
                      FadeOut(cap), FadeOut(hi), FadeOut(lab_prev), run_time=0.9)
            self.play(Indicate(out, color=C_PRED, scale_factor=1.12), run_time=0.9)

        # ============================================================ b03 : memory starts empty every time
        with self.beat('c08_b03'):
            self.play(*[s.animate.set_fill(opacity=0.1) for s in (*l1.sq, *l2.sq)],
                      out.box.animate.set_fill(PANEL, opacity=1).set_stroke(C_PRED, width=2), FadeOut(pred), run_time=1.2)
            mem_lab = T('memory: empty', 22, MEM).next_to(VGroup(l1, l2), DOWN, buff=0.35)
            self.play(FadeIn(mem_lab, shift=UP * 0.1), run_time=0.6)
            self.until(self.cue('Nothing is carried over'))
            win_b = window_grid(WIN_B).move_to(win)
            next_lab = W('next window: another trial, another time', 20, DIM, width=22, align=ORIGIN).next_to(win_b, UP, buff=0.2)
            self.play(win.animate.shift(LEFT * 1.2).set_opacity(0), FadeIn(win_b, shift=RIGHT * 1.2), FadeIn(next_lab),
                      run_time=1.2)
            self.remove(win)
            win = win_b
            self.until(self.cue('and since the windows are shuffled'))
            chips = VGroup()
            for i in range(5):
                stripes = VGroup(*[Rectangle(width=0.5, height=0.045, stroke_width=0, fill_color=SIG_C[r],
                                             fill_opacity=0.35 + 0.6 * ((i * 3 + r * 2) % 5) / 4) for r in range(5)]
                                 ).arrange(DOWN, buff=0.02)
                chips.add(VGroup(RoundedRectangle(corner_radius=0.05, width=0.68, height=0.5, stroke_color=FAINT,
                                                  stroke_width=1.5, fill_color=PANEL, fill_opacity=1), stripes))
            chips.arrange(RIGHT, buff=0.22).move_to([0, -1.3, 0])
            q_lab = T('windows come in shuffled order', 20, DIM).next_to(chips, DOWN, buff=0.18)
            self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.1) for c in chips], lag_ratio=0.12), FadeIn(q_lab),
                      run_time=1.0)
            pos = [c.get_center() for c in chips]
            for perm in ([2, 4, 0, 3, 1], [3, 0, 4, 1, 2]):
                self.play(*[c.animate.move_to(pos[p]) for c, p in zip(chips, perm)], run_time=0.8)
            self.until(self.cue('Each prediction is made'))
            self.play(FadeOut(next_lab), run_time=0.3)
            brace = VGroup(Line(win.get_corner(UL) + UP * 0.2, win.get_corner(UR) + UP * 0.2),
                           Line(win.get_corner(UL) + UP * 0.1, win.get_corner(UL) + UP * 0.3),
                           Line(win.get_corner(UR) + UP * 0.1, win.get_corner(UR) + UP * 0.3)
                           ).set_stroke(C_HI, 2.5)
            b_lab = T('0.2 s', 24, C_HI, weight='BOLD').next_to(brace, UP, buff=0.1)
            note = M('memory starts empty for every window:\nthe network sees <b>0.2 s</b>, nothing else', 28)
            note.move_to([0, -2.9, 0])
            self.play(Create(brace), FadeIn(b_lab), FadeIn(note, shift=UP * 0.1), run_time=1.0)
        self.play(FadeOut(VGroup(chips, q_lab, brace, b_lab, note, mem_lab)), run_time=0.5)

        # ============================================================ b04 : dropout
        with self.beat('c08_b04'):
            xs2 = centers([2.3, 2.1, 1.2, 2.1, 1.2, 1.8], 0.42)

            def dropout_box():
                box = RoundedRectangle(corner_radius=0.14, width=1.2, height=1.8, stroke_color=C_BAD, stroke_width=2,
                                       fill_color=PANEL, fill_opacity=1)
                t1 = T('dropout', 20, INK).move_to(box.get_top() + DOWN * 0.3)
                t2 = T('0.3', 26, C_BAD, weight='BOLD').next_to(t1, DOWN, buff=0.08)
                dots = VGroup(*[Circle(0.085, stroke_width=0, fill_color=MEM, fill_opacity=0.65) for _ in range(8)])
                dots.arrange_in_grid(4, 2, buff=0.1).move_to(box.get_bottom() + UP * 0.52)
                g = VGroup(box, t1, t2, dots)
                g.dots = dots
                return g

            d1, d2 = dropout_box(), dropout_box()
            d1.move_to([xs2[2], CY, 0])
            d2.move_to([xs2[4], CY, 0])
            old_ars = ars
            new_pos = [xs2[0], xs2[1], xs2[3], xs2[5]]
            tgt = {'win': win, 'inp_lab': inp_lab, 'l1': l1, 'l2': l2, 'out': out}
            self.play(FadeOut(old_ars), win.animate.move_to([new_pos[0], CY, 0]),
                      inp_lab.animate.move_to([new_pos[0], CY - 1.45, 0]),
                      l1.animate.move_to([new_pos[1], CY, 0]), l2.animate.move_to([new_pos[2], CY, 0]),
                      out.animate.move_to([new_pos[3], CY, 0]), run_time=1.0)
            chain = [win, l1, d1, l2, d2, out]
            ars2 = VGroup(*[arrow(a.get_right(), b.get_left()) for a, b in zip(chain[:-1], chain[1:])])
            self.play(FadeIn(d1, shift=DOWN * 0.2), FadeIn(d2, shift=DOWN * 0.2), FadeIn(ars2), run_time=1.0)
            # mode switch
            m_train = T('training', 24, INK, weight='BOLD')
            m_check = T('checking or predicting', 24, INK, weight='BOLD')
            modes = VGroup(m_train, m_check).arrange(RIGHT, buff=0.9).move_to([0, -1.75, 0])
            pill = RoundedRectangle(corner_radius=0.2, width=m_train.width + 0.5, height=0.6, stroke_color=C_HI,
                                    stroke_width=2.5, fill_color=C_HI, fill_opacity=0.18).move_to(m_train)
            self.until(self.cue('During training'))
            self.play(FadeIn(modes), FadeIn(pill), run_time=0.6)
            rule = T('every batch: each signal has a 30 % chance of being switched off', 22, DIM).move_to([0, -2.55, 0])
            self.play(FadeIn(rule), run_time=0.5)
            rng = np.random.default_rng(30)
            t_stop = self._beat[0] + self.cue('When the model is checked') * self._beat[1] - 0.6
            first, scale_lab = True, None
            while self.renderer.time < t_stop - 0.9:
                anims = []
                for d in (d1, d2):
                    mask = rng.random(8) >= 0.3
                    if mask.all():
                        mask[rng.integers(8)] = False
                    anims += [dot.animate.set_fill(opacity=1.0 if m else 0.07) for dot, m in zip(d.dots, mask)]
                self.play(*anims, run_time=0.3)
                if first:
                    first = False
                    scale_lab = T('the rest are scaled up', 22, MEM).move_to([(d1.get_x() + d2.get_x()) / 2, CY + 1.45, 0])
                    self.play(FadeIn(scale_lab), run_time=0.4)
                self.wait(0.65)
            self.until(self.cue('When the model is checked'))
            all_on = [dot.animate.set_fill(opacity=0.65) for d in (d1, d2) for dot in d.dots]
            self.play(pill.animate.move_to(m_check).stretch_to_fit_width(m_check.width + 0.5), *all_on,
                      FadeOut(rule), *([FadeOut(scale_lab)] if scale_lab is not None else []), run_time=0.9)
            ok = T('dropout is off: everything is used', 22, DIM).move_to([0, -2.55, 0])
            self.play(FadeIn(ok), run_time=0.5)
        self.play(FadeOut(VGroup(win, inp_lab, l1, l2, out, d1, d2, ars2, modes, pill, ok)), run_time=0.5)

        # ============================================================ b05 : size in context
        with self.beat('c08_b05'):
            title = T('memory cells per layer', 28, INK).move_to([0, 2.6, 0])
            g = bars([8, 50, 200], ['ours (baseline)', 'Zhang et al. 2022', 'Kim et al. 2020'], [C_HI, DIM, DIM],
                     width=7.0, height=3.2, vmax=215, value_fmt=lambda v: f'{v:.0f}', label_size=22, value_size=30,
                     gap=0.9)
            g.move_to([0, -0.35, 0])
            self.play(FadeIn(title), run_time=0.5)
            self.play(FadeIn(g.axis), GrowFromEdge(g.bars[0], DOWN), FadeIn(g.vals[0]), FadeIn(g.labels[0]), run_time=0.8)
            self.until(self.cue('The LSTM papers we built on used fifty'))
            self.play(GrowFromEdge(g.bars[1], DOWN), FadeIn(g.vals[1]), FadeIn(g.labels[1]), run_time=0.8)
            self.play(GrowFromEdge(g.bars[2], DOWN), FadeIn(g.vals[2]), FadeIn(g.labels[2]), run_time=1.0)
            self.until(self.cue('We will come back'))
            later = T('does size matter? we come back to it later', 24, DIM).move_to([0, -3.35, 0])
            self.play(FadeIn(later), run_time=0.6)
