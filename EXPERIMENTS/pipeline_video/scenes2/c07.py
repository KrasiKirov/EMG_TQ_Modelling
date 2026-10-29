"""c07: Who learns from what (fit / validation / test; the boundary; why validation cannot see the session change)."""
from manim import *

from style2 import *

F = facts()
CNT = F['y02']['eg_counts']
N_WIN = CNT['windows_per_trial'][0]            # 8,891
N_FIT = CNT['fit_per_trial'][0]                # 7,112
N_VAL = CNT['val_per_trial'][0]                # 1,778
TOT_FIT, TOT_VAL, TOT_TEST = CNT['n_fit'], CNT['n_val'], CNT['n_test']
ANGLES = F['x01']['eg_S1_angles']              # EG's session-1 positions, toes down -> toes up
BND = F['y02']['boundary']
LAST_FIT = BND['last_fit_window_samples']      # [7111, 7130]
FIRST_VAL = BND['first_val_window_samples']    # [7112, 7131]
SHARED = BND['n_shared_samples']               # 19
REACH = F['x02']['imp90'][1]                   # 0.376 s


def until_t(scene, sec):
    """Wait until `sec` seconds of the current beat's narration have played (audio-measured onsets)."""
    scene.until(sec / BEATS[scene.beat_log_current]['seconds'])


def n(v):
    return f'{v:,}'


def dashed_arrow(a, b, color, sw=4):
    ln = DashedVMobject(Line(a, b, stroke_color=color, stroke_width=sw), num_dashes=22)
    tip = Triangle(fill_color=color, fill_opacity=1, stroke_width=0).scale(0.13)
    d = np.array(b) - np.array(a)
    tip.rotate(np.arctan2(d[1], d[0]) - PI / 2).move_to(b)
    return VGroup(ln, tip)


def icon_wrench(color):
    head = Arc(radius=0.27, start_angle=PI / 2 + 0.55, angle=TAU - 1.1, stroke_color=color, stroke_width=11)
    handle = Line([0, -0.27, 0], [0, -1.0, 0], stroke_color=color, stroke_width=11)
    return VGroup(head, handle).rotate(-PI / 4)


def icon_stopwatch(color):
    body = Circle(radius=0.5, stroke_color=color, stroke_width=6)
    knob = Line([0, 0.5, 0], [0, 0.72, 0], stroke_color=color, stroke_width=9)
    cap = Line([-0.15, 0.72, 0], [0.15, 0.72, 0], stroke_color=color, stroke_width=7)
    hand = Line([0, 0, 0], [0.22, 0.3, 0], stroke_color=color, stroke_width=5)
    return VGroup(body, knob, cap, hand, Dot(radius=0.05, color=color))


def icon_flag(color):
    pole = Line([-0.3, -0.55, 0], [-0.3, 0.6, 0], stroke_color=color, stroke_width=6)
    cloth = Polygon([-0.3, 0.6, 0], [0.55, 0.32, 0], [-0.3, 0.04, 0], fill_color=color, fill_opacity=1, stroke_width=0)
    return VGroup(pole, cloth)


def role_card(icon, title, count, body, color, cx):
    box = RoundedRectangle(corner_radius=0.16, width=4.1, height=4.6, fill_color=PANEL, fill_opacity=1,
                           stroke_color=color, stroke_width=2.5).move_to([cx, 0.2, 0])
    icon.move_to([cx, 1.65, 0])
    ti = T(title, 32, color, weight='BOLD').move_to([cx, 0.75, 0])
    co = T(count, 22, DIM).move_to([cx, 0.25, 0])
    bo = W(body, 24, INK, width=chars_for(3.55, 24), align=ORIGIN).move_to([cx, -0.95, 0])
    return VGroup(box, icon, ti, co, bo)


class Ch07(BeatScene):
    def construct(self):
        # ── b01: the 80 / 20 split inside each session-1 trial ──────────────────────────────────────────────
        with self.beat('c07_b01'):
            tag = self.intro('c07')
            X0, WID = -3.0, 8.6
            sx = lambda w: X0 + WID * w / N_WIN
            ys = [2.25 - i * 0.46 for i in range(8)]
            head = T('Session 1 · EG · one bar per trial', 24, DIM).move_to([X0 + WID / 2, 3.2, 0])
            pos_h = T('position (rad)', 20, DIM).move_to([-3.35, 2.8, 0], aligned_edge=RIGHT)
            base, fit_r, val_r, sl_r, labs = VGroup(), VGroup(), VGroup(), VGroup(), VGroup()
            for i, (y, a) in enumerate(zip(ys, ANGLES)):
                base.add(Rectangle(width=WID, height=0.3, fill_color=FAINT, fill_opacity=1, stroke_width=0)
                         .move_to([X0 + WID / 2, y, 0]))
                fit_r.add(Rectangle(width=sx(N_FIT) - X0, height=0.3, fill_color=C_FIT, fill_opacity=0.95, stroke_width=0)
                          .move_to([(X0 + sx(N_FIT)) / 2, y, 0]))
                x_v1 = X0 + WID - 0.06
                val_r.add(Rectangle(width=x_v1 - sx(N_FIT), height=0.3, fill_color=C_VAL, fill_opacity=0.95, stroke_width=0)
                          .move_to([(sx(N_FIT) + x_v1) / 2, y, 0]))
                sl_r.add(Rectangle(width=0.06, height=0.3, fill_color=C_BAD, fill_opacity=1, stroke_width=0)
                         .move_to([X0 + WID - 0.03, y, 0]))
                lab = VGroup(Dot(radius=0.1, color=ANGLE_C[i]), T(fmt(a, 3), 20, INK)).arrange(RIGHT, buff=0.15)
                lab.move_to([-3.35, y, 0], aligned_edge=RIGHT)
                labs.add(lab)
            self.play(FadeIn(head), FadeIn(pos_h), run_time=0.4)
            self.until(self.cue('In session 1'))
            self.play(LaggedStart(*[FadeIn(b) for b in base], lag_ratio=0.12), FadeIn(labs), run_time=1.5)
            until_t(self, 7.9)
            foot_f = T('first 80 % of the windows', 22, C_FIT).move_to([(X0 + sx(N_FIT)) / 2, -1.45, 0])
            self.play(*[GrowFromEdge(r, LEFT) for r in fit_r], FadeIn(foot_f), run_time=1.2)
            until_t(self, 12.9)
            top_f = T(f'fit: {n(N_FIT)} windows', 26, C_FIT, weight='BOLD').move_to([(X0 + sx(N_FIT)) / 2, 2.75, 0])
            self.play(FadeIn(top_f), run_time=0.5)
            until_t(self, 16.4)
            foot_v = T('last 20 %: ≈ 18 s', 22, C_VAL).move_to([sx(N_FIT) + (X0 + WID - sx(N_FIT)) / 2 - 0.9, -1.45, 0])
            top_v = T(f'validation: {n(N_VAL)}', 26, C_VAL, weight='BOLD').move_to([X0 + WID - 1.2, 2.75, 0])
            self.play(*[GrowFromEdge(r, LEFT) for r in val_r], FadeIn(foot_v), FadeIn(top_v), run_time=1.0)
            until_t(self, 22.9)
            un = T('1 window unused', 22, C_BAD).move_to([X0 + WID - 0.2, -2.45, 0], aligned_edge=RIGHT)
            un_a = Arrow(un.get_top() + UP * 0.05 + RIGHT * 0.9, sl_r[7].get_bottom() + DOWN * 0.04, buff=0.03,
                         color=C_BAD, stroke_width=3, max_tip_length_to_length_ratio=0.3)
            self.play(FadeIn(sl_r), FadeIn(un), Create(un_a), run_time=0.9)

        # ── b02: session 2 is the test; totals ──────────────────────────────────────────────────────────────
        with self.beat('c07_b02'):
            CX0, CW = -5.9, 7.0
            cys1 = [2.45 - i * 0.22 for i in range(8)]
            cys2 = [-0.5 - i * 0.22 for i in range(8)]
            csx = lambda w: CX0 + CW * w / N_WIN
            c_fit, c_val, c_sl, c_test = VGroup(), VGroup(), VGroup(), VGroup()
            for y in cys1:
                c_fit.add(Rectangle(width=csx(N_FIT) - CX0, height=0.15, fill_color=C_FIT, fill_opacity=0.95, stroke_width=0)
                          .move_to([(CX0 + csx(N_FIT)) / 2, y, 0]))
                c_val.add(Rectangle(width=CX0 + CW - 0.05 - csx(N_FIT), height=0.15, fill_color=C_VAL, fill_opacity=0.95,
                                    stroke_width=0).move_to([(csx(N_FIT) + CX0 + CW - 0.05) / 2, y, 0]))
                c_sl.add(Rectangle(width=0.05, height=0.15, fill_color=C_BAD, fill_opacity=1, stroke_width=0)
                         .move_to([CX0 + CW - 0.025, y, 0]))
            for y in cys2:
                c_test.add(Rectangle(width=CW, height=0.15, fill_color=C_TEST, fill_opacity=0.95, stroke_width=0)
                           .move_to([CX0 + CW / 2, y, 0]))
            old = VGroup(base, labs, head, pos_h, foot_f, foot_v, top_f, top_v, un, un_a)
            self.play(FadeOut(old), ReplacementTransform(fit_r, c_fit), ReplacementTransform(val_r, c_val),
                      ReplacementTransform(sl_r, c_sl), run_time=1.2)
            l1 = T('Session 1: learns', 24, C_S1, weight='BOLD').move_to([CX0, 3.05, 0], aligned_edge=LEFT)
            self.play(FadeIn(l1), run_time=0.4)
            self.until(self.cue('Session 2 is never'))
            l2 = T('Session 2: never used for training', 24, C_S2, weight='BOLD').move_to([CX0, 0.0, 0], aligned_edge=LEFT)
            self.play(FadeIn(l2), LaggedStart(*[GrowFromEdge(r, LEFT) for r in c_test], lag_ratio=0.1), run_time=1.2)
            until_t(self, 3.1)
            lt = T('all eight trials: test', 22, C_TEST).move_to([CX0 + CW / 2, -2.45, 0])
            self.play(FadeIn(lt), run_time=0.5)
            until_t(self, 5.2)
            cx = 1.8
            r_fit = VGroup(T(f'fit: {n(TOT_FIT)}', 34, C_FIT, weight='BOLD'),
                           T(f'8 × {n(N_FIT)} windows', 22, DIM)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            r_val = VGroup(T(f'validation: {n(TOT_VAL)}', 34, C_VAL, weight='BOLD'),
                           T(f'8 × {n(N_VAL)} windows', 22, DIM)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            r_test = VGroup(T(f'test: {n(TOT_TEST)}', 34, C_TEST, weight='BOLD'),
                            T(f'8 × {n(N_WIN)} windows', 22, DIM)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            r_fit.move_to([cx, 2.55, 0], aligned_edge=LEFT)
            r_val.move_to([cx, 1.35, 0], aligned_edge=LEFT)
            r_test.move_to([cx, -1.0, 0], aligned_edge=LEFT)
            until_t(self, 7.6)
            self.play(FadeIn(r_fit, shift=LEFT * 0.2), run_time=0.6)
            until_t(self, 11.0)
            self.play(FadeIn(r_val, shift=LEFT * 0.2), run_time=0.6)
            until_t(self, 14.5)
            self.play(FadeIn(r_test, shift=LEFT * 0.2), run_time=0.6)
            until_t(self, 17.9)
            foot = T('windows per person · the same for all six people', 24, INK).move_to([0, -3.2, 0])
            self.play(FadeIn(foot), run_time=0.6)

        self.clear_all(keep=[tag])

        # ── b03: the other direction ────────────────────────────────────────────────────────────────────────
        with self.beat('c07_b03'):
            def pair(y, a_txt, a_col, b_txt, b_col, arrow_col, cap, bright):
                a = card([T(a_txt, 26, a_col, weight='BOLD'), T('learns', 22, INK)], width=3.4, color=a_col)
                b = card([T(b_txt, 26, b_col, weight='BOLD'), T('tested', 22, INK)], width=3.4, color=b_col)
                a.move_to([-4.3, y, 0])
                b.move_to([4.3, y, 0])
                ar = Arrow(a.get_right(), b.get_left(), buff=0.15, color=arrow_col, stroke_width=6)
                cp = T(cap, 24, arrow_col).next_to(ar, UP, buff=0.12)
                g = VGroup(a, b, ar, cp)
                if not bright:
                    g.set_opacity(0.55)
                return g
            p1 = pair(1.2, 'Session 1', C_S1, 'Session 2', C_S2, C_GOOD, 'the baseline runs here', True)
            p2 = pair(-1.6, 'Session 2', C_S2, 'Session 1', C_S1, DIM, 'some later experiments', False)
            self.until(self.cue('Some later'))
            self.play(FadeIn(p2), run_time=0.8)
            self.until(self.cue('The baseline runs'))
            self.play(FadeIn(p1), p2.animate.set_opacity(0.35), run_time=0.8)

        self.clear_all(keep=[tag])

        # ── b04: three jobs ─────────────────────────────────────────────────────────────────────────────────
        with self.beat('c07_b04'):
            c1 = role_card(icon_wrench(C_FIT), 'Fit', f'{n(TOT_FIT)} windows', 'the network adjusts itself on these',
                           C_FIT, -4.5)
            c2 = role_card(icon_stopwatch(C_VAL), 'Validation', f'{n(TOT_VAL)} windows',
                           'checked after every pass: when to slow down, when to stop, which version to keep', C_VAL, 0)
            c3 = role_card(icon_flag(C_TEST), 'Test', f'{n(TOT_TEST)} windows',
                           'used only at the very end, to score the finished model', C_TEST, 4.5)
            self.until(self.cue('The network adjusts'))
            self.play(FadeIn(c1, shift=UP * 0.2), run_time=0.7)
            self.until(self.cue('The validation windows'))
            self.play(FadeIn(c2, shift=UP * 0.2), run_time=0.7)
            self.until(self.cue('And the test windows'))
            self.play(FadeIn(c3, shift=UP * 0.2), run_time=0.7)

        self.clear_all(keep=[tag])

        # ── b05: the boundary between fit and validation ────────────────────────────────────────────────────
        with self.beat('c07_b05'):
            S0 = 7121                                           # strip centre sample
            cwid = 0.105
            sxs = lambda s: (s - S0) * cwid
            # context bar: one trial, split at 80 %
            CB_X0, CB_W, CB_Y = -3.3, 8.6, 2.95
            ctx_f = Rectangle(width=CB_W * N_FIT / N_WIN, height=0.28, fill_color=C_FIT, fill_opacity=0.95, stroke_width=0)
            ctx_f.move_to([CB_X0 + ctx_f.width / 2, CB_Y, 0])
            ctx_v = Rectangle(width=CB_W * (1 - N_FIT / N_WIN), height=0.28, fill_color=C_VAL, fill_opacity=0.95, stroke_width=0)
            ctx_v.move_to([CB_X0 + CB_W - ctx_v.width / 2, CB_Y, 0])
            ctx_l = T('one session-1 trial', 20, DIM).move_to([CB_X0 - 0.2, CB_Y, 0], aligned_edge=RIGHT)
            bx = CB_X0 + CB_W * N_FIT / N_WIN
            strip_y = 1.4
            cells = VGroup(*[Rectangle(width=cwid, height=0.3, fill_color=FAINT, fill_opacity=1, stroke_color=BG, stroke_width=0.6)
                             .move_to([sxs(s), strip_y, 0]) for s in range(S0 - 50, S0 + 50)])
            zoom = VGroup(Line([bx - 0.03, CB_Y - 0.14, 0], [sxs(S0 - 50) - cwid / 2, strip_y + 0.18, 0], stroke_color=FAINT, stroke_width=1.5),
                          Line([bx + 0.03, CB_Y - 0.14, 0], [sxs(S0 + 49) + cwid / 2, strip_y + 0.18, 0], stroke_color=FAINT, stroke_width=1.5))
            strip_l = T('samples, 100 per second', 18, DIM).move_to([sxs(S0 - 50), strip_y - 0.38, 0], aligned_edge=LEFT)
            self.play(FadeIn(ctx_f), FadeIn(ctx_v), FadeIn(ctx_l), run_time=0.6)
            self.play(Create(zoom), FadeIn(cells), FadeIn(strip_l), run_time=1.0)
            self.until(self.cue('there is no gap'))

            def wbar(rng, y, color, op):
                s0, s1 = rng
                return Rectangle(width=(s1 - s0 + 1) * cwid, height=0.28, fill_color=color, fill_opacity=op, stroke_color=color,
                                 stroke_width=2).move_to([(sxs(s0) + sxs(s1)) / 2, y, 0])
            wf = wbar(LAST_FIT, 0.65, C_FIT, 0.85)
            wv = wbar(FIRST_VAL, 0.25, C_VAL, 0.85)
            lf = T(f'last fit window: samples {n(LAST_FIT[0])}–{n(LAST_FIT[1])}', 20, C_FIT).move_to([wf.get_left()[0] - 0.2, 0.65, 0], aligned_edge=RIGHT)
            lv = T(f'first validation window: {n(FIRST_VAL[0])}–{n(FIRST_VAL[1])}', 20, C_VAL).move_to([wv.get_right()[0] + 0.2, 0.25, 0], aligned_edge=LEFT)
            self.play(FadeIn(wf), FadeIn(lf), run_time=0.6)
            self.play(FadeIn(wv), FadeIn(lv), run_time=0.6)
            self.until(self.cue('share nineteen'))
            shared = VGroup(*[cells[s - (S0 - 50)].copy().set_fill(C_HI, 1) for s in range(FIRST_VAL[0], LAST_FIT[1] + 1)])
            sh_l = T(f'{SHARED} of 20 samples shared', 22, C_HI, weight='BOLD').move_to([0, -0.3, 0])
            self.play(FadeIn(shared), FadeIn(sh_l), run_time=0.8)
            self.until(self.cue('The filters have also'))
            c_b = (LAST_FIT[1] + FIRST_VAL[1]) / 2 - 0.5 + 0.5
            band = Rectangle(width=2 * REACH * 100 * cwid, height=0.5, fill_color=C_BAD, fill_opacity=0.16, stroke_width=0)
            band.move_to([sxs(c_b), strip_y, 0])
            band_l = T(f'the filters blur the signal across the boundary: ±{fmt(REACH, 1)} s', 20, C_BAD).move_to([0, -0.85, 0])
            bline = Line([sxs(c_b), strip_y - 0.25, 0], [sxs(c_b), strip_y + 0.25, 0], stroke_color=INK, stroke_width=3)
            bnd_l = T('boundary', 18, INK).move_to([sxs(c_b), strip_y + 0.47, 0])
            self.play(FadeIn(band), FadeIn(band_l), FadeIn(bline), FadeIn(bnd_l), run_time=0.8)
            self.until(self.cue('Only the first few dozen'))
            aff = T(f'no gap · affects only the first few dozen of {n(N_VAL)} validation windows', 24, INK).move_to([0, -0.3, 0])
            self.play(FadeOut(sh_l), FadeOut(band_l), run_time=0.4)
            self.play(FadeIn(aff), run_time=0.6)
            self.until(self.cue('and it never touches'))
            nt = T('and it never touches the test (session 2)', 24, C_TEST).move_to([0, -1.1, 0])
            self.play(FadeIn(nt), run_time=0.6)

        self.clear_all(keep=[tag])

        # ── b06: validation cannot see the change between sessions ──────────────────────────────────────────
        with self.beat('c07_b06'):
            box = RoundedRectangle(corner_radius=0.16, width=12.6, height=2.3, fill_color=PANEL, fill_opacity=1,
                                   stroke_color=C_VAL, stroke_width=2).move_to([0, 1.95, 0])
            ttl = T('Validation = the end of the same trials', 30, C_VAL, weight='BOLD').move_to([0, 2.75, 0])
            chips = []
            for txt, (cx, cy) in zip(['the same moment of the visit', 'the same electrode contact', 'the same passive table',
                                      'the same angle for each trial'], [(-5.9, 2.1), (0.6, 2.1), (-5.9, 1.4), (0.6, 1.4)]):
                chips.append(VGroup(Dot(radius=0.08, color=C_VAL), T(txt, 24, INK)).arrange(RIGHT, buff=0.18)
                             .move_to([cx, cy, 0], aligned_edge=LEFT))
            self.until(self.cue('Validation is just'))
            self.play(FadeIn(box), FadeIn(ttl), run_time=0.7)
            for txt, ph in zip(chips, ['the same moment', 'the same electrode', 'the same passive', 'the same angle']):
                self.until(self.cue(ph))
                self.play(FadeIn(txt, shift=RIGHT * 0.15), run_time=0.4)
            self.until(self.cue('So validation can tell'))
            by = -1.0
            fit_b = Rectangle(width=4.8, height=0.55, fill_color=C_FIT, fill_opacity=0.95, stroke_width=0).move_to([-3.8, by, 0])
            val_b = Rectangle(width=1.4, height=0.55, fill_color=C_VAL, fill_opacity=0.95, stroke_width=0)
            val_b.move_to([fit_b.get_right()[0] + 0.7, by, 0])
            s1l = T('Session 1', 24, C_S1, weight='BOLD').next_to(fit_b, DOWN, buff=0.2).align_to(fit_b, LEFT)
            fit_t = T('fit', 22, BG, weight='BOLD').move_to(fit_b)
            val_t = T('val', 22, BG, weight='BOLD').move_to(val_b)
            tst_b = Rectangle(width=3.8, height=0.55, fill_color=C_TEST, fill_opacity=0.95, stroke_width=0).move_to([4.5, by, 0])
            s2l = T('Session 2', 24, C_S2, weight='BOLD').next_to(tst_b, DOWN, buff=0.2).align_to(tst_b, LEFT)
            tst_t = T('test', 22, BG, weight='BOLD').move_to(tst_b)
            a1 = Arrow([fit_b.get_center()[0] + 0.5, by + 0.5, 0], [val_b.get_center()[0] - 0.1, by + 0.5, 0], buff=0, color=C_GOOD,
                       stroke_width=5, max_tip_length_to_length_ratio=0.25)
            a1l = T('fit → validation: same session', 22, C_GOOD).move_to([-3.3, by + 0.95, 0])
            self.play(FadeIn(VGroup(fit_b, val_b, fit_t, val_t, s1l)), run_time=0.6)
            self.play(Create(a1), FadeIn(a1l), run_time=0.7)
            self.until(self.cue('It cannot tell'))
            self.play(FadeIn(VGroup(tst_b, tst_t, s2l)), run_time=0.5)
            a2 = dashed_arrow([fit_b.get_center()[0] + 1.0, by - 0.62, 0] if False else [-2.2, by - 1.0, 0], [3.9, by - 1.0, 0], C_BAD)
            q = T('?', 54, C_BAD, weight='BOLD').move_to([0.9, by - 1.0, 0]).set_stroke(BG, 10, background=True)
            a2.set_z_index(0)
            a2l = T('session 1 → session 2: validation cannot see this', 22, C_BAD).move_to([0.9, by - 1.75, 0])
            self.play(Create(a2), FadeIn(q), FadeIn(a2l), run_time=0.9)
            self.until(self.cue('Hold on'))
            self.play(Indicate(a2l, color=C_HI, scale_factor=1.08), run_time=0.8)
