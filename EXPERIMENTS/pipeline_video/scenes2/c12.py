"""Chapter 12 - The offset trap."""
from manim import *
from style2 import *

C_OFF = C_BAD          # constant offset
C_SHAPE = '#8FA3BF'    # shape-only part
C_MAE = '#FF9F43'      # MAE


def stat(label, value, color, width=3.5, vsize=36):
    box = RoundedRectangle(corner_radius=0.14, width=width, height=1.2, fill_color=PANEL, fill_opacity=1,
                           stroke_color=color, stroke_width=2.5)
    g = VGroup(T(label, 20, DIM), T(value, vsize, color, weight='BOLD')).arrange(DOWN, buff=0.08).move_to(box)
    return VGroup(box, g)


def bracket_under(x0, x1, y, h=0.14, color=C_HI):
    return VMobject(stroke_color=color, stroke_width=3).set_points_as_corners(
        [[x0, y + h, 0], [x0, y, 0], [x1, y, 0], [x1, y + h, 0]])


def put(plot, cx, cy):
    """Centre the plot FRAME (not the labels) at (cx, cy)."""
    x0, x1 = plot.x_range
    y0, y1 = plot.y_range
    mid = (plot.p(x0, y0) + plot.p(x1, y1)) / 2
    plot.shift(np.array([cx, cy, 0]) - mid)
    return plot


def legend_row(items, size=20):
    """items: [(color, text, kind)] kind 'line' or 'box'."""
    g = VGroup()
    for col, txt, kind in items:
        mark = Line(LEFT * 0.22, RIGHT * 0.22, stroke_color=col, stroke_width=4) if kind == 'line' else \
            Rectangle(width=0.26, height=0.26, fill_color=col, fill_opacity=1, stroke_width=0)
        g.add(VGroup(mark, T(txt, size, INK)).arrange(RIGHT, buff=0.14))
    return g.arrange(RIGHT, buff=0.55)


class Ch12(BeatScene):
    def construct(self):
        F = facts()
        Y3 = F['y03']
        D3 = load('y03')
        Y7 = F['y07']
        Y7_ARR = load('y07')
        yes = Y3['YES_minus0p486_seed1']
        t = D3['t_trial_s']
        key, pred = D3['yes_m486_key'], D3['yes_m486_pred'][1]
        bias, sd, mae = yes['bias_nm'], yes['sd_truth_nm'], yes['mae_nm']
        r2, r2s, pen = yes['r2'], yes['r2_shape'], yes['penalty_bias2_over_var']
        mk, mp = float(np.mean(key)), float(np.mean(pred))

        # ---------------------------------------------------------------- b01
        with self.beat('c12_b01'):
            tag = self.intro('c12')
            plot = Plot((0, 90), (-4.2, -0.2), 11.0, 3.2, xticks=(0, 20, 40, 60, 80), yticks=(-4, -3, -2, -1),
                        xlabel='time in the trial (s)', ylabel='active torque (Nm)', tick_size=18)
            put(plot, 0.4, 1.3)
            title = T('YES  ·  −0.486 rad  ·  session 2  ·  seed 1', 24, INK).move_to([1.6, 3.2, 0])
            leg = legend_row([(C_TQ, 'truth (answer key)', 'line'), (C_PRED, 'model prediction', 'line')])
            leg.move_to(plot.p(45, -0.6))
            self.play(FadeIn(plot), FadeIn(title), FadeIn(leg), run_time=0.8)
            self.until(self.cue('This is YES'))
            c_key = envelope_curve(plot, t, key, C_TQ, 2.0)
            c_pred = envelope_curve(plot, t, pred, C_PRED, 2.0)
            self.play(Create(c_key), Create(c_pred), run_time=1.6)

            self.until(self.cue('Its standard deviation'))
            band = plot.band(0, 90, C_HI, 0.2, y0=mk - sd, y1=mk + sd)
            card_sd = stat('spread of the truth (SD)', f'{sd:.2f} Nm', C_HI).move_to([-3.75, -2.1, 0])
            self.play(FadeIn(band), FadeIn(card_sd, shift=UP * 0.2), run_time=0.8)

            self.until(self.cue('but on average'))
            ln_k = plot.hline(mk, C_TQ, 1.5)
            ln_p = plot.hline(mp, C_PRED, 1.5)
            # the offset is drawn just outside the right end of the plot, with guide lines out to it
            ax_ = plot.p(90, 0)[0] + 0.4
            arr = DoubleArrow([ax_, plot.p(0, mk)[1], 0], [ax_, plot.p(0, mp)[1], 0], buff=0, color=C_OFF,
                              stroke_width=5, tip_length=0.1, max_tip_length_to_length_ratio=0.45)
            gl_k = Line([plot.p(90, 0)[0], plot.p(0, mk)[1], 0], [ax_ + 0.18, plot.p(0, mk)[1], 0],
                        stroke_color=C_TQ, stroke_width=1.5)
            gl_p = Line([plot.p(90, 0)[0], plot.p(0, mp)[1], 0], [ax_ + 0.18, plot.p(0, mp)[1], 0],
                        stroke_color=C_PRED, stroke_width=1.5)
            arr_l = T('offset', 18, C_OFF, weight='BOLD').next_to(arr, DOWN, buff=0.1)
            arr_l.set_x(min(ax_, 6.8 - arr_l.width / 2))
            card_off = stat('offset (average gap)', f'{fmt(bias, 2, True)} Nm', C_OFF).move_to([0.45, -2.1, 0])
            self.play(Create(ln_k), Create(ln_p), run_time=0.6)
            self.play(Create(gl_k), Create(gl_p), GrowFromCenter(arr), FadeIn(arr_l),
                      FadeIn(card_off, shift=UP * 0.2), run_time=0.7)

            self.until(self.cue('Its typical miss'))
            sticks = VGroup(*[Line(plot.p(x, float(np.interp(x, t, key))), plot.p(x, float(np.interp(x, t, pred))),
                                   stroke_color=C_MAE, stroke_width=2.5) for x in np.arange(1.5, 89, 2.0)])
            card_mae = stat('MAE (typical miss)', f'{mae:.2f} Nm', C_MAE).move_to([4.35, -2.1, 0])
            self.play(Create(sticks, lag_ratio=0.03), FadeIn(card_mae, shift=UP * 0.2), run_time=1.2)

            self.until(self.cue('Notice that'))
            br = bracket_under(0.45, 4.35, -2.85)
            br_t = T('almost the same number', 26, C_HI).next_to(br, DOWN, buff=0.25)
            self.play(Create(br), FadeIn(br_t), run_time=0.8)
            ann1 = VGroup(band, card_sd, arr, gl_k, gl_p, arr_l, card_off, sticks, card_mae, br, br_t)

        # ---------------------------------------------------------------- b02
        with self.beat('c12_b02'):
            self.play(FadeOut(ann1), run_time=0.5)
            badge_old = stat('R² (old score)', fmt(r2, 2), C_BAD, width=3.3, vsize=40).move_to([-3.6, -2.2, 0])
            self.play(FadeIn(badge_old, shift=UP * 0.2), run_time=0.5)
            dy = plot.p(0, -bias) - plot.p(0, 0)
            mover = VGroup(c_pred, ln_p)
            self.play(mover.animate.shift(dy), run_time=2.2)

            self.until(self.cue('The R² of the shifted curve'))
            badge_new = stat('shape-only R²', fmt(r2s, 2), C_SHAPE, width=3.6, vsize=40).move_to([0.9, -2.2, 0])
            arr2 = Arrow(badge_old.get_right(), badge_new.get_left(), buff=0.12, color=C_HI, stroke_width=4)
            self.play(GrowArrow(arr2), FadeIn(badge_new, shift=UP * 0.2), run_time=0.8)

            self.until(self.cue("Last semester's code"))
            vaf = Y7['vaf']['vaf']
            vaf_t = T(f'= VAF {vaf:.0f} %, already printed by the old code', 24, INK).move_to([2.3, -3.25, 0])
            self.play(FadeIn(vaf_t, shift=UP * 0.15), run_time=0.7)

        # ---------------------------------------------------------------- b03
        with self.beat('c12_b03'):
            self.clear_all(keep=[tag])
            eq = VGroup(T('R²', 42), T('=', 42), T('shape-only R²', 42, C_SHAPE), T('−', 42), T('penalty', 42, C_OFF)
                        ).arrange(RIGHT, buff=0.35).move_to([0, 2.4, 0])
            rule = T('penalty  =  (offset ÷ spread of the truth)²', 30, INK).move_to([0, 1.5, 0])
            self.play(FadeIn(eq, shift=UP * 0.15), run_time=0.8)
            self.until(self.cue('To get the penalty'))
            self.play(FadeIn(rule, shift=UP * 0.15), run_time=0.7)

            self.until(self.cue('Here the offset'))
            pieces = [T(f'{r2s:.3f}', 34, C_SHAPE), T('−', 34), T(f'({bias:.3f} ÷ {sd:.3f})²', 34, C_OFF),
                      T('=', 34), T(f'{r2s:.3f}', 34, C_SHAPE), T('−', 34), T(f'{pen:.3f}', 34, C_OFF),
                      T('=', 34), T(fmt(r2s - pen, 3), 34, INK, weight='BOLD')]
            nums = VGroup(*pieces).arrange(RIGHT, buff=0.22).move_to([0, 0.5, 0])
            self.play(FadeIn(nums, shift=UP * 0.15), run_time=0.8)

            self.until(self.cue('So the ratio'))
            # waterfall on a number line
            x0, x1 = -0.5, 1.0
            sc = 9.0 / (x1 - x0)
            ox = -4.5 - x0 * sc

            def X(v):
                return ox + v * sc
            axis = Line([X(x0), -3.1, 0], [X(x1), -3.1, 0], stroke_color=FAINT, stroke_width=1.8)
            ticks = VGroup(*[VGroup(Line([X(v), -3.1, 0], [X(v), -3.2, 0], stroke_color=FAINT, stroke_width=1.5),
                                    T(fmt(v, 1), 18, DIM).move_to([X(v), -3.45, 0])) for v in (-0.5, 0, 0.5, 1.0)])
            zero = DashedVMobject(Line([X(0), -3.1, 0], [X(0), -1.1, 0], stroke_color=DIM, stroke_width=1.5),
                                  num_dashes=14)
            bar1 = Rectangle(width=r2s * sc, height=0.5, fill_color=C_SHAPE, fill_opacity=0.95, stroke_width=0
                             ).move_to([X(r2s / 2), -1.55, 0])
            bar2 = Rectangle(width=pen * sc, height=0.5, fill_color=C_OFF, fill_opacity=0.95, stroke_width=0
                             ).move_to([X(r2s - pen / 2), -2.25, 0])
            l1 = T(f'shape-only  {r2s:.2f}', 22, INK).next_to(bar1, RIGHT, buff=0.2)
            l2 = T(f'penalty  {pen:.2f}', 22, INK).next_to(bar2, RIGHT, buff=0.2)
            drop = DashedVMobject(Line([X(r2s), -1.8, 0], [X(r2s), -2.0, 0], stroke_color=DIM, stroke_width=1.5),
                                  num_dashes=3)
            self.play(FadeIn(axis), FadeIn(ticks), FadeIn(zero), run_time=0.5)
            self.play(GrowFromEdge(bar1, LEFT), FadeIn(l1), run_time=0.7)
            self.until(self.cue('Take 0.9 away'))
            self.play(GrowFromEdge(bar2, RIGHT), FadeIn(l2), run_time=0.9)
            endm = Dot([X(r2s - pen), -2.25, 0], radius=0.08, color=C_HI)
            endl = T(f'R² = {fmt(r2s - pen, 2)}', 24, C_HI, weight='BOLD').next_to(bar2, DOWN, buff=0.14)
            endl.set_x(X(r2s - pen))
            self.play(FadeIn(endm), FadeIn(endl), run_time=0.5)

        # ---------------------------------------------------------------- b04
        with self.beat('c12_b04'):
            self.clear_all(keep=[tag])
            S = 0.5                    # frame units per Nm
            bx = -4.2
            # row A: toes-down YES
            ttl_a = T('YES at −0.49 rad  (toes-down)', 26, INK, weight='BOLD').move_to([0, 2.95, 0]).align_to([-6.2, 0, 0], LEFT)
            sp_a = Rectangle(width=sd * S, height=0.32, fill_color=C_HI, fill_opacity=0.95, stroke_width=0
                             ).move_to([bx + sd * S / 2, 2.3, 0])
            of_a = Rectangle(width=bias * S, height=0.32, fill_color=C_OFF, fill_opacity=0.95, stroke_width=0
                             ).move_to([bx + bias * S / 2, 1.85, 0])
            lsp_a = T(f'spread of the truth  {sd:.2f} Nm', 22, INK).next_to(sp_a, RIGHT, buff=0.2)
            lof_a = T(f'offset  {bias:.2f} Nm', 22, INK).next_to(of_a, RIGHT, buff=0.2)
            pen_a = T(f'penalty = ({bias:.3f} ÷ {sd:.3f})² = {pen:.2f}', 32, C_OFF, weight='BOLD').move_to([0, 1.2, 0]).align_to([-6.2, 0, 0], LEFT)
            self.play(FadeIn(ttl_a), run_time=0.4)
            self.play(GrowFromEdge(sp_a, LEFT), FadeIn(lsp_a), GrowFromEdge(of_a, LEFT), FadeIn(lof_a), run_time=0.9)
            self.play(FadeIn(pen_a, shift=UP * 0.1), run_time=0.6)

            self.until(self.cue('Put the same offset'))
            spread_b = 10.0
            ttl_b = T('a strong position', 26, INK, weight='BOLD').move_to([0, 0.35, 0]).align_to([-6.2, 0, 0], LEFT)
            sp_b = Rectangle(width=spread_b * S, height=0.32, fill_color=C_HI, fill_opacity=0.95, stroke_width=0
                             ).move_to([bx + spread_b * S / 2, -0.3, 0])
            of_b = Rectangle(width=bias * S, height=0.32, fill_color=C_OFF, fill_opacity=0.95, stroke_width=0
                             ).move_to([bx + bias * S / 2, -0.75, 0])
            lsp_b = T('spread of the truth  ≈ 10 Nm', 22, INK).next_to(sp_b, RIGHT, buff=0.2)
            lof_b = T(f'offset  {bias:.2f} Nm  (the same)', 22, INK).next_to(of_b, RIGHT, buff=0.2)
            pen_b_val = (bias / spread_b) ** 2
            pen_b = T(f'penalty = ({bias:.2f} ÷ 10)² ≈ {pen_b_val:.4f}', 32, C_OFF, weight='BOLD').move_to([0, -1.4, 0]).align_to([-6.2, 0, 0], LEFT)
            inv = T('invisible', 26, DIM).next_to(pen_b, RIGHT, buff=0.35)
            self.play(FadeIn(ttl_b), run_time=0.4)
            self.play(GrowFromEdge(sp_b, LEFT), FadeIn(lsp_b), GrowFromEdge(of_b, LEFT), FadeIn(lof_b), run_time=1.0)
            self.play(FadeIn(pen_b, shift=UP * 0.1), FadeIn(inv), run_time=0.6)

            self.until(self.cue('The offsets there are not bigger'))
            b = np.abs(D3['bias'])
            td = np.broadcast_to(D3['toes_down'].astype(bool)[:, None, :], b.shape)
            m_td, m_ot = float(np.median(b[td])), float(np.median(b[~td]))
            note = card([T('typical size of the offset in a run', 22, DIM),
                         VGroup(T(f'toes-down  {m_td:.2f} Nm', 26, C_OFF, weight='BOLD'),
                                T(f'elsewhere  {m_ot:.2f} Nm', 26, C_OFF, weight='BOLD')).arrange(RIGHT, buff=0.7)],
                        width=8.8, size=22).move_to([0, -2.9, 0])
            self.play(FadeIn(note, shift=UP * 0.15), run_time=0.8)

        # ---------------------------------------------------------------- b05
        with self.beat('c12_b05'):
            self.clear_all(keep=[tag])
            pos, r2a, rsa = D3['positions_rad'], D3['r2'], D3['r2_shape']
            share = (rsa - r2a) / (1 - r2a)
            cells = [(s, p) for s in range(6) for p in range(2)]
            pl = Plot((0, 13.5), (0, 100), 11.4, 2.5, yticks=(0, 50, 100), ylabel='share of 1 − R² (%)', tick_size=18)
            put(pl, 0.6, 1.75)
            leg = legend_row([(C_OFF, 'constant offset', 'box'), (C_SHAPE, 'shape errors', 'box')])
            leg.move_to([2.0, 3.25, 0])
            cols_off, cols_sh, labs = VGroup(), VGroup(), VGroup()
            for i, (s, p) in enumerate(cells):
                xc = i + (i // 2) * 0.3 + 0.5
                f = float(np.median(share[s, :, p])) * 100
                w = 0.68 * pl.p(1, 0)[0] - 0.68 * pl.p(0, 0)[0]
                bo = Rectangle(width=w, height=pl.p(0, f)[1] - pl.p(0, 0)[1], fill_color=C_OFF, fill_opacity=0.95,
                               stroke_width=0).move_to([pl.p(xc, 0)[0], (pl.p(0, f)[1] + pl.p(0, 0)[1]) / 2, 0])
                bs = Rectangle(width=w, height=pl.p(0, 100 - f)[1] - pl.p(0, 0)[1], fill_color=C_SHAPE,
                               fill_opacity=0.95, stroke_width=0).move_to(
                    [pl.p(xc, 0)[0], pl.p(0, f)[1] + (pl.p(0, 100 - f)[1] - pl.p(0, 0)[1]) / 2, 0])
                cols_off.add(bo)
                cols_sh.add(bs)
                labs.add(VGroup(T(str(D3['subjects'][s]), 18, INK), T(fmt(pos[s, p], 2), 18, DIM)
                                ).arrange(DOWN, buff=0.04).move_to([pl.p(xc, 0)[0], pl.p(0, 0)[1] - 0.45, 0]))
            self.play(FadeIn(pl), FadeIn(leg), FadeIn(labs), run_time=0.8)
            self.until(self.cue('In a typical toes-down run'))
            self.play(*[GrowFromEdge(r, DOWN) for r in cols_off], *[FadeIn(r) for r in cols_sh], run_time=1.4)
            dr = F['y01']['D_loss_decomposition_median_per_run']
            c1 = card([T('typical toes-down run', 22, DIM), T('offset ≈ half', 34, C_OFF, weight='BOLD'),
                       T(f"of the shortfall  ({dr['toes_down']['offset_share'] * 100:.0f} %)", 22, INK)],
                      width=5.3, size=22).move_to([-3.1, -2.3, 0])
            c2 = card([T('other positions (JM bug aside)', 22, DIM), T('offset ≈ a fifth', 34, C_OFF, weight='BOLD'),
                       T(f"of the shortfall  ({dr['other']['offset_share'] * 100:.0f} %)", 22, INK)],
                      width=5.3, size=22).move_to([3.1, -2.3, 0])
            self.until(self.cue('In a typical toes-down run'))
            self.play(FadeIn(c1, shift=UP * 0.15), run_time=0.7)
            self.until(self.cue('At the other positions'))
            self.play(FadeIn(c2, shift=UP * 0.15), run_time=0.7)

            self.until(self.cue('And if you removed'))
            self.play(FadeOut(c1), FadeOut(c2), run_time=0.5)
            gs = F['y01']['B_group_summaries']['X00_baseline']['toes_down_12']
            lab = T('toes-down median R²', 24, DIM).move_to([0, -1.5, 0])
            a = T(fmt(gs['r2_floor0'], 2), 56, C_BAD, weight='BOLD')
            ar = Arrow(LEFT * 0.8, RIGHT * 0.8, color=C_HI, stroke_width=5)
            b2 = T(fmt(gs['r2_demeaned'], 2), 56, C_GOOD, weight='BOLD')
            row = VGroup(a, ar, b2).arrange(RIGHT, buff=0.5).move_to([0, -2.5, 0])
            a_cap = T('as is', 18, DIM).next_to(a, DOWN, buff=0.15)
            b_cap = T('offsets removed', 18, DIM).next_to(b2, DOWN, buff=0.15)
            self.play(FadeIn(lab), FadeIn(a), FadeIn(a_cap), run_time=0.5)
            self.play(GrowArrow(ar), FadeIn(b2, shift=RIGHT * 0.2), FadeIn(b_cap), run_time=0.8)


        # ---------------------------------------------------------------- b06
        with self.beat('c12_b06'):
            self.clear_all(keep=[tag])
            rs_med, pos = D3['r2_shape_med'], D3['positions_rad']
            xs = np.linspace(0, 1, 160)

            def bumps(amp, cen, wid):
                y = np.zeros_like(xs)
                for a_, c_, w_ in zip(amp, cen, wid):
                    y -= a_ * np.exp(-((xs - c_) / w_) ** 2)
                return y
            main = bumps([1.0, 0.8, 0.9], [0.17, 0.5, 0.82], [0.07, 0.08, 0.07])
            wig = -0.32 * np.abs(np.sin(2 * np.pi * 11 * xs)) * (np.abs(main) < 0.25)
            specs = [  # subject idx, position idx, truth, prediction, caption (narration order: EG, JM, YES)
                (1, 0, main, 1.5 * main, 'push size too big  (×1.5)'),
                (4, 1, main, 0.4 * main, 'push size too small  (×0.4)'),
                (5, 0, main + wig, main, 'part of the wiggle not followed')]
            cards, sk_all, cap_all = [], [], []
            for i, (s, p, tr, pr, cap) in enumerate(specs):
                cx = -4.4 + i * 4.4
                box = RoundedRectangle(corner_radius=0.14, width=4.1, height=4.3, fill_color=PANEL, fill_opacity=1,
                                       stroke_color=FAINT, stroke_width=2).move_to([cx, 0.75, 0])
                name = T(f'{D3["subjects"][s]}  {fmt(pos[s, p], 3)} rad', 26, INK, weight='BOLD').move_to([cx, 2.55, 0])
                lab = T('shape-only R²', 20, DIM).move_to([cx, 1.98, 0])
                val = T(fmt(rs_med[s, p], 2), 44, C_SHAPE, weight='BOLD').move_to([cx, 1.4, 0])
                sc_ = 1.0 / max(np.abs(tr).max(), np.abs(pr).max())
                y0 = 0.8

                def pts(arr, cx=cx, sc_=sc_, y0=y0):
                    return [[cx - 1.7 + 3.4 * x, y0 + a_ * sc_, 0] for x, a_ in zip(xs, arr)]
                ln0 = Line([cx - 1.7, y0, 0], [cx + 1.7, y0, 0], stroke_color=FAINT, stroke_width=1.2)
                ck = VMobject(stroke_color=C_TQ, stroke_width=2.6).set_points_as_corners(pts(tr))
                cp = VMobject(stroke_color=C_PRED, stroke_width=2.6).set_points_as_corners(pts(pr))
                capt = W(cap, 22, INK, width=chars_for(3.7, 22), align=ORIGIN).move_to([cx, -0.75, 0])
                cards.append(VGroup(box, name, lab, val))
                sk_all.append(VGroup(ln0, ck, cp))
                cap_all.append(capt)
            self.until(self.cue('At the three worst'))
            self.play(*[FadeIn(c) for c in cards], run_time=1.0)
            self.until(self.cue('The predictions rise and fall'))
            self.play(*[FadeIn(s_) for s_ in sk_all], run_time=1.0)
            sketch_tag = T('curves are schematic sketches, not data', 18, DIM).move_to([0, -1.7, 0])
            timing = T('timing is right:  best lag ≤ 0.03 s', 24, INK).move_to([0, -2.2, 0])
            self.play(FadeIn(sketch_tag), FadeIn(timing, shift=UP * 0.1), run_time=0.6)
            self.until(self.cue('too big for EG'))
            self.play(FadeIn(cap_all[0], shift=UP * 0.1), run_time=0.5)
            self.until(self.cue('too small for JM'))
            self.play(FadeIn(cap_all[1], shift=UP * 0.1), run_time=0.5)
            self.until(self.cue('And for YES'))
            self.play(FadeIn(cap_all[2], shift=UP * 0.1), run_time=0.5)
            self.until(self.cue('Offset is the biggest'))
            fin = T('Offset is the biggest problem, not the only one.', 30, C_HI, weight='BOLD').move_to([0, -3.0, 0])
            self.play(FadeIn(fin, shift=UP * 0.1), run_time=0.7)

        # ---------------------------------------------------------------- b07
        with self.beat('c12_b07'):
            self.clear_all(keep=[tag])
            sub7 = [str(s) for s in Y7_ARR['td_subject']]
            pos7 = Y7_ARR['td_pos']
            ref7, lstm7 = Y7_ARR['td_ref_rmse'], Y7_ARR['td_lstm_rmse']
            ymax = 4.4
            pl = Plot((0, 12), (0, ymax), 11.4, 3.0, yticks=(0, 1, 2, 3, 4), ylabel='typical error, RMSE (Nm)',
                      tick_size=18)
            put(pl, 0.5, 1.1)
            leg = legend_row([(C_PASSIVE, "guess: session 1's average here, no EMG", 'box'),
                              (C_PRED, 'baseline LSTM (median of 10 seeds)', 'box')])
            leg.move_to([1.4, 3.2, 0])
            bw = 0.36 * (pl.p(1, 0)[0] - pl.p(0, 0)[0])
            grey, green, glabs, top_labels = VGroup(), VGroup(), VGroup(), VGroup()
            for i in range(12):
                xc = i + 0.5
                for off, v, col, grp in ((-0.2, ref7[i], C_PASSIVE, grey), (0.2, lstm7[i], C_PRED, green)):
                    h = pl.p(0, min(v, ymax))[1] - pl.p(0, 0)[1]
                    r = Rectangle(width=bw, height=h, fill_color=col, fill_opacity=0.95, stroke_width=0
                                  ).move_to([pl.p(xc + off, 0)[0], pl.p(0, 0)[1] + h / 2, 0])
                    grp.add(r)
                    if v > ymax:
                        top_labels.add(T(f'{v:.1f}', 18, INK).next_to(r, UP, buff=0.06))
                glabs.add(VGroup(T(sub7[i], 18, INK), T(fmt(pos7[i], 3), 18, DIM)).arrange(DOWN, buff=0.04
                          ).move_to([pl.p(xc, 0)[0], pl.p(0, 0)[1] - 0.45, 0]))
            self.play(FadeIn(pl), FadeIn(glabs), FadeIn(leg), run_time=0.8)
            self.until(self.cue('predict session'))
            self.play(*[GrowFromEdge(r, DOWN) for r in grey], FadeIn(top_labels), run_time=1.2)
            self.until(self.cue('The baseline beats'))
            self.play(*[GrowFromEdge(r, DOWN) for r in green], run_time=1.2)

            def cell_card(sub, p_, cx, verdict, colr):
                i = [k for k in range(12) if sub7[k] == sub and abs(pos7[k] - p_) < 1e-6][0]
                # the frame also wraps the two row labels under the bars
                frame = Rectangle(width=0.95, height=4.0, stroke_color=C_HI, stroke_width=3, fill_opacity=0
                                  ).move_to([pl.p(i + 0.5, 0)[0], pl.p(0, 0)[1] + 1.25, 0])
                cd = card([T(f'{sub}  {fmt(p_, 3)} rad', 22, C_HI, weight='BOLD'),
                           VGroup(T(f'guess {ref7[i]:.2f}', 22, C_PASSIVE), T('vs', 20, DIM),
                                  T(f'model {lstm7[i]:.2f}', 22, C_PRED)).arrange(RIGHT, buff=0.2),
                           T(verdict, 20, colr)], width=4.1, size=22, pad=0.2).move_to([cx, -2.45, 0])
                # a thin elbow connector ties the card to its frame (cards sit in the same left-to-right
                # order as the framed bars, so the lines never cross)
                x_f = frame.get_center()[0]
                y_a = frame.get_bottom()[1]
                y_b = cd.get_top()[1]
                y_m = (y_a + y_b) / 2
                con = VMobject(stroke_color=C_HI, stroke_width=2).set_points_as_corners(
                    [[x_f, y_a, 0], [x_f, y_m, 0], [cx, y_m, 0], [cx, y_b, 0]])
                return frame, cd, con
            fr_eg, cd_eg, cn_eg = cell_card('EG', -0.739, -4.4, 'only a little better', C_HI)
            fr_hm, cd_hm, cn_hm = cell_card('HM', -0.789, 0.0, 'the model clearly wins', C_GOOD)
            fr_yes, cd_yes, cn_yes = cell_card('YES', -0.486, 4.4, 'the constant guess wins', C_BAD)
            self.until(self.cue("at HM's most"))
            self.play(Create(fr_hm), Create(cn_hm), FadeIn(cd_hm, shift=UP * 0.1), run_time=0.7)
            self.until(self.cue("At EG's,"))
            self.play(Create(fr_eg), Create(cn_eg), FadeIn(cd_eg, shift=UP * 0.1), run_time=0.7)
            self.until(self.cue("And at YES's,"))
            self.play(Create(fr_yes), Create(cn_yes), FadeIn(cd_yes, shift=UP * 0.1), run_time=0.7)
            self.until(self.cue('At those cells'))
            cap = T('a guess the model could have made', 22, DIM).move_to([0, -3.65, 0])
            self.play(FadeIn(cap), run_time=0.5)

        # ---------------------------------------------------------------- b08
        with self.beat('c12_b08'):
            self.clear_all(keep=[tag])
            jm = Y3['JM_minus0p078_seed0']
            jk, jp = D3['jm_m078_key'], D3['jm_m078_pred_seed0']
            pl = Plot((0, 90), (-18, 1.5), 8.4, 3.6, xticks=(0, 20, 40, 60, 80), yticks=(-15, -10, -5, 0),
                      xlabel='time in the trial (s)', ylabel='active torque (Nm)', tick_size=18)
            put(pl, -1.3, 1.0)
            title = T('JM  ·  −0.078 rad  ·  session 2  ·  seed 0', 24, INK).move_to([0.9, 3.2, 0])
            leg = legend_row([(C_TQ, 'truth', 'line'), (C_PRED, 'prediction', 'line')]).move_to(pl.p(38, -9.5))
            self.play(FadeIn(pl), FadeIn(title), FadeIn(leg), run_time=0.8)
            c_k = envelope_curve(pl, t, jk, C_TQ, 2.0)
            c_p = envelope_curve(pl, t, jp, C_PRED, 2.0)
            self.play(Create(c_k), Create(c_p), run_time=1.6)
            mk_, mp_ = float(np.mean(jk)), float(np.mean(jp))

            self.until(self.cue('with an R²'))
            c_r2 = card([T('R² (seed 0)', 20, DIM), T(fmt(jm['r2'], 1), 36, C_BAD, weight='BOLD'),
                         T(f'median of 10 seeds: {fmt(jm["median_r2_over_seeds"], 0)}', 20, INK)],
                        width=3.4, size=20, pad=0.2).move_to([5.1, 2.05, 0])
            self.play(FadeIn(c_r2, shift=LEFT * 0.2), run_time=0.7)

            self.until(self.cue('sits about twelve'))
            ln_k = pl.hline(mk_, C_TQ, 1.5)
            ln_p = pl.hline(mp_, C_PRED, 1.5)
            arr = DoubleArrow(pl.p(84, mk_), pl.p(84, mp_), buff=0, color=C_OFF, stroke_width=5, tip_length=0.18)
            c_off = stat('offset (seed 0)', f'{fmt(jm["bias_nm"], 1)} Nm', C_OFF, width=3.4).move_to([5.1, 0.45, 0])
            self.play(Create(ln_k), Create(ln_p), run_time=0.5)
            self.play(GrowFromCenter(arr), FadeIn(c_off, shift=LEFT * 0.2), run_time=0.7)

            self.until(self.cue('where the pushes peak'))
            pk = pl.hline(-jm['peak_abs_truth_nm'], C_HI, 2.0)
            c_pk = stat('truth pushes peak at', f'≈ {jm["peak_abs_truth_nm"]:.0f} Nm', C_HI, width=3.4).move_to([5.1, -0.95, 0])
            self.play(Create(pk), FadeIn(c_pk, shift=LEFT * 0.2), run_time=0.7)

            self.until(self.cue('That one is not'))
            st = VGroup(RoundedRectangle(corner_radius=0.12, width=5.0, height=0.8, fill_color=PANEL, fill_opacity=1,
                                         stroke_color=C_HI, stroke_width=3),
                        T('a bug  →  bugs section', 28, C_HI, weight='BOLD'))
            st[1].move_to(st[0])
            st.rotate(-0.05).move_to([-0.8, -3.0, 0])
            self.play(FadeIn(st, shift=UP * 0.2, scale=1.1), run_time=0.7)
