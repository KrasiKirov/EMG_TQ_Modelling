"""Chapter 10: Seeds: same code, different answers."""
from manim import *
from style2 import *

TD_C, OT_C = C_HI, DIM          # toes-down cells highlighted, other cells grey

PIPS = {1: [(0, 0)], 2: [(-1, 1), (1, -1)], 3: [(-1, 1), (0, 0), (1, -1)], 4: [(-1, 1), (1, 1), (-1, -1), (1, -1)],
        5: [(-1, 1), (1, 1), (0, 0), (-1, -1), (1, -1)], 6: [(-1, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (1, -1)]}


def pips(n, size, color):
    return VGroup(*[Dot([x * size * 0.24, y * size * 0.24, 0], radius=size * 0.075, color=color) for x, y in PIPS[n]])


def die(n, size=1.5, color=C_HI):
    box = RoundedRectangle(corner_radius=0.2, width=size, height=size, fill_color=PANEL, fill_opacity=1,
                           stroke_color=color, stroke_width=3)
    g = VGroup(box)
    g.box, g.size = box, size
    g.pips = pips(n, size, INK).move_to(box)
    g.add(g.pips)
    return g


def key_icon(color=C_HI, s=1.0):
    ring = Circle(radius=0.28 * s, stroke_color=color, stroke_width=5)
    shaft = Line(ring.get_right(), ring.get_right() + RIGHT * 0.9 * s, stroke_color=color, stroke_width=5)
    t1 = Line(shaft.get_end(), shaft.get_end() + DOWN * 0.26 * s, stroke_color=color, stroke_width=5)
    t2 = Line(shaft.get_end() + LEFT * 0.25 * s, shaft.get_end() + LEFT * 0.25 * s + DOWN * 0.2 * s,
              stroke_color=color, stroke_width=5)
    return VGroup(ring, shaft, t1, t2)


def spread_dots(values, y_axis_plot, x, color, radius=0.05):
    return VGroup(*[Dot(y_axis_plot.p(x, v), radius=radius, color=color) for v in values])


class Ch10(BeatScene):
    def construct(self):
        d = load('y03')
        F = facts()
        y3, y1 = F['y03'], F['y01']
        disp = y3['display']

        # ───────────────────────── b01: three dice and a key ─────────────────────────
        with self.beat('c10_b01'):
            tag = self.intro('c10')
            xs = [-4.3, 0.0, 4.3]
            names = ['starting weights', 'order of the windows', 'which signals dropout switches off']
            faces = [[4, 2, 6, 3], [3, 5, 1, 6], [6, 1, 4, 5]]
            dice, labels = [], []
            for x, nm, fc in zip(xs, names, faces):
                dd = die(fc[-1], 2.0).move_to([x, 0.9, 0])
                lb = W(nm, 24, INK, width=chars_for(3.6, 24), align=ORIGIN).next_to(dd, DOWN, buff=0.35)
                dice.append(dd)
                labels.append(lb)

            def roll(dd, fc):
                for n in fc:
                    new = die(n, 2.0).move_to(dd.box)
                    self.play(Transform(dd.pips, new.pips), run_time=0.13)

            for i, ph in enumerate(['The weights start', 'The order in which', "And dropout's"]):
                self.until(self.cue(ph))
                self.play(FadeIn(dice[i], shift=UP * 0.2), FadeIn(labels[i], shift=UP * 0.2), run_time=0.5)
                roll(dice[i], faces[i])
            self.until(self.cue('A seed is a number'))
            key = key_icon(C_HI, 1.2)
            kl = T('seed', 36, C_HI, weight='BOLD')
            kg = VGroup(key, kl).arrange(RIGHT, buff=0.35).move_to([-1.6, -2.6, 0])
            note = T('fixes all the luck: the run can be repeated', 26, INK).move_to([2.6, -2.6, 0])
            note.next_to(kg, RIGHT, buff=0.5)
            VGroup(kg, note).move_to([0, -2.95, 0])
            frame = Rectangle(width=13.2, height=4.2, stroke_color=C_HI, stroke_width=2).move_to([0, 0.35, 0])
            self.play(FadeIn(kg, shift=UP * 0.2), run_time=0.6)
            self.play(Create(frame), *[dd.pips.animate.set_color(C_HI) for dd in dice], FadeIn(note), run_time=0.9)

        # ───────────────────────── b02: no seed last semester → X00 ─────────────────────────
        with self.beat('c10_b02'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            c1 = card([W("Last semester's code set no seed", 30, INK, width=chars_for(10.8, 30)),
                       W('its exact runs can never be reproduced', 28, C_BAD, width=chars_for(10.8, 28))],
                      width=12.0, color=C_BAD, pad=0.35).move_to([0, 1.7, 0])
            self.play(FadeIn(c1, shift=UP * 0.2), run_time=0.7)
            self.until(self.cue('To study the baseline'))
            chips = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.1, width=0.62, height=0.62, fill_color=PANEL,
                                                     fill_opacity=1, stroke_color=C_HI, stroke_width=2),
                                    T(str(i), 26, C_HI, weight='BOLD')) for i in range(10)])
            for ch in chips:
                ch[1].move_to(ch[0])
            chips.arrange(RIGHT, buff=0.2)
            cap = T('rerun with ten fixed seeds, nothing else changed', 28, INK)
            nm = M('this set of runs is called <b>X00</b>  <span foreground="#8B949E">(what this video shows)</span>', 30)
            c2 = card([VGroup(T('seeds', 26, DIM), chips).arrange(RIGHT, buff=0.4), cap],
                      width=12.0, color=C_GOOD, pad=0.35)
            c2.move_to([0, -0.9, 0])
            self.play(FadeIn(c2, shift=UP * 0.2), run_time=0.7)
            self.until(self.cue('That set of runs is called'))
            nm.next_to(c2, DOWN, buff=0.4)
            self.play(FadeIn(nm, shift=UP * 0.15), run_time=0.6)

        # ───────────────────────── b03: ten seeds, two positions ─────────────────────────
        with self.beat('c10_b03'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            t = d['t_window_s']
            i0, i1 = 1400, 3400
            tt = t[i0:i1] - t[i0]

            def make_panel(kname, yr, yticks, cx, wd=5.0):
                pl = Plot((0, 20), yr, wd, 2.9, xticks=(0, 10, 20), yticks=yticks, xlabel='time (s)', tick_size=18)
                pl.move_to([cx, 0.65, 0])
                key = d[kname + '_key'][i0:i1]
                pr = d[kname + '_pred'][:, i0:i1]
                truth = pl.curve(tt, key, C_TQ, 2.6)
                lines = VGroup(*[pl.curve(tt[::2], pr[k][::2], C_PRED, 1.4).set_stroke(opacity=0.75)
                                 for k in range(10)])
                spread = float(np.mean(d[kname + '_pred'].max(0) - d[kname + '_pred'].min(0)))
                return pl, truth, lines, spread

            pl_l, tr_l, ln_l, sp_l = make_panel('yes_strong', (-40, 5), (-40, -20, 0), -3.45)
            pl_r, tr_r, ln_r, sp_r = make_panel('yes_m486', (-4.2, -0.8), (-4, -3, -2, -1), 3.45)
            yl = T('torque (Nm)', 18, DIM).rotate(PI / 2).next_to(pl_l, LEFT, buff=0.12)
            yr_ = T('torque (Nm)', 18, DIM).rotate(PI / 2).next_to(pl_r, LEFT, buff=0.12)
            ttl_l = W('YES at +0.138 rad: torque swings most', 22, INK, width=chars_for(5.9, 22), align=ORIGIN)
            ttl_r = W('YES at −0.486 rad: most toes-down', 22, INK, width=chars_for(5.9, 22), align=ORIGIN)
            ttl_l.move_to([-3.45, 2.95, 0])
            ttl_r.move_to([3.45, 2.95, 0])
            sc_l = T(f'(panel height: {fmt(45, 0)} Nm)', 20, C_HI).move_to([-3.45, 2.5, 0])
            sc_r = T(f'(panel height: {fmt(3.4, 1)} Nm)', 20, C_HI).move_to([3.45, 2.5, 0])
            leg = VGroup(VGroup(Line(ORIGIN, RIGHT * 0.5, stroke_color=C_TQ, stroke_width=3), T('answer key', 20, INK))
                         .arrange(RIGHT, buff=0.15),
                         VGroup(Line(ORIGIN, RIGHT * 0.5, stroke_color=C_PRED, stroke_width=3),
                                T('prediction of one seed (ten lines)', 20, INK)).arrange(RIGHT, buff=0.15)
                         ).arrange(RIGHT, buff=0.7).move_to([0, -1.95, 0])
            self.play(FadeIn(pl_l), FadeIn(pl_r), FadeIn(yl), FadeIn(yr_), FadeIn(ttl_l), FadeIn(ttl_r), run_time=0.7)
            self.play(Create(tr_l), Create(tr_r), FadeIn(leg), run_time=1.2)
            self.until(self.cue('On the left'))
            self.play(LaggedStart(*[Create(l) for l in ln_l], lag_ratio=0.08), run_time=1.8)
            cap_l = W(f'ten seeds, R² {fmt(y3["YES_strong_cell"]["r2_per_seed"] and min(y3["YES_strong_cell"]["r2_per_seed"]), 3)}'
                      f'–{fmt(max(y3["YES_strong_cell"]["r2_per_seed"]), 3)}: almost on top of each other', 22, INK,
                      width=chars_for(5.9, 22), align=ORIGIN).move_to([-3.45, -2.75, 0])
            self.play(FadeIn(cap_l, shift=UP * 0.15), run_time=0.5)
            self.until(self.cue('On the right'))
            self.play(LaggedStart(*[Create(l) for l in ln_r], lag_ratio=0.08), run_time=1.8)
            self.until(self.cue('They all follow'))
            cap_r = W('timing right, but different heights, every one too high', 22, INK,
                      width=chars_for(5.9, 22), align=ORIGIN).move_to([3.45, -2.75, 0])
            self.play(FadeIn(cap_r, shift=UP * 0.15), run_time=0.5)
            self.until(self.cue('Mind the scales'))
            self.play(FadeIn(sc_l), FadeIn(sc_r), run_time=0.5)
            self.until(self.cue('In newton metres'))
            sp = VGroup(T(f'average spread of the ten lines: {fmt(sp_l, 1)} Nm', 20, C_HI).move_to([-3.45, -3.5, 0]),
                        T(f'average spread of the ten lines: {fmt(sp_r, 1)} Nm', 20, C_HI).move_to([3.45, -3.5, 0]))
            self.play(FadeIn(sp, shift=UP * 0.1), run_time=0.6)

        # ───────────────────────── b04: all 48 cells ─────────────────────────
        with self.beat('c10_b04'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            r2 = d['r2']
            td = d['toes_down']
            subj = [str(s) for s in d['subjects']]
            colx = lambda s, p: s * 8 + p + s * 0.9
            XR = (-1.0, 52.0)
            pl = Plot(XR, (-2.2, 1.1), 11.2, 2.4, yticks=(-2, -1, 0, 1), tick_size=18)
            pl.move_to([0.3, 1.7, 0])
            ttl = T('R² of every run (axis cut at −2)', 22, INK).move_to([0, 3.12, 0]).align_to(pl, LEFT)
            ttl.shift(RIGHT * 0.0)
            bands = VGroup(*[pl.band(colx(s, 0) - 0.55, colx(s, 1) + 0.55, C_HI, 0.10) for s in range(6)])
            zero = pl.hline(0, FAINT, 1.2)
            plab = VGroup(*[T(subj[s], 20, DIM).move_to([pl.p(colx(s, 3.5), -2.2)[0], pl.p(0, -2.2)[1] - 0.3, 0])
                            for s in range(6)])
            self.play(FadeIn(pl), FadeIn(ttl), FadeIn(zero), FadeIn(plab), run_time=0.8)
            groups = []
            jm_arrow_x = None
            for s in range(6):
                g = VGroup()
                for p in range(8):
                    col = TD_C if td[s, p] else OT_C
                    for k in range(10):
                        v = r2[s, k, p]
                        if v < -2.2:
                            continue
                        g.add(Dot(pl.p(colx(s, p) + (k - 4.5) * 0.045, v), radius=0.036, color=col))
                groups.append(g)
            self.until(self.cue('Now all forty-eight'))
            self.play(LaggedStart(*[FadeIn(g) for g in groups], lag_ratio=0.25), run_time=2.2)
            # JM -0.078 arrow
            jx = pl.p(colx(4, 5), -1.55)
            jm_arrow = Arrow(jx, pl.p(colx(4, 5), -2.2), buff=0, stroke_width=4, color=C_BAD,
                             max_tip_length_to_length_ratio=0.5, max_stroke_width_to_length_ratio=8)
            jm_lab = T('JM −0.078 rad: R² ≈ −54, off the scale', 18, C_BAD).next_to(jm_arrow, LEFT, buff=0.15)
            jm_lab.shift(UP * 0.12)
            self.play(FadeIn(jm_arrow), FadeIn(jm_lab), run_time=0.5)
            # a cell
            self.until(self.cue('A cell is one person'))
            cell_box = Rectangle(width=pl.p(1.0, 0)[0] - pl.p(0, 0)[0], height=pl.p(0, 1.1)[1] - pl.p(0, -2.2)[1],
                                 stroke_color=C_GOOD, stroke_width=2).move_to(
                [pl.p(colx(2, 4), 0)[0], (pl.p(0, 1.1)[1] + pl.p(0, -2.2)[1]) / 2, 0])
            cell_lab = W('one cell = one person at one position', 20, C_GOOD,
                         width=chars_for(4.2, 20)).next_to(pl, DOWN, buff=0.55).align_to(cell_box, LEFT)
            cell_lab.shift(LEFT * 0.0).set_x(cell_box.get_x())
            self.play(Create(cell_box), FadeIn(cell_lab), run_time=0.6)
            self.until(self.cue('By toes-down'))
            self.play(FadeOut(cell_box), FadeOut(cell_lab), FadeIn(bands), run_time=0.6)
            tdl = T('toes-down: each person\'s two most toes-down positions', 20, C_HI).move_to([0, -0.1, 0])
            self.play(FadeIn(tdl), run_time=0.4)
            # chips
            hd = T('best minus worst of 10 seeds, typical cell (R²)', 24, INK).move_to([0, -1.0, 0])
            chip_o = card([M(f'<b>{disp["range_other"]}</b>', 44, OT_C if False else INK),
                           T('other positions (36 cells)', 22, DIM)], width=5.4, color=OT_C, pad=0.25)
            chip_t = card([M(f'<b>{disp["range_toes_down"]}</b>', 44, C_HI),
                           T('toes-down (12 cells)', 22, DIM)], width=5.4, color=C_HI, pad=0.25)
            for c in (chip_o, chip_t):
                c.content.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
                c.content.move_to(c.box).align_to(c.box, LEFT).shift(RIGHT * 0.3)
            VGroup(chip_o, chip_t).arrange(RIGHT, buff=0.5).move_to([0, -2.4, 0])
            self.until(self.cue('At the other positions'))
            self.play(FadeIn(hd), FadeIn(chip_o, shift=UP * 0.15), run_time=0.6)
            self.until(self.cue('At the toes-down cells'))
            self.play(FadeIn(chip_t, shift=UP * 0.15), run_time=0.6)
            self.until(self.cue("And at YES's most"))
            cx = pl.p(colx(5, 0), 0)[0]
            call = T(f'YES −0.486 rad: from {disp["yes_m486_min"].replace("-", "−")} to {disp["yes_m486_max"]}', 20,
                     C_BAD).move_to([0, -0.12, 0])
            call.set_x(6.6 - call.width / 2)
            lead = Line(pl.p(colx(5, 0), -2.2) + DOWN * 0.02, [cx, call.get_top()[1] + 0.02, 0], stroke_color=C_BAD,
                        stroke_width=2)
            ring = Circle(radius=0.28, stroke_color=C_BAD, stroke_width=3).move_to(pl.p(colx(5, 0), -0.95))
            ring = Rectangle(width=0.4, height=pl.p(0, 0.1)[1] - pl.p(0, -2.15)[1], stroke_color=C_BAD,
                             stroke_width=2).move_to([cx, (pl.p(0, 0.1)[1] + pl.p(0, -2.15)[1]) / 2, 0])
            self.play(FadeOut(tdl), Create(ring), FadeIn(call), run_time=0.7)

        # ───────────────────────── b05: the same spread in Nm ─────────────────────────
        with self.beat('c10_b05'):
            self.play(*[FadeOut(m) for m in (hd, chip_o, chip_t, call, ring)], run_time=0.5)
            mae = d['mae']
            sd1 = mae.std(axis=1, ddof=1)
            med_td = float(np.median(sd1[td.astype(bool)]))
            med_ot = float(np.median(sd1[~td.astype(bool)]))
            ref_td = y1['F_seed_spread']['X00_baseline']['toes_down']['median_mae_sd_nm']
            ref_ot = y1['F_seed_spread']['X00_baseline']['other']['median_mae_sd_nm']
            assert abs(med_td - ref_td) < 0.002 and abs(med_ot - ref_ot) < 0.002
            pl2 = Plot(XR, (0.0, 0.7), 11.2, 1.7, yticks=(0, 0.2, 0.4, 0.6), tick_size=18)
            pl2.move_to([0.3, -2.3, 0])
            pl2.shift(UP * 0.0)
            ttl2 = T('seed-to-seed wobble of the error (Nm)', 22, INK).align_to(pl2, LEFT)
            ttl2.set_y(pl2.get_top()[1] + 0.35)
            dots2 = VGroup(*[Dot(pl2.p(colx(s, p), sd1[s, p]), radius=0.06, color=TD_C if td[s, p] else OT_C)
                             for s in range(6) for p in range(8)])
            l_td = pl2.hline(med_td, C_HI, 2.0)
            l_ot = pl2.hline(med_ot, OT_C, 2.0)
            chips2 = VGroup(VGroup(Dot(radius=0.07, color=C_HI), T(f'toes-down {fmt(ref_td, 2)} Nm', 20, C_HI))
                            .arrange(RIGHT, buff=0.12),
                            VGroup(Dot(radius=0.07, color=OT_C), T(f'other {fmt(ref_ot, 2)} Nm', 20, INK))
                            .arrange(RIGHT, buff=0.12)).arrange(RIGHT, buff=0.5)
            chips2.align_to(pl2, RIGHT).set_y(ttl2.get_y())
            self.until(self.cue('But measure'))
            self.play(FadeIn(pl2), FadeIn(ttl2), run_time=0.6)
            self.play(FadeIn(dots2, lag_ratio=0.02), run_time=1.4)
            self.until(self.cue('Across seeds'))
            self.play(Create(l_td), Create(l_ot), FadeIn(chips2), run_time=0.9)
            self.until(self.cue('R² simply magnifies'))
            note = T('R² magnifies the wobble where there is little to predict', 24, C_HI).move_to([0, -0.2, 0])
            self.play(FadeIn(note, shift=UP * 0.1), *[g.animate.set_opacity(0.4) for g in groups],
                      bands.animate.set_opacity(0.05), run_time=0.8)

        # ───────────────────────── b06: rebuilt copy gives different answers ─────────────────────────
        with self.beat('c10_b06'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            bx_a = card([T('frozen code', 24, INK), T('X00', 34, INK, weight='BOLD')], width=3.8, color=INK, pad=0.2)
            bx_b = card([T('rebuilt copy', 24, INK), T('X00f', 34, C_HI, weight='BOLD')], width=3.8, color=C_HI,
                        pad=0.2)
            for c in (bx_a, bx_b):
                c.content.arrange(DOWN, buff=0.1).move_to(c.box)
            bx_a.move_to([-3.6, 2.6, 0])
            bx_b.move_to([3.6, 2.6, 0])
            same = W('same data, up to rounding: 0.0000019 to 0.0000076 Nm apart', 20, INK,
                     width=chars_for(10.0, 20), align=ORIGIN).move_to([0, 1.5, 0])
            arrow_l = Line([-1.65, 2.45, 0], [-1.45, 2.45, 0], stroke_width=0)
            self.play(FadeIn(bx_a), FadeIn(bx_b), run_time=0.6)
            self.until(self.cue('It produces the same data'))
            self.play(FadeIn(same), run_time=0.6)
            # table
            colx_ = {'case': -6.3, 'a': 0.0, 'b': 3.6, 'v': 6.0}
            hdr = VGroup(T('R² of one run', 20, DIM).move_to([-3.6, 0.85, 0]),
                         T('X00', 22, INK, weight='BOLD').move_to([colx_['a'] + 0.4, 0.85, 0]),
                         T('X00f', 22, C_HI, weight='BOLD').move_to([colx_['b'] + 0.4, 0.85, 0]))
            hdr[0].align_to([-6.3, 0, 0], LEFT)
            line = Line([-6.4, 0.55, 0], [6.5, 0.55, 0], stroke_color=FAINT, stroke_width=1.5)
            yes_x00 = load('y06')['yes_m486_r2_X00'][1]
            yes_x00f = load('y06')['yes_m486_r2_X00f'][1]

            def row(y, case, a, b, same_, col):
                c = T(case, 22, INK).move_to([0, y, 0]).align_to([-6.3, 0, 0], LEFT)
                ta = T(a, 30, INK, weight='BOLD').move_to([colx_['a'] + 0.4, y, 0])
                tb = T(b, 30, INK, weight='BOLD').move_to([colx_['b'] + 0.4, y, 0])
                mark = T('different' if not same_ else 'identical', 20, col).move_to([6.0, y, 0])
                mark.set_x(min(6.0, 6.8 - mark.width / 2))
                return VGroup(c, ta, tb, mark)

            r_jm = row(0.05, 'JM −0.787 rad, seed 0', fmt(0.66, 2), fmt(-1.92, 2), False, C_BAD)
            r_yes = row(-0.85, 'YES −0.486 rad, seed 1', fmt(yes_x00, 2), fmt(yes_x00f, 2), False, C_BAD)
            r_ies = row(-1.75, 'IES, identical data', 'same scores', 'same scores', True, C_GOOD)
            for rr in (r_ies,):
                rr[1].become(T('=', 34, C_GOOD).move_to(rr[1]))
                rr[2].become(T('=', 34, C_GOOD).move_to(rr[2]))
            ies_cap = T('R² equal to 4 decimals', 20, C_GOOD).move_to([colx_['a'] + 2.0, -2.3, 0])
            self.play(FadeIn(hdr), Create(line), run_time=0.5)
            self.until(self.cue('With the same seeds'))
            self.play(FadeIn(r_jm, shift=UP * 0.1), run_time=0.6)
            self.play(FadeIn(r_yes, shift=UP * 0.1), run_time=0.6)
            self.until(self.cue('Only IES'))
            self.play(FadeIn(r_ies, shift=UP * 0.1), FadeIn(ies_cap), run_time=0.6)
            self.until(self.cue('Training is chaotic'))
            fin = T('tiny changes lead to a different final model', 30, C_HI, weight='BOLD').move_to([0, -3.1, 0])
            self.play(FadeIn(fin, shift=UP * 0.1), run_time=0.6)

        # ───────────────────────── b07: the loss pie ─────────────────────────
        with self.beat('c10_b07'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            ssd = load('y05')['loss_YES_share_ssd']
            sqe = load('y05')['loss_YES_share_sqerr']
            disp5 = F['y05']['display']['loss_share_toes_down_two_pct']['YES']
            assert abs((ssd[:2].sum() * 100) - disp5['target_variation']) < 0.06
            assert abs((sqe[:2].sum() * 100) - disp5['squared_error']) < 0.06
            R = 1.45
            cy = 0.55

            def pie_with(fracs, cx):
                pg = pie(fracs, ANGLE_C, radius=R).move_to([cx, cy, 0])
                wedge = Sector(radius=R + 0.12, start_angle=PI / 2, angle=-PI / 2, stroke_color=INK, stroke_width=3,
                               fill_opacity=0).move_to([cx, cy, 0])
                wedge.move_to([cx, cy, 0], aligned_edge=ORIGIN)
                wedge = DashedVMobject(
                    Sector(radius=R + 0.1, start_angle=PI / 2, angle=-PI / 2, arc_center=[cx, cy, 0],
                           stroke_color=INK, stroke_width=3, fill_opacity=0), num_dashes=40)
                return pg, wedge

            pie_a, wedge_a = pie_with(ssd, -3.4)
            pie_b, wedge_b = pie_with(sqe, 3.4)
            ta_ = W('share of the push variation in the training data', 22, INK, width=chars_for(5.6, 22),
                    align=ORIGIN).move_to([-3.4, -1.65, 0])
            tb_ = W('share of the error left after training', 22, INK, width=chars_for(5.6, 22),
                    align=ORIGIN).move_to([3.4, -1.65, 0])
            self.play(FadeIn(pie_a), FadeIn(ta_), run_time=0.8)
            self.play(FadeIn(pie_b), FadeIn(tb_), run_time=0.8)
            strip = VGroup(*[Rectangle(width=0.3, height=0.14, stroke_width=0, fill_color=ANGLE_C[i], fill_opacity=1)
                             for i in range(8)]).arrange(RIGHT, buff=0)
            ramp = VGroup(T('toes-down', 18, DIM), strip, T('toes-up', 18, DIM))
            ramp.arrange(RIGHT, buff=0.15)
            ramp_title = T('slices: the eight positions', 18, DIM)
            VGroup(ramp_title, ramp).arrange(DOWN, buff=0.12).move_to([0, -0.2, 0])
            ramp_title.move_to([-0.2, 2.9, 0]); ramp.move_to([-0.2, 2.55, 0])
            run_cap = T('YES, rebuilt copy, one run', 20, DIM).move_to([-0.2, 3.2, 0])
            self.play(FadeIn(run_cap), FadeIn(ramp_title), FadeIn(ramp), run_time=0.4)
            self.until(self.cue('For YES'))
            lab_a = T(f'≈ {fmt(disp5["target_variation"], 0)} %', 30, C_HI, weight='BOLD').move_to([-3.4 - 0.3, cy + R + 0.4, 0])
            lab_b = T(f'≈ {fmt(disp5["squared_error"], 0)} %', 30, C_HI, weight='BOLD').move_to([3.4 - 0.3, cy + R + 0.4, 0])
            la = Line(lab_a.get_bottom() + DOWN * 0.03, [-3.4 + 0.04, cy + R - 0.05, 0], stroke_color=C_HI, stroke_width=2)
            lb = Line(lab_b.get_bottom() + DOWN * 0.03, [3.4 + 0.04, cy + R - 0.05, 0], stroke_color=C_HI, stroke_width=2)
            self.until(self.cue('the two most toes-down'))
            self.play(FadeIn(lab_a), Create(la), FadeIn(lab_b), Create(lb), run_time=0.8)
            self.until(self.cue('where an equal share'))
            eq = T('equal share = 25 %', 18, INK)
            eq_a = eq.copy().move_to([0, cy + 1.4, 0]).align_to([-2.35, 0, 0], LEFT)
            eq_b = eq.copy().move_to([0, cy + 1.4, 0]).align_to([4.45, 0, 0], LEFT)
            self.play(Create(wedge_a), Create(wedge_b), FadeIn(eq_a), FadeIn(eq_b), run_time=0.9)
            self.until(self.cue('In line with that'))
            note = card([W('Test with equal weight for every position (rebuilt copy, 3 seeds): the seed-to-seed '
                           'wobble of the error was roughly halved, and cut further at toes-down.', 22, INK,
                           width=chars_for(11.2, 22))], width=12.4, color=C_GOOD, pad=0.25)
            note.move_to([0, -2.95, 0])
            self.play(FadeIn(note, shift=UP * 0.15), run_time=0.7)

        # ───────────────────────── b08: lesson ─────────────────────────
        with self.beat('c10_b08'):
            self.play(*[FadeOut(m) for m in self.mobjects if m is not tag], run_time=0.5)
            yc = y3['YES_minus0p486_across_seeds']
            vals = yc['r2_per_seed']
            ps = Plot((-2.4, 0.4), (0, 1), 8.6, 0.01, xticks=(-2, -1, 0), xlabel='R² of one run (YES, −0.486 rad)',
                      tick_size=18)
            ps.move_to([0, -1.55, 0])
            line0 = Line(ps.p(-2.4, 0.5), ps.p(0.4, 0.5), stroke_color=FAINT, stroke_width=1.5)
            line0.set_y(ps.p(0, 0)[1] + 0.8)
            yline = ps.p(0, 0)[1] + 0.9
            dts = [Dot([ps.p(v, 0)[0], yline, 0], radius=0.11, color=C_PRED) for v in vals]
            one = dts[1]
            ttl = W('One run says very little.', 40, INK, width=40).move_to([0, 2.5, 0])
            ttl2_ = W('Compare over many seeds. Report the median and the spread.', 30, INK,
                      width=chars_for(12.6, 30)).next_to(ttl, DOWN, buff=0.35)
            self.play(FadeIn(ps), FadeIn(line0), FadeIn(ttl), run_time=0.6)
            self.play(FadeIn(one, scale=2), run_time=0.4)
            lab1 = T(f'one run: {fmt(vals[1], 2)}', 22, C_PRED).next_to(one, UP, buff=0.3)
            self.play(FadeIn(lab1), run_time=0.3)
            self.until(self.cue('Any comparison'))
            self.play(FadeIn(ttl2_), *[FadeIn(dd, scale=2) for dd in dts if dd is not one], run_time=0.9)
            self.until(self.cue('and the spread'))
            med = float(np.median(vals))
            mk = Line([ps.p(med, 0)[0], yline - 0.35, 0], [ps.p(med, 0)[0], yline + 0.35, 0], stroke_color=C_HI,
                      stroke_width=5)
            ml = T(f'median {fmt(med, 2)}', 22, C_HI).next_to(mk, DOWN, buff=0.1)
            sp = Line([ps.p(min(vals), 0)[0], yline + 0.55, 0], [ps.p(max(vals), 0)[0], yline + 0.55, 0],
                      stroke_color=C_HI, stroke_width=3)
            spl = T(f'spread {fmt(min(vals), 1)} to {fmt(max(vals), 2)}', 22, C_HI).next_to(sp, UP, buff=0.08)
            self.play(FadeOut(lab1), Create(mk), FadeIn(ml), Create(sp), FadeIn(spl), run_time=0.8)
