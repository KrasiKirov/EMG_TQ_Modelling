"""Chapter 17: potential techniques, equal weight per position and longer windows (beats c17_b01 .. c17_b10)."""
import textwrap

from manim import *

from style2 import *

F5 = facts()['x05']
X5 = load('x05')
Y5 = load('y05')
Y6 = load('y06')
Y2 = load('y02')
F5y = facts()['y05']
GREY_B = '#6E7681'


# ── local helpers ────────────────────────────────────────────────────────────
def checkmark(color=C_GOOD, s=0.3):
    pts = [np.array([-0.5, 0.0, 0]) * s, np.array([-0.15, -0.42, 0]) * s, np.array([0.55, 0.5, 0]) * s]
    return VMobject(stroke_color=color, stroke_width=7).set_points_as_corners(pts)


def check_row(text, size=26, width_chars=50, mark=None):
    mark = mark or checkmark()
    txt = W(text, size, INK, width=width_chars)
    row = VGroup(mark, txt).arrange(RIGHT, buff=0.3, aligned_edge=UP)
    mark.align_to(txt, UP).shift(DOWN * 0.08)
    return row


def rr(w, h, color=FAINT, fill=PANEL, r=0.14):
    return RoundedRectangle(corner_radius=r, width=w, height=h, fill_color=fill, fill_opacity=1,
                            stroke_color=color, stroke_width=2)


def chip(text, w, h, color=FAINT, size=24, tcolor=INK, fill=PANEL):
    box = rr(w, h, color, fill)
    return VGroup(box, T(text, size, tcolor).move_to(box))


def place_plot(P, cx, cy):
    P.shift(np.array([cx, cy, 0]) - P.ax.get_center())
    return P


def gbars(pairs, labels, cb, ca, width, height, vmax, nd=2, label_size=22, vsize=20, bw=None):
    n = len(pairs)
    slot = width / n
    bw = bw or min(0.8, slot * 0.36)
    g = VGroup()
    before, after, bv, av, labs = VGroup(), VGroup(), VGroup(), VGroup(), VGroup()
    for i, ((b, a), lab) in enumerate(zip(pairs, labels)):
        xc = -width / 2 + slot * (i + 0.5)
        for v, grp, vals, xo, col in ((b, before, bv, -bw * 0.55, cb[i]), (a, after, av, bw * 0.55, ca[i])):
            h = max(0.03, v / vmax * height)
            r = Rectangle(width=bw, height=h, fill_color=col, fill_opacity=0.92, stroke_width=0)
            r.move_to([xc + xo, h / 2, 0])
            grp.add(r)
            vals.add(T(fmt(v, nd), vsize, INK).next_to(r, UP, buff=0.08))
        labs.add(T(lab, label_size, DIM).move_to([xc, -0.34, 0]))
    axis = Line([-width / 2 - 0.1, 0, 0], [width / 2 + 0.1, 0, 0], stroke_color=FAINT, stroke_width=1.6)
    g.add(axis, before, after, bv, av, labs)
    g.axis, g.before, g.after, g.bv, g.av, g.labs = axis, before, after, bv, av, labs
    return g


def put(g, x, y):
    g.shift(np.array([x, y, 0]) - g.axis.get_center())
    return g


def legend(items, size=20, buff=0.5):
    row = VGroup()
    for col, txt in items:
        row.add(VGroup(Rectangle(width=0.26, height=0.26, fill_color=col, fill_opacity=0.95, stroke_width=0),
                       T(txt, size, DIM)).arrange(RIGHT, buff=0.14))
    return row.arrange(RIGHT, buff=buff)


def warn_triangle(color=C_HI, s=0.5):
    tri = Polygon([0, s, 0], [-s * 0.95, -s * 0.75, 0], [s * 0.95, -s * 0.75, 0], stroke_color=color, stroke_width=5,
                  fill_opacity=0)
    bar = Line([0, s * 0.45, 0], [0, -s * 0.1, 0], stroke_color=color, stroke_width=6)
    dot = Dot([0, -s * 0.45, 0], radius=0.05, color=color)
    return VGroup(tri, bar, dot)


def wrap_keep(s, chars, size=24, color=INK, keep=()):
    """Like W, but never breaks inside a hyphenated word or inside the phrases listed in ``keep``."""
    for k in keep:
        s = s.replace(k, k.replace(' ', chr(1)))
    lines = textwrap.wrap(s, chars, break_on_hyphens=False)
    return T(chr(10).join(l.replace(chr(1), ' ') for l in lines), size, color, line_spacing=0.8)


class Ch17(BeatScene):
    def intro_long(self, cid):
        """Like BeatScene.intro, but the long title is split over two lines so it fits the frame."""
        n = int(cid[1:])
        a, b = CHAPTER_TITLE[cid].split(': ')
        big = VGroup(T(f'{n:02d}', 72, C_HI, weight='BOLD'), T(a + ':', 44, INK), T(b, 44, INK)).arrange(DOWN, buff=0.22)
        self.play(FadeIn(big, shift=UP * 0.2), run_time=0.6)
        self.wait(1.3)
        tag = chapter_tag(cid)
        self.play(ReplacementTransform(big, tag), run_time=0.7)
        return tag

    def construct(self):
        # ───────────────────────── b01: naming + the problem ─────────────────────────
        with self.beat('c17_b01'):
            tag = self.intro_long('c17')
            head = T('Equal weight per position', 38, INK, weight='BOLD').move_to([0, 2.85, 0])
            self.play(FadeIn(head, shift=DOWN * 0.15), run_time=0.6)
            self.until(self.cue('Careful with names'))
            nm = VGroup(T('Names on the dashboard', 22, DIM),
                        VGroup(T('“Equal weight per position”', 24, C_GM, weight='BOLD'),
                               T('= this change alone', 22, INK)).arrange(DOWN, buff=0.08, aligned_edge=LEFT),
                        VGroup(T('“Balanced positions”', 24, C_HI, weight='BOLD'),
                               T('= the final combined model', 22, INK)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
                        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            nbox = rr(nm.width + 0.7, nm.height + 0.6, FAINT)
            nmg = VGroup(nbox, nm.move_to(nbox)).move_to([0, 0.6, 0])
            self.play(FadeIn(nmg, shift=UP * 0.15), run_time=0.7)
            self.until(self.cue('The problem it targets'))
            shares = Y5['loss_YES_share_sqerr']
            pi = pie(shares, ANGLE_C, radius=1.45).move_to([-3.6, 0.4, 0])
            ptt = T('YES · share of the training error', 22, DIM).move_to([-3.6, 2.2, 0])
            ramp = VGroup(*[Dot(radius=0.09, color=c) for c in ANGLE_C]).arrange(RIGHT, buff=0.12)
            lo_l, hi_l = T('toes-down', 20, ANGLE_C[0]), T('toes-up', 20, ANGLE_C[-1])
            rampg = VGroup(lo_l, ramp, hi_l).arrange(RIGHT, buff=0.25).move_to([-3.6, -1.7, 0])
            small = shares[0] + shares[1]
            call = T(f'two toes-down slices: {small * 100:.0f} % of the error', 24, ANGLE_C[1]).move_to([-3.6, -2.5, 0])
            self.play(nmg.animate.move_to([3.4, 1.0, 0]), FadeIn(ptt), FadeIn(pi, scale=0.9), FadeIn(rampg), run_time=0.9)
            self.play(FadeIn(call), run_time=0.5)
            prob = W('problem: the loss ignores weak positions', 30, C_BAD, width=chars_for(5.0, 30), weight='BOLD')
            prob.move_to([3.4, -1.5, 0])
            self.play(FadeIn(prob, shift=UP * 0.1), run_time=0.7)
            sub = T('the strong positions drown them out', 24, INK).move_to([3.4, -2.6, 0])
            self.play(FadeIn(sub), run_time=0.5)

        self.play(FadeOut(VGroup(head, nmg, ptt, pi, rampg, call, prob, sub)), run_time=0.5)

        # ───────────────────────── b02: divide by the spread ─────────────────────────
        with self.beat('c17_b02'):
            sd = np.asarray(X5['sd'])
            kn = np.asarray(X5['knots'])
            ttl = T('EG · session 1 · spread (SD) of the training targets, in Nm', 22, DIM).move_to([0, 2.9, 0])
            bg = bars(list(sd), [fmt(k, 2) for k in kn], ANGLE_C, width=8.6, height=2.2, vmax=float(sd.max()),
                      value_fmt=lambda v: f'{v:.1f}', label_size=18, value_size=20, gap=0.3)
            bg.shift(np.array([-0.2, 0.1, 0]) - bg.axis.get_center())
            xl = T('position (rad), toes-down → toes-up', 18, DIM).move_to([-0.2, -0.75, 0])
            self.play(FadeIn(ttl), FadeIn(bg.axis), FadeIn(bg.labels), FadeIn(xl), run_time=0.5)
            self.until(self.cue('For EG'))
            self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in bg.bars], lag_ratio=0.1), FadeIn(bg.vals), run_time=1.6)
            self.until(self.cue('Divide every target'))
            ones = bars([1.0] * 8, [fmt(k, 2) for k in kn], ANGLE_C, width=8.6, height=1.4, vmax=1.0,
                        value_fmt=lambda v: f'{v:.1f}', label_size=18, value_size=20, gap=0.3)
            ones.shift(np.array([-0.2, 0.1, 0]) - ones.axis.get_center())
            ttl2 = T('after dividing: the spread is 1 at every position', 22, C_HI).move_to([0, 2.9, 0])
            self.play(ReplacementTransform(bg.bars, ones.bars), ReplacementTransform(bg.vals, ones.vals),
                      ReplacementTransform(ttl, ttl2), run_time=1.2)
            self.until(self.cue('train on those'))
            steps = [('target ÷ spread of its position', C_HI), ('network learns the scaled targets', INK),
                     ('prediction × spread of the nearest training position', C_HI)]
            boxes = VGroup()
            for s, c in steps:
                box = rr(3.95, 1.2, FAINT)
                boxes.add(VGroup(box, W(s, 20, c, width=chars_for(3.5, 20), align=ORIGIN).move_to(box)))
            boxes.arrange(RIGHT, buff=0.55).move_to([0, -1.6, 0])
            arrs = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.08, color=DIM, stroke_width=4,
                                  tip_length=0.18) for i in range(2)])
            self.play(FadeIn(boxes[0]), run_time=0.5)
            self.play(FadeIn(boxes[1]), GrowArrow(arrs[0]), run_time=0.6)
            self.until(self.cue('multiply the predictions back'))
            self.play(FadeIn(boxes[2]), GrowArrow(arrs[1]), run_time=0.6)
            self.until(self.cue('Why does this matter'))
            r = F5['sd2_ratio_max_min']
            ratio = float(sd.max() / sd.min())
            f1 = T(f'spread {ratio:.0f}× bigger → about {int(r // 10 * 10)}× more weight in squared error', 24, INK).move_to([0, -2.85, 0])
            self.play(FadeIn(f1, shift=UP * 0.1), run_time=0.6)
            self.until(self.cue('After this change'))
            f2 = T('after scaling, every position counts about equally', 24, C_GOOD).move_to([0, -3.45, 0])
            self.play(FadeIn(f2, shift=UP * 0.1), run_time=0.6)

        self.play(FadeOut(VGroup(ttl2, bg.axis, bg.labels, xl, ones.bars, ones.vals, boxes, arrs, f1, f2)), run_time=0.5)

        # ───────────────────────── b03: cheating? ─────────────────────────
        with self.beat('c17_b03'):
            q = T('Is it cheating?', 40, INK, weight='BOLD').move_to([-3.6, 2.2, 0])
            ans = T('no', 30, C_GOOD, weight='BOLD').next_to(q, RIGHT, buff=0.4).align_to(q, DOWN)
            rows = VGroup(*[check_row(s, 28, 52) for s in
                            ["spreads computed from session 1's fit part only",
                             'at test, the multiplier is picked by the angle, which is already an input']
                            ]).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to([0, 0.2, 0]).align_to(q, LEFT)
            self.play(FadeIn(q), FadeIn(ans), run_time=0.6)
            self.until(self.cue('The standard deviations'))
            self.play(FadeIn(rows[0], shift=RIGHT * 0.2), run_time=0.6)
            self.until(self.cue('and at test time'))
            self.play(FadeIn(rows[1], shift=RIGHT * 0.2), run_time=0.6)

        self.play(FadeOut(VGroup(q, ans, rows)), run_time=0.5)

        # ───────────────────────── b04: result ─────────────────────────
        with self.beat('c17_b04'):
            g4 = Y6['x04_group']
            r2b, r2a, mb, ma = g4[0, 0, 0], g4[0, 0, 1], g4[0, 1, 0], g4[0, 1, 1]
            ttl = T('toes-down positions · 6 people · seeds 0–2', 24, DIM).move_to([-3.1, 2.85, 0])
            lg = legend([(GREY_B, 'rebuilt baseline (X00f)'), (C_GOOD, 'equal weight (X04)')], 20).move_to([-3.3, 2.3, 0])
            gA = gbars([(r2b, r2a)], ['R²'], [GREY_B], [C_GOOD], 2.0, 2.2, 1.1, label_size=22, vsize=22, bw=0.8)
            gB = gbars([(mb, ma)], ['error (Nm)'], [GREY_B], [C_GOOD], 2.6, 2.2, 1.1, label_size=22, vsize=22, bw=0.8)
            put(gA, -5.0, -0.1)
            put(gB, -2.1, -0.1)
            self.play(FadeIn(ttl), FadeIn(lg), run_time=0.5)
            for g in (gA, gB):
                self.play(FadeIn(g.axis), FadeIn(g.labs),
                          LaggedStart(*[GrowFromEdge(r, DOWN) for r in (g.before[0], g.after[0])], lag_ratio=0.25),
                          FadeIn(g.bv), FadeIn(g.av), run_time=1.2)
                if g is gA:
                    self.until(self.cue('and the error'))
            self.until(self.cue('But not for everyone'))
            subj = [str(s) for s in Y6['x04_subjects']]
            jm = subj.index('JM')
            jb, ja = Y6['x04_td_r2_before'][jm], Y6['x04_td_r2_after'][jm]
            pos = list(np.round(Y6['jm_x04_pos'], 2))
            k36 = pos.index(-0.36)
            cb, ca = Y6['jm_x04_r2_floor0_before'][k36], Y6['jm_x04_r2_floor0_after'][k36]
            flag = VGroup(
                T('JM got worse', 28, C_BAD, weight='BOLD'),
                T(f'toes-down R²   {fmt(jb)} → {fmt(ja)}', 24, INK),
                T(f'at −0.36 rad   {fmt(cb)} → {fmt(ca)}', 24, INK)).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
            fb = rr(flag.width + 0.7, flag.height + 0.6, C_BAD)
            flg = VGroup(fb, flag.move_to(fb)).move_to([3.7, 1.2, 0])
            self.play(FadeIn(flg, shift=LEFT * 0.15), run_time=0.7)
            self.until(self.cue('And it cannot fix'))
            bug = Y6['jm_x04_r2_raw_after'][pos.index(-0.08)]
            cav = VGroup(
                T('wrong levels in the labels stay wrong', 26, C_HI, weight='BOLD'),
                T(f'JM cell with the passive-table bug: still R² ≈ {fmt(bug, 0)}    ·    YES at −0.49 rad: still negative',
                  22, INK)).arrange(DOWN, buff=0.2)
            cb_ = rr(12.4, cav.height + 0.5, C_HI)
            cvg = VGroup(cb_, cav.move_to(cb_)).move_to([0, -2.55, 0])
            self.play(FadeIn(cvg, shift=UP * 0.15), run_time=0.7)

        self.play(FadeOut(VGroup(ttl, lg, gA, gB, flg, cvg)), run_time=0.5)

        # ───────────────────────── b05: not techniques ─────────────────────────
        with self.beat('c17_b05'):
            hd = T('Not techniques', 38, INK, weight='BOLD').move_to([0, 2.85, 0])
            self.play(FadeIn(hd), run_time=0.4)
            # card 1: clean table
            c1 = rr(6.2, 5.3, FAINT)
            t1 = T('Clean passive table', 28, C_HI, weight='BOLD')
            d1 = W('accepts only still, steady holds', 24, INK, width=chars_for(5.5, 24))
            dots = VGroup(*[Dot(radius=0.13, color=ANGLE_C[i]) for i in range(8)]).arrange(RIGHT, buff=0.35)
            dots[5].set_color(C_BAD).scale(1.5)
            dl = T("JM's broken entry", 20, C_BAD).next_to(dots[5], DOWN, buff=0.2)
            ch = W('changes only that entry', 24, INK, width=chars_for(5.5, 24))
            v1 = T('→ a bug fix', 28, C_GOOD, weight='BOLD')
            g1 = VGroup(t1, d1, VGroup(dots, dl), ch, v1).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            g1.move_to(c1).align_to(c1, LEFT).shift(RIGHT * 0.4).align_to(c1, UP).shift(DOWN * 0.4)
            card1 = VGroup(c1, g1)
            # card 2: retest table
            c2 = rr(6.2, 5.3, FAINT)
            t2 = T('Retest passive table', 28, C_HI, weight='BOLD')
            d2 = W("session 2's own passive trials make session 2's answer key", 24, INK, width=chars_for(5.5, 24))
            key = chip('answer key: changes', 2.9, 0.55, C_HI, 20, C_HI)
            net = chip('trained network: identical', 3.3, 0.55, C_GOOD, 20, C_GOOD)
            kn_ = VGroup(key, net).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            ch2 = W('the baseline already did this', 24, INK, width=chars_for(5.5, 24))
            v2 = T('→ not a model change', 28, C_GOOD, weight='BOLD')
            g2 = VGroup(t2, d2, kn_, ch2, v2).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            g2.move_to(c2).align_to(c2, LEFT).shift(RIGHT * 0.4).align_to(c2, UP).shift(DOWN * 0.4)
            card2 = VGroup(c2, g2)
            VGroup(card1, card2).arrange(RIGHT, buff=0.35).move_to([0, -0.3, 0])
            self.until(self.cue('The clean passive table'))
            self.play(FadeIn(card1, shift=UP * 0.15), run_time=0.7)
            self.until(self.cue("And last week's"))
            self.play(FadeIn(card2, shift=UP * 0.15), run_time=0.7)

        self.play(FadeOut(VGroup(hd, card1, card2)), run_time=0.5)

        # ───────────────────────── b06: D01 ─────────────────────────
        with self.beat('c17_b06'):
            parts = [chip('clean table', 2.5, 0.6, FAINT, 22), chip('auto-gain', 2.2, 0.6, C_S2, 22, C_S2),
                     chip('equal weight per position', 4.0, 0.6, C_GM, 22, C_GM)]
            plus = [T('+', 30, DIM), T('+', 30, DIM)]
            eq = T('=', 30, DIM)
            d01 = chip('D01', 1.4, 0.7, C_HI, 30, C_HI)
            row = VGroup(parts[0], plus[0], parts[1], plus[1], parts[2], eq, d01).arrange(RIGHT, buff=0.25)
            row.move_to([0, 2.85, 0])
            self.until(self.cue('So we put'))
            self.play(LaggedStart(*[FadeIn(m) for m in row], lag_ratio=0.15), run_time=1.6)
            self.until(self.cue('which the dashboard calls'))
            dash = T("dashboard name: “Balanced positions”", 22, C_HI).move_to([0, 2.1, 0])
            self.play(FadeIn(dash), run_time=0.5)
            self.until(self.cue('One more naming trap'))
            trap = W("the dashboard's description mentions only the clean table and equal weight; the model also has auto-gain",
                     22, C_BAD, width=chars_for(11.0, 22), align=ORIGIN).move_to([0, 1.45, 0])
            ring = SurroundingRectangle(parts[1], color=C_BAD, buff=0.1, corner_radius=0.12)
            self.play(FadeIn(trap), Create(ring), run_time=0.7)
            self.until(self.cue('Against the baseline'))
            G = Y6['d01_vs_x00_group']          # [td|other][R2|MAE][before|after]
            sp0 = facts()['y06']['group_numbers']['X00_baseline_seeds0-9']['toes_down']['r2_range_median']
            sp1 = facts()['y06']['group_numbers']['D01_with_C01_for_EG_YES_seeds0-9']['toes_down']['r2_range_median']
            spec = [('toes-down R²', G[0, 0, 0], G[0, 0, 1], 1.0, 2), ('toes-down error (Nm)', G[0, 1, 0], G[0, 1, 1], 1.1, 2),
                    ('other R²', G[1, 0, 0], G[1, 0, 1], 1.0, 2), ('other error (Nm)', G[1, 1, 0], G[1, 1, 1], 1.1, 2),
                    ('toes-down seed spread (R² range)', sp0, sp1, 1.0, 2)]
            x0, L = -2.4, 5.2
            rows_g = VGroup()
            for i, (lab, b, a, vm, nd) in enumerate(spec):
                y = 0.75 - i * 0.8
                if i == 4:      # two short lines instead of a wrap that strands '(R² range)'
                    lb = VGroup(T('toes-down seed spread', 22, INK), T('R² range across seeds', 18, DIM)).arrange(
                        DOWN, buff=0.05, aligned_edge=LEFT)
                else:
                    lb = W(lab, 22, INK if i in (0, 1) else DIM, width=chars_for(3.8, 22))
                lb.move_to([-6.6 + lb.width / 2, y, 0])
                rb_ = Rectangle(width=max(0.03, b / vm * L), height=0.27, fill_color=GREY_B, fill_opacity=0.92, stroke_width=0)
                ra_ = Rectangle(width=max(0.03, a / vm * L), height=0.27, fill_color=C_GOOD, fill_opacity=0.92, stroke_width=0)
                rb_.move_to([x0 + rb_.width / 2, y + 0.16, 0])
                ra_.move_to([x0 + ra_.width / 2, y - 0.16, 0])
                vb = T(fmt(b, nd), 20, INK).next_to(rb_, RIGHT, buff=0.12)
                va = T(fmt(a, nd), 20, INK).next_to(ra_, RIGHT, buff=0.12)
                rows_g.add(VGroup(lb, rb_, ra_, vb, va))
            axl = Line([x0, 1.3, 0], [x0, -2.75, 0], stroke_color=FAINT, stroke_width=1.6)
            lgd = legend([(GREY_B, 'baseline (X00)'), (C_GOOD, 'D01')], 20).move_to([3.9, 1.6, 0])
            note = T('10 seeds each · identical C01 runs used for EG and YES', 20, DIM).move_to([0, -3.4, 0])
            self.play(FadeOut(VGroup(trap, ring, dash)), FadeIn(axl), FadeIn(lgd), FadeIn(note), run_time=0.5)
            self.play(FadeIn(rows_g[0], shift=RIGHT * 0.2), run_time=0.6)
            self.until(self.cue('The error goes'))
            self.play(FadeIn(rows_g[1], shift=RIGHT * 0.2), run_time=0.6)
            self.play(FadeIn(rows_g[2], shift=RIGHT * 0.2), FadeIn(rows_g[3], shift=RIGHT * 0.2), run_time=0.6)
            self.until(self.cue('And the seed spread'))
            self.play(FadeIn(rows_g[4], shift=RIGHT * 0.2), run_time=0.6)

        self.play(FadeOut(VGroup(row, rows_g, axl, lgd, note)), run_time=0.5)

        # ───────────────────────── b07: caveats on D01 ─────────────────────────
        with self.beat('c17_b07'):
            hd = T('Be careful how you describe it', 34, INK, weight='BOLD').move_to([-2.4, 2.85, 0])
            items = ['no single change passed the pre-written rules',
                     'combination chosen by judgment, after seeing session-2 results',
                     'only IES and HM never influenced a decision',
                     'YES at −0.49 rad got worse: −0.68 → −0.97']
            bl = bullets([wrap_keep(s, chars_for(5.6, 24), keep=('−0.68 → −0.97', 'influenced a decision')) for s in items], 24, INK)
            bl.move_to([0, 0.0, 0]).align_to([-6.4, 0, 0], LEFT)
            for k, it in enumerate(bl):
                if k == 3:
                    it[1].set_color(C_BAD)
            P = Plot((-2.2, 0.4), (0, 2), 5.4, 2.0, xticks=(-2, -1, 0), xlabel='R² (one dot per seed)', tick_size=18)
            place_plot(P, 3.7, 0.2)
            x00 = Y6['yes_m486_r2_X00']
            c01 = Y6['yes_m486_r2_C01']
            zero = P.vline(0, C_HI, 2.0, dashed=True)
            zl = T('0', 18, C_HI).next_to(P.p(0, 2), UP, buff=0.08)
            dots0 = VGroup(*[Dot(P.p(max(v, -2.2), 1.5), radius=0.1, color=GREY_B) for v in x00])
            dots1 = VGroup(*[Dot(P.p(v, 0.5), radius=0.1, color=C_GOOD) for v in c01])
            l0 = T('baseline', 20, GREY_B).next_to(P.p(-2.2, 1.5), LEFT, buff=0.15)
            l1 = T('D01', 20, C_GOOD).next_to(P.p(-2.2, 0.5), LEFT, buff=0.15)
            med0 = T(f'median {fmt(np.median(x00))}', 20, GREY_B).move_to(P.p(-1.1, 1.95))
            med1 = T(f'median {fmt(np.median(c01))}', 20, C_GOOD).move_to(P.p(-1.1, 0.95))
            ptl = T('YES · −0.49 rad · 10 seeds each', 22, DIM).move_to([3.7, 2.6, 0])
            bad = T('all 10 D01 seeds below zero', 22, C_BAD).move_to([3.7, -1.9, 0])
            self.play(FadeIn(hd), run_time=0.4)
            self.until(self.cue('No single change'))
            self.play(FadeIn(bl[0], shift=RIGHT * 0.15), run_time=0.5)
            self.until(self.cue('The combination was chosen'))
            self.play(FadeIn(bl[1], shift=RIGHT * 0.15), run_time=0.5)
            self.until(self.cue('and only IES'))
            self.play(FadeIn(bl[2], shift=RIGHT * 0.15), run_time=0.5)
            self.until(self.cue('And YES at'))
            self.play(FadeIn(bl[3], shift=RIGHT * 0.15), FadeIn(P), FadeIn(zero), FadeIn(zl), FadeIn(ptl), FadeIn(l0),
                      FadeIn(l1), run_time=0.7)
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots0], lag_ratio=0.05), FadeIn(med0), run_time=0.8)
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots1], lag_ratio=0.05), FadeIn(med1), run_time=0.8)
            self.until(self.cue('with all ten seeds below zero'))
            self.play(FadeIn(bad), run_time=0.5)

        self.play(FadeOut(VGroup(hd, bl, P, zero, zl, dots0, dots1, l0, l1, med0, med1, ptl, bad)), run_time=0.5)

        # ───────────────────────── b08: longer window, the idea ─────────────────────────
        with self.beat('c17_b08'):
            t = Y2['s1_t'] - 1.0
            sol = Y2['s1_sol']
            e = 1246
            tend = t[e]
            hd = T('Another idea, on top of D01: a longer window', 28, INK, weight='BOLD').move_to([0, 3.0, 0])
            P = Plot((8, 16), (0, 0.7), 10.0, 1.9, xticks=(8, 10, 12, 14, 16), yticks=(0, 0.3, 0.6), xlabel='time in trial (s)',
                     ylabel='EMG envelope', tick_size=18, yfmt=lambda v: f'{v:.1f}')
            place_plot(P, -0.3, 1.2)
            m = (t >= 8) & (t <= 16)
            cv = P.curve(t[m][::2], sol[m][::2], C_SOL, 2.0)
            note = T('EG · soleus · session 1 · 8 s of the trial', 20, DIM).move_to([-0.3, 2.45, 0])
            self.play(FadeIn(hd), run_time=0.5)
            self.play(FadeIn(P), FadeIn(note), run_time=0.5)
            self.play(Create(cv), run_time=1.6)
            self.until(self.cue('The envelope is so smooth'))

            def wbox(n):
                a, b = P.p(tend - (n - 1) / 100, 0), P.p(tend, 0)
                top = P.p(0, 0.7)[1]
                r = Rectangle(width=max(0.05, b[0] - a[0]), height=top - a[1], stroke_color=C_HI, stroke_width=3,
                              fill_color=C_HI, fill_opacity=0.25)
                return r.move_to([(a[0] + b[0]) / 2, (top + a[1]) / 2, 0])
            box = wbox(20)
            # inset: what the window contains
            Q = Plot((-0.5, 0.0), (0, 0.7), 5.6, 1.7, xticks=(-0.5, -0.4, -0.3, -0.2, -0.1, 0), yticks=(0, 0.3, 0.6),
                     xlabel='seconds before now', ylabel='envelope', tick_size=18, yfmt=lambda v: f'{v:.1f}')
            place_plot(Q, -2.3, -1.9)

            def win_curve(n, col):
                idx = np.arange(e - n + 1, e + 1)
                xs = (idx - e) / 100.0
                return VGroup(Q.curve(xs, sol[idx], col, 4.0), *[Dot(Q.p(x, y), radius=0.045, color=col) for x, y in zip(xs, sol[idx])])
            w20 = win_curve(20, C_HI)
            lab20 = T('0.2 s = 20 samples: a short, almost straight piece', 22, C_HI).move_to([4.0, -1.2, 0])
            lab20 = W('0.2 s = 20 samples: a short, almost straight piece', 22, C_HI, width=chars_for(3.8, 22)).move_to([4.2, -1.9, 0])
            self.play(FadeIn(box), run_time=0.5)
            self.play(FadeIn(Q), FadeIn(w20), FadeIn(lab20), run_time=0.8)
            self.until(self.cue('It tells the network'))
            self.until(self.cue('Half a second'))
            box50 = wbox(50)
            w50 = win_curve(50, C_GOOD)
            lab50 = W('0.5 s = 50 samples: shows the start of the rise', 22, C_GOOD, width=chars_for(3.8, 22)).move_to([4.2, -1.9, 0])
            self.play(Transform(box, box50), ReplacementTransform(w20, w50), ReplacementTransform(lab20, lab50), run_time=1.2)

        self.play(FadeOut(VGroup(hd, P, cv, note, box, Q, w50, lab50)), run_time=0.5)

        # ───────────────────────── b09: evidence ─────────────────────────
        with self.beat('c17_b09'):
            hd = T('The evidence is modest', 34, INK, weight='BOLD').move_to([-3.6, 2.95, 0])
            D2 = Y6['d02_group']
            # X06 cells
            td = Y6['x06_cell_toes_down'] == 1
            mbf, maf = Y6['x06_cell_mae_before'][td], Y6['x06_cell_mae_after'][td]
            nworse = int((maf > mbf).sum())
            x6 = Y6['x06_group']
            cards = []
            specs = [
                ('0.5 s on top of D01', 'D02 · 6 people · 3 seeds · one direction',
                 f'toes-down error {fmt(D2[0, 1, 0])} → {fmt(D2[0, 1, 1])} Nm,  R² {fmt(D2[0, 0, 0], 3)} → {fmt(D2[0, 0, 1], 3)}',
                 'modest gain', C_HI),
                ('0.4 s on the plain baseline', 'X06 · YES + EG · 2 seeds',
                 f'{nworse} of {int(td.sum())} toes-down cells worse, error up {fmt(x6[0, 1, 0])} → {fmt(x6[0, 1, 1])} Nm',
                 'dropped', C_BAD),
                ("last week's 0.5 s choice", "last week's window check",
                 'chosen by looking at session-2 scores', 'selection bias', C_HI)]
            for k, (a, b, c, tagt, col) in enumerate(specs):
                box = rr(12.6, 1.55, FAINT)
                tt = VGroup(T(a, 26, INK, weight='BOLD'), T(b, 18, DIM), T(c, 22, INK)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
                tt.move_to(box).align_to(box, LEFT).shift(RIGHT * 0.4)
                badge = chip(tagt, 2.6, 0.55, col, 22, col).move_to(box.get_right() + LEFT * 1.55)
                g = VGroup(box, tt, badge)
                if k == 1:
                    sq = VGroup(*[Square(0.3, fill_color=C_BAD if w else C_GOOD, fill_opacity=0.9, stroke_width=0)
                                  for w in (maf > mbf)]).arrange(RIGHT, buff=0.1).move_to(box.get_right() + LEFT * 3.9)
                    g.add(sq)
                cards.append(g)
            cg = VGroup(*cards).arrange(DOWN, buff=0.25).move_to([0, -0.35, 0])
            self.play(FadeIn(hd), run_time=0.4)
            self.until(self.cue('The evidence is modest'))
            self.play(FadeIn(cards[0], shift=UP * 0.15), run_time=0.7)
            self.until(self.cue('On the plain baseline'))
            self.play(FadeIn(cards[1], shift=UP * 0.15), run_time=0.7)
            self.until(self.cue("And last week's"))
            self.play(FadeIn(cards[2], shift=UP * 0.15), run_time=0.7)

        self.play(FadeOut(VGroup(hd, cg)), run_time=0.5)

        # ───────────────────────── b10: cheating? selection bias ─────────────────────────
        with self.beat('c17_b10'):
            q = T('Is a longer window cheating?', 36, INK, weight='BOLD').move_to([-3.0, 2.7, 0])
            rows = VGroup(*[check_row(s, 28, 48) for s in ['still looks only backwards',
                                                           'nothing from session 2 enters training']]
                          ).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to([0, 1.0, 0]).align_to(q, LEFT)
            tri = warn_triangle(C_HI, 0.55)
            wt = VGroup(T('selection bias', 32, C_HI, weight='BOLD'),
                        W('choosing a setting by its test score: no leak into the model, but an optimistic score', 24, INK,
                          width=chars_for(9.8, 24))).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
            wrow = VGroup(tri, wt).arrange(RIGHT, buff=0.5)
            wbox = rr(wrow.width + 0.9, wrow.height + 0.7, C_HI)
            wg = VGroup(wbox, wrow.move_to(wbox)).move_to([0, -1.5, 0])
            self.play(FadeIn(q), run_time=0.5)
            self.until(self.cue('it still only looks backwards'))
            self.play(FadeIn(rows[0], shift=RIGHT * 0.2), run_time=0.6)
            self.until(self.cue('and nothing from session 2'))
            self.play(FadeIn(rows[1], shift=RIGHT * 0.2), run_time=0.6)
            self.until(self.cue('But picking any setting'))
            self.play(FadeIn(wg, shift=UP * 0.15), run_time=0.8)
