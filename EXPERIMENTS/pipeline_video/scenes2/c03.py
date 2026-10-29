"""Chapter 3: what the model must predict (active torque = measured torque - passive table)."""
import os

from manim import *
from style2 import *

Y = load('y02')
X4 = load('x04')
F2 = facts()['y02']
ACT = C_HI                      # active torque / target
PAS = C_PASSIVE                 # passive torque


# ── small helpers ────────────────────────────────────────────────────────────
def seg(x0, w, y, h, color, opacity=0.95, stroke=None):
    """Rectangle whose left edge is x0 (stacked-bar segment)."""
    r = Rectangle(width=w, height=h, fill_color=color, fill_opacity=opacity,
                  stroke_color=stroke or color, stroke_width=2)
    return r.move_to([x0 + w / 2, y, 0])


def dashed_rect(x0, w, y, h, color=DIM):
    r = Rectangle(width=w, height=h, stroke_color=color, stroke_width=2, fill_opacity=0)
    return DashedVMobject(r, num_dashes=30).move_to([x0 + w / 2, y, 0])


def legend_row(color, text, size=22):
    sw = Square(0.24, fill_color=color, fill_opacity=1, stroke_width=0)
    return VGroup(sw, T(text, size, INK)).arrange(RIGHT, buff=0.2)


def ankle_cartoon():
    """Shin, foot, pedal and torque sensor as plain shapes (parts returned as attributes)."""
    shin = RoundedRectangle(corner_radius=0.2, width=0.55, height=2.4, fill_color=PANEL, fill_opacity=1,
                            stroke_color=DIM, stroke_width=3).move_to([-3.4, 1.5, 0])
    foot = RoundedRectangle(corner_radius=0.18, width=2.4, height=0.45, fill_color=PANEL, fill_opacity=1,
                            stroke_color=DIM, stroke_width=3).move_to([-2.65, 0.05, 0])
    joint = Circle(0.13, fill_color=INK, fill_opacity=1, stroke_width=0).move_to([-3.4, 0.28, 0])
    pedal = Rectangle(width=3.2, height=0.12, fill_color=FAINT, fill_opacity=1, stroke_color=DIM,
                      stroke_width=2).move_to([-2.65, -0.28, 0])
    stem = Line([-3.4, -0.34, 0], [-3.4, -0.6, 0], stroke_color=DIM, stroke_width=3)
    sensor = RoundedRectangle(corner_radius=0.08, width=0.75, height=0.4, fill_color=PANEL, fill_opacity=1,
                              stroke_color=INK, stroke_width=2.5).move_to([-3.4, -0.8, 0])
    slab = T('torque sensor', 20, DIM).next_to(sensor, DOWN, buff=0.15)
    g = VGroup(shin, foot, joint, pedal, stem, sensor, slab)
    g.sensor = sensor
    return g


def spring_zigzag(x=-3.95, y0=1.75, y1=0.3, teeth=8, amp=0.13, color=PAS):
    ys = np.linspace(y0, y1, 2 * teeth + 1)
    pts = [[x, ys[0], 0]] + [[x + (amp if i % 2 else -amp), ys[i], 0] for i in range(1, 2 * teeth)] + [[x, ys[-1], 0]]
    return VMobject(stroke_color=color, stroke_width=4).set_points_as_corners(pts)


def trace(arr, x0, x1, yc, h, color, width=2.0, n=300):
    """Envelope strip scaled to [yc - h/2, yc + h/2] (0..max of the strip)."""
    arr = np.asarray(arr, float)
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    a = arr[idx] / max(arr.max(), 1e-9)
    xs = np.linspace(x0, x1, n)
    pts = [[x, yc - h / 2 + v * h, 0] for x, v in zip(xs, a)]
    return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(pts)


def spoken_frac(say, phrase):
    """Fraction of the narration (by characters, with pauses at punctuation) spoken before ``phrase``."""
    words, target = say.split(), phrase.split()
    norm = lambda w: w.strip('.,:;!?"').lower()
    w = [len(x) + (3 if x[-1] in ',:;' else 6 if x[-1] in '.?!' else 0) + 1 for x in words]
    for i in range(len(words)):
        if [norm(x) for x in words[i:i + len(target)]] == [norm(x) for x in target]:
            return sum(w[:i]) / sum(w)
    raise ValueError(phrase)


class Ch03(BeatScene):
    def at(self, phrase, lead=0.4):
        """Wait until ``lead`` seconds before ``phrase`` is spoken (character-weighted estimate)."""
        t0, dur = self._beat
        rest = t0 + spoken_frac(SAY[self.beat_log_current], phrase) * dur - lead - self.renderer.time
        if rest > 0.02:
            self.wait(rest)
        if os.environ.get('DBG_T'):
            print('DBG', self.beat_log_current, phrase, round(self.renderer.time - t0, 1), '/', round(dur, 1))


    def construct(self):
        # ── b01: total = active + passive ───────────────────────────────────
        with self.beat('c03_b01'):
            tag = self.intro('c03')
            cart = ankle_cartoon()
            self.play(FadeIn(cart), run_time=0.8)
            x0, wa, wp, h = 1.3, 2.8, 1.4, 0.75
            yA, yB = 1.85, -0.15
            self.at('The sensor at the pedal')
            self.play(Indicate(cart.sensor, color=INK, scale_factor=1.25), run_time=0.9)
            self.at('measures the total torque')
            totA = Rectangle(width=wa + wp, height=h, stroke_color=C_TQ, stroke_width=3, fill_opacity=0
                             ).move_to([x0 + (wa + wp) / 2, yA, 0])
            totL = T('measured torque (total)', 24, C_TQ).next_to(totA, UP, buff=0.2).align_to(totA, LEFT)
            self.play(Create(totA), FadeIn(totL), run_time=0.9)
            self.at('Part of that')
            actA = seg(x0, wa, yA, h, ACT)
            actT = T('active', 26, BG, weight='BOLD').move_to(actA)
            muscle = Ellipse(width=0.32, height=1.4, fill_color=ACT, fill_opacity=0.85, stroke_width=0
                             ).move_to([-3.4, 1.45, 0])
            self.play(FadeIn(actA, shift=RIGHT * 0.2), FadeIn(actT), FadeIn(muscle), run_time=0.9)
            self.at('But some torque')
            relL = T('person completely relaxed', 22, DIM).move_to([x0, yB + 0.85, 0]).align_to(totA, LEFT)
            relA = dashed_rect(x0, wa, yB, h, DIM)
            relAt = T('active = 0', 22, DIM).move_to([x0 + wa / 2, yB, 0])
            relP = seg(x0 + wa, wp, yB, h, PAS)
            relPt = T('passive', 22, BG, weight='BOLD').move_to(relP)
            relN = T('measured torque is not zero', 24, C_TQ).move_to([x0, yB - 0.75, 0]).align_to(totA, LEFT)
            self.play(FadeIn(relL), FadeIn(relA), FadeIn(relAt), FadeOut(muscle), run_time=0.7)
            self.play(FadeIn(relP, shift=LEFT * 0.2), FadeIn(relPt), FadeIn(relN), run_time=0.8)
            self.at('Stretched tendons')
            spring = spring_zigzag()
            springL = VGroup(T('stretched tendons,', 20, PAS), T('muscles, skin', 20, PAS)
                             ).arrange(DOWN, buff=0.06, aligned_edge=RIGHT).next_to(spring, LEFT, buff=0.2)
            self.play(Create(spring), FadeIn(springL), run_time=1.0)
            self.at('and the weight')
            wt = Polygon([-2.55, 0.28, 0], [-2.05, 0.28, 0], [-2.15, 0.8, 0], [-2.45, 0.8, 0], fill_color=FAINT,
                         fill_opacity=1, stroke_color=PAS, stroke_width=3)
            wtA = Arrow([-2.3, 1.45, 0], [-2.3, 0.88, 0], buff=0, color=PAS, stroke_width=4,
                        max_tip_length_to_length_ratio=0.35)
            wtL = VGroup(T('weight of foot', 20, PAS), T('and pedal', 20, PAS)).arrange(DOWN, buff=0.06, aligned_edge=LEFT
                                                                                        ).move_to([-1.0, 0.78, 0])
            self.play(FadeIn(wt), FadeIn(wtA), FadeIn(wtL), run_time=0.9)
            self.at('We call that whole')
            pasA = seg(x0 + wa, wp, yA, h, PAS)
            pasT = T('passive', 24, BG, weight='BOLD').move_to(pasA)
            lg = VGroup(legend_row(ACT, 'active: muscles contracting'),
                        legend_row(PAS, 'passive: relaxed reading')).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
            lg.move_to([x0, -2.65, 0]).align_to(totA, LEFT)
            self.play(FadeIn(pasA), FadeIn(pasT), run_time=0.6)
            self.play(FadeIn(lg), Indicate(relP, color=INK, scale_factor=1.1), run_time=0.8)

        # ── b02: the target = measured - passive ────────────────────────────
        with self.beat('c03_b02'):
            self.clear_all(0.5, keep=(tag,))
            panel = RoundedRectangle(corner_radius=0.14, width=3.7, height=3.2, fill_color=PANEL, fill_opacity=1,
                                     stroke_color=FAINT, stroke_width=2).move_to([-4.6, 1.0, 0])
            ptitle = T('EMG envelopes', 22, INK, weight='BOLD').move_to(panel.get_top() + DOWN * 0.38)
            ps = Y['push_start_samples']
            a = int(ps[2]) - 80
            rows = VGroup()
            for i, (k, nm) in enumerate((('s1_gm', 'GM'), ('s1_gl', 'GL'), ('s1_sol', 'SOL'), ('s1_ta', 'TA'))):
                yc = 1.85 - i * 0.6
                lab = T(nm, 20, MUSCLE[nm]).move_to([-5.95, yc, 0])
                rows.add(lab, trace(Y[k][a:a + 600], -5.5, -3.0, yc, 0.42, MUSCLE[nm], 2.0))
            emg = VGroup(panel, ptitle, rows)
            lab_t = T('measured torque', 22, C_TQ).move_to([4.3, 2.55, 0])
            blk_a = RoundedRectangle(corner_radius=0.1, width=3.0, height=0.85, fill_color=ACT, fill_opacity=0.95,
                                     stroke_width=0).move_to([4.3, 1.75, 0])
            blk_at = T('active torque', 24, BG, weight='BOLD').move_to(blk_a)
            blk_p = RoundedRectangle(corner_radius=0.1, width=3.0, height=0.85, fill_color=PAS, fill_opacity=0.95,
                                     stroke_width=0).move_to([4.3, 0.78, 0])
            blk_pt = T('passive torque', 24, BG, weight='BOLD').move_to(blk_p)
            torque = VGroup(lab_t, blk_a, blk_at, blk_p, blk_pt)
            self.play(FadeIn(emg), FadeIn(torque), run_time=0.9)
            self.at('The EMG mainly reflects')
            src = np.array([-2.7, 1.3, 0])
            arr_a = Arrow(src, blk_a.get_left() + LEFT * 0.08, buff=0.05, color=INK, stroke_width=5)
            self.play(GrowArrow(arr_a), run_time=0.8)
            self.at('a stretched tendon')
            arr_p = DashedVMobject(Line(src, blk_p.get_left() + LEFT * 0.08, stroke_color=C_BAD, stroke_width=5),
                                   num_dashes=14)
            mid = (src + blk_p.get_left()) / 2 + np.array([-0.1, 0.0, 0])
            cross = VGroup(Line(mid + UL * 0.22, mid + DR * 0.22), Line(mid + UR * 0.22, mid + DL * 0.22)
                           ).set_stroke(C_BAD, 6)
            why = T('a stretched tendon gives off no EMG', 22, C_BAD).move_to([0.2, -0.28, 0])
            self.play(Create(arr_p), run_time=0.5)
            self.play(Create(cross), FadeIn(why), run_time=0.6)
            self.at('That is probably')
            note = T('a design choice; the code states no reason', 20, DIM).move_to([0, -3.3, 0])
            self.play(FadeIn(note), run_time=0.6)
            self.at('It asks for the active torque only')
            eq = M('<span foreground="#F0E442">target</span>  =  measured torque  −  '
                   '<span foreground="#8B949E">passive torque</span>', 34).move_to([0, -1.9, 0])
            self.play(FadeIn(eq, shift=UP * 0.2), run_time=0.9)

        # ── b03: where the passive torque comes from ────────────────────────
        with self.beat('c03_b03'):
            self.clear_all(0.5, keep=(tag,))
            n_hold = int(round(20.1 * 100))
            ht, hq, hp = X4['p20_t'][:n_hold], X4['p20_tq'][:n_hold], X4['p20_pos'][:n_hold]
            mean_tq = float(np.mean(hq))
            mean_pos = float(np.mean(hp))
            PA = Plot((0, 20.1), (-0.05, 0.0), 7.8, 1.4, yticks=(-0.03,), tick_size=18,
                      yfmt=lambda v: f'{v:.2f}').move_to([-1.9, 2.0, 0])
            PA.add(T('ankle angle (rad)', 18, DIM).move_to(PA.p(0.3, -0.0095), aligned_edge=LEFT))
            PT = Plot((0, 20.1), (-5, -2), 7.8, 2.5, xticks=(0, 5, 10, 15, 20), yticks=(-5, -4, -3),
                      xlabel='time in the passive trial (s)', ylabel='torque (Nm)', tick_size=18,
                      grid=True).move_to([-1.9, -0.7, 0])
            ttl = T('EG · session 1 · a passive trial', 22, C_S1).move_to([-1.9, 3.05, 0])
            self.play(FadeIn(ttl), FadeIn(PA), FadeIn(PT), run_time=0.7)
            self.at('From the start of the passive trials')
            ca = PA.curve(ht, hp, C_ANGLE, 2.5)
            ct = PT.curve(ht, hq, C_TQ, 2.5)
            side = W('the machine holds the relaxed foot still for the first 20 s', 22, INK,
                     width=chars_for(2.9, 22)).move_to([5.0, 1.6, 0])
            self.play(Create(ca), Create(ct), FadeIn(side), run_time=1.6)
            self.at('takes the average torque')
            mline = PT.hline(mean_tq, C_HI, 2.5, dashed=True)
            mdot = Dot(PT.p(20.1, mean_tq), radius=0.12, color=C_HI)
            mlab = VGroup(T('average torque', 22, C_HI), T(f'{fmt(mean_tq)} Nm', 28, C_HI, weight='BOLD')
                          ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to([5.0, -0.9, 0])
            self.play(Create(mline), FadeIn(mdot, scale=1.8), FadeIn(mlab), run_time=1.0)
            # table plot
            self.at('roughly one number per position')
            tx = lambda c: Plot((-0.85, 0.35), (-12, 5), 4.8, 3.3, xticks=(-0.6, -0.3, 0, 0.3),
                                yticks=(-10, -5, 0), xlabel='ankle angle (rad)', ylabel='passive torque (Nm)',
                                tick_size=18, grid=True, xfmt=lambda v: f'{v:g}').move_to([c, -0.35, 0])
            T1 = tx(0.0)
            pts1 = Y['passive_s1']
            dots1 = VGroup(*[Dot(T1.p(a_, b_), radius=0.09, color=C_S1) for a_, b_ in pts1])
            i_eg = int(np.argmin(np.abs(pts1[:, 0] - mean_pos)))
            tt1 = T('EG · session 1 passive table', 24, C_S1).move_to([0, 2.55, 0])
            self.play(FadeOut(VGroup(PA, PT, ca, ct, mline, side, mlab, ttl)), FadeIn(T1), FadeIn(tt1), run_time=0.7)
            self.play(ReplacementTransform(mdot, dots1[i_eg]), run_time=0.9)
            others = VGroup(*[d_ for i, d_ in enumerate(dots1) if i != i_eg])
            self.play(LaggedStart(*[FadeIn(d_, scale=2.0) for d_ in others], lag_ratio=0.15), run_time=1.5)
            self.at('Join those numbers')
            line1 = VMobject(stroke_color=C_S1, stroke_width=3).set_points_as_corners([T1.p(a_, b_) for a_, b_ in pts1])
            self.play(Create(line1), run_time=1.2)
            self.at('Between two measured angles')
            a_q = 0.5 * (pts1[4, 0] + pts1[5, 0])
            b_q = float(np.interp(a_q, pts1[:, 0], pts1[:, 1]))
            q_dot = Dot(T1.p(a_q, b_q), radius=0.11, color=C_HI)
            q_v = DashedVMobject(Line(T1.p(a_q, -12), T1.p(a_q, b_q), stroke_color=C_HI, stroke_width=2), num_dashes=14)
            q_h = DashedVMobject(Line(T1.p(-0.85, b_q), T1.p(a_q, b_q), stroke_color=C_HI, stroke_width=2),
                                 num_dashes=14)
            self.play(Create(q_v), run_time=0.5)
            self.play(FadeIn(q_dot, scale=1.8), Create(q_h), run_time=0.7)
            self.at('Each session gets', lead=1.3)
            grp1 = VGroup(T1, tt1, dots1, line1, q_dot, q_v, q_h)
            T2 = tx(0.0)
            pts2 = Y['passive_s2']
            dots2 = VGroup(*[Dot(T2.p(a_, b_), radius=0.09, color=C_S2) for a_, b_ in pts2])
            line2 = VMobject(stroke_color=C_S2, stroke_width=3).set_points_as_corners([T2.p(a_, b_) for a_, b_ in pts2])
            tt2 = T('EG · session 2 passive table', 24, C_S2).move_to([0, 2.55, 0])
            grp2 = VGroup(T2, tt2, dots2, line2)
            grp2.shift(RIGHT * 3.5)
            self.play(grp1.animate.shift(LEFT * 3.5), run_time=0.8)
            self.play(FadeIn(VGroup(T2, tt2)), LaggedStart(*[FadeIn(d_, scale=1.8) for d_ in dots2], lag_ratio=0.1),
                      run_time=1.2)
            self.play(Create(line2), run_time=0.9)

        # ── b04: subtract the table from an active trial ────────────────────
        with self.beat('c03_b04'):
            self.clear_all(0.5, keep=(tag,))
            t, tq, pas, tgt = Y['s1_t'], Y['s1_torque'], Y['s1_passive'], Y['s1_target']
            P = Plot((0, 91), (-35, 5), 7.2, 4.5, xticks=(0, 20, 40, 60, 80), yticks=(-30, -20, -10, 0),
                     xlabel='time in the active trial (s)', ylabel='torque (Nm)', tick_size=18,
                     grid=True).move_to([-2.75, -0.45, 0])
            ttl = T('EG · session 1 · active trial at −0.038 rad', 22, C_S1).move_to([-2.75, 3.0, 0])
            self.play(FadeIn(ttl), FadeIn(P), run_time=0.6)
            self.at('for every sample of an active trial')
            c_tq = envelope_curve(P, t, tq, C_TQ, 2.0)
            lg1 = VGroup(Line(LEFT * 0.3, RIGHT * 0.3, stroke_color=C_TQ, stroke_width=4),
                         T('measured torque', 22, C_TQ)).arrange(RIGHT, buff=0.2)
            lg2 = VGroup(Line(LEFT * 0.3, RIGHT * 0.3, stroke_color=PAS, stroke_width=5),
                         T('passive (table)', 22, PAS)).arrange(RIGHT, buff=0.2)
            lg3 = VGroup(Line(LEFT * 0.3, RIGHT * 0.3, stroke_color=ACT, stroke_width=5),
                         T('target (active torque)', 22, ACT)).arrange(RIGHT, buff=0.2)
            legend = VGroup(lg1, lg2, lg3).arrange(DOWN, buff=0.22, aligned_edge=LEFT).move_to([4.7, 2.5, 0])
            legend.align_to([2.5, 0, 0], LEFT)
            self.play(Create(c_tq), FadeIn(lg1), run_time=1.5)
            self.at('looks up the passive torque')
            c_pas = P.curve(t, pas, PAS, 4)
            self.play(Create(c_pas), FadeIn(lg2), run_time=1.0)
            self.at('For EG at this position')
            p1, p2 = F2['display']['eg_passive_S1_Nm'], F2['display']['eg_passive_S2_Nm']
            y_p = P.p(91, p1)[1]
            plab = T(f'≈ {fmt(p1, 1)} Nm', 30, PAS, weight='BOLD').move_to([4.4, y_p - 0.05, 0])
            plab.align_to([2.5, 0, 0], LEFT)
            parr = Arrow(plab.get_left() + LEFT * 0.05, P.p(91, p1) + RIGHT * 0.03, buff=0.05, color=PAS,
                         stroke_width=3, max_tip_length_to_length_ratio=0.35)
            self.play(FadeIn(plab), GrowArrow(parr), run_time=0.7)
            self.at('In session 2')
            n2 = card([W('session 2, same position:', 22, C_S2, width=chars_for(3.6, 22)),
                       T(f'≈ {fmt(p2, 1)} Nm', 34, C_S2, weight='BOLD')], width=4.2, color=C_S2, pad=0.25
                      ).move_to([4.6, -1.1, 0]).align_to([2.5, 0, 0], LEFT)
            self.play(FadeIn(n2, shift=LEFT * 0.2), run_time=0.8)
            self.at('What remains is the target', lead=2.2)
            Pz = Plot((24, 44), (-35, 5), 7.2, 4.5, xticks=(25, 30, 35, 40), yticks=(-30, -20, -10, 0),
                      xlabel='time in the active trial (s)', ylabel='torque (Nm)', tick_size=18,
                      grid=True).move_to(P.get_center())
            mz = (t >= 24) & (t <= 44)
            c_tqz = envelope_curve(Pz, t[mz], tq[mz], C_TQ, 2.0)
            c_pasz = Pz.curve(t[mz], pas[mz], PAS, 4)
            self.play(ReplacementTransform(P, Pz), ReplacementTransform(c_tq, c_tqz),
                      ReplacementTransform(c_pas, c_pasz), run_time=1.2)
            c_tg = envelope_curve(Pz, t[mz], tgt[mz], ACT, 3)
            self.at('What remains is the target', lead=0.0)
            self.play(Create(c_tg), FadeIn(lg3), run_time=1.3)

        # ── b05: the network predicts the active torque only ────────────────
        with self.beat('c03_b05'):
            self.clear_all(0.5, keep=(tag,))
            net = VGroup(RoundedRectangle(corner_radius=0.14, width=2.5, height=0.9, fill_color=PANEL, fill_opacity=1,
                                          stroke_color=C_PRED, stroke_width=3), T('network', 28, C_PRED))
            net[1].move_to(net[0])
            r1 = VGroup(net, Arrow(LEFT * 0.7, RIGHT * 0.7, buff=0, color=INK, stroke_width=5),
                        T('active torque (Nm)', 32, ACT, weight='BOLD')).arrange(RIGHT, buff=0.35).move_to([0, 2.45, 0])
            self.play(FadeIn(net), run_time=0.5)
            self.play(GrowArrow(r1[1]), FadeIn(r1[2]), run_time=0.7)
            self.at('does the network predict the total')
            a1 = T('active torque', 26, ACT)
            ar = Arrow(LEFT * 1.0, RIGHT * 1.0, buff=0, color=DIM, stroke_width=4)
            a2 = T('total torque', 26, DIM)
            r2 = VGroup(a1, ar, a2).arrange(RIGHT, buff=0.45).move_to([0, 1.05, 0])
            plus = T('+ passive', 22, PAS).next_to(ar, UP, buff=0.4)
            self.play(FadeIn(VGroup(a1, ar, a2, plus)), run_time=0.7)
            self.at('No. It outputs')
            xm = ar.get_center()
            cross = VGroup(Line(xm + UL * 0.3, xm + DR * 0.3), Line(xm + UR * 0.3, xm + DL * 0.3)).set_stroke(C_BAD, 7)
            self.play(Create(cross), VGroup(a2, plus).animate.set_opacity(0.7), run_time=0.7)
            self.at('The ground truth')
            key = card([T('answer key  (ground truth)', 26, C_HI, weight='BOLD'),
                        M('=  measured torque in session 2  −  <span foreground="#FF7A45">session 2</span>'
                          "'s own passive table", 28)], width=11.4, color=C_HI, pad=0.3).move_to([0, -0.7, 0])
            self.play(FadeIn(key, shift=UP * 0.2), run_time=0.9)
            self.at('One detail')
            foot = card([W('for scoring, session 2’s table is rebuilt from the still, relaxed holds only', 22, INK,
                           width=chars_for(10.6, 22)),
                         W('identical for 5 people  ·  differs for JM', 22, DIM, width=chars_for(10.6, 22))],
                        width=11.4, color=FAINT, pad=0.28).move_to([0, -2.65, 0])
            self.play(FadeIn(foot, shift=UP * 0.2), run_time=0.9)

        # ── b06: two caveats ─────────────────────────────────────────────────
        with self.beat('c03_b06'):
            self.clear_all(0.5, keep=(tag,))
            # IES: empty table
            ep = Plot((0, 1), (0, 1), 1.5, 1.0, axis_color=DIM)
            ex = VGroup(Line(ep.p(0.35, 0.3), ep.p(0.65, 0.7), stroke_color=C_BAD, stroke_width=5),
                        Line(ep.p(0.35, 0.7), ep.p(0.65, 0.3), stroke_color=C_BAD, stroke_width=5))
            ies_icon = VGroup(ep, ex)
            ies = VGroup(T('IES', 36, C_HI, weight='BOLD'), ies_icon,
                         W('no passive trials at all: the passive table is empty', 24, INK, width=chars_for(5.2, 24)),
                         W('nothing is subtracted', 24, INK, width=chars_for(5.2, 24)),
                         T('target = total torque', 26, C_HI, weight='BOLD')).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
            ies_icon.align_to(ies, LEFT)
            ies_box = RoundedRectangle(corner_radius=0.14, width=6.1, height=ies.height + 0.7, fill_color=PANEL,
                                       fill_opacity=1, stroke_color=FAINT, stroke_width=2)
            ies.move_to(ies_box)
            ies_g = VGroup(ies_box, ies).move_to([-3.4, 0.0, 0])
            # YES: typed in values
            sheet = RoundedRectangle(corner_radius=0.08, width=1.2, height=1.0, fill_color=BG, fill_opacity=1,
                                     stroke_color=DIM, stroke_width=2)
            lines_ = VGroup(*[Line(LEFT * 0.4, RIGHT * 0.4, stroke_color=DIM, stroke_width=2) for _ in range(4)]
                            ).arrange(DOWN, buff=0.17).move_to(sheet)
            yes_icon = VGroup(sheet, lines_)
            yes = VGroup(T('YES', 36, C_HI, weight='BOLD'), yes_icon,
                         W('passive values typed in by hand from the lab worksheet', 24, INK, width=chars_for(5.2, 24)),
                         W('they agree with the holds in the recording to within 0.15 Nm', 24, INK,
                           width=chars_for(5.2, 24)),
                         ).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
            yes_icon.align_to(yes, LEFT)
            yes_box = RoundedRectangle(corner_radius=0.14, width=6.1, height=ies_box.height, fill_color=PANEL,
                                       fill_opacity=1, stroke_color=FAINT, stroke_width=2)
            yes.move_to(yes_box)
            yes_g = VGroup(yes_box, yes).move_to([3.4, 0.0, 0])
            self.at('IES has no passive')
            self.play(FadeIn(ies_g, shift=RIGHT * 0.2), run_time=0.9)
            self.at('For YES')
            self.play(FadeIn(yes_g, shift=LEFT * 0.2), run_time=0.9)

        # ── b07: one sentence to remember ────────────────────────────────────
        with self.beat('c03_b07'):
            self.clear_all(0.5, keep=(tag,))
            bx = lambda txt, col, dashed=False: VGroup(
                RoundedRectangle(corner_radius=0.12, width=3.1, height=0.8, fill_color=PANEL, fill_opacity=1,
                                 stroke_color=col, stroke_width=2.5), T(txt, 24, col))
            b1, b2, b3 = bx('measurement', C_TQ), bx('estimate', C_BAD), bx('answer key', C_HI)
            for b in (b1, b2, b3):
                b[1].move_to(b[0])
            eq = VGroup(b1, T('−', 40, INK), b2, T('=', 40, INK), b3).arrange(RIGHT, buff=0.3).move_to([0, 2.2, 0])
            l1 = T('The ground truth is a measurement\nminus an estimate.', 40, INK, line_spacing=0.8)
            l2 = T('Any error in the passive table\ngoes straight into the answer key.', 40, C_HI, weight='BOLD',
                   line_spacing=0.8)
            txt = VGroup(l1, l2).arrange(DOWN, buff=0.55, aligned_edge=LEFT).move_to([0, -0.9, 0])
            eq.move_to([0, 2.0, 0])
            self.at('The ground truth is not')
            self.play(FadeIn(eq, shift=DOWN * 0.15), FadeIn(l1), run_time=0.9)
            self.at('Any error')
            self.play(Indicate(b2, color=C_BAD, scale_factor=1.12), FadeIn(l2, shift=UP * 0.15), run_time=1.0)
            self.at('Keep that in mind')
            self.play(Flash(b2, color=C_BAD, flash_radius=0.7, line_length=0.2), run_time=0.8)
            self.wait(self.left())
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)
