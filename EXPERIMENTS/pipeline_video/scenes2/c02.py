"""c02 | The experiment: set-up, electrodes, eight positions, six people and two sessions, trial kinds, one trial, raw EMG."""
from manim import *
from style2 import *

F1 = facts()['x01']
X1 = load('x01')

ANG = F1['eg_S1_angles']                     # EG session 1, 8 angles (rad)
PEAK = F1['eg_S1_peak_abs_tq_Nm']            # largest |torque| per position (Nm)
LEG_FILL, LEG_EDGE = '#232A35', '#6B7684'
SUBJECTS = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES']


# ── cartoon leg ──────────────────────────────────────────────────────────────
def make_leg(J, s=1.0, full=True, pedal_color='#8B949E'):
    """Side view of the lower leg and foot (toes to the right, calf on the left), ankle at J.
    Returns a VGroup with named parts: .shank .foot .pedal .straps .axis and .rot (foot+pedal+straps, rotate
    it about J to change the ankle angle: positive = toes up)."""
    def poly(pts, fill=LEG_FILL, stroke=LEG_EDGE, r=0.15, sw=2.5):
        p = Polygon(*[np.array([x, y, 0.0]) for x, y in pts], fill_color=fill, fill_opacity=1, stroke_color=stroke,
                    stroke_width=sw)
        return p.round_corners(r)
    if full:
        shank = poly([(-0.28, 0.0), (-0.30, 0.45), (-0.55, 1.0), (-0.74, 1.5), (-0.72, 1.95), (-0.55, 2.4),
                      (-0.42, 2.8), (0.40, 2.8), (0.37, 1.5), (0.34, 0.6), (0.32, 0.0)], r=0.22)
        knee = Circle(radius=0.44, fill_color=LEG_FILL, fill_opacity=1, stroke_color=LEG_EDGE, stroke_width=2.5)
        knee.move_to([0, 2.85, 0])
        shank = VGroup(knee, shank)
    else:
        shank = poly([(-0.30, 0.0), (-0.34, 1.25), (0.38, 1.25), (0.32, 0.0)], r=0.1)
    foot = poly([(-0.38, 0.05), (-0.5, -0.3), (-0.3, -0.6), (1.75, -0.6), (1.95, -0.48), (1.9, -0.3), (1.2, -0.15),
                 (0.4, 0.02)], r=0.12)
    pedal = Rectangle(width=2.9, height=0.2, fill_color=pedal_color, fill_opacity=1, stroke_color=INK,
                      stroke_width=1.5).move_to([0.82, -0.72, 0])
    straps = VGroup(Rectangle(width=0.13, height=0.6, fill_color='#C9D1D9', fill_opacity=0.95, stroke_width=0)
                      .move_to([0.85, -0.32, 0]),
                    Rectangle(width=0.13, height=0.46, fill_color='#C9D1D9', fill_opacity=0.95, stroke_width=0)
                      .move_to([1.55, -0.38, 0]))
    axis = VGroup(Circle(radius=0.2, stroke_color=C_HI, stroke_width=3, fill_opacity=0),
                  Dot([0, 0, 0], radius=0.09, color=C_HI))
    rot = VGroup(foot, pedal, straps)
    g = VGroup(shank, rot, axis)
    g.shank, g.foot, g.pedal, g.straps, g.axis, g.rot = shank, foot, pedal, straps, axis, rot
    g.scale(s, about_point=ORIGIN).shift(J)
    g.J = np.array(J, float)
    return g


def label_line(text, size, color, anchor, pos, side=LEFT, tcolor=None):
    """Text at ``pos`` with a thin leader line to ``anchor``."""
    t = T(text, size, tcolor or color)
    t.move_to(pos)
    edge = t.get_right() if anchor[0] > pos[0] else t.get_left()
    ln = Line(edge + (RIGHT if anchor[0] > pos[0] else LEFT) * 0.08, anchor, stroke_color=color, stroke_width=2)
    return VGroup(ln, t)


class Ch02(BeatScene):
    def construct(self):
        # ── b01: the set-up ──────────────────────────────────────────────────
        with self.beat('c02_b01'):
            tag = self.tag_ref = self.intro('c02', hold=0.9)
            S = self.S = 0.85
            J = np.array([-1.2, 0.2, 0.0])
            leg = make_leg(J, S)
            shaft = Line(J, J + DOWN * 1.7, stroke_color='#4A525E', stroke_width=12).set_z_index(-1)
            box = RoundedRectangle(corner_radius=0.14, width=5.4, height=1.5, fill_color=PANEL, fill_opacity=1,
                                   stroke_color=C_S1, stroke_width=3).move_to(J + np.array([-0.8, -2.45, 0]))
            box_t = VGroup(T('dynamometer', 28, C_S1, weight='BOLD'),
                           T('holds the angle, measures the torque', 20, INK)).arrange(DOWN, buff=0.12).move_to(box)
            box.set_z_index(-1)
            arc = Arc(radius=2.15, start_angle=12 * DEGREES, angle=-52 * DEGREES, arc_center=J, color=C_HI,
                      stroke_width=7).add_tip(tip_length=0.3)
            tq_lab = VGroup(T('torque', 30, C_HI, weight='BOLD'), T('in newton metres (Nm)', 22, C_HI)).arrange(
                DOWN, buff=0.1, aligned_edge=LEFT).move_to(J + np.array([4.2, -0.6, 0]))
            strap_lab = label_line('strap', 22, DIM, J + np.array([1.32, -0.12, 0]), J + np.array([2.4, 0.95, 0]),
                                   tcolor=INK)
            axis_lab = label_line('pedal axis = ankle', 22, C_HI, J, J + np.array([-3.5, 0.0, 0]), tcolor=C_HI)
            pedal_lab = label_line('pedal', 22, DIM, J + np.array([-0.5, -0.61, 0]), J + np.array([-2.7, -0.95, 0]),
                                   tcolor=INK)
            self.until(self.cue("A person's foot is strapped"))
            self.play(FadeIn(leg.shank), FadeIn(leg.foot), run_time=0.8)
            self.play(FadeIn(leg.pedal), FadeIn(leg.straps), FadeIn(strap_lab), FadeIn(pedal_lab), run_time=0.8)
            self.until(self.cue("with the pedal's axis lined up"))
            self.play(FadeIn(leg.axis, scale=2.2), FadeIn(axis_lab), FadeIn(shaft), run_time=0.8)
            self.until(self.cue('The pedal belongs to a dynamometer'))
            self.play(FadeIn(box), FadeIn(box_t), run_time=0.8)
            self.until(self.cue('and measures the twisting force'))
            self.play(FadeIn(arc), run_time=0.8)
            self.play(FadeIn(tq_lab[0]), run_time=0.4)
            self.until(self.cue('Torque is measured in newton metres'))
            self.play(FadeIn(tq_lab[1]), run_time=0.6)
            self.leg_cache = VGroup(leg, shaft)
            self.b01_extra = VGroup(box, box_t, arc, tq_lab, strap_lab, axis_lab, pedal_lab)
            self.leg, self.J = leg, J

        # ── b02: four electrodes ─────────────────────────────────────────────
        with self.beat('c02_b02'):
            leg, J = self.leg, self.J
            shaft = self.leg_cache[1]
            self.play(FadeOut(self.b01_extra), FadeOut(shaft), run_time=0.6)
            shift = np.array([1.2, -0.55, 0.0])
            self.play(leg.animate.shift(shift), run_time=0.8)
            J2 = J + shift
            S = self.S
            spots = {'GM': (-0.56, 1.80), 'GL': (-0.64, 1.30), 'SOL': (-0.38, 0.72), 'TA': (0.36, 1.35)}
            els, labs = {}, {}
            for name, (dx, dy) in spots.items():
                p = J2 + S * np.array([dx, dy, 0])
                dot = Dot(p, radius=0.13, color=MUSCLE[name]).set_stroke(INK, 2)
                if name == 'TA':
                    lab = label_line(name, 26, MUSCLE[name], p, p + np.array([1.7, 0.35, 0]))
                else:
                    lab = label_line(name, 26, MUSCLE[name], p, p + np.array([-1.9, 0.0, 0]))
                els[name], labs[name] = dot, lab
            calf_t = VGroup(T('calf', 28, INK, weight='BOLD'), T('pushes the toes down', 22, DIM)).arrange(
                DOWN, buff=0.1, aligned_edge=LEFT).move_to([-5.0, 0.3, 0])
            shin_t = VGroup(T('shin', 28, INK, weight='BOLD'), T('pulls the toes up', 22, DIM)).arrange(
                DOWN, buff=0.1, aligned_edge=LEFT).move_to([4.7, 1.2, 0])
            emg_note = M('<b>EMG</b> = electromyography: the tiny electrical signal a muscle gives off '
                         'when told to contract', 22, DIM).move_to([0, -3.35, 0])
            self.until(self.cue('On the leg sit four'))
            self.play(LaggedStart(*[FadeIn(els[k], scale=2) for k in ('GM', 'GL', 'SOL', 'TA')], lag_ratio=0.25),
                      run_time=1.0)
            self.until(self.cue('They record the EMG'))
            self.play(FadeIn(emg_note), run_time=0.7)
            self.until(self.cue('the two bulges of the calf muscle'))
            self.play(FadeIn(labs['GM']), Indicate(els['GM'], color=C_HI, scale_factor=1.8), run_time=0.5)
            self.play(FadeIn(labs['GL']), Indicate(els['GL'], color=C_HI, scale_factor=1.8), run_time=0.5)
            self.until(self.cue('and the soleus underneath'))
            self.play(FadeIn(labs['SOL']), Indicate(els['SOL'], color=C_HI, scale_factor=1.8), run_time=0.6)
            self.until(self.cue('These three push the toes down'))
            self.play(FadeIn(calf_t, shift=RIGHT * 0.2), run_time=0.7)
            self.until(self.cue('The fourth sits on the shin'))
            self.play(FadeIn(labs['TA']), Indicate(els['TA'], color=C_HI, scale_factor=1.8), run_time=0.6)
            self.until(self.cue('It pulls the toes up'))
            self.play(FadeIn(shin_t, shift=LEFT * 0.2), run_time=0.7)

        # ── b03: eight positions ─────────────────────────────────────────────
        with self.beat('c02_b03'):
            self.clear_all(0.6, keep=(self.tag_ref,))
            xs = [-5.6 + 1.6 * k + 0.15 for k in range(8)]       # icon centres
            icons, lab_a = VGroup(), VGroup()
            rots = []
            for k, (x, a) in enumerate(zip(xs, ANG)):
                Jk = np.array([x - 0.35, 1.95, 0.0])
                ic = make_leg(Jk, 0.5, full=False, pedal_color=ANGLE_C[k])
                ic.axis.set_opacity(0.0)
                icons.add(ic)
                lab_a.add(T(fmt(a, 2, signed=True) + ' rad', 20, ANGLE_C[k], weight='BOLD').move_to([x, 0.5, 0]))
            self.play(LaggedStart(*[FadeIn(ic) for ic in icons], lag_ratio=0.12), run_time=1.4)
            self.until(self.cue('eight different settings'))
            self.play(LaggedStart(*[Rotate(ic.rot, a, about_point=ic.J) for ic, a in zip(icons, ANG)],
                                  lag_ratio=0.1), run_time=1.6)
            self.until(self.cue('We call each setting a position'))
            brace = Brace(VGroup(lab_a[0], lab_a[-1]), DOWN, buff=0.1, color=DIM)
            pos_t = T('position', 32, C_HI, weight='BOLD').next_to(brace, DOWN, buff=0.06)
            self.play(GrowFromCenter(brace), FadeIn(pos_t), run_time=0.7)
            self.until(self.cue('Angles are in radians'))
            self.play(FadeIn(lab_a), run_time=0.8)
            # two arrows (left = negative, right = positive)
            ay = -2.85
            a_left = Arrow([-0.4, ay, 0], [-6.4, ay, 0], buff=0, color=ANGLE_C[0], stroke_width=5)
            a_right = Arrow([0.4, ay, 0], [6.4, ay, 0], buff=0, color=ANGLE_C[7], stroke_width=5)
            t_left = VGroup(T('negative = toes down', 24, INK), T('plantarflexion', 22, DIM)).arrange(
                DOWN, buff=0.08).next_to(a_left, DOWN, buff=0.12).align_to(a_left, LEFT)
            t_right = VGroup(T('positive = toes up', 24, INK), T('dorsiflexion', 22, DIM)).arrange(
                DOWN, buff=0.08).next_to(a_right, DOWN, buff=0.12).align_to(a_right, RIGHT)
            self.until(self.cue('Negative means toes down'))
            self.play(GrowArrow(a_left), FadeIn(t_left), run_time=0.8)
            self.until(self.cue('Positive means toes up'))
            self.play(GrowArrow(a_right), FadeIn(t_right), run_time=0.8)
            self.until(self.cue('For EG, the eight positions run'))
            hi = VGroup(SurroundingRectangle(lab_a[0], color=C_HI, buff=0.08, corner_radius=0.08),
                        SurroundingRectangle(lab_a[-1], color=C_HI, buff=0.08, corner_radius=0.08))
            eg_t = T('EG, session 1', 22, DIM).next_to(icons, UP, buff=0.0).align_to(icons, LEFT)
            eg_t.move_to([0, 3.1, 0])
            self.play(Create(hi), FadeIn(eg_t), run_time=0.8)
            # peak torque bars
            base_y, h_max = -2.2, 1.2
            bars_g, vals_g = VGroup(), VGroup()
            for k, (x, pk) in enumerate(zip(xs, PEAK)):
                h = pk / 41.0 * h_max
                r = Rectangle(width=0.8, height=h, fill_color=ANGLE_C[k], fill_opacity=0.9, stroke_width=0)
                r.move_to([x, base_y + h / 2, 0])
                bars_g.add(r)
                vals_g.add(T(f'{pk:.1f}', 20, INK).next_to(r, UP, buff=0.06))
            base = Line([-6.4, base_y, 0], [6.4, base_y, 0], stroke_color=FAINT, stroke_width=1.6)
            bar_t = T('largest push at each position (Nm)', 22, DIM).move_to([-3.95, -0.9, 0])
            weak = T('weak', 30, ANGLE_C[0], weight='BOLD').next_to(vals_g[0:2], UP, buff=0.15)
            strong = T('strong', 30, ANGLE_C[7], weight='BOLD').next_to(vals_g[7], UP, buff=0.15)
            self.until(self.cue('With the toes down, the calf'))
            self.play(FadeIn(base), FadeIn(bar_t), *[GrowFromEdge(b, DOWN) for b in bars_g[:4]],
                      LaggedStart(*[FadeIn(v) for v in vals_g[:4]], lag_ratio=0.2), run_time=1.4)
            self.play(FadeIn(weak), run_time=0.5)
            self.until(self.cue('With the toes up, people push'))
            self.play(*[GrowFromEdge(b, DOWN) for b in bars_g[4:]],
                      LaggedStart(*[FadeIn(v) for v in vals_g[4:]], lag_ratio=0.2), run_time=1.4)
            self.play(FadeIn(strong), run_time=0.5)

        # ── b04: six people, two sessions, the question ──────────────────────
        with self.beat('c02_b04'):
            self.clear_all(0.6, keep=(self.tag_ref,))
            badges = VGroup()
            for c in SUBJECTS:
                bx = RoundedRectangle(corner_radius=0.16, width=1.55, height=0.95, fill_color=PANEL, fill_opacity=1,
                                      stroke_color=FAINT, stroke_width=2)
                badges.add(VGroup(bx, T(c, 34, INK, weight='BOLD').move_to(bx)))
            badges.arrange(RIGHT, buff=0.35).move_to([0, 2.3, 0])
            self.until(self.cue('We have recordings from six people'))
            self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in badges], lag_ratio=0.12), run_time=1.6)
            # timeline of one visit
            wblk, hblk, y_t = 5.3, 1.15, -0.2
            s1 = RoundedRectangle(corner_radius=0.12, width=wblk, height=hblk, fill_color=C_S1, fill_opacity=0.16,
                                  stroke_color=C_S1, stroke_width=3)
            s2 = RoundedRectangle(corner_radius=0.12, width=wblk, height=hblk, fill_color=C_S2, fill_opacity=0.16,
                                  stroke_color=C_S2, stroke_width=3)
            VGroup(s1, s2).arrange(RIGHT, buff=0.5).move_to([0, y_t, 0])
            s1_t = VGroup(T('session 1', 30, C_S1, weight='BOLD'), T('files call it “test”', 22, INK)).arrange(
                DOWN, buff=0.1).move_to(s1)
            s2_t = VGroup(T('session 2', 30, C_S2, weight='BOLD'), T('files call it “retest”', 22, INK)).arrange(
                DOWN, buff=0.1).move_to(s2)
            visit = Brace(VGroup(s1, s2), UP, buff=0.15, color=DIM)
            visit_t = T('same visit · electrodes left on (audit notes)', 24, INK).next_to(visit, UP, buff=0.1)
            self.until(self.cue('Each was recorded in two sessions'))
            self.play(FadeIn(s1), FadeIn(s1_t), run_time=0.7)
            self.play(FadeIn(s2), FadeIn(s2_t), run_time=0.7)
            self.until(self.cue('during the same visit'))
            self.play(GrowFromCenter(visit), FadeIn(visit_t), run_time=0.8)
            q = M(f'Learn from <span foreground="{C_S1}"><b>session 1</b></span> → predict '
                  f'<span foreground="{C_S2}"><b>session 2</b></span> ?', 42).move_to([0, -2.5, 0])
            self.until(self.cue('The whole project rests on one question'))
            self.play(FadeIn(q, shift=UP * 0.2), run_time=0.9)
            ar = CurvedArrow(s1.get_bottom() + DOWN * 0.05, s2.get_bottom() + DOWN * 0.05, angle=TAU / 7,
                             color=C_HI, stroke_width=4)
            self.play(Create(ar), run_time=0.9)

        # ── b05: three kinds of trial ────────────────────────────────────────
        with self.beat('c02_b05'):
            self.clear_all(0.6, keep=(self.tag_ref,))
            trials = F1['eg_trials']
            cw, ch_, gap = 0.27, 0.62, 0.035

            def cells(items):
                g = VGroup()
                for t in items:
                    if t['kind'] == 'mvc':
                        col, op = '#6E7681', 0.85
                    else:
                        col = C_S1 if t['session'] == 'test' else C_S2
                        op = 0.9 if t['kind'] == 'active' else 0.0
                    g.add(Rectangle(width=cw, height=ch_, fill_color=col, fill_opacity=op, stroke_color=col,
                                    stroke_width=2.5))
                return g.arrange(RIGHT, buff=gap)
            g_mvc = cells([t for t in trials if t['kind'] == 'mvc'])
            g_s1 = cells([t for t in trials if t['session'] == 'test'])
            g_s2 = cells([t for t in trials if t['session'] == 'retest'])
            line = VGroup(g_mvc, g_s1, g_s2).arrange(RIGHT, buff=0.45, aligned_edge=DOWN).move_to([0.15, 1.45, 0])
            l_mvc = VGroup(T('8 × MVC', 24, INK, weight='BOLD'), T('30 s each · max effort', 20, DIM), T('recorded once', 20, DIM)).arrange(
                DOWN, buff=0.06).next_to(g_mvc, UP, buff=0.18)
            l_s1 = VGroup(T('session 1', 24, C_S1, weight='BOLD'), T('8 passive + 8 active', 20, INK)).arrange(
                DOWN, buff=0.06).next_to(g_s1, UP, buff=0.18)
            l_s2 = VGroup(T('session 2', 24, C_S2, weight='BOLD'), T('8 passive + 8 active', 20, INK)).arrange(
                DOWN, buff=0.06).next_to(g_s2, UP, buff=0.18)
            unused = VGroup(RoundedRectangle(corner_radius=0.08, width=3.5, height=0.46, stroke_color=C_BAD,
                                             stroke_width=2.5, fill_opacity=0),
                            T('not used by the baseline', 20, C_BAD)).move_to(g_mvc.get_bottom() + DOWN * 0.55)
            unused.shift(RIGHT * (max(-6.6 - unused[0].get_left()[0], 0)))
            unused[1].move_to(unused[0])
            passive_card = card([W('first 20–30 s: relaxed, foot held still', 24, INK, width=40),
                                 W('after that: something else, which the code has to cut away', 24, INK,
                                   width=40)], width=5.9, title='passive trial', color=C_S1, size=24)
            active_card = card([W('90 s of repeated pushes at one position', 24, INK, width=40),
                                W('one trial per position, so 8 per session', 24, INK, width=40)], width=5.9,
                               title='active trial', color=C_GOOD, size=24, title_color=C_GOOD)
            key = VGroup(*[VGroup(Rectangle(width=cw, height=ch_ * 0.6, fill_color=INK, fill_opacity=op, stroke_color=INK,
                                            stroke_width=2.5), T(nm, 20, INK)).arrange(RIGHT, buff=0.15)
                           for nm, op in (('passive', 0.0), ('active', 0.9))]).arrange(RIGHT, buff=0.6)
            key.move_to([(g_s1.get_left()[0] + g_s2.get_right()[0]) / 2, line.get_bottom()[1] - 0.45, 0])
            cards = VGroup(passive_card, active_card).arrange(RIGHT, buff=0.4, aligned_edge=UP).move_to([0, -1.75, 0])
            self.until(self.cue('First, maximum-effort trials'))
            self.play(FadeIn(g_mvc), FadeIn(l_mvc), run_time=0.9)
            self.until(self.cue('The baseline does not use them'))
            self.play(g_mvc.animate.set_opacity(0.4), FadeIn(unused), run_time=0.8)
            self.until(self.cue('Then, in each session'))
            self.play(FadeIn(g_s1), FadeIn(l_s1), FadeIn(g_s2), FadeIn(l_s2), FadeIn(key), run_time=1.0)
            passive_cells = VGroup(*[c for c, t in zip(list(g_s1) + list(g_s2),
                                                       [t for t in trials if t['session']]) if t['kind'] == 'passive'])
            active_cells = VGroup(*[c for c, t in zip(list(g_s1) + list(g_s2),
                                                      [t for t in trials if t['session']]) if t['kind'] == 'active'])
            self.until(self.cue('A passive trial begins'))
            self.play(FadeIn(passive_card, shift=UP * 0.2), Indicate(passive_cells, color=C_HI, scale_factor=1.15),
                      run_time=1.0)
            self.until(self.cue('And the active trials'))
            self.play(FadeIn(active_card, shift=UP * 0.2), Indicate(active_cells, color=C_HI, scale_factor=1.15),
                      run_time=1.0)

        # ── b06: one active trial ────────────────────────────────────────────
        with self.beat('c02_b06'):
            self.clear_all(0.6, keep=(self.tag_ref,))
            tt, tq, pos = X1['trial_t'], X1['trial_tq'], X1['trial_pos']
            runs = X1['push_runs']
            pt = Plot((0, 91), (-36, 4), 10.4, 3.0, xticks=range(0, 91, 10), yticks=(-30, -20, -10, 0),
                      xlabel='time in trial (s)', ylabel='torque (Nm)', tick_size=18, grid=True)
            pt.shift(np.array([-5.0, 0.7, 0]) - pt.p(0, 4))
            pa = Plot((0, 91), (-0.8, 0.3), 10.4, 1.15, yticks=(-0.5, 0), ylabel='rad', tick_size=18,
                      yfmt=lambda v: f'{v:g}')
            pa.shift(np.array([-5.0, 2.7, 0]) - pa.p(0, 0.3))
            ttl = M(f'<b>EG</b> · <span foreground="{C_S1}">session 1</span> · position '
                    f'{fmt(F1["eg_angle"], 3)} rad', 26).move_to([0.2, 3.15, 0])
            ct = envelope_curve(pt, tt, tq, C_TQ, 2.0, 1800)
            ca = pa.curve(tt, pos, C_ANGLE, 2.4, 1500)
            self.until(self.cue('This is one active trial'))
            self.play(FadeIn(ttl), FadeIn(pt), run_time=0.7)
            self.until(self.cue('On this machine, a calf push shows up'))
            self.play(Create(ct), run_time=3.5, rate_func=linear)
            n = len(runs)
            cnt = ValueTracker(0)
            counter = always_redraw(lambda: T(f'{int(round(cnt.get_value()))} pushes', 30, C_HI, weight='BOLD')
                                    .move_to([5.2, 1.2, 0]))
            marks = VGroup(*[Triangle(color=C_HI, fill_opacity=1).scale(0.09).rotate(PI)
                             .move_to(pt.p(float(tt[a]), 1.8)) for a, b in runs])
            self.add(counter)
            self.until(self.cue('There are twenty-two of them'))
            self.play(LaggedStart(*[FadeIn(m, shift=DOWN * 0.1) for m in marks], lag_ratio=0.1),
                      cnt.animate.set_value(n), run_time=2.6, rate_func=linear)
            mid = np.where((tt > 40) & (tt < 60))[0]
            i = int(mid[np.argmin(tq[mid])])
            rest = float(np.median(tq[mid][tq[mid] > -6]))  # resting level between pushes (about -2.8 Nm)
            xd = 50.6                                         # relaxed gap just before this dip's falling edge
            dep = DoubleArrow(pt.p(xd, rest), pt.p(xd, float(tq[i])), buff=0, color=C_HI, stroke_width=4,
                              tip_length=0.18)
            dep_t = T('size of a push = how deep the dip goes', 24, C_HI).move_to([-2.2, 1.2, 0])
            self.until(self.cue('When we talk about how strong'))
            self.play(Create(dep), FadeIn(dep_t), run_time=0.9)
            self.until(self.cue('On top, the angle is flat'))
            ang_t = T('moves by only a few mrad', 22, C_ANGLE).move_to(pa.p(45, -0.45))
            self.play(FadeIn(pa), run_time=0.5)
            self.play(Create(ca), FadeIn(ang_t), run_time=1.2)
            self.until(self.cue('These are called isometric'))
            iso = M('<b>isometric contractions</b>: the muscles work, the joint does not move', 24, INK) \
                .move_to([0.2, -3.5, 0])
            self.play(FadeIn(iso, shift=UP * 0.15), run_time=0.8)

        # ── b07: raw EMG ─────────────────────────────────────────────────────
        with self.beat('c02_b07'):
            self.clear_all(0.6, keep=(self.tag_ref,))
            t = X1['raw_t'].astype(float)
            t0, t1 = float(t[0]), float(t[-1])
            rows = [('raw_torque', 'torque (Nm)', C_TQ, 1.0), ('raw_gm', 'GM', C_GM, 0.62), ('raw_gl', 'GL', C_GL, 0.62),
                    ('raw_sol', 'SOL', C_SOL, 0.62), ('raw_ta', 'TA', C_TA, 0.62)]
            group, plots, curves, labs = VGroup(), [], VGroup(), VGroup()
            y = 2.75
            for key, name, col, h in rows:
                v = X1[key].astype(float)
                lo, hi = float(v.min()), float(v.max())
                pad = 0.1 * (hi - lo + 1e-9)
                yt = (-5, -25) if key == 'raw_torque' else ()
                pl = Plot((t0, t1), (lo - pad, hi + pad), 9.0, h, axis_color=BG, yticks=yt, tick_size=18)
                pl.shift(np.array([-3.6, y - h / 2, 0]) - pl.p(t0, (lo + hi) / 2))
                y -= h + 0.2
                cv = envelope_curve(pl, t, v, col, 1.5, 1600)
                lab = T(name, 22, col, weight='BOLD').move_to(np.array([-4.4, y + 0.2 + h / 2, 0]), aligned_edge=RIGHT)
                plots.append(pl)
                curves.add(cv)
                labs.add(lab)
            axis_y = y + 0.05
            ax = Line([plots[0].p(t0, 0)[0], axis_y, 0], [plots[0].p(t1, 0)[0], axis_y, 0], stroke_color=FAINT,
                      stroke_width=1.6)
            ticks = VGroup()
            for v in np.arange(np.ceil(t0 * 2) / 2, t1 + 0.01, 0.5):
                x = plots[0].p(v, 0)[0]
                ticks.add(Line([x, axis_y, 0], [x, axis_y - 0.07, 0], stroke_color=FAINT, stroke_width=1.5),
                          T(f'{v:.1f}', 18, DIM).move_to([x, axis_y - 0.3, 0]))
            xlab = T('time in trial (s)', 18, DIM).move_to([plots[0].get_center()[0], axis_y - 0.62, 0])
            # the push inside this window
            tq = X1['raw_torque'].astype(float)
            on = np.where(tq < tq[0] - 4)[0]
            p0, p1 = float(t[on[0]]), t1
            top = plots[0].p(p0, 0)[1] + 0.5
            band = Rectangle(width=plots[0].p(p1, 0)[0] - plots[0].p(p0, 0)[0], height=top - (axis_y + 0.02),
                             fill_color=C_HI, fill_opacity=0.10, stroke_width=0).move_to(
                [(plots[0].p(p0, 0)[0] + plots[0].p(p1, 0)[0]) / 2, (top + axis_y + 0.02) / 2, 0])
            brk = T('bigger push → bigger scribble', 24, C_HI, weight='BOLD').next_to(band, UP, buff=0.0)
            brk.move_to([plots[0].p((p0 + p1) / 2, 0)[0] - 0.2, top + 0.22, 0])
            samp = T('1 sample = 1 recorded value · 1,000 per second', 22, INK).move_to([0.5, axis_y - 1.15, 0])
            chain = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.1, width=w, height=0.6, fill_color=PANEL,
                                                     fill_opacity=1, stroke_color=c, stroke_width=2.5),
                                    T(s, 22, INK)) for s, w, c in [('raw EMG', 1.9, C_GL), ('smooth effort', 2.5, C_HI),
                                                                   ('… more steps …', 2.5, FAINT),
                                                                   ('prediction', 2.1, C_PRED)]])
            for c in chain:
                c[1].move_to(c[0])
            chain.arrange(RIGHT, buff=0.55).move_to([0.2, -3.3, 0])
            chain_ar = VGroup(*[Arrow(chain[i].get_right(), chain[i + 1].get_left(), buff=0.05, color=DIM,
                                      stroke_width=3, max_tip_length_to_length_ratio=0.3) for i in range(3)])
            self.until(self.cue('And this is the raw EMG'))
            self.play(FadeIn(plots[0]), FadeIn(labs[0]), Create(curves[0]), run_time=0.8)
            self.play(LaggedStart(*[AnimationGroup(FadeIn(labs[i]), Create(curves[i])) for i in range(1, 5)],
                                  lag_ratio=0.25), FadeIn(ax), FadeIn(ticks), FadeIn(xlab), run_time=1.8)
            self.until(self.cue('It is recorded as a thousand values per second'))
            self.play(FadeIn(samp), run_time=0.7)
            self.until(self.cue('What matters is its size'))
            self.play(FadeIn(band), FadeIn(brk), run_time=0.8)
            self.until(self.cue('Turning that scribble'))
            self.play(FadeOut(samp), FadeIn(chain), run_time=0.7)
            self.play(LaggedStart(*[GrowArrow(a) for a in chain_ar], lag_ratio=0.3), run_time=1.0)
            self.play(chain[1][0].animate.set_stroke(C_HI, 4), run_time=0.5)
