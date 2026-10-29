"""c06: Cutting the recording into windows (EG, session 1, trial 21 at -0.038 rad; baseline settings)."""
from manim import *

from style2 import *

F = facts()
D = load('y02')
N_WIN = F['y02']['eg_counts']['windows_per_trial'][0]    # 8,891
NAMES = ['GM', 'GL', 'SOL', 'TA']
KEYS = ['gm', 'gl', 'sol', 'ta']
T_ = D['s1_t']
SIG = [D[f's1_{k}'] for k in KEYS] + [D['s1_angle']]
TGT = D['s1_target'].astype(float)
EX = F['y02']['example_window']
W0, W1 = EX['samples']                                   # window samples 419 .. 438
ANG = F['y02']['eg_S1_trial']['angle_mean_rad']
ROW_Y = [2.85, 2.32, 1.79, 1.26, 0.73]
ROW_H = 0.40
TGT_Y, TGT_H = -1.2, 1.8
ROW_COL = MUSCLE_C + [C_ANGLE]
ROW_NAME = NAMES + ['angle']
assert abs(TGT[W1] - EX['target_Nm']) < 1e-3


class Rows:
    """Five aligned signal rows plus the target row for samples i0..i1 of the trial."""

    def __init__(self, i0, i1, width, cx, xticks, side_labels=False, label_size=20):
        self.i0, self.i1 = i0, i1
        ts, te = T_[i0], T_[i1]
        self.plots, self.curves, self.labels = [], VGroup(), VGroup()
        idx = np.arange(i0, i1 + 1)
        for k in range(5):
            yr = (0, 1) if k < 4 else (-0.12, 0.04)
            pl = Plot((ts, te), yr, width, ROW_H, axis_color=FAINT).move_to([cx, ROW_Y[k], 0])
            self.plots.append(pl)
            self.curves.add(envelope_curve(pl, T_[idx], SIG[k][idx], ROW_COL[k], 2.0, 700))
            self.labels.add(T(ROW_NAME[k], label_size, ROW_COL[k]).next_to(pl, LEFT, buff=0.15))
            if side_labels:
                txt = '0–1' if k < 4 else f'≈ {fmt(ANG, 3)} rad'
                self.labels.add(T(txt, 18, DIM).next_to(pl, RIGHT, buff=0.2))
        self.tp = Plot((ts, te), (-33, 4), width, TGT_H, xticks=xticks, yticks=[-30, -15, 0], xlabel='time in trial (s)',
                       ylabel='active torque (Nm)', tick_size=18).move_to([cx, TGT_Y, 0])
        self.tp.shift(RIGHT * (self.plots[0].frame.get_center()[0] - self.tp.frame.get_center()[0]))  # same time axis
        self.tcurve = envelope_curve(self.tp, T_[idx], TGT[idx], C_TQ, 2.2, 700)
        self.tlabel = T('target', label_size, C_TQ).move_to(
            [self.plots[0].frame.get_left()[0] - 0.15, self.tp.frame.get_top()[1] + 0.22, 0], aligned_edge=RIGHT)
        self.frames = VGroup(*self.plots, *self.labels)
        self.top, self.bot = ROW_Y[0] + ROW_H / 2 + 0.07, ROW_Y[4] - ROW_H / 2 - 0.07

    def x(self, i):
        return self.plots[0].p(T_[i], 0)[0]

    def box(self, j, color=C_HI, sw=3.0):
        """Rectangle around the 20-sample window that ends at sample j."""
        half = 0.5 * (self.x(1) - self.x(0))
        x0, x1 = self.x(j - 19) - half, self.x(j) + half
        r = Rectangle(width=x1 - x0, height=self.top - self.bot, stroke_color=color, stroke_width=sw,
                      fill_color=color, fill_opacity=0.12).move_to([(x0 + x1) / 2, (self.top + self.bot) / 2, 0])
        return r

    def dot(self, j, radius=0.09):
        return Dot(self.tp.p(T_[j], TGT[j]), radius=radius, color=C_HI)


def until_t(scene, sec):
    """Wait until `sec` seconds of the current beat's narration have played (audio-measured onsets)."""
    scene.until(sec / BEATS[scene.beat_log_current]['seconds'])


def clear_except(scene, tag, rt=0.5):
    gone = [m for m in scene.mobjects if m is not tag and not isinstance(m, ValueTracker)]
    for m in list(scene.mobjects):
        if isinstance(m, ValueTracker):
            scene.remove(m)
    if gone:
        scene.play(*[FadeOut(m) for m in gone], run_time=rt)


class Ch06(BeatScene):
    def construct(self):
        # ── b01: the five signals and the target ────────────────────────────────────────────────────────────
        with self.beat('c06_b01'):
            tag = self.intro('c06')
            R = Rows(0, 1000, 9.8, 0.0, [1, 3, 5, 7, 9, 11], side_labels=True)
            self.play(FadeIn(R.frames), run_time=0.5)
            self.play(*[Create(c) for c in R.curves], run_time=1.6)
            self.until(self.cue('Below them is the target'))
            tl = R.tlabel
            self.play(FadeIn(R.tp), FadeIn(tl), run_time=0.6)
            self.play(Create(R.tcurve), run_time=1.4)
            self.until(self.cue('How do we turn'))
            q = T('how do we turn this into examples the network can learn from?', 24, C_HI).move_to([0, -3.4, 0])
            self.play(FadeIn(q), run_time=0.6)

        clear_except(self, tag)

        # ── b02: a window is a little table ─────────────────────────────────────────────────────────────────
        with self.beat('c06_b02'):
            i0, i1 = W1 - 109, W1 + 91                      # a 2 s zoom around the window
            R = Rows(i0, i1, 6.1, -2.55, [4.5, 5, 5.5, 6])
            tl = R.tlabel
            self.play(FadeIn(R.frames), FadeIn(R.tp), FadeIn(tl), run_time=0.5)
            self.play(*[Create(c) for c in R.curves], Create(R.tcurve), run_time=0.9)
            self.until(self.cue('Each window is twenty'))
            box, dot = R.box(W1), R.dot(W1)
            link = DashedVMobject(Line([R.x(W1), R.bot, 0], dot.get_center(), stroke_color=C_HI, stroke_width=2),
                                  num_dashes=12)
            span = T('20 samples = 0.2 s', 20, C_HI).move_to([R.x(W1) - 0.3, 0.15, 0], aligned_edge=RIGHT)
            self.play(FadeIn(box), run_time=0.5)
            self.play(Create(link), FadeIn(dot, scale=2), FadeIn(span), run_time=0.8)
            self.until(self.cue('So one example is'))
            # the table: 20 time steps x 5 signals, rows aligned with the signal rows
            Wd = D['window']
            cw, chh = 0.25, 0.48
            tx0 = 1.7
            table = VGroup()
            for k in range(5):
                lab = T(ROW_NAME[k], 20, ROW_COL[k]).move_to([tx0 - 0.1, ROW_Y[k], 0], aligned_edge=RIGHT)
                table.add(lab)
                for i in range(20):
                    v = float(Wd[i, k])
                    col = (interpolate_color(ManimColor(PANEL), ManimColor(ROW_COL[k]), float(np.clip(v / 0.7, 0, 1)))
                           if k < 4 else ManimColor('#4A525C'))
                    table.add(Rectangle(width=cw - 0.02, height=chh, stroke_width=0, fill_color=col, fill_opacity=1)
                              .move_to([tx0 + 0.15 + i * cw, ROW_Y[k], 0]))
            cap = T('input: 20 time steps × 5 signals', 22, C_HI).move_to([4.2, 0.15, 0])
            self.play(TransformFromCopy(box, table), run_time=1.3)
            self.play(FadeIn(cap), run_time=0.5)
            self.until(self.cue('Its answer is'))
            ans = VGroup(T('answer: active torque', 22, INK), T('at the last time step', 22, INK),
                         T(f'{fmt(EX["target_Nm"], 1)} Nm', 36, C_HI, weight='BOLD')).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            ans.move_to([1.2, -1.3, 0], aligned_edge=LEFT)
            arr = Arrow(dot.get_center() + RIGHT * 0.15, ans.get_left() + LEFT * 0.1, buff=0.05, color=C_HI, stroke_width=3,
                        max_tip_length_to_length_ratio=0.15, path_arc=-0.3)
            self.play(Create(arr), FadeIn(ans), run_time=0.9)
            self.until(self.cue('In plain words'))
            plain = W('given the last 0.2 s of muscle activity, and the angle: what is the torque right now?', 22, DIM,
                      width=chars_for(5.0, 22)).move_to([1.7, -2.95, 0], aligned_edge=LEFT)
            self.play(FadeIn(plain), run_time=0.6)

        clear_except(self, tag)

        # ── b03: slide one sample at a time ─────────────────────────────────────────────────────────────────
        with self.beat('c06_b03'):
            j0, j1 = W1 - 28, W1 + 372
            R = Rows(j0, j1, 6.1, -2.55, [5, 6, 7, 8, 9])
            tl = R.tlabel
            jt = ValueTracker(W1)
            n = ValueTracker(W1 - 18)                   # number of the window that ends at sample W1
            win = always_redraw(lambda: R.box(int(round(jt.get_value()))))
            head = always_redraw(lambda: R.dot(int(round(jt.get_value()))))

            def trail_fn():
                j = int(round(jt.get_value()))
                pts = [R.tp.p(T_[i], TGT[i]) for i in range(W1, j + 1, 2)] or [R.tp.p(T_[W1], TGT[W1])] * 2
                if len(pts) < 2:
                    pts = pts * 2
                return VMobject(stroke_color=C_HI, stroke_width=4).set_points_as_corners(pts)
            trail = always_redraw(trail_fn)
            counter = always_redraw(lambda: T(f'{int(round(n.get_value())):,}', 54, C_HI, weight='BOLD')
                                    .move_to([1.7, 2.95, 0], aligned_edge=LEFT))
            cl = T('window number in this trial', 22, DIM).move_to([1.7, 2.4, 0], aligned_edge=LEFT)
            # two neighbouring windows as strips of 20 samples
            cA, cB = VGroup(), VGroup()
            for i in range(21):
                if i < 20:
                    cA.add(Rectangle(width=0.2, height=0.3, stroke_width=1, stroke_color=BG, fill_opacity=1,
                                     fill_color=C_HI if i >= 1 else '#6B6A2A').move_to([1.9 + i * 0.2, 1.3, 0]))
                if i >= 1:
                    cB.add(Rectangle(width=0.2, height=0.3, stroke_width=1, stroke_color=BG, fill_opacity=1,
                                     fill_color=C_HI if i <= 19 else '#6B6A2A').move_to([1.9 + i * 0.2, 0.9, 0]))
            la = T('this window', 20, DIM).move_to([1.7, 1.75, 0], aligned_edge=LEFT)
            lb = T('next window: 1 sample (10 ms) later', 20, DIM).move_to([1.7, 0.45, 0], aligned_edge=LEFT)
            share = T('share 19 of 20 samples', 24, C_HI).move_to([1.7, -0.1, 0], aligned_edge=LEFT)
            self.play(FadeIn(R.frames), FadeIn(R.tp), FadeIn(tl), run_time=0.5)
            self.play(*[Create(c) for c in R.curves], Create(R.tcurve), run_time=0.8)
            self.add(win, head, trail, counter)
            self.play(FadeIn(cl), FadeIn(cA), FadeIn(la), run_time=0.5)
            self.play(FadeIn(cB), FadeIn(lb), run_time=0.5)
            # slide, accelerating; the second part starts when the narration says the neighbours are nearly identical
            # the counter follows the box: window number = (last sample of the window) - 18
            self.play(jt.animate.set_value(W1 + 60), n.animate.set_value(W1 + 42), run_time=2.4,
                      rate_func=rate_functions.ease_in_quad)
            until_t(self, 6.0)              # 'Neighbouring windows share nineteen ...' (spoken-time cue)
            self.play(FadeIn(share), jt.animate.set_value(W1 + 180), n.animate.set_value(W1 + 162), run_time=2.2,
                      rate_func=rate_functions.ease_in_quad)
            until_t(self, 10.6)             # 'A trial of eight thousand ...' starts at about 11.0 s
            self.play(jt.animate.set_value(W1 + 370), n.animate.set_value(W1 + 352), run_time=1.8,
                      rate_func=rate_functions.ease_out_cubic)
            self.play(FadeOut(win), FadeOut(head), run_time=0.3)
            eq = T('8,910 − 20 + 1 = 8,891', 30, INK).move_to([1.7, -1.0, 0], aligned_edge=LEFT)
            cl2 = T('windows in this trial', 22, INK).move_to([1.7, 2.4, 0], aligned_edge=LEFT)
            self.play(FadeOut(cl), FadeIn(cl2), FadeIn(eq), n.animate.set_value(N_WIN), run_time=1.0,
                      rate_func=rate_functions.ease_out_cubic)

        clear_except(self, tag)

        # ── b04: windows only look backwards; the filters peek ahead ────────────────────────────────────────
        with self.beat('c06_b04'):
            lo, hi = W1 - 60, W1 + 60
            idx = np.arange(lo, hi + 1)
            rel = T_[idx] - T_[W1]
            pl = Plot((-0.6, 0.6), (0, 1), 10.6, 2.3, xticks=[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
                      xlabel='time relative to the target moment (s)', yticks=[0, 1], tick_size=18,
                      xfmt=lambda v: f'{v:+.1f}' if abs(v) > 1e-9 else '0',
                      ylabel='EMG envelope (0–1)').move_to([0.4, -0.45, 0])
            cur = VGroup(*[pl.curve(rel, SIG[k][idx], MUSCLE_C[k], 2.2) for k in range(4)])
            xs = lambda t: pl.p(t, 0)[0]
            half = 0.005
            yb, yt = pl.p(0, 0)[1], pl.p(0, 1)[1]
            win_r = Rectangle(width=xs(half) - xs(-19 * 0.01 - half), height=yt - yb, fill_color=C_HI, fill_opacity=0.18,
                              stroke_width=0).move_to([(xs(half) + xs(-19 * 0.01 - half)) / 2, (yt + yb) / 2, 0])
            now = Line([xs(0), yb, 0], [xs(0), yt, 0], stroke_color=C_HI, stroke_width=2.5)
            now_up = DashedVMobject(Line([xs(0), yt, 0], [xs(0), 2.55, 0], stroke_color=C_HI, stroke_width=2), num_dashes=14)
            win_l = T('window: only the past', 20, C_HI).move_to([xs(0) - 0.15, yt + 0.25, 0], aligned_edge=RIGHT)
            now_l = T('target moment', 20, C_HI).move_to([xs(0) + 0.15, yt + 0.25, 0], aligned_edge=LEFT)
            fut = Rectangle(width=xs(0.6) - xs(half), height=yt - yb, fill_color=C_BAD, fill_opacity=0.09, stroke_width=0)
            fut.move_to([(xs(0.6) + xs(half)) / 2, (yt + yb) / 2, 0])
            fut_l = T('future samples: not in the window', 20, C_BAD).move_to([xs(0.3), yt - 0.22, 0])
            tarr = VGroup(T('time', 20, DIM), Arrow(ORIGIN, RIGHT * 0.7, buff=0, color=DIM, stroke_width=3,
                                                    max_tip_length_to_length_ratio=0.3)).arrange(RIGHT, buff=0.1)
            tarr.move_to([5.6, pl.xlabel.get_center()[1], 0])
            leg = VGroup(*[VGroup(Rectangle(width=0.28, height=0.28, fill_color=c, fill_opacity=1, stroke_width=0), T(nm, 20, c))
                            .arrange(RIGHT, buff=0.12) for nm, c in zip(NAMES, MUSCLE_C)]).arrange(RIGHT, buff=0.5)
            leg.move_to([-1.6, -2.65, 0])
            self.play(FadeIn(pl), FadeIn(leg), run_time=0.5)
            self.play(*[Create(c) for c in cur], FadeIn(win_r), FadeIn(win_l), FadeIn(tarr), run_time=1.2)
            self.play(Create(now), FadeIn(now_l), run_time=0.6)
            self.until(self.cue('The moment-by-moment'))
            self.play(FadeIn(fut), FadeIn(fut_l), run_time=0.7)
            self.until(self.cue('mainly the forward-backward'))
            reach = F['x02']['imp90'][1]                      # 0.376 s
            e_y, d_y = 1.85, 2.45

            def bracket(t_end, y, color):
                ln = Line([xs(0), y, 0], [xs(t_end), y, 0], stroke_color=color, stroke_width=4)
                ends = VGroup(Line([xs(0), y - 0.1, 0], [xs(0), y + 0.1, 0], stroke_color=color, stroke_width=3),
                              Line([xs(t_end), y - 0.1, 0], [xs(t_end), y + 0.1, 0], stroke_color=color, stroke_width=3))
                return VGroup(ln, ends)
            eb = bracket(reach, e_y, C_BAD)
            el = T(f'envelope filter looks ≈ {fmt(reach, 1)} s ahead', 20, C_BAD).move_to([xs(0) - 0.2, e_y, 0], aligned_edge=RIGHT)
            self.play(Create(eb), FadeIn(el), Create(now_up), run_time=0.9)
            self.until(self.cue('a much shorter one'))
            db = bracket(0.1, d_y, C_BAD)
            dl = T('downsampling ≈ 0.1 s ahead', 20, C_BAD).move_to([xs(0) - 0.2, d_y, 0], aligned_edge=RIGHT)
            self.play(Create(db), FadeIn(dl), run_time=0.8)
            self.until(self.cue('Two constants'))
            note = T("two constants use the whole recording: the trial's average and session 1's maximum", 22, INK) \
                .move_to([0, -3.35, 0])
            self.play(FadeIn(note), run_time=0.6)
