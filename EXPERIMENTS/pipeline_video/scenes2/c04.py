"""Chapter 4: from raw EMG to a smooth envelope (six steps on 3 s of EG's soleus)."""
import os

from manim import *
from style2 import *

X = load('x02')
XT = load('c04_trim')
F = facts()['x02']

STEPS = ['remove the\naverage', 'band-pass\n30–300 Hz', 'rectify', 'low-pass\n2 Hz', 'trim\n1 s', '1000 → 100\nsamples/s']
SIGNAL = interpolate_color(ManimColor(C_SOL), ManimColor(BG), 0.35)


def spoken_frac(say, phrase):
    """Fraction of the narration (by characters, with pauses at punctuation) spoken before ``phrase``."""
    words, target = say.split(), phrase.split()
    norm = lambda w: w.strip('.,:;!?"').lower()
    w = [len(x) + (3 if x[-1] in ',:;' else 6 if x[-1] in '.?!' else 0) + 1 for x in words]
    for i in range(len(words)):
        if [norm(x) for x in words[i:i + len(target)]] == [norm(x) for x in target]:
            return sum(w[:i]) / sum(w)
    raise ValueError(phrase)


def env_idx(y, n=1500):
    """Indices that keep the min and max of every bin (so spikes survive); shared by all stages of the piece."""
    m = n // 2
    edges = np.linspace(0, len(y), m + 1).astype(int)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b]
        idx += sorted({a + int(np.argmin(seg)), a + int(np.argmax(seg))})
    return np.array(idx)


def align_plots(new, old, old_corner, new_corner):
    """Shift ``new`` so that its data point new_corner sits where old's old_corner sits."""
    new.shift(old.p(*old_corner) - new.p(*new_corner))
    return new


class Strip(VGroup):
    """The six step boxes along the top; ``light(i)`` returns the animations that make step i the active one."""

    def __init__(self):
        super().__init__()
        self.boxes, self.texts, self.nums = [], [], []
        for i, s in enumerate(STEPS):
            r = RoundedRectangle(corner_radius=0.1, width=2.0, height=0.95, fill_color=PANEL, fill_opacity=1,
                                 stroke_color=FAINT, stroke_width=2)
            num = T(str(i + 1), 22, DIM, weight='BOLD')
            txt = T(s, 18, DIM, line_spacing=0.7)
            row = VGroup(num, txt).arrange(RIGHT, buff=0.14).move_to(r)
            self.boxes.append(r)
            self.texts.append(txt)
            self.nums.append(num)
            self.add(VGroup(r, row))
        self.arrange(RIGHT, buff=0.24).move_to([0, 2.75, 0])

    def light(self, i):
        anims = []
        for k, (r, tx, nm) in enumerate(zip(self.boxes, self.texts, self.nums)):
            if k == i:
                anims += [r.animate.set_stroke(C_HI, 3.5).set_fill(C_HI, 0.2), tx.animate.set_color(INK),
                          nm.animate.set_color(C_HI)]
            elif k < i:
                anims += [r.animate.set_stroke(DIM, 2).set_fill(PANEL, 1), tx.animate.set_color(INK),
                          nm.animate.set_color(DIM)]
            else:
                anims += [r.animate.set_stroke(FAINT, 2).set_fill(PANEL, 1), tx.animate.set_color(DIM),
                          nm.animate.set_color(DIM)]
        return anims


class Ch04(BeatScene):
    def at(self, phrase, lead=0.4):
        """Wait until ``lead`` seconds before ``phrase`` is spoken (character-weighted estimate)."""
        t0, dur = self._beat
        rest = t0 + spoken_frac(SAY[self.beat_log_current], phrase) * dur - lead - self.renderer.time
        if rest > 0.02:
            self.wait(rest)
        if os.environ.get('DBG_T'):
            print('DBG', self.beat_log_current, phrase, round(self.renderer.time - t0, 1), '/', round(dur, 1))

    def construct(self):
        t = X['piece_t']
        t0, t1 = float(t[0]), float(t[-1])
        idx = env_idx(X['piece_band'])
        XT_TICKS = (4, 5, 6)

        def make_plot(y_range, yticks, yfmt=None):
            return Plot((t0, t1), y_range, 7.4, 3.2, xticks=XT_TICKS, yticks=yticks, xlabel='time in the trial (s)',
                        ylabel='soleus EMG (a.u.)', tick_size=18, xfmt=lambda v: f'{v:.0f}', yfmt=yfmt)

        P = make_plot((-0.3, 0.3), (-0.2, 0, 0.2), lambda v: f'{v:.1f}').move_to([-2.5, -0.5, 0])
        zero = P.hline(0, FAINT, 1.5, dashed=True)

        def mk(arr, color=SIGNAL, width=1.6, plot=None):
            pl = plot or P
            lo, hi = pl.y_range
            pts = [pl.p(t[i], float(np.clip(arr[i], lo, hi))) for i in idx]
            return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(pts)

        cap = T('EG · session 1 · −0.038 rad · soleus, 3 s', 20, DIM)

        # ── b01: the six steps ───────────────────────────────────────────────
        with self.beat('c04_b01'):
            tag = self.intro('c04')
            strip = Strip()
            self.at('Every channel goes through')
            self.play(LaggedStart(*[FadeIn(g, shift=DOWN * 0.15) for g in strip], lag_ratio=0.15), run_time=1.6)
            self.at('We will follow')
            cap.next_to(P, UP, buff=0.12).align_to(P.frame, LEFT)
            trace = mk(X['piece_raw'])
            self.play(FadeIn(P), FadeIn(zero), FadeIn(cap), run_time=0.7)
            self.play(Create(trace), run_time=1.6)

        # ── b02: remove the average ──────────────────────────────────────────
        with self.beat('c04_b02'):
            self.play(*strip.light(0), run_time=0.5)
            mean = F['mean_offset']
            mline = P.hline(mean, C_HI, 3.5, dashed=True, x0=t0, x1=t1 + 0.12).set_z_index(2)
            mlab = VGroup(T('average of the', 20, C_HI), T('whole trial', 20, C_HI),
                          T(f'average = {fmt(mean, 3)} a.u.', 20, C_HI, weight='BOLD')
                          ).arrange(DOWN, buff=0.05, aligned_edge=LEFT).next_to(P.p(t1 + 0.12, mean), RIGHT, buff=0.3)
            self.at('so the code subtracts')
            self.play(Create(mline), FadeIn(mlab), run_time=0.8)
            self.at('The signal is now centred')
            shift_up = P.p(t0, 0) - P.p(t0, mean)
            self.play(Transform(trace, mk(X['piece_demeaned'])), mline.animate.shift(shift_up),
                      mlab.animate.shift(shift_up), run_time=1.4)
            self.play(FadeOut(mline), FadeOut(mlab), run_time=0.4)

        # ── b03: band-pass ───────────────────────────────────────────────────
        with self.beat('c04_b03'):
            self.play(*strip.light(1), run_time=0.5)
            G = Plot((0, np.log10(499)), (0, 1.1), 3.7, 1.9, xticks=[np.log10(v) for v in (10, 30, 100, 300)],
                     yticks=(0, 1), xlabel='frequency (Hz)', ylabel='gain', tick_size=18,
                     xfmt=lambda v: f'{10 ** v:.0f}').move_to([4.55, 0.35, 0])
            self.at('Think of it as a sieve')
            self.play(FadeIn(G), run_time=0.6)
            lo_band = G.band(0, np.log10(30), C_BAD, 0.16, 0, 1.1)
            lo_lab = VGroup(T('cable and skin', 18, C_BAD), T('movement', 18, C_BAD)
                            ).arrange(DOWN, buff=0.05).move_to(G.p(np.log10(5.5), 0.62))
            self.at('Very slow wiggles')
            self.play(FadeIn(lo_band), FadeIn(lo_lab), run_time=0.8)
            keep = G.band(np.log10(30), np.log10(300), C_GOOD, 0.2, 0, 1.1)
            keep_lab = T('kept', 22, C_GOOD, weight='BOLD').move_to(G.p(np.log10(95), 0.5))
            gc = G.curve(np.log10(X['f']), X['gain_bp'], C_HI, 3)
            self.at('Most of the')
            self.play(FadeIn(keep), FadeIn(keep_lab), Create(gc), run_time=1.0)
            self.at('so the code keeps')
            self.play(Transform(trace, mk(X['piece_band'])), run_time=1.3)
            zh = T('same band as Zhang et al. 2022', 20, DIM).move_to([4.55, -1.6, 0])
            self.at('the same band as')
            self.play(FadeIn(zh), run_time=0.5)
            lost = T('above 300 Hz: also thrown away', 20, DIM).move_to([4.55, -2.2, 0])
            self.at('Some genuine muscle signal')
            self.play(FadeIn(lost), run_time=0.5)
            col3 = VGroup(G, lo_band, lo_lab, keep, keep_lab, gc, zh, lost)

        # ── b04: rectify ─────────────────────────────────────────────────────
        with self.beat('c04_b04'):
            self.play(*strip.light(2), FadeOut(col3), run_time=0.5)
            self.at('The signal swings')
            self.play(Indicate(zero, color=C_HI, scale_factor=1.0), run_time=0.7)
            self.at('So every negative value')
            self.play(Transform(trace, mk(X['piece_rect'])), run_time=1.8)
            note = VGroup(T('sign: no information', 24, DIM), T('size: how hard the muscle works', 24, INK)
                          ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([4.3, 0.2, 0])
            self.at('Only the size does')
            self.play(FadeIn(note, shift=LEFT * 0.2), run_time=0.7)

        # ── b05: low-pass = envelope ─────────────────────────────────────────
        with self.beat('c04_b05'):
            self.play(*strip.light(3), FadeOut(note), run_time=0.5)
            L = Plot((0, np.log10(499)), (0, 1.1), 3.7, 1.9, xticks=[np.log10(v) for v in (2, 10, 100)],
                     yticks=(0, 1), xlabel='frequency (Hz)', ylabel='gain', tick_size=18,
                     xfmt=lambda v: f'{10 ** v:.0f}').move_to([4.55, 0.35, 0])
            lc = L.curve(np.log10(X['f']), X['gain_lp'], C_SOL, 3)
            lv = L.vline(np.log10(2), C_HI, 2)
            lt = T('2 Hz', 20, C_HI).next_to(L.p(np.log10(2), 1.1), UR, buff=0.05)
            self.at('a low-pass filter')
            self.play(FadeIn(L), run_time=0.5)
            self.play(Create(lc), Create(lv), FadeIn(lt), run_time=0.9)
            env = mk(X['piece_env'], C_SOL, 5)
            self.at('draws a smooth curve')
            self.play(trace.animate.set_stroke(opacity=0.4), Create(env), run_time=1.6)
            elab = T('envelope', 28, C_SOL, weight='BOLD').move_to([4.55, -1.2, 0])
            earr = Arrow(elab.get_left() + LEFT * 0.05, P.p(6.55, float(X['piece_env'][-150])) + RIGHT * 0.1,
                         buff=0.05, color=C_SOL, stroke_width=3, max_tip_length_to_length_ratio=0.3)
            self.at('That curve is called')
            self.play(FadeIn(elab), GrowArrow(earr), run_time=0.7)
            # zoom in on the envelope itself
            P2 = Plot((t0, t1), (0, 0.06), 7.4, 3.2, xticks=XT_TICKS, yticks=(0, 0.02, 0.04, 0.06),
                      xlabel='time in the trial (s)', ylabel='soleus envelope (mV)', tick_size=18,
                      xfmt=lambda v: f'{v:.0f}', yfmt=lambda v: f'{v:.2f}')
            align_plots(P2, P, (t0, -0.3), (t0, 0.0))
            env2 = mk(X['piece_env'], C_SOL, 5, plot=P2)
            self.at('It rises when the muscle')
            self.play(FadeOut(trace), FadeOut(zero), FadeOut(earr), FadeOut(elab), FadeOut(cap),
                      ReplacementTransform(P, P2), Transform(env, env2), run_time=1.4)
            up = T('works harder', 20, INK).move_to(P2.p(4.75, 0.034))
            dn = T('relaxes', 20, INK).move_to(P2.p(4.3, 0.015))
            self.play(FadeIn(up), FadeIn(dn), run_time=0.5)
            self.at('This is the signal')
            net = VGroup(RoundedRectangle(corner_radius=0.12, width=2.9, height=0.8, fill_color=PANEL, fill_opacity=1,
                                          stroke_color=C_PRED, stroke_width=2.5),
                         T('network input', 24, C_PRED)).move_to([4.55, -1.6, 0])
            net[1].move_to(net[0])
            self.play(FadeIn(net, shift=LEFT * 0.2), env.animate.set_stroke(width=6.5), run_time=0.8)
            self.P = P2
            col5 = VGroup(L, lc, lv, lt, up, dn, net)

        P = self.P

        # ── b06: the filter reaches into the future ──────────────────────────
        with self.beat('c04_b06'):
            self.clear_all(0.5, keep=(tag, strip))
            self.play(*strip.light(3), run_time=0.3)
            sel = np.abs(X['imp_t']) <= 1.5
            W_ = Plot((-1.5, 1.5), (-0.2, 1.05), 7.6, 3.3, xticks=(-1, -0.5, 0, 0.5, 1), yticks=(0, 1),
                      xlabel='time relative to “now” (s)', ylabel='weight in the envelope', tick_size=18,
                      xfmt=lambda v: f'{v:g}' if v else '0').move_to([-2.5, -0.55, 0])
            fut = W_.band(0, 1.5, C_HI, 0.13)
            fut_l = T('future', 24, C_HI, weight='BOLD').move_to(W_.p(0.8, 0.9))
            past_l = T('past', 24, DIM, weight='BOLD').move_to(W_.p(-0.8, 0.9))
            now = W_.vline(0, INK, 2)
            now_l = T('now', 22, INK).next_to(W_.p(0, 1.05), UP, buff=0.05)
            wc = W_.curve(X['imp_t'][sel], X['imp_h'][sel], C_SOL, 3.5)
            self.at('Both filters are run')
            self.play(FadeIn(W_), FadeIn(fut), FadeIn(fut_l), FadeIn(past_l), FadeIn(now), FadeIn(now_l), run_time=0.8)
            self.play(Create(wc), run_time=1.4)
            a90, a99 = F['imp90'][1], F['imp99'][1]
            m90 = W_.vline(a90, C_HI, 2.5, y0=-0.2, y1=0.55)
            l90 = T(f'{fmt(a90, 1)} s', 22, C_HI, weight='BOLD').next_to(W_.p(a90, 0.55), UP, buff=0.05)
            m99 = W_.vline(a99, DIM, 2, y0=-0.2, y1=0.2)
            l99 = T(f'{fmt(a99, 1)} s', 22, DIM).next_to(W_.p(a99, 0.2), UP, buff=0.05)
            self.at('the next few tenths')
            self.play(Create(m90), FadeIn(l90), run_time=0.7)
            self.play(Create(m99), FadeIn(l99), run_time=0.6)
            cw = chars_for(3.8, 24)
            b1 = T('forward + backward:\nno delay', 24, INK, line_spacing=0.8)
            b2 = W(f'but it reaches ≈ {fmt(a90, 1)} s into the future', 24, C_HI, width=cw)
            b3 = W(f'(a faint tail to ≈ {fmt(a99, 1)} s)', 22, DIM, width=cw)
            b4 = T('→ offline only', 30, C_BAD, weight='BOLD')
            badge = card([b1, b2, b3, b4], width=4.4, color=C_BAD, pad=0.25).move_to([4.5, 0.05, 0])
            self.at('That is fine for analysing')
            self.play(FadeIn(badge, shift=LEFT * 0.2), run_time=0.8)

        # ── b07: the first second is cut ─────────────────────────────────────
        with self.beat('c04_b07'):
            self.clear_all(0.5, keep=(tag, strip))
            self.play(*strip.light(4), run_time=0.3)
            tl, el = XT['t_start'], XT['env_start']
            tr, er = XT['t_end'], XT['env_end']
            ymax = 0.07
            A = Plot((0, 4), (0, ymax), 5.6, 3.0, xticks=(0, 1, 2, 3, 4), yticks=(0, 0.02, 0.04, 0.06),
                     xlabel='time in the trial (s)', ylabel='soleus envelope (mV)', tick_size=18,
                     xfmt=lambda v: f'{v:.0f}', yfmt=lambda v: f'{v:.2f}').move_to([-3.6, -0.55, 0])
            B = Plot((float(tr[0]), float(tr[0]) + 4), (0, ymax), 5.0, 3.0, xticks=(87, 88, 89, 90),
                     xlabel='time in the trial (s)', tick_size=18, xfmt=lambda v: f'{v:.0f}')
            align_plots(B, A, (0, 0), (float(tr[0]), 0)).shift(RIGHT * 7.0)
            tA = T('start of the trial', 24, INK).next_to(A, UP, buff=0.12).align_to(A.frame, LEFT)
            tB = T('end of the trial', 24, INK).move_to([0, 0, 0]).next_to(B, UP, buff=0.12).align_to(B.frame, LEFT)
            tB.set_y(tA.get_y())
            m_c = tl >= 1.0
            cutc = A.curve(tl[~m_c], el[~m_c], C_BAD, 2.5)
            keepc = envelope_curve(A, tl[m_c], el[m_c], C_SOL, 2.5)
            endc = envelope_curve(B, tr, er, C_SOL, 2.5)
            self.play(FadeIn(A), FadeIn(B), FadeIn(tA), FadeIn(tB), run_time=0.7)
            self.play(Create(cutc), Create(keepc), Create(endc), run_time=1.4)
            self.at('The first second')
            shade = A.band(0, 1, C_BAD, 0.25)
            cut_l = T('cut', 26, C_BAD, weight='BOLD').move_to(A.p(0.5, 0.062))
            self.play(FadeIn(shade), FadeIn(cut_l), run_time=0.7)
            self.play(cutc.animate.set_stroke(opacity=0.35), run_time=0.4)
            cmt = W('code comment: removes a start-up effect of the filters', 22, DIM, width=chars_for(4.8, 22)
                    ).move_to([-3.6, -3.3, 0])
            self.at('comment says this')
            self.play(FadeIn(cmt), run_time=0.6)
            self.at('Oddly, the end')
            x_end = float(tr[-1])
            end_box = DashedVMobject(Rectangle(width=abs(B.p(x_end, 0)[0] - B.p(x_end - 1, 0)[0]), height=3.0 * 0.06 / ymax,
                                               stroke_color=C_HI, stroke_width=2.5, fill_opacity=0
                                               ).move_to([(B.p(x_end, 0)[0] + B.p(x_end - 1, 0)[0]) / 2,
                                                          (B.p(x_end, 0)[1] + B.p(x_end, 0.06)[1]) / 2, 0]),
                                     num_dashes=40)
            nt = T('not trimmed', 26, C_HI, weight='BOLD').move_to(B.p(x_end - 1.3, 0.066))
            qm = T('?', 60, C_HI, weight='BOLD').move_to(B.p(x_end - 0.5, 0.05))
            self.play(Create(end_box), FadeIn(nt), FadeIn(qm), run_time=0.9)

        # ── b08: 1000 -> 100 samples per second ──────────────────────────────
        with self.beat('c04_b08'):
            self.clear_all(0.5, keep=(tag, strip))
            self.play(*strip.light(5), run_time=0.3)
            env = X['piece_env']
            win = 300
            starts = np.arange(0, len(env) - win, 10)
            j0 = int(starts[np.argmax([env[s:s + win].max() - env[s:s + win].min() for s in starts])])
            j0 -= j0 % 10
            et, ev = t[j0:j0 + win], env[j0:j0 + win]
            lo, hi = float(ev.min()), float(ev.max())
            pad = 0.08 * (hi - lo)
            Q = Plot((float(et[0]), float(et[-1])), (lo - pad, hi + pad), 8.2, 2.0,
                     xticks=[round(float(et[0]) + 0.1 * k, 2) for k in range(4)], xlabel='time (s)',
                     ylabel='envelope (mV)', tick_size=18, xfmt=lambda v: f'{v:.2f}'
                     ).move_to([-1.9, 0.4, 0])
            dense = VGroup(*[Dot(Q.p(a, b), radius=0.022, color=C_SOL) for a, b in zip(et, ev)])
            keep_ids = list(range(0, win, 10))
            lab_d = T('1000 samples per second', 24, C_SOL).move_to([4.6, 1.5, 0]).align_to([3.0, 0, 0], LEFT)
            self.at('from a thousand')
            self.play(FadeIn(Q), run_time=0.5)
            self.play(FadeIn(dense, lag_ratio=0.0), FadeIn(lab_d), run_time=1.0)
            rest = VGroup(*[d for i, d in enumerate(dense) if i not in keep_ids])
            kept = VGroup(*[dense[i] for i in keep_ids])
            lab_k = T('100 samples per second', 24, C_HI, weight='BOLD').move_to(lab_d).align_to(lab_d, LEFT)
            self.at('The code smooths slightly')
            self.play(FadeOut(rest), run_time=0.9)
            self.play(*[d.animate.scale(3.2).set_color(C_HI) for d in kept], ReplacementTransform(lab_d, lab_k), run_time=0.8)
            self.at('A ninety-second trial')
            c1 = T(f'90.1 s → {F["seconds_100hz"]:.1f} s', 34, INK).move_to([4.6, 0.55, 0]).align_to([3.0, 0, 0], LEFT)
            c1b = T('after the 1 s trim', 20, DIM).next_to(c1, DOWN, buff=0.1).align_to(c1, LEFT)
            c2 = T(f'{F["n_100hz"]:,} samples', 38, C_HI, weight='BOLD').move_to([4.6, -0.5, 0]).align_to([3.0, 0, 0], LEFT)
            c2b = T('per muscle channel, per trial', 20, DIM).next_to(c2, DOWN, buff=0.1).align_to(c2, LEFT)
            self.play(FadeIn(c1), FadeIn(c1b), run_time=0.6)
            self.play(FadeIn(c2), FadeIn(c2b), run_time=0.6)
            self.at('The angle and the torque')
            rows = VGroup()
            guides = VGroup()
            for k, (nm, yy) in enumerate((('angle', -2.45), ('torque', -3.1))):
                lab = T(nm, 22, ANGLE_C[3] if nm == 'angle' else C_TQ).move_to([-6.0, yy, 0])
                lab.next_to([Q.p(float(et[-1]), 0)[0], yy, 0], RIGHT, buff=0.3)
                col = ANGLE_C[3] if nm == 'angle' else C_TQ
                dd = VGroup(*[Dot([Q.p(a, 0)[0], yy, 0], radius=0.022, color=col) for a in et])
                rows.add(VGroup(lab, dd))
            self.play(FadeIn(rows), run_time=0.7)
            for r in rows:
                dd = r[1]
                r_rest = VGroup(*[d for i, d in enumerate(dd) if i not in keep_ids])
                r_keep = VGroup(*[dd[i] for i in keep_ids])
                r.parts = (r_rest, r_keep)
            self.play(*[FadeOut(r.parts[0]) for r in rows], *[d.animate.scale(3.2) for r in rows for d in r.parts[1]],
                      run_time=0.9)
            xs = [Q.p(float(et[i]), 0)[0] for i in keep_ids]
            for xg in xs[::3]:
                guides.add(Line([xg, Q.xlabel.get_bottom()[1] - 0.12, 0], [xg, -3.05, 0], stroke_color=FAINT, stroke_width=1))
            guides.set_z_index(-1)
            self.play(FadeIn(guides), run_time=0.6)
            self.wait(self.left())
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)
