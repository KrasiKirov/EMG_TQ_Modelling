from manim import *

from style import (ANGLE_C, C_ANGLE, C_BAD, C_GOOD, C_HI, C_PRED, C_S1, C_S2, C_TQ, DIM, FAINT, INK, MUSCLE_C, PANEL, V,
                   BeatScene, CodePanel, Plot, T, chapter_tag, code_at_bottom, facts, load)

FF = facts()
F7, F6 = FF['x07'], FF['x06_probe']
X7, X6 = load('x07'), load('x06_probe')
NAMES = ['GM', 'GL', 'SOL', 'TA', 'angle']
GATE_C = {'i': '#4FA8E8', 'f': '#F5D547', 'g': '#9BE564', 'o': '#F15BB5'}


def sgn(v, d=2):
    return f'{v:.{d}f}'.replace('-', '−')


def heat(M, w=0.28, h=0.22, vmax=None):
    """rows = units, columns = steps; blue negative, orange positive."""
    vmax = vmax or float(np.max(np.abs(M))) or 1.0
    g = VGroup()
    for u in range(M.shape[1]):
        for t in range(M.shape[0]):
            v = float(M[t, u]) / vmax
            col = interpolate_color(ManimColor(PANEL), ManimColor('#FF7A45' if v > 0 else '#4C9AFF'), min(1.0, abs(v)))
            g.add(Rectangle(width=w, height=h, stroke_width=0.4, stroke_color=PANEL, fill_color=col, fill_opacity=1)
                  .move_to([t * w, -u * h, 0]))
    return g


class Ch09(BeatScene):
    def construct(self):
        tag = chapter_tag(9, 'Inside the LSTM')
        self.play(FadeIn(tag), run_time=0.4)

        # 9.1 the model
        with self.beat('c09_b01') as d:
            specs = [('input', '20 steps × 5', DIM), ('LSTM 8', 'every step', C_S1), ('Dropout', '30 %', DIM),
                     ('LSTM 8', 'last step only', C_S1), ('Dropout', '30 %', DIM), ('Dense 1', 'one number', C_HI)]
            boxes = VGroup()
            for name, sub, col in specs:
                r = RoundedRectangle(corner_radius=0.1, width=1.85, height=1.1, stroke_color=col, fill_color=PANEL, fill_opacity=1)
                boxes.add(VGroup(r, VGroup(T(name, 20, INK), T(sub, 14, DIM)).arrange(DOWN, buff=0.08).move_to(r)))
            boxes.arrange(RIGHT, buff=0.38).move_to([0, 1.9, 0])
            arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.04, color=DIM,
                                    max_tip_length_to_length_ratio=0.3) for i in range(5)])
            pc = VGroup(T('4 × 8 × (5 + 8 + 1) = 448', 15, C_S1).next_to(boxes[1], DOWN, buff=0.2),
                        T('4 × 8 × (8 + 8 + 1) = 544', 15, C_S1).next_to(boxes[3], DOWN, buff=0.2),
                        T('8 + 1 = 9', 15, C_HI).next_to(boxes[5], DOWN, buff=0.2))
            tot = T(f"{V['n_params']} trainable numbers", 24, INK).move_to([0, 0.15, 0])
            code = code_at_bottom(CodePanel('builder'))
            self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.1) for b in boxes], lag_ratio=0.15),
                      LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.15), run_time=2.0)
            self.show_code(code)
            self.mark(code, 57, 60)
            self.play(FadeIn(pc), run_time=0.8)
            self.play(FadeIn(tot), run_time=0.6)
        self.play(FadeOut(VGroup(boxes, arrows, pc, tot, code)), run_time=0.5)

        # 9.2 the cell
        with self.beat('c09_b02') as d:
            eqs = VGroup(T('z = x · W  +  h₋₁ · U  +  b          (4 × 8 numbers)', 26, INK, font='Consolas'),
                         T('i = σ(z_i)   f = σ(z_f)   g = tanh(z_g)   o = σ(z_o)', 26, INK, font='Consolas'),
                         T('c = f × c₋₁  +  i × g      (element by element)', 26, C_HI, font='Consolas'),
                         T('h = o × tanh(c)', 26, C_HI, font='Consolas')).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
            eqs.move_to([0, 1.6, 0])
            gl = VGroup(*[T(t, 15, GATE_C[k]) for k, t in (('i', 'i: input gate'), ('f', 'f: forget gate'),
                                                           ('g', 'g: candidate'), ('o', 'o: output gate'))]).arrange(RIGHT, buff=0.6)
            gl.next_to(eqs, DOWN, buff=0.35)
            code = code_at_bottom(CodePanel('lstm_call'))
            self.play(FadeIn(eqs[0]), run_time=0.7)
            self.show_code(code)
            self.mark(code, 277, 283)
            self.play(FadeIn(eqs[1]), FadeIn(gl), run_time=0.7)
            self.wait(max(0.1, d * 0.45 - 2.7))
            code2 = code_at_bottom(CodePanel('lstm_cell'))
            self.play(FadeOut(code), FadeIn(code2), FadeIn(eqs[2]), run_time=0.7)
            self.mark(code2, 226)
            self.wait(max(0.1, d * 0.25 - 1.0))
            self.play(FadeIn(eqs[3]), run_time=0.6)
        self.play(FadeOut(VGroup(eqs, gl, code2)), run_time=0.5)

        # 9.3 one window, step one, real numbers
        w = F7['window']
        u = w['unit']
        x1 = np.array(w['x1'])
        with self.beat('c09_b03') as d:
            hdr = T(f"EG · session 2 · −0.038 rad · t = {w['t']:.2f} s · trained weights · layer 1, unit {u}, step 1",
                    18, DIM).move_to([0, 3.2, 0])
            xin = VGroup(*[T(f'{n:>5}  {sgn(v, 3)}', 20, MUSCLE_C[k] if k < 4 else C_ANGLE, font='Consolas')
                           for k, (n, v) in enumerate(zip(NAMES, x1))]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            xin.move_to([-5.2, 1.3, 0])
            xl = T('inputs x (step 1)', 15, DIM).next_to(xin, UP, buff=0.15)
            wi = np.array(w['kernel_unit']['i'])
            terms = ' + '.join(f'{sgn(a, 2)}·{sgn(b, 2)}' for a, b in zip(x1, wi))
            line1 = T(f"z_i = {terms} + {sgn(w['bias_unit']['i'], 2)}", 17, INK, font='Consolas')
            line2 = T(f"    = {w['z1']['i']:.2f}   (h₋₁ = 0 at the first step)".replace('-', '−'), 17, INK, font='Consolas')
            line3 = T(f"i = σ({sgn(w['z1']['i'])}) = {w['gates1']['i']:.2f}", 20, GATE_C['i'], font='Consolas')
            blk = VGroup(line1, line2, line3).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            blk.scale_to_fit_width(min(blk.width, 9.0)).move_to([1.6, 1.85, 0])
            rows = VGroup()
            for k, fn in (('f', 'σ'), ('g', 'tanh'), ('o', 'σ')):
                rows.add(T(f"{k} = {fn}({sgn(w['z1'][k])}) = {sgn(w['gates1'][k])}", 20, GATE_C[k], font='Consolas'))
            rows.arrange(RIGHT, buff=0.6).move_to([1.6, 0.55, 0])
            cc = T(f"c = f·0 + i·g = {w['gates1']['i']:.2f} × {sgn(w['gates1']['g'])} = {sgn(w['gates1']['c'], 3)}", 20, C_HI,
                   font='Consolas')
            hh = T(f"h = o · tanh(c) = {w['gates1']['o']:.2f} × {sgn(np.tanh(w['gates1']['c']), 3)} = {sgn(w['gates1']['h'], 3)}",
                   20, C_HI, font='Consolas')
            ch = VGroup(cc, hh).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([1.6, -0.45, 0])
            self.play(FadeIn(hdr), FadeIn(xl), LaggedStart(*[FadeIn(r) for r in xin], lag_ratio=0.1), run_time=1.2)
            self.play(FadeIn(line1), run_time=0.8)
            self.wait(max(0.1, d * 0.22 - 2.0))
            self.play(FadeIn(line2), FadeIn(line3), run_time=0.8)
            self.wait(max(0.1, d * 0.18 - 0.8))
            self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.3), run_time=1.2)
            self.wait(max(0.1, d * 0.15 - 1.2))
            self.play(FadeIn(cc), run_time=0.6)
            self.play(FadeIn(hh), run_time=0.6)
        self.play(FadeOut(VGroup(hdr, xl, xin, blk, rows, ch)), run_time=0.5)

        # 9.4 twenty steps, then layer 2, dense, × SD
        with self.beat('c09_b04') as d:
            Hh = heat(X7['L1_h'], vmax=1.0).move_to([-2.6, 2.0, 0])
            Hc = heat(X7['L1_c'], vmax=float(np.max(np.abs(X7['L1_c'])))).move_to([-2.6, -0.4, 0])
            lh = T('layer 1 outputs h (8 units × 20 steps)', 15, DIM).next_to(Hh, UP, buff=0.1)
            lc = T('layer 1 memories c', 15, DIM).next_to(Hc, UP, buff=0.1)
            st = VGroup(T('step 1', 12, DIM).next_to(Hh, DOWN, buff=0.06).align_to(Hh, LEFT),
                        T('step 20', 12, DIM).next_to(Hh, DOWN, buff=0.06).align_to(Hh, RIGHT))
            h2 = np.array(w['h2_last'])
            dw = np.array(w['dense_w'])
            col2 = VGroup(*[T(f'{sgn(a, 3)} × {sgn(b, 3)}', 15, INK, font='Consolas') for a, b in zip(h2, dw)])
            col2.arrange(DOWN, aligned_edge=LEFT, buff=0.06).move_to([2.6, 1.75, 0])
            l2 = T('layer 2, last step: h₂ × dense weight', 15, DIM).next_to(col2, UP, buff=0.12)
            res = VGroup(T(f"sum + bias {sgn(w['dense_b'], 3)}  →  ŷ = {V['cell_yhat']}", 17, INK, font='Consolas'),
                         T(f"× SD {V['cell_sd']} Nm = {V['cell_pred']} Nm", 19, C_PRED, font='Consolas'),
                         T(f"measured active torque: {V['cell_true']} Nm", 16, C_TQ))
            res.arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(col2, DOWN, buff=0.25).align_to(col2, LEFT)
            code = code_at_bottom(CodePanel('pred_scale'))
            self.play(FadeIn(lh), LaggedStart(*[FadeIn(c) for c in Hh], lag_ratio=0.004), FadeIn(st), run_time=2.0)
            self.play(FadeIn(lc), LaggedStart(*[FadeIn(c) for c in Hc], lag_ratio=0.004), run_time=1.6)
            self.wait(max(0.1, d * 0.3 - 3.6))
            self.play(FadeIn(l2), LaggedStart(*[FadeIn(r) for r in col2], lag_ratio=0.1), run_time=1.2)
            self.play(FadeIn(res[0]), run_time=0.6)
            self.show_code(code)
            self.mark(code, 183, 184)
            self.play(FadeIn(res[1]), run_time=0.6)
            self.play(FadeIn(res[2]), run_time=0.6)
        self.play(FadeOut(VGroup(Hh, Hc, lh, lc, st, col2, l2, res, code)), run_time=0.5)

        # 9.5 the seed and the starting weights
        with self.beat('c09_b05') as d:
            code = code_at_bottom(CodePanel('seed_keras'))
            k0 = X7['w0_kernel1'].ravel()
            lim = F7['init']['glorot_limit_l1']
            Hp = Plot((-0.45, 0.45), (0, 22), 5.6, 2.4, xticks=(-0.4, -0.2, 0, 0.2, 0.4), yticks=(0, 10, 20), tick_size=13,
                      xlabel='layer-1 input weights at the start', ylabel='count', grid=True).move_to([-3.3, 1.3, 0])
            cnt, edges = np.histogram(k0, bins=18, range=(-0.45, 0.45))
            hb = VGroup(*[Rectangle(width=Hp.p(edges[1], 0)[0] - Hp.p(edges[0], 0)[0] - 0.02,
                                    height=max(0.005, Hp.p(0, c)[1] - Hp.p(0, 0)[1]), stroke_width=0, fill_color=C_S1,
                                    fill_opacity=0.9).move_to(Hp.p((a + b) / 2, 0), aligned_edge=DOWN)
                          for a, b, c in zip(edges[:-1], edges[1:], cnt)])
            lims = VGroup(*[DashedLine(Hp.p(v, 0), Hp.p(v, 22), color=C_HI, dash_length=0.06) for v in (-lim, lim)])
            ll = T(f"±√(6 / (5 + 32)) = ±{lim:.3f}", 15, C_HI).next_to(Hp, UP, buff=0.08)
            fb = np.array(F7['init']['forget_bias'])
            right = VGroup(T('Python random → weight initialisers', 17, INK), T('                  + both dropout masks', 17, INK),
                           T('TensorFlow seed → epoch shuffle', 17, INK),
                           T(f"forget-gate biases: {', '.join(f'{v:.0f}' for v in fb)}", 16, GATE_C['f']),
                           T('other biases: 0 · recurrent weights: orthogonal', 16, DIM))
            right.arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([3.4, 1.4, 0])
            self.show_code(code)
            self.mark(code, 63, 66)
            self.play(FadeIn(right[:3]), run_time=0.8)
            self.play(FadeIn(Hp), LaggedStart(*[GrowFromEdge(b, DOWN) for b in hb], lag_ratio=0.03), run_time=1.2)
            self.play(Create(lims), FadeIn(ll), run_time=0.6)
            self.play(FadeIn(right[3:]), run_time=0.6)
        self.play(FadeOut(VGroup(code, Hp, hb, lims, ll, right)), run_time=0.5)

        # 9.6 step 1: a real batch of 8
        b1 = F7['batch1']
        with self.beat('c09_b06') as d:
            head = ['trial', 'angle (rad)', 't (s)', 'target (Nm)', 'SD (Nm)', 'target ÷ SD', 'prediction', 'squared error']
            data = [[str(tr), sgn(a, 3), f'{t:.2f}', sgn(y, 2), f'{s:.2f}', sgn(ys, 3), sgn(p, 3), f'{q:.3f}']
                    for tr, a, t, y, s, ys, p, q in zip(b1['trial'], b1['angle'], b1['t'], b1['y_Nm'], b1['sd'],
                                                       b1['y_scaled'], b1['y_pred'], b1['sq'])]
            cols_x = [-6.1, -4.6, -3.05, -1.5, 0.0, 1.45, 3.05, 4.75]
            tab = VGroup()
            hd = VGroup(*[T(h, 14, DIM).move_to([x, 2.95, 0]) for h, x in zip(head, cols_x)])
            for r, row in enumerate(data):
                tab.add(VGroup(*[T(c, 16, ANGLE_C[b1['knot'][r]] if j == 1 else INK, font='Consolas').move_to([x, 2.5 - r * 0.36, 0])
                                 for j, (c, x) in enumerate(zip(row, cols_x))]))
            mean = T(f"mean of the 8 = loss = {V['batch_loss']}   (recomputed: {b1['loss_recomputed']:.3f})", 19, C_HI)
            mean.move_to([0, -0.55, 0])
            drop = T(f"dropout in this step: {100 * b1['drop1_zero_frac']:.0f} % of layer-1 outputs and "
                     f"{100 * b1['drop2_zero_frac']:.0f} % of layer-2 outputs set to 0", 15, DIM).next_to(mean, DOWN, buff=0.1)
            code = code_at_bottom(CodePanel('train_step'), )
            self.play(FadeIn(hd), LaggedStart(*[FadeIn(r) for r in tab], lag_ratio=0.12), run_time=1.6)
            self.show_code(code)
            self.mark(code, 59)
            self.wait(max(0.1, d * 0.55 - 2.6))
            self.mark(code, 62, 68, color=C_S2)
            self.play(FadeIn(mean), FadeIn(drop), run_time=0.7)
        self.play(FadeOut(VGroup(hd, tab, mean, drop, code)), run_time=0.5)

        # 9.7 one Nadam update
        nd = F7['nadam1']
        with self.beat('c09_b07') as d:
            u1, u2 = nd['u1'], nd['u2']
            lines = [f"g      = {nd['g']:.4f}          dense weight from layer-2 unit {nd['index'] - 992 + 1}",
                     f"m      = 0.1 × g      = {nd['m']:.4f}",
                     f"v      = 0.001 × g²   = {nd['v']:.6f}",
                     f"m̂      = u₂·m/(1 − u₁u₂) + (1 − u₁)·g/(1 − u₁) = {nd['m_hat']:.4f}    u₁ = {u1:.3f}, u₂ = {u2:.3f}",
                     f"v̂      = v / (1 − 0.999)  = {nd['v_hat']:.4f}",
                     f"step   = 0.003 × m̂ / (√v̂ + 1e−7) = {nd['step']:.5f}",
                     f"weight : {sgn(nd['w0'], 4)}  →  {sgn(nd['w1'], 4)}"]
            eq = VGroup(*[T(l, 18, INK, font='Consolas') for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            eq[-1].set_color(C_HI)
            eq.scale_to_fit_width(min(eq.width, 8.6)).move_to([-2.0, 1.4, 0])
            sa = X7['step1_abs']
            Hs = Plot((0.00310, 0.00320), (0, 1001), 3.6, 2.4, xticks=(0.0031, 0.0032), yticks=(0, 500, 1000), tick_size=12,
                      xlabel='|change| of every weight at step 1', xfmt=lambda v: f'{v:.4f}', grid=True).move_to([4.9, 1.45, 0])
            cnt, ed = np.histogram(sa, bins=10, range=(0.00310, 0.00320))
            hb = VGroup(*[Rectangle(width=Hs.p(ed[1], 0)[0] - Hs.p(ed[0], 0)[0] - 0.02,
                                    height=max(0.005, Hs.p(0, c)[1] - Hs.p(0, 0)[1]), stroke_width=0, fill_color=C_HI,
                                    fill_opacity=0.9).move_to(Hs.p((a + b) / 2, 0), aligned_edge=DOWN)
                          for a, b, c in zip(ed[:-1], ed[1:], cnt)])
            code = code_at_bottom(CodePanel('nadam', 129, 139))
            self.show_code(code)
            for i, l in enumerate(eq):
                self.play(FadeIn(l, shift=RIGHT * 0.1), run_time=max(0.35, min(0.9, (d * 0.6) / 9)))
                if i == 2:
                    self.mark(code, 129, 134)
                if i == 4:
                    self.mark(code, 135, 139, color=C_S2)
            code2 = code_at_bottom(CodePanel('nadam', 141, 146))
            self.play(FadeOut(code), FadeIn(code2), FadeIn(Hs), LaggedStart(*[GrowFromEdge(b, DOWN) for b in hb], lag_ratio=0.05),
                      run_time=1.0)
            chk = T('formula reproduces every recorded weight exactly' if nd['formula_max_err'] == 0 else
                    f"formula vs recorded weights: max difference {nd['formula_max_err']:.1e}", 14, C_GOOD).next_to(Hs, UP, buff=0.1)
            self.play(FadeIn(chk), run_time=0.4)
        self.play(FadeOut(VGroup(eq, Hs, hb, code2, chk)), run_time=0.5)

        # 9.8 the first five steps
        fs = F7['first_steps']
        with self.beat('c09_b08') as d:
            names = list(fs['weights'].keys())
            head = VGroup(T('step', 15, DIM), T('batch loss', 15, DIM), *[T(n, 13, DIM) for n in names])
            cx = [-5.4, -3.6, -1.1, 2.0, 5.0]
            for h, x in zip(head, cx):
                h.move_to([x, 2.8, 0])
            rows = VGroup()
            for s in range(6):
                vals = ['start' if s == 0 else str(s), '' if s == 0 else f"{fs['loss'][s - 1]:.3f}"] + \
                       [sgn(fs['weights'][n][s], 5) for n in names]
                rows.add(VGroup(*[T(v, 17, INK, font='Consolas').move_to([x, 2.35 - s * 0.38, 0]) for v, x in zip(vals, cx)]))
            ep = T(f"one epoch = {V['n_steps_epoch']} such steps", 20, C_HI).move_to([0, -0.45, 0])
            self.play(FadeIn(head), LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.15), run_time=min(3.0, d * 0.5))
            self.play(FadeIn(ep), run_time=0.5)
        self.play(FadeOut(VGroup(head, rows, ep)), run_time=0.5)

        # 9.9 epochs, LR drops, early stopping, restore
        with self.beat('c09_b09') as d:
            ep_, lo, va = X6['epoch'], X6['loss'], X6['val_loss']
            n = len(ep_)
            P = Plot((0.5, n + 0.5), (0, 0.11), 10.0, 2.2, xticks=(1, 5, 10, 13, 15, 20, 23, 25, 30, 33, 35, 38), yticks=(0, 0.05, 0.10),
                     tick_size=12, ylabel='loss (scaled targets)', grid=True, yfmt=lambda v: f'{v:.2f}').move_to([0.2, 1.95, 0])
            cl = P.curve(ep_, lo, C_S1, 2.5)
            cv = P.curve(ep_, va, C_S2, 2.5)
            dl = VGroup(*[Dot(P.p(e, v), radius=0.035, color=C_S1) for e, v in zip(ep_, lo)])
            dv = VGroup(*[Dot(P.p(e, v), radius=0.035, color=C_S2) for e, v in zip(ep_, va)])
            ll = VGroup(T('training loss (dropout on, epoch average)', 14, C_S1), T('validation loss (dropout off)', 14, C_S2))
            ll.arrange(RIGHT, buff=0.6).next_to(P, UP, buff=0.08)
            LR = Plot((0.5, n + 0.5), (0.0012, 0.0032), 10.0, 0.5, yticks=(0.0015, 0.003), tick_size=11, ylabel='LR',
                      yfmt=lambda v: f'{v:.4f}').next_to(P, DOWN, buff=0.42)
            lr_x = np.repeat(np.arange(1, n + 1), 2) + np.tile([-0.5, 0.5], n)
            lr_y = np.repeat(X6['lr_begin'], 2)
            clr = LR.curve(lr_x, lr_y, C_HI, 2.5)
            best = F7['epochs']['best']
            bm = VGroup(Circle(radius=0.12, color=C_GOOD).move_to(P.p(best, float(va[best - 1]))),
                        T(f"best: epoch {best}, {V['best_val']}", 14, C_GOOD).next_to(P.p(best, float(va[best - 1])), DOWN, buff=0.18))
            cuts = VGroup(*[DashedLine(P.p(c + 0.5, 0), P.p(c + 0.5, 0.11), color=C_HI, dash_length=0.05) for c in F7['epochs']['cuts']])
            cl_ = VGroup(*[T(f'LR × 0.7', 12, C_HI).next_to(P.p(c + 0.5, 0.1), LEFT, buff=0.06)
                           for c in F7['epochs']['cuts']])
            stop = VGroup(Line(P.p(n, 0), P.p(n, 0.11), color=C_BAD, stroke_width=3),
                          T('stop', 12, C_BAD).next_to(P.p(n, 0.1), RIGHT, buff=0.06))
            rest = CurvedArrow(P.p(n, 0.045), P.p(best, 0.03), angle=-PI / 5, color=C_GOOD)
            rl = T(f'weights of epoch {best} restored', 14, C_GOOD).next_to(rest, UP, buff=0.02).shift(RIGHT * 1.5)
            code = code_at_bottom(CodePanel('callbacks'))
            self.play(FadeIn(P), FadeIn(ll), run_time=0.5)
            self.play(Create(cl), Create(cv), FadeIn(dl), FadeIn(dv), run_time=2.0)
            self.show_code(code)
            self.play(Create(bm[0]), FadeIn(bm[1]), run_time=0.6)
            self.wait(max(0.1, d * 0.3 - 3.6))
            self.play(FadeIn(LR), Create(clr), Create(cuts), FadeIn(cl_), run_time=1.0)
            self.mark(code, 174, 175)
            self.wait(max(0.1, d * 0.25 - 1.4))
            code2 = code_at_bottom(CodePanel('es_end', 123, 126))
            self.play(FadeOut(code), FadeIn(code2), Create(stop[0]), FadeIn(stop[1]), run_time=0.8)
            self.mark(code2, 123, 126)
            code3 = code_at_bottom(CodePanel('es_end', 133, 140))
            self.play(FadeOut(code2), FadeIn(code3), Create(rest), FadeIn(rl), run_time=0.9)
            self.mark(code3, 133, 140, color=C_GOOD)
            code2 = code3
        self.play(FadeOut(VGroup(P, cl, cv, dl, dv, ll, LR, clr, bm, cuts, cl_, stop, rest, rl, code2)), run_time=0.5)

        # 9.10 this rerun is the saved run
        with self.beat('c09_b10') as d:
            ck = F6
            rows = [('epochs', str(ck['epochs']), '38'), ('best epoch', str(ck['best_epoch']), '13'),
                    ('best validation loss', f"{ck['best_val_loss']:.12f}", '0.014504178427'),
                    ('max |prediction − saved| on 71,128 windows', f"{ck['max_abs_pred_diff_Nm']:.1f} Nm", '')]
            tab = VGroup()
            for i, (a, b, c) in enumerate(rows):
                tab.add(VGroup(T(a, 20, DIM).move_to([-3.2, 1.9 - i * 0.6, 0]),
                               T(b, 22, INK, font='Consolas').move_to([2.4, 1.9 - i * 0.6, 0]),
                               T('✓', 26, C_GOOD).move_to([4.6, 1.9 - i * 0.6, 0])))
            ttl = T('instrumented rerun (x06_train_probe.py) vs the saved run', 20, INK).move_to([0, 2.75, 0])
            self.play(FadeIn(ttl), LaggedStart(*[FadeIn(r) for r in tab], lag_ratio=0.25), run_time=min(3.0, d * 0.6))
        self.play(FadeOut(VGroup(ttl, tab, tag)), run_time=0.6)
