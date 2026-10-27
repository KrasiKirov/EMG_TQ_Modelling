"""Display strings for every number in the narration and on screen, built from data/facts_*.json.

The narration's {placeholders} and the scenes both use values(), so a spoken number and the number
on screen are the same string. Pure Python (runs in either venv).
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'
MINUS = '−'


def facts():
    out = {}
    for p in sorted(DATA.glob('facts_*.json')):
        out.update(json.loads(p.read_text(encoding='utf-8')))
    return out


def num(v, d=2, plus=False):
    s = f'{abs(v):.{d}f}'
    if v < 0 and float(s) != 0:
        return MINUS + s
    return ('+' + s) if plus and v > 0 else s


def whole(v):
    return f'{int(round(v)):,}'


def values():
    F = facts()
    V = {}
    x1, x2, x3, x4, x5, x8, x9 = (F.get(k, {}) for k in ('x01', 'x02', 'x03', 'x04', 'x05', 'x08', 'x09'))
    if x1:
        V.update(eg_n_trials=str(x1['eg_n_trials']), eg_n_mvc=str(x1['eg_n_mvc']), eg_mvc_s=num(x1['eg_mvc_seconds'][0], 1),
                 eg_angle=num(x1['eg_angle'], 3), eg_angle_sd=num(x1['eg_angle_sd_mrad'], 1),
                 eg_pushes=str(x1['eg_push_count']), eg_tq_min=num(x1['eg_tq_min_Nm'], 1),
                 eg_raw_s=num(x1['eg_active_seconds'][0], 1),
                 eg_ang_min=num(x1['eg_S1_angles'][0], 2), eg_ang_max=num(x1['eg_S1_angles'][-1], 2, plus=True),
                 eg_peak_min=num(x1['eg_S1_peak_abs_tq_Nm'][0], 1), eg_peak_max=num(x1['eg_S1_peak_abs_tq_Nm'][-1], 1))
    if x2:
        V.update(env_mean=num(x2['mean_offset'], 3), imp90=num(x2['imp90'][1], 2), imp99=num(x2['imp99'][1], 2),
                 n100=whole(x2['n_100hz']), sec100=num(x2['seconds_100hz'], 1))
        V['eg_active_s'] = V['sec100']
    if x3:
        ps = x3['per_subject']
        eg, ch = ps['EG']['push_directions'], ps['CH']['push_directions']
        V.update(eg_dir_hi=num(max(eg), 0), eg_dir_lo=num(min(eg), 0), ch_dir_hi=num(max(ch), 0, plus=True),
                 ch_dir_lo=num(min(ch), 0), eg_despiked=whole(ps['EG']['despiked']))
        jm = next(r for r in x3['size_ratio']['JM'] if abs(r['angle'] + 0.36) < 0.005)
        V.update(jm_angle=num(jm['angle'], 2), jm_gm_before=num(jm['before'][0]), jm_gl_before=num(jm['before'][1]),
                 jm_sol_before=num(jm['before'][2]), jm_gm_after=num(jm['after'][0]), jm_gl_after=num(jm['after'][1]),
                 jm_sol_after=num(jm['after'][2]))
        w = x3['jm_example']
        V.update(wx_t=num(w['t'], 2), wx_t_u=num(w['t_u'], 1), wx_n_lo=whole(w['n_lo']), wx_n_hi=whole(w['n_hi']),
                 wx_w_lo=num(w['w_lo'], 2), wx_w_hi=num(w['w_hi'], 3), wx_p10=num(w['p10'], 4), wx_p95=num(w['p95'], 4),
                 wx_prior_lo=num(w['prior_lo'], 4), wx_prior_hi=num(w['prior_hi'], 4), wx_floor=num(w['floor'], 4),
                 wx_top=num(w['top'], 4), wx_range=num(w['range'], 4), wx_rmin=num(w['rmin'], 4),
                 wx_prior_range=num(w['prior_range'], 4), wx_x=num(w['x'], 4), wx_a=num(w['a'], 3))
        mis = x3['size_mismatch']
        V.update(eg_mis_before=num(mis['EG']['before'], 2), eg_mis_after=num(mis['EG']['after'], 2),
                 jm_mis_before=num(mis['JM']['before'], 2), jm_mis_after=num(mis['JM']['after'], 2))
        c = x3['eg_cancel']
        V.update(cancel_pct=num(100 * c['fraction_differing'], 1), cancel_trials=str(len(c['trials'])))
    if x4:
        maps = x4['maps']['EG']['static']
        V.update(eg_map_low=num(maps['S1'][0][1], 1, plus=True), eg_map_high=num(maps['S1'][-1][1], 1),
                 eg_map_low_s2=num(maps['S2'][0][1], 1, plus=True), eg_map_high_s2=num(maps['S2'][-1][1], 1),
                 p_s1=num(x4['eg_S1_P_median'], 2), p_s2=num(x4['eg_S2_P_median'], 2),
                 eg_kept=str(sum(p['keep'] for p in x4['eg_pieces'])), eg_pieces=str(len(x4['eg_pieces'])),
                 hold_s=num(next(p['seconds'] for p in x4['eg_pieces'] if p['keep']), 1))
    if x5:
        sd, knots = x5['sd'], x5['knots']
        V.update(sd_min=num(min(sd)), sd_max=num(max(sd)), sd_min_angle=num(knots[x5['sd_argmin']], 2),
                 sd_max_angle=num(knots[x5['sd_argmax']], 2, plus=True), sd2_ratio=whole(x5['sd2_ratio_max_min']),
                 n_windows=whole(x5['trial']['n_windows']), n_train=whole(x5['trial']['n_train']),
                 n_val=whole(x5['trial']['n_val']), x_train=whole(x5['shapes']['train'][0]),
                 x_val=whole(x5['shapes']['val'][0]), x_test=whole(x5['shapes']['test'][0]))
    if x8:
        lv, ag = x8['eg_live'], x8['aggregate']['S1toS2']
        V.update(live_n=whole(lv['n']), live_ss_res=whole(lv['ss_res']), live_ss_tot=whole(lv['ss_tot']),
                 live_r2=num(lv['r2'], 3), live_mae=num(lv['mae'], 2), live_peak=num(lv['peak'], 1),
                 live_mean=num(lv['mean'], 2), live_bias=num(lv['bias'], 2, plus=True),
                 cells=str(ag['all']['cells']), head_r2=num(ag['all']['r2'], 3), head_mae=num(ag['all']['mae'], 2),
                 toes_r2=num(ag['toes']['r2'], 3), other_r2=num(ag['other']['r2'], 3), other_cells=str(ag['other']['cells']),
                 toes_mae=num(ag['toes']['mae'], 2), other_mae=num(ag['other']['mae'], 2))
    if x9:
        V.update(lw_r2=num(x9['eg_m0038']['r2'], 3), lw_mae=num(x9['eg_m0038']['mae'], 2))
    x6, x7 = F.get('x06_probe'), F.get('x07')
    if x6:
        V.update(n_params=whole(x6['n_params']), best_epoch=str(x6['best_epoch']), n_epochs=str(x6['epochs']),
                 best_val=num(x6['best_val_loss'], 4), n_steps_epoch=whole(x6['steps_per_epoch']))
    if x7:
        V.update({k: x7['display'][k] for k in x7['display']})
    return V


if __name__ == '__main__':
    for k, v in sorted(values().items()):
        print(f'{k:16s} {v}')
