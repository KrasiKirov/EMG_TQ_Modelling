"""Chapter 4 data: one real 3 s piece of EG's soleus through every preprocessing step.

The steps repeat ML/preprocessing/emg_envelope.py:120-144 line by line (same butter / sosfiltfilt
calls) on the WHOLE trial, as the code does, and the result is checked against extract_envelope()
and against the cached 100 Hz bundle (trim 1 s, decimate, data.py:90-104). Writes data/x02.npz.
"""
import os

import numpy as np
from scipy.signal import butter, decimate, sosfiltfilt, sosfreqz

from common import save_facts, save_npz, use_d01_code

use_d01_code()
from ML.config import DATA_DIR, SUBJECTS as REGISTRY          # noqa: E402
from ML.preprocessing import emg_envelope as ee              # noqa: E402
from ML.preprocessing.flb_reader import read_flb             # noqa: E402
from ML.research import config as RC                         # noqa: E402
from ML.research.data import load_subject                    # noqa: E402

IDX = 21                                                      # EG, session 1, -0.038 rad
fn, sid = REGISTRY['EG']
df = read_flb(os.path.join(DATA_DIR, fn), subject_id=sid)[IDX]
x = df['sol'].values.astype(np.float64)
fs, nyq = ee.DEFAULT_FS, 0.5 * ee.DEFAULT_FS

mean = float(np.mean(x))                                      # emg_envelope.py:124
demeaned = x - mean
sos_bp = butter(ee.DEFAULT_ORDER, [ee.DEFAULT_BP_LOW / nyq, ee.DEFAULT_BP_HIGH / nyq], btype='band', output='sos')
band = sosfiltfilt(sos_bp, demeaned)                          # emg_envelope.py:129-130
rect = np.abs(band)                                           # emg_envelope.py:133
sos_lp = butter(ee.DEFAULT_ORDER, ee.DEFAULT_LP_CUTOFF / nyq, btype='low', output='sos')
env = np.clip(sosfiltfilt(sos_lp, rect), 0, None)             # emg_envelope.py:136-140
check_env = float(np.max(np.abs(env - ee.extract_envelope(x))))

# downsampling exactly as data.py:94-98 and the cached bundle
s = RC.TRIM_START
env100 = decimate(env[s:], RC.DOWNSAMPLE, ftype='fir', zero_phase=True)
b = load_subject('EG')
rec = next(r for r in b['active'] if r['idx'] == IDX)
n = len(rec['t'])
check_bundle = float(np.max(np.abs(env100[:n] - rec['env_raw'][:, 2])))

# the 3 s piece: around the push onset used in chapter 1 (2nd half of x01's window)
tq = rec['tq']
mid = 0.5 * (np.percentile(tq, 5) + np.percentile(tq, 95))
below = np.r_[False, tq < mid, False]
edges = np.flatnonzero(np.diff(below.astype(int)))
runs = [(a, e) for a, e in zip(edges[::2], edges[1::2]) if e - a >= 50]
onset_raw = s + runs[2][0] * RC.DOWNSAMPLE
p0, p1 = onset_raw - 1500, onset_raw + 1500
piece = dict(t=np.arange(p0, p1) / fs, raw=x[p0:p1], demeaned=demeaned[p0:p1], band=band[p0:p1],
             rect=rect[p0:p1], env=env[p0:p1])
j0, j1 = (p0 - s) // RC.DOWNSAMPLE, (p1 - s) // RC.DOWNSAMPLE + 1
piece.update(t100=rec['t'][j0:j1], env100=rec['env_raw'][j0:j1, 2])

# filter gain (applied forward and backward -> squared magnitude)
f = np.logspace(0, np.log10(499), 600)
_, h_bp = sosfreqz(sos_bp, worN=f, fs=fs)
_, h_lp = sosfreqz(sos_lp, worN=f, fs=fs)
# two-sided impulse response of the zero-phase 2 Hz low-pass
imp = np.zeros(20001)
imp[10000] = 1.0
h = sosfiltfilt(sos_lp, imp)
tt = (np.arange(len(h)) - 10000) / fs
c = np.cumsum(np.abs(h)) / np.sum(np.abs(h))
q = lambda p: float(tt[np.searchsorted(c, p)])

save_npz('x02', f=f, gain_bp=np.abs(h_bp) ** 2, gain_lp=np.abs(h_lp) ** 2, imp_t=tt[::10], imp_h=h[::10] / np.max(h),
         **{f'piece_{k}': v for k, v in piece.items()})
save_facts('x02', {'x02': dict(
    trial_idx=IDX, mean_offset=mean, bp=[ee.DEFAULT_BP_LOW, ee.DEFAULT_BP_HIGH], lp=ee.DEFAULT_LP_CUTOFF,
    order=ee.DEFAULT_ORDER, piece_s=[p0 / fs, p1 / fs], check_env_vs_extract_envelope=check_env,
    check_decimated_vs_bundle=check_bundle, imp90=[q(0.05), q(0.95)], imp99=[q(0.005), q(0.995)],
    n_100hz=n, seconds_100hz=n / RC.FS, decimate_taps=20 * RC.DOWNSAMPLE + 1)})
print('mean %.5f  env check %.2e  bundle check %.2e  imp90 %.3f..%.3f s' % (mean, check_env, check_bundle, q(.05), q(.95)))
