"""Extra data for chapter 4 beat 7: EG trial 21 soleus envelope BEFORE the 1 s trim (1 kHz), first and last 4 s.
Same filter calls as extract/x02_envelope.py (emg_envelope.py:120-144). Writes data2/c04_trim.npz."""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'extract'))
from common import use_d01_code                              # noqa: E402

use_d01_code()
from ML.config import DATA_DIR, SUBJECTS as REGISTRY          # noqa: E402
from ML.preprocessing import emg_envelope as ee              # noqa: E402
from ML.preprocessing.flb_reader import read_flb             # noqa: E402

fn, sid = REGISTRY['EG']
df = read_flb(os.path.join(DATA_DIR, fn), subject_id=sid)[21]
x = df['sol'].values.astype(np.float64)
env = ee.extract_envelope(x)
fs = 1000
n = len(env)
k = 4 * fs
out = Path(__file__).resolve().parent.parent / 'data2' / 'c04_trim.npz'
np.savez_compressed(out, n=n, t_start=np.arange(k) / fs, env_start=env[:k], t_end=np.arange(n - k, n) / fs,
                    env_end=env[-k:], env_all_1hz=env[::fs], max_all=env.max())
print('n', n, 'env start', env[:5], 'start max', env[:k].max(), 'end max', env[-k:].max(), 'all max', env.max())
