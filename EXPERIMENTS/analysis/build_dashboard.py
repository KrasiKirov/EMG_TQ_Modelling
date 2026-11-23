"""Rebuild the study v2 dashboard in one command: the confirmatory analysis, the page data, the waveforms, the page.

    python -m EXPERIMENTS.analysis.build_dashboard [--skip-confirm] [--no-npm]

1. confirm          EXPERIMENTS/dashboard_react/src/data/confirm.json      (needs every final run)
2. study_v2_export  .../src/data/study_v2.json
3. study_v2_waves   .../src/data/study-v2-waves.json
4. npm run build    EXPERIMENTS/dashboard_react/dist/index.html           (one self-contained file)

Stops at the first step that fails and names it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from EXPERIMENTS import paths as P
from EXPERIMENTS.analysis import confirm, study_v2_export, study_v2_waves

STEPS = {'confirm': confirm, 'study_v2_export': study_v2_export, 'study_v2_waves': study_v2_waves}
DASHBOARD = P.EXP / 'dashboard_react'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-confirm', action='store_true', help='keep the current confirm.json')
    ap.add_argument('--no-npm', action='store_true', help='update the data files only')
    a = ap.parse_args(argv)
    for name, module in STEPS.items():
        if name == 'confirm' and a.skip_confirm:
            continue
        print(f'== {name}', flush=True)
        try:
            module.main([])
        except Exception as e:
            raise SystemExit(f'build_dashboard: step {name} failed: {e}') from e
    if a.no_npm:
        return
    print('== npm run build', flush=True)
    npm = 'npm.cmd' if sys.platform == 'win32' else 'npm'
    r = subprocess.run([npm, 'run', 'build'], cwd=DASHBOARD)
    if r.returncode != 0:
        raise SystemExit(f'build_dashboard: step npm run build failed (exit {r.returncode})')


if __name__ == '__main__':
    main()
