"""Deprecated: the normalization entries are now experiment specs (EXPERIMENTS/configs/*.json with a dashboard block).

This shim re-exports them. Use instead:  python -m EXPERIMENTS.pipeline.dashboard export
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from EXPERIMENTS.pipeline import dashboard  # noqa: E402

if __name__ == '__main__':
    print('import_normalization.py is deprecated; running: python -m EXPERIMENTS.pipeline.dashboard export', file=sys.stderr)
    dashboard.main(['export', *[s.id for s in dashboard.S.load_all() if s.dashboard and s.dashboard.id.startswith('norm/')]])
