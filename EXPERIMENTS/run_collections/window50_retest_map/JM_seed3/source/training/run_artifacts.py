"""Create exclusive run directories and capture the code/environment used."""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from ML.evaluation.report import write_json


def create_run(output, configuration):
    if output:
        root = Path(output)
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        root = Path(__file__).resolve().parents[1] / 'runs' / f'{stamp}-{uuid.uuid4().hex[:6]}'
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else f'Unavailable: {result.stderr}'
    write_json(root / 'config.json', asdict(configuration) if is_dataclass(configuration) else configuration)
    write_json(root / 'provenance.json', dict(created=datetime.now(timezone.utc).isoformat(),
               python=sys.version, executable=sys.executable, platform=platform.platform(),
               git_head=git('rev-parse', 'HEAD').strip(), branch=git('branch', '--show-current').strip(),
               packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()}))
    (root / 'source.diff').write_text(git('diff', 'HEAD', '--', 'ML'))
    # A diff cannot represent untracked modules: snapshot all Python sources too.
    import shutil
    if isinstance(configuration, dict) and configuration.get('overrides'):
        shutil.copy2(configuration['overrides'], root / 'reviewed_overrides.json')
    for source in (repo / 'ML').rglob('*.py'):
        relative = source.relative_to(repo / 'ML')
        if any(part in ('runs', '.venv', '__pycache__') for part in relative.parts):
            continue
        destination = root / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return root
