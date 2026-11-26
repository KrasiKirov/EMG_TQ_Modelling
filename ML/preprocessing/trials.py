"""Trial classification by comment, and the split of a passive trial into its static holds (1 kHz recordings)."""

import re

import numpy as np

POSITION_COL = 'position'   # ankle angle column name (rad)
TORQUE_COL = 'torque'       # measured torque column name (Nm)

# Active trials with a larger torque std are ramps / dynamic movements, not holds; excluded.
ACTIVE_STD_THRESHOLD = 30.0   # Nm
POSITION_TOLERANCE = 0.05     # rad — positions within this band are merged into one entry
_STEP_VEL_THRESHOLD = 5.0     # rad/s — single-sample velocity spike flags a device step
_SETTLE_SAMPLES = 1000        # samples to skip after a step (1 s at 1000 Hz)


def _parse_comment(comment: str):
    """Parse a trial comment into (trial_type 'mvc'|'passive'|'active', session 'test'|'retest'|None, position 'p1'..'p8'|None)."""
    comment = comment.lower().strip()

    # Position: match "p3" style OR "pos indx 3" / "indx 3" style
    m = re.search(r'p(\d+)', comment)
    if m:
        pos = m.group(0)
    else:
        m2 = re.search(r'(?:pos\s*)?indx\s*(\d+)', comment)
        pos = f"p{m2.group(1)}" if m2 else None

    # Determine session (retest must be checked before test)
    if 'retest' in comment:
        session = 'retest'
    elif 'test' in comment or 'repeat' in comment:
        session = 'test'
    else:
        session = None

    if comment.startswith('mvc'):
        return 'mvc', session, pos
    elif 'passive' in comment:
        return 'passive', session, pos
    else:
        return 'active', session, pos


def classify_trials(trials):
    """Split trials into (mvc, passive, active) lists and tag each with ``attrs['session']`` from its comment."""
    mvc, passive, active = [], [], []
    for df in trials:
        t_type, session, _ = _parse_comment(df.attrs.get('comment', ''))
        df.attrs['session'] = session
        if t_type == 'mvc':
            mvc.append(df)
        elif t_type == 'passive':
            passive.append(df)
        else:
            active.append(df)
    return mvc, passive, active


def split_passive_segments(df, position_col=POSITION_COL):
    """
    Split a passive trial at single-sample position steps (two back-to-back static holds in one trial).

    Splits at jumps above ``_STEP_VEL_THRESHOLD`` rad/s and drops ``_SETTLE_SAMPLES`` after each jump.
    Returns one DataFrame slice per static-hold segment.
    """
    pos   = df[position_col].values
    dt    = df.attrs.get('domainIncr', 0.001)
    speed = np.abs(np.diff(pos)) / dt          # instantaneous speed, length T-1
    jumps = np.where(speed > _STEP_VEL_THRESHOLD)[0]   # sample just before jump

    if len(jumps) == 0:
        return [df]

    segments, start = [], 0
    for j in jumps:
        seg = df.iloc[start: j + 1]
        if len(seg) > _SETTLE_SAMPLES:
            segments.append(seg)
        start = j + 1 + _SETTLE_SAMPLES       # skip post-step transient

    if start < len(df):
        seg = df.iloc[start:]
        if len(seg) > _SETTLE_SAMPLES:
            segments.append(seg)

    return segments
