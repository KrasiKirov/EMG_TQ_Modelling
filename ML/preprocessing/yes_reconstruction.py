"""Explicit, opt-in reconstruction of the historical eight-position YES analysis.

The report shows eight retest angles and documents manual label corrections,
but the original trial-selection script is unavailable. This mapping matches
the measured angles and acquisition order; it is not an independently verified
acquisition log. Never select a mapping based on model performance.
"""
import hashlib
from pathlib import Path

import numpy as np

FLB_SHA256 = '21c48d08cd902ff649091f0a4679b6ac18e702b0ce6df27c8685ee19700ca4a7'
WORKSHEET_SHA256 = '9c905c91c83d69e5eb4994fb2975edcd59fb7e1ef60fe94137165e66e54ec851'
# One-based trial, intended worksheet position, expected measured angle,
# original comment, and retest passive torque transcribed from the worksheet.
RETEST_RECORDS = (
    (27, 'p1', -.278, 'passive retest, p2', +3.50),
    (29, 'p2', +.245, 'passive retest, p3', -5.00),
    (31, 'p3', -.381, 'passive retest, p4', +6.70),
    (33, 'p4', -.071, 'passive retest, p4', +.73),
    (35, 'p5', -.176, 'retest, p5', +2.40),
    (37, 'p6', -.486, 'retest, p6', +10.60),
    (39, 'p7', +.031, 'retest, p7', -.38),
    (41, 'p8', +.138, 'retest, p8', -2.50),
)
# Passive torques transcribed from the independent Test column of the YES
# worksheet, in the same p1--p8 order as RETEST_RECORDS.  These values are
# kept separate from the retest map so session-matched target construction can
# use session 1 for training/validation and session 2 for retest.
TEST_PASSIVE_TORQUES = (+2.60, -10.10, +4.30, +0.35, +1.80, +7.90, -2.60, -3.50)
SOURCE = 'YES retest worksheet map; eight-position historical reconstruction (trial identity inferred)'


def _sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reconstruct_yes(trials, flb_path):
    """Return new frames, passive map and ledger; never mutate source frames/file."""
    worksheet = Path(flb_path).with_name('reklab_trial_worksheet_YES_20241128.pdf')
    if _sha256(flb_path) != FLB_SHA256 or _sha256(worksheet) != WORKSHEET_SHA256:
        raise ValueError('YES reconstruction requires the reviewed FLB and worksheet hashes')
    if len(trials) != 41:
        raise ValueError('YES reconstruction expected exactly 41 trials')
    frames = [frame.copy(deep=True) for frame in trials]
    ledger, passive = [], []
    for number, position_id, expected_angle, original_comment, torque in RETEST_RECORDS:
        frame = frames[number - 1]
        angle = float(frame.position.mean())
        if (frame.attrs['trial_index'] != number - 1
                or frame.attrs['comment'] != original_comment
                or abs(angle - expected_angle) > .005
                or not np.isfinite(frame.to_numpy()).all()
                or frame.position.std() > .01
                or frame.torque.std() > 30):
            raise ValueError(f'YES trial {number} differs from the reviewed recording')
        frame.attrs['original_comment'] = original_comment
        frame.attrs['comment'] = f'retest, {position_id}'
        frame.attrs['correction_source'] = SOURCE
        passive.append((angle, torque))
        ledger.append(dict(trial=number, position_id=position_id,
                           mean_angle_rad=angle, original_comment=original_comment,
                           analysis_comment=frame.attrs['comment'],
                           torque_sd_nm=float(frame.torque.std()),
                           passive_torque_nm=torque,
                           status='reconstructed' if number <= 33 else 'original active label'))
    test_passive = sorted((entry['mean_angle_rad'], torque)
                          for entry, torque in zip(ledger, TEST_PASSIVE_TORQUES))
    audit = dict(protocol='YES eight-position reconstruction',
                 confidence='historical angle matching and worksheet; exact trial identity inferred',
                 source_sha256=FLB_SHA256, worksheet_sha256=WORKSHEET_SHA256,
                 worksheet=str(worksheet), retest_records=ledger,
                 excluded_unstable_active_trials=[20, 22, 26, 30],
                 passive_map_source=SOURCE,
                 test_passive_entries=test_passive,
                 test_passive_map_source='YES test worksheet passive column')
    return frames, sorted(passive), audit
