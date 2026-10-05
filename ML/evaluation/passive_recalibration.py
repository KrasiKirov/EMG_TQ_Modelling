"""Audit session-to-session passive-torque calibration drift.

The audit compares the production/source passive curve with a curve fitted from
included passive plateaus in a selected session, normally the retest session.
It never reads active trials or active torque values and does not alter model
predictions. The result is evidence for deciding whether an active-torque target
is stable enough to interpret across sessions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ML.evaluation.report import write_json
from ML.preprocessing.calibration import fit_passive, fit_passive_session
from ML.preprocessing.trial_manifest import load_inventory


def audit_passive_recalibration(trials, manifest, data_dir=None, session="retest",
                                drift_flag_nm=0.5):
    """Compare source calibration with an independently fitted session curve.

    Parameters
    ----------
    trials, manifest:
        The raw frames and immutable inventory returned by ``load_inventory``.
    data_dir:
        Required only when the production calibration uses a documented source
        file whose hash must be checked.
    session:
        Session whose passive plateaus are audited. ``retest`` is the normal
        deployment-style check.
    drift_flag_nm:
        Informational threshold for the per-position drift flag. It does not
        change any target or model output.
    """
    if not 0 < drift_flag_nm:
        raise ValueError("drift_flag_nm must be positive")

    source = fit_passive(trials, manifest, data_dir)
    session_calibration = fit_passive_session(trials, manifest, session=session)
    rows = []
    for position, session_torque in zip(session_calibration.positions,
                                         session_calibration.torques):
        source_torque = float(source.predict(np.asarray([position], dtype=float))[0])
        supported = bool(np.isfinite(source_torque))
        delta = float(session_torque - source_torque) if supported else None
        rows.append({
            "session": session,
            "position_rad": float(position),
            "source_passive_nm": source_torque if supported else None,
            "session_passive_nm": float(session_torque),
            "delta_session_minus_source_nm": delta,
            "abs_delta_nm": abs(delta) if delta is not None else None,
            "source_supported": supported,
            "drift_flag": bool(delta is not None and abs(delta) >= drift_flag_nm),
            "evidence": [e for e in session_calibration.evidence
                         if np.isclose(e.get("position", np.nan), position)],
        })

    supported_rows = [r for r in rows if r["source_supported"]]
    deltas = np.asarray([r["delta_session_minus_source_nm"] for r in supported_rows],
                        dtype=float)
    endpoint = min(rows, key=lambda row: row["position_rad"]) if rows else None
    summary = {
        "session": session,
        "source_point_count": len(source.positions),
        "session_point_count": len(session_calibration.positions),
        "supported_point_count": len(supported_rows),
        "unsupported_point_count": len(rows) - len(supported_rows),
        "mean_signed_drift_nm": float(np.mean(deltas)) if len(deltas) else None,
        "rms_drift_nm": float(np.sqrt(np.mean(deltas ** 2))) if len(deltas) else None,
        "max_abs_drift_nm": float(np.max(np.abs(deltas))) if len(deltas) else None,
        "flagged_point_count": int(sum(r["drift_flag"] for r in rows)),
        "most_negative_position": endpoint,
    }
    return {
        "schema_version": 1,
        "subject": manifest["subject"],
        "session": session,
        "audit_only": True,
        "active_trials_used": 0,
        "active_torque_used": False,
        "drift_flag_threshold_nm": float(drift_flag_nm),
        "source_calibration": source.to_dict(),
        "session_calibration": session_calibration.to_dict(),
        "summary": summary,
        "records": rows,
    }


def markdown_report(report):
    """Render a compact audit report suitable for review artifacts."""
    summary = report["summary"]
    lines = [
        "# Passive recalibration audit", "",
        f"Subject: `{report['subject']}`  ",
        f"Session: `{report['session']}`  ",
        "This is an audit only: active trials and active torque values were not used.",
        "",
        "## Summary", "",
        f"- Supported points: {summary['supported_point_count']} / "
        f"{summary['session_point_count']}",
        f"- RMS passive drift: {summary['rms_drift_nm']:.3f} Nm"
        if summary["rms_drift_nm"] is not None else "- RMS passive drift: unavailable",
        f"- Maximum absolute drift: {summary['max_abs_drift_nm']:.3f} Nm"
        if summary["max_abs_drift_nm"] is not None else "- Maximum absolute drift: unavailable",
        f"- Flagged positions: {summary['flagged_point_count']}",
        "",
        "## Position comparison", "",
        "| Position (rad) | Source passive (Nm) | Session passive (Nm) | Drift (Nm) | Supported | Flag |",
        "| ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    for row in report["records"]:
        source = "—" if row["source_passive_nm"] is None else f"{row['source_passive_nm']:.3f}"
        delta = "—" if row["delta_session_minus_source_nm"] is None else f"{row['delta_session_minus_source_nm']:+.3f}"
        lines.append(f"| {row['position_rad']:+.4f} | {source} | "
                     f"{row['session_passive_nm']:.3f} | {delta} | "
                     f"{'yes' if row['source_supported'] else 'no'} | "
                     f"{'yes' if row['drift_flag'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--overrides")
    parser.add_argument("--session", choices=("test", "retest"), default="retest")
    parser.add_argument("--drift-flag-nm", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    args = parser.parse_args(argv)

    trials, manifest = load_inventory(args.subject, args.data_dir, args.overrides)
    report = audit_passive_recalibration(trials, manifest, args.data_dir, args.session,
                                         args.drift_flag_nm)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    write_json(output, report)
    if args.markdown:
        markdown = Path(args.markdown)
        if markdown.exists():
            raise FileExistsError(markdown)
        markdown.write_text(markdown_report(report))
    print(f"Audited {len(report['records'])} passive {args.session} positions for "
          f"{args.subject}")


if __name__ == "__main__":
    main()
