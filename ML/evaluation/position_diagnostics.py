"""Signed-position convention audit and per-position error diagnostics.

The project files use signed ankle positions, but a sign convention in source
code is not the same thing as an independent physical polarity verification.
This module therefore keeps the numeric groups (negative/near-zero/positive)
primary and attaches the project-intended anatomical labels only as qualified
metadata.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ML.evaluation.metrics import compute_metrics
from ML.evaluation.report import write_json


DEFAULT_CONVENTION_FILES = (
    "exp_model/initialize_input_waveform.m",
    "ES_sandbox/EMG2TQ_TIDynNLBiLinModel_prbsTQ_HM01_20250411.m",
)


def signed_group(angle_rad: float, zero_tolerance: float = 1e-6) -> str:
    """Return a direction-neutral group from a signed angle."""
    if not np.isfinite(angle_rad):
        return "unknown"
    if angle_rad < -zero_tolerance:
        return "negative"
    if angle_rad > zero_tolerance:
        return "positive"
    return "near_zero"


def _project_label(group: str) -> str | None:
    return {
        "negative": "plantarflexion (project-intended; physical polarity unverified)",
        "positive": "dorsiflexion (project-intended; physical polarity unverified)",
        "near_zero": "near-neutral angle",
    }.get(group)


def audit_project_convention(project_root, source_files=DEFAULT_CONVENTION_FILES):
    """Audit signed constants in the original MATLAB source.

    This deliberately reports intended software semantics, not a claim that a
    transducer's physical positive direction has been independently calibrated.
    """
    root = Path(project_root)
    evidence = []
    missing = []
    for relative in source_files:
        path = root / relative
        if not path.exists():
            missing.append(relative)
            continue
        text = path.read_text(errors="replace")
        values = {}
        for name in ("maxPF", "maxDF"):
            match = re.search(rf"\b{name}\s*=\s*([-+]?\d+(?:\.\d+)?)", text)
            if match:
                values[name] = float(match.group(1))
        evidence.append({"file": relative, "values": values})

    observed_pf = [item["values"]["maxPF"] for item in evidence if "maxPF" in item["values"]]
    observed_df = [item["values"]["maxDF"] for item in evidence if "maxDF" in item["values"]]
    consistent = bool(observed_pf and observed_df and all(v < 0 for v in observed_pf)
                      and all(v > 0 for v in observed_df))
    return {
        "status": "intended_convention_found" if consistent else "incomplete_or_inconsistent",
        "negative_angle_intended_label": "plantarflexion" if consistent else None,
        "positive_angle_intended_label": "dorsiflexion" if consistent else None,
        "physical_sensor_polarity_verified": False,
        "evidence": evidence,
        "missing_files": missing,
        "interpretation": (
            "The original MATLAB source declares maxPF as negative and maxDF as positive. "
            "This establishes the project's intended software convention, but does not "
            "independently verify raw-sensor polarity."
            if consistent else
            "The expected signed constants were not consistently found in the requested source files."
        ),
    }


def _manifest_position_angles(run_root: Path):
    """Map subject/position IDs to manifest means and observed spreads."""
    import json

    result = {}
    for manifest_path in run_root.glob("*_manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        subject = manifest.get("subject", manifest_path.stem.removesuffix("_manifest"))
        values = {}
        for record in manifest.get("trials", []):
            if record.get("kind") != "active" or record.get("position_id") is None:
                continue
            angle = record.get("mean_position_rad")
            if angle is None or not np.isfinite(angle):
                continue
            values.setdefault(record["position_id"], []).append(float(angle))
        result[subject] = {
            position: {"angle_rad": float(np.mean(angles)),
                       "angle_spread_rad": float(np.ptp(angles)),
                       "manifest_trials": len(angles)}
            for position, angles in values.items()
        }
    return result


def _metric_row(y_true, y_pred, subject, run_name, position_id, angle, zero_tolerance):
    group = signed_group(angle, zero_tolerance)
    row = compute_metrics(y_true, y_pred)
    row.update({
        "subject": str(subject),
        "run": run_name,
        "position_id": str(position_id),
        "angle_rad": angle,
        "signed_group": group,
        "project_intended_label": _project_label(group),
        "n": int(len(y_true)),
    })
    return row


def diagnose_run(run, project_root=None, split="test", names=None,
                 zero_tolerance=1e-6):
    """Create a strict-JSON-ready signed-position diagnostic for a saved run."""
    import json

    root = Path(run)
    if not root.is_dir():
        raise FileNotFoundError(root)
    angles = _manifest_position_angles(root)
    selected = set(names) if names else None
    rows = []
    warnings = []
    for prediction_path in sorted(root.glob(f"*/{split}_predictions.npz")):
        run_name = prediction_path.parent.name
        if selected and run_name not in selected:
            continue
        package_path = prediction_path.parent / "package.json"
        package = json.loads(package_path.read_text()) if package_path.exists() else {}
        with np.load(prediction_path, allow_pickle=False) as saved:
            required = {"y_true", "y_pred", "position_id", "subject", "trial"}
            missing = required - set(saved.files)
            if missing:
                warnings.append(f"{run_name}: missing fields {sorted(missing)}")
                continue
            y_true, y_pred = saved["y_true"], saved["y_pred"]
            subject_values = saved["subject"]
            subject = str(subject_values[0]) if len(subject_values) else run_name.split("_")[0]
            position_values = saved["position_id"]
            for position_id in sorted(np.unique(position_values).tolist()):
                mask = position_values == position_id
                position_id = str(position_id)
                details = angles.get(subject, {}).get(position_id)
                if details is None:
                    warnings.append(f"{run_name}/{position_id}: no manifest angle")
                    angle = float("nan")
                    angle_spread = None
                    manifest_trials = 0
                else:
                    angle = details["angle_rad"]
                    angle_spread = details["angle_spread_rad"]
                    manifest_trials = details["manifest_trials"]
                row = _metric_row(y_true[mask], y_pred[mask], subject, run_name,
                                  position_id, angle, zero_tolerance)
                row.update({
                    "trial_count": int(np.unique(saved["trial"][mask]).size),
                    "manifest_trial_count": manifest_trials,
                    "angle_spread_rad": angle_spread,
                    "model_kind": package.get("kind"),
                    "seed": package.get("seed"),
                    "low_variance": bool(row["target_sd"] < .1),
                })
                rows.append(row)

    rows.sort(key=lambda row: (row["subject"], row["angle_rad"]
                               if np.isfinite(row["angle_rad"]) else np.inf,
                               row["run"], row["position_id"]))
    grouped = {}
    for group in ("negative", "near_zero", "positive", "unknown"):
        group_rows = [row for row in rows if row["signed_group"] == group]
        if not group_rows:
            continue
        grouped[group] = {
            "n_windows": int(sum(row["n"] for row in group_rows)),
            "positions": sorted({row["position_id"] for row in group_rows}),
            "mean_rmse": float(np.mean([row["rmse"] for row in group_rows])),
            "worst_rmse": float(np.max([row["rmse"] for row in group_rows])),
        }

    subject_summary = {}
    for subject in sorted({row["subject"] for row in rows}):
        subject_rows = [row for row in rows if row["subject"] == subject]
        subject_summary[subject] = {
            "run_count": len({row["run"] for row in subject_rows}),
            "position_count": len({row["position_id"] for row in subject_rows}),
            "mean_rmse": float(np.mean([row["rmse"] for row in subject_rows])),
            "worst_position_rmse": float(np.max([row["rmse"] for row in subject_rows])),
        }
    worst = sorted(rows, key=lambda row: row["rmse"], reverse=True)
    if selected and not rows:
        raise ValueError("Requested packages with saved predictions were not found")
    if project_root is None:
        project_root = root.parents[2] if len(root.parents) >= 3 else root
    return {
        "schema_version": 1,
        "run": str(root),
        "split": split,
        "position_convention": audit_project_convention(project_root),
        "zero_tolerance_rad": zero_tolerance,
        "records": rows,
        "signed_groups": grouped,
        "subjects": subject_summary,
        "worst_positions": worst[:20],
        "warnings": warnings,
    }


def markdown_report(report):
    """Render a compact human-readable report without hiding missing coverage."""
    lines = ["# Signed-position diagnostic", "",
             f"Run: `{report['run']}`", f"Split: `{report['split']}`", "",
             "The numeric grouping is authoritative: negative, near-zero, and positive. "
             "Anatomical labels are project-intended and physical polarity remains unverified.", "",
             "## Worst positions", "",
             "| Subject | Run | Position | Angle (rad) | Signed group | RMSE (Nm) | MAE (Nm) | n |",
             "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for row in report["worst_positions"]:
        angle = "unknown" if not np.isfinite(row["angle_rad"]) else f"{row['angle_rad']:+.4f}"
        lines.append(f"| {row['subject']} | {row['run']} | {row['position_id']} | {angle} | "
                     f"{row['signed_group']} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['n']} |")
    lines.extend(["", "## Subject summary", "",
                  "| Subject | Runs | Positions | Mean RMSE (Nm) | Worst position RMSE (Nm) |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for subject, summary in report["subjects"].items():
        lines.append(f"| {subject} | {summary['run_count']} | {summary['position_count']} | "
                     f"{summary['mean_rmse']:.4f} | {summary['worst_position_rmse']:.4f} |")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Saved benchmark run directory")
    parser.add_argument("--project-root", required=True,
                        help="Repository root containing the original MATLAB sources")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--names", nargs="+", help="Specific saved package names")
    parser.add_argument("--zero-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--markdown", help="Optional Markdown output path")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = diagnose_run(args.run, args.project_root, args.split, args.names,
                          args.zero_tolerance)
    write_json(output, report)
    if args.markdown:
        markdown = Path(args.markdown)
        if markdown.exists():
            raise FileExistsError(markdown)
        markdown.write_text(markdown_report(report))
    print(f"Wrote {len(report['records'])} position records to {output}")


if __name__ == "__main__":
    main()
