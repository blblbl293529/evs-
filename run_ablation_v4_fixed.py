#!/usr/bin/env python3
"""Corrected V4 ablations. Run in the original V4 Python environment.

Every sequence passes an independent original-vs-all-on equivalence gate before
evaluation. Outputs must be new directories below this script's directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import traceback

SCRIPT_DIR = Path(__file__).resolve().parent
FUSION_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(FUSION_DIR))

import numpy as np
import pandas as pd
import fusion_v4 as original
import comparison_protocol as protocol
import v4_ablation_core as core
from v4_ablation_variants import make_variants, planned_contrasts

VERSION = "v4-ablation-equivalence-and-factorial-20261009-r1"
ATOL = RTOL = 1e-12
BRANCHES = {"L": "local_scale_likelihood", "M": "mesoscopic_scale_likelihood",
            "G": "global_scale_likelihood"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def numeric_check(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, None, max(a.size, b.size)
    if a.dtype.kind == "b" or b.dtype.kind == "b":
        equal = a == b
    else:
        equal = np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
    finite = np.isfinite(a) & np.isfinite(b)
    maximum = float(np.max(np.abs(a[finite].astype(float) - b[finite].astype(float)))) if np.any(finite) else 0.0
    return bool(np.all(equal)), maximum, int(np.sum(~equal))


def metadata_differences(expected, actual, prefix="meta"):
    differences = []
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                differences.append(f"{prefix}.{key}: missing")
            else:
                differences.extend(metadata_differences(value, actual[key], f"{prefix}.{key}"))
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            differences.append(f"{prefix}: different length/type")
        else:
            for i, (left, right) in enumerate(zip(expected, actual)):
                differences.extend(metadata_differences(left, right, f"{prefix}[{i}]"))
    elif isinstance(expected, (float, int, np.number)) and not isinstance(expected, bool):
        if actual is None or not np.isclose(expected, actual, rtol=RTOL, atol=ATOL, equal_nan=True):
            differences.append(f"{prefix}: {expected!r} != {actual!r}")
    elif expected != actual:
        differences.append(f"{prefix}: {expected!r} != {actual!r}")
    return differences


def equivalence_gate(reference_curve, reference_meta, curve, meta, key):
    rows = []
    if list(reference_curve.columns) != list(curve.columns):
        raise AssertionError("All-on curve column schema differs from original V4")
    for column in reference_curve.columns:
        passed, maximum, count = numeric_check(reference_curve[column], curve[column])
        rows.append(dict(**key, check="all_on_curve", field=column, passed=passed,
                         max_abs_difference=maximum, mismatched_samples=count))
    differences = metadata_differences(reference_meta, meta)
    rows.append(dict(**key, check="all_on_metadata", field="all original metadata keys",
                     passed=not differences, max_abs_difference=None,
                     mismatched_samples=len(differences), details=" | ".join(differences)))
    return rows


def branch_checks(full, full_meta, curve, meta, variant, key):
    checks = []
    for column in ["is_candidate"] + [BRANCHES[b] for b in variant.unchanged_branches]:
        passed, maximum, count = numeric_check(full[column], curve[column])
        checks.append(dict(**key, variant=variant.name, check="unchanged_upstream_field",
                           field=column, passed=passed, max_abs_difference=maximum,
                           mismatched_samples=count))
    if set(variant.unchanged_branches) == {"L", "M", "G"}:
        for field in ("evidence_weights", "structural_consensus", "structural_reliability_features"):
            differences = metadata_differences(full_meta.get(field), meta.get(field), field)
            checks.append(dict(**key, variant=variant.name, check="unchanged_upstream_metadata",
                field=field, passed=not differences, max_abs_difference=None,
                mismatched_samples=len(differences), details=" | ".join(differences)))
    if variant.name == "Factor refs=both scales=LMG":
        passed, maximum, count = numeric_check(full["fusion_v4_score"], curve["fusion_v4_score"])
        checks.append(dict(**key, variant=variant.name, check="factorial_full_anchor",
            field="fusion_v4_score", passed=passed, max_abs_difference=maximum, mismatched_samples=count))
    return checks


def estimate(meta, frame):
    if not meta.get("valid_focus", False):
        return None
    index = float(meta["focus_index_float"])
    if not np.isfinite(index):
        raise AssertionError("Valid focus has a nonfinite index")
    return float(np.interp(index, np.arange(len(frame)), frame["timestamp_us"].to_numpy(float)))


def result_row(dataset, sequence, variant, timestamp, runtime_ms, meta, evaluation, mapper, data):
    gt = float(evaluation["gt_timestamp_us"])
    gt_position = evaluation.get("gt_focus_position")
    position = None if mapper is None or timestamp is None else float(mapper(timestamp))
    error = None if timestamp is None else (timestamp - gt) / 1000.0
    position_error = None if position is None or gt_position is None else position - float(gt_position)
    unit = evaluation.get("position_unit")
    primary = "position" if dataset in {"EAD", "EvHF"} else "timestamp"
    weights = meta.get("scale_posterior_weights", {})
    return dict(
        dataset=dataset, sequence=sequence, condition=evaluation.get("condition", ""),
        variant=variant.name, family=variant.family, legacy_name=variant.legacy_name,
        valid=timestamp is not None, predicted_timestamp_us=timestamp, gt_timestamp_us=gt,
        signed_error_ms=error, abs_error_ms=None if error is None else abs(error),
        predicted_focus_position=position, gt_focus_position=gt_position,
        signed_position_error=position_error,
        abs_position_error=None if position_error is None else abs(position_error),
        position_unit=unit, primary_metric=primary,
        primary_unit=unit if primary == "position" else "ms",
        primary_abs_error=(None if position_error is None else abs(position_error)) if primary == "position"
                          else (None if error is None else abs(error)),
        runtime_ms=runtime_ms, event_count_used=int(data.total.sum()), bin_count=len(data.total),
        spatial_scope="official_roi" if data.roi is not None else "full_sensor", roi=str(data.roi),
        local_weight=weights.get("local"), mesoscopic_weight=weights.get("mesoscopic"),
        global_weight=weights.get("global"), focus_index_float=meta.get("focus_index_float"),
    )


def describe(values, prefix):
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {f"{prefix}_n": int(len(values)), f"{prefix}_mae": values.mean(),
            f"{prefix}_median": values.median(), f"{prefix}_p90": values.quantile(.90),
            f"{prefix}_max": values.max()}


def summaries(results, group_keys):
    rows = []
    for group, frame in results.groupby(group_keys, dropna=False, sort=False):
        group = group if isinstance(group, tuple) else (group,)
        row = dict(zip(group_keys, group))
        row.update(sequence_count=len(frame), valid_count=int(frame.valid.sum()),
                   valid_output_rate=float(frame.valid.mean()), runtime_mean_ms=frame.runtime_ms.mean(),
                   primary_metric=frame.primary_metric.iloc[0], primary_unit=frame.primary_unit.iloc[0])
        if frame.primary_unit.dropna().nunique() > 1:
            raise ValueError("Cannot aggregate different primary position units")
        row.update(describe(frame.abs_error_ms, "timestamp"))
        row.update(describe(frame.abs_position_error, "position"))
        row.update(describe(frame.primary_abs_error, "primary"))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tables(results, contrasts):
    details, summary = [], []
    for question, name_a, name_b in contrasts:
        a = results[results.variant == name_a]
        b = results[results.variant == name_b]
        if a.empty or b.empty:
            continue
        merged = a.merge(b, on=["dataset", "sequence"], suffixes=("_A", "_B"), validate="one_to_one")
        for dataset, group in merged.groupby("dataset", sort=False):
            for metric, field in [("timestamp", "abs_error_ms"), ("primary", "primary_abs_error")]:
                left = pd.to_numeric(group[field + "_A"], errors="coerce")
                right = pd.to_numeric(group[field + "_B"], errors="coerce")
                valid = np.isfinite(left) & np.isfinite(right)
                delta = right[valid] - left[valid]
                summary.append(dict(question=question, dataset=dataset, metric=metric, A=name_a, B=name_b,
                    unit="ms" if metric == "timestamp" else group.primary_unit_A.iloc[0],
                    matched_sequences=len(group), paired_valid=int(valid.sum()),
                    A_missing=int((~np.isfinite(left)).sum()), B_missing=int((~np.isfinite(right)).sum()),
                    A_mae_paired=left[valid].mean(), B_mae_paired=right[valid].mean(),
                    mean_delta_B_minus_A=delta.mean(), median_delta_B_minus_A=delta.median(),
                    B_better=int((delta < -1e-9).sum()), tie=int((np.abs(delta) <= 1e-9).sum()),
                    B_worse=int((delta > 1e-9).sum())))
                for idx in group.index:
                    details.append(dict(question=question, dataset=dataset, sequence=group.loc[idx, "sequence"],
                        metric=metric, A=name_a, B=name_b, error_A=left.loc[idx], error_B=right.loc[idx],
                        delta_B_minus_A=right.loc[idx] - left.loc[idx], paired_valid=bool(valid.loc[idx])))
    return pd.DataFrame(details), pd.DataFrame(summary)


def interaction_tables(results):
    """Difference of differences; descriptive interactions, not causal attribution."""
    records = []
    for dataset, group in results.groupby("dataset", sort=False):
        table = group.pivot(index="sequence", columns="variant", values="primary_abs_error")
        for scales in ("L", "LM", "LG", "LMG"):
            names = [f"Factor refs={ref} scales={scales}" for ref in ("both", "sym", "rate", "none")]
            if all(name in table for name in names):
                values = table[names].dropna()
                delta = values[names[0]] - values[names[1]] - values[names[2]] + values[names[3]]
                records.append(dict(dataset=dataset, interaction=f"symmetry x rate; scales={scales}",
                    formula="E(both)-E(sym)-E(rate)+E(none)", paired_valid=len(delta),
                    mean_interaction=delta.mean(), median_interaction=delta.median(), unit=group.primary_unit.iloc[0]))
        for refs in ("none", "sym", "rate", "both"):
            names = [f"Factor refs={refs} scales={scales}" for scales in ("LMG", "LM", "LG", "L")]
            if all(name in table for name in names):
                values = table[names].dropna()
                delta = values[names[0]] - values[names[1]] - values[names[2]] + values[names[3]]
                records.append(dict(dataset=dataset, interaction=f"M x G; refs={refs}",
                    formula="E(LMG)-E(LM)-E(LG)+E(L)", paired_valid=len(delta),
                    mean_interaction=delta.mean(), median_interaction=delta.median(), unit=group.primary_unit.iloc[0]))
    return pd.DataFrame(records)


def historical_comparison(results, old_path, output):
    if not old_path.is_file():
        write_json(output / "historical_comparison_status.json", {"status": "old_results_missing", "path": old_path})
        return
    old = pd.read_csv(old_path)
    now = results[results.legacy_name.notna()].copy()
    old["sequence"] = old.sequence.str.replace("\\", "/", regex=False)
    now["sequence"] = now.sequence.str.replace("\\", "/", regex=False)
    joined = now.merge(old, left_on=["dataset", "sequence", "legacy_name"],
                       right_on=["dataset", "sequence", "variant"], suffixes=("_new", "_old"),
                       how="left", validate="one_to_one", indicator=True)
    for field in ("predicted_timestamp_us", "abs_error_ms", "abs_position_error"):
        joined[field + "_new_minus_old"] = joined[field + "_new"] - joined[field + "_old"]
    joined["same_input_count_and_gt"] = (
        (joined.event_count_used_new == joined.event_count_used_old)
        & (joined.bin_count_new == joined.bin_count_old)
        & np.isclose(joined.gt_timestamp_us_new, joined.gt_timestamp_us_old, atol=1e-6, rtol=0)
        & (joined.roi_new.fillna("None") == joined.roi_old.fillna("None")))
    joined.to_csv(output / "historical_result_comparison.csv", index=False, encoding="utf-8-sig")
    rows = []
    for (dataset, variant), g in joined.groupby(["dataset", "variant_new"], sort=False):
        paired = g[g.same_input_count_and_gt & (g["_merge"] == "both")]
        rows.append(dict(dataset=dataset, variant=variant, matched_compatible=len(paired),
                         unmatched_or_incompatible=len(g)-len(paired),
                         old_timestamp_mae=paired.abs_error_ms_old.mean(),
                         new_timestamp_mae=paired.abs_error_ms_new.mean(),
                         old_position_mae=paired.abs_position_error_old.mean(),
                         new_position_mae=paired.abs_position_error_new.mean(),
                         max_prediction_shift_us=paired.predicted_timestamp_us_new_minus_old.abs().max()))
    pd.DataFrame(rows).to_csv(output / "historical_summary_comparison.csv", index=False, encoding="utf-8-sig")
    write_json(output / "historical_comparison_status.json", {
        "status": "completed", "path": old_path, "sha256": digest(old_path),
        "note": "Count/bin/ROI/GT agreement is a compatibility check, not proof of identical raw events. "
                "Legacy compound definitions are labelled explicitly; this table measures correction effects, not module contributions."})


def write_outputs(output, rows, checks, args):
    pd.DataFrame(checks).to_csv(output / "equivalence_checks.csv", index=False, encoding="utf-8-sig")
    if not rows:
        return
    results = pd.DataFrame(rows)
    if results.duplicated(["dataset", "sequence", "variant"]).any():
        raise AssertionError("Duplicate result keys")
    results.to_csv(output / "all_ablation_results.csv", index=False, encoding="utf-8-sig")
    summaries(results, ["dataset", "variant"]).to_csv(output / "summary_by_dataset.csv", index=False, encoding="utf-8-sig")
    summaries(results, ["dataset", "condition", "variant"]).to_csv(output / "summary_by_condition.csv", index=False, encoding="utf-8-sig")
    contrasts = [("Ablation versus Full V4", "Full V4", name) for name in results.variant.unique() if name != "Full V4"]
    contrasts.extend(planned_contrasts())
    detail, summary = paired_tables(results, contrasts)
    detail.to_csv(output / "paired_differences.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "paired_summary.csv", index=False, encoding="utf-8-sig")
    interaction_tables(results).to_csv(output / "factorial_interactions.csv", index=False, encoding="utf-8-sig")
    historical_comparison(results, args.legacy_results, output)
    write_json(output / "result_interpretation.json", {
        "delta_definition": "B absolute error minus A absolute error; negative favors B",
        "valid_output_rate": "Fraction producing an estimate; NOT autofocus accuracy/success probability",
        "primary_metrics": {"SYN": "timestamp ms", "DAVIS": "timestamp ms", "EVK4": "timestamp ms",
                            "EvHF": "position; preserve protocol unit", "EAD": "position; preserve protocol unit"},
        "warnings": ["All MAE statistics exclude missing errors; consult valid/missing counts.",
                     "Do not pool position errors with different units or pool EAD raw ms with the other datasets.",
                     "Runtime includes computing disabled branches for audit; not a deployable speed comparison.",
                     "Frozen V4 coefficients were jointly calibrated on this public suite; these are not independent generalization tests.",
                     "Factorial effects are conditional on shared preprocessing and frozen gates, not additive causal percentages."]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=FUSION_DIR.parent / "_datasets")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--suite", choices=["all", "check", "legacy", "contributions", "components"], default="all")
    parser.add_argument("--datasets", nargs="+", choices=protocol.DATASETS, default=list(protocol.DATASETS))
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--save-curves", action="store_true")
    parser.add_argument("--legacy-results", type=Path,
                        default=SCRIPT_DIR / "all_dataset_full_curves" / "all_ablation_results.csv")
    args = parser.parse_args()
    if args.max_sequences is not None and args.max_sequences < 1:
        parser.error("--max-sequences must be positive")
    args.datasets = list(dict.fromkeys(args.datasets))
    output = (args.output or SCRIPT_DIR / ("results_fixed_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    if not output.is_relative_to(SCRIPT_DIR) or output == SCRIPT_DIR:
        parser.error("Output must be a new child directory of the requested ablation folder")
    if output.exists():
        parser.error("Output already exists. Choose a new directory to preserve previous results.")
    if digest(original.__file__) != core.SOURCE_SHA256:
        parser.error("fusion_v4.py differs from the reviewed r11 source. Reconcile the core before running.")
    output.mkdir(parents=True)
    variants = make_variants(args.suite)
    if len({v.name for v in variants}) != len(variants):
        raise AssertionError("Duplicate variant names")
    manifest = dict(version=VERSION, status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        arguments=vars(args), python=sys.version, platform=platform.platform(),
        protocol_version=protocol.PROTOCOL_VERSION, original_version=original.VERSION,
        equivalence_atol=ATOL, equivalence_rtol=RTOL, all_on_uses_original_shortcut=False,
        coefficients_refitted=False, variants=[asdict(v) for v in variants],
        packages={name: package_version(name) for name in
                  ("numpy", "pandas", "scipy", "PyWavelets", "matplotlib")}, source_files={})
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in [Path(__file__), Path(core.__file__), SCRIPT_DIR / "v4_ablation_variants.py",
                 Path(original.__file__), Path(protocol.__file__)]:
        shutil.copy2(path, snapshot / path.name)
        manifest["source_files"][path.name] = {"path": str(path), "sha256": digest(path)}
    write_json(output / "run_manifest.json", manifest)
    pd.DataFrame([dict(name=v.name, family=v.family, purpose=v.purpose, legacy_name=v.legacy_name,
                       offsets=str(v.offsets), **asdict(v.switches)) for v in variants]).to_csv(
        output / "variant_definitions.csv", index=False, encoding="utf-8-sig")
    rows, checks = [], []
    current = {}
    discovered_total = 0
    try:
        for dataset in args.datasets:
            dataset_root = args.data_root.resolve() / dataset
            if not dataset_root.is_dir():
                raise FileNotFoundError(dataset_root)
            directories = sorted(protocol.discover_event_sequences(dataset_root), key=lambda p: str(p).lower())
            if args.max_sequences is not None:
                directories = directories[:args.max_sequences]
            if not directories:
                raise ValueError(f"No event sequences discovered in {dataset_root}")
            discovered_total += len(directories)
            print(f"[{dataset}] {len(directories)} sequences, {len(variants)} variants", flush=True)
            for number, directory in enumerate(directories, 1):
                current = dict(dataset=dataset, directory=str(directory))
                data = protocol.load_sequence(dataset, directory, dataset_root)
                frame = protocol.frame_from_data(data)
                key = dict(dataset=dataset, sequence=data.relative_name)
                current.update(key)
                # Hash suffix prevents collisions in sanitized sequence names.
                safe = protocol.safe_sequence_name(data.relative_name)[:110]
                safe += "_" + hashlib.sha256(data.relative_name.encode()).hexdigest()[:10]
                seqout = output / dataset / "sequences" / safe
                seqout.mkdir(parents=True)
                positive, negative = data.positive.copy(), data.negative.copy()
                duration = data.duration_us.copy() * 1e-6
                for array in (positive, negative, duration):
                    array.flags.writeable = False
                reference_curve, reference_meta = original.focus_from_event_counts(positive, negative, duration)
                started = time.perf_counter()
                full_curve, full_meta = core.focus_from_event_counts(positive, negative, duration,
                                        offsets=variants[0].offsets, switches=variants[0].switches)
                full_runtime = (time.perf_counter() - started) * 1000
                gate = equivalence_gate(reference_curve, reference_meta, full_curve, full_meta, key)
                checks.extend(gate)
                write_json(seqout / "all_on_equivalence.json", gate)
                if not all(r["passed"] for r in gate):
                    reference_curve.to_csv(seqout / "FAILED_original_curve.csv", index=False)
                    full_curve.to_csv(seqout / "FAILED_all_on_curve.csv", index=False)
                    raise AssertionError(f"All-on equivalence failed: {key}")
                predictions, metadata, runtimes = {}, {}, {}
                for variant in variants:
                    current["variant"] = variant.name
                    if variant.name == "Full V4":
                        curve, meta, runtime = full_curve, full_meta, full_runtime
                    else:
                        started = time.perf_counter()
                        curve, meta = core.focus_from_event_counts(positive, negative, duration,
                                      offsets=variant.offsets, switches=variant.switches)
                        runtime = (time.perf_counter() - started) * 1000
                    branch = branch_checks(full_curve, full_meta, curve, meta, variant, key)
                    checks.extend(branch)
                    if not all(r["passed"] for r in branch):
                        write_json(seqout / "FAILED_branch_checks.json", branch)
                        raise AssertionError(f"Unexpected upstream branch change: {current}")
                    predictions[variant.name] = estimate(meta, frame)
                    metadata[variant.name], runtimes[variant.name] = meta, runtime
                    if args.save_curves:
                        (seqout / "curves").mkdir(exist_ok=True)
                        saved = curve.copy()
                        saved.insert(1, "timestamp_us", frame["timestamp_us"].to_numpy())
                        saved.to_csv(seqout / "curves" / (slug(variant.name) + ".csv"), index=False, encoding="utf-8-sig")
                # No evaluation labels are requested until ALL variant predictions exist.
                evaluation = protocol.evaluation_metadata(data)
                mapper = evaluation.pop("position_mapper", None)
                sequence_rows = [result_row(dataset, data.relative_name, variant, predictions[variant.name],
                    runtimes[variant.name], metadata[variant.name], evaluation, mapper, data) for variant in variants]
                rows.extend(sequence_rows)
                pd.DataFrame(sequence_rows).to_csv(seqout / "ablation_results.csv", index=False, encoding="utf-8-sig")
                input_hash = hashlib.sha256()
                for array in (positive, negative, duration, frame["timestamp_us"].to_numpy()):
                    input_hash.update(np.ascontiguousarray(array).tobytes())
                write_json(seqout / "ablation_audit.json", dict(**key, event_file=data.event_path,
                    event_file_size=data.event_path.stat().st_size, event_file_mtime_ns=data.event_path.stat().st_mtime_ns,
                    input_arrays_sha256=input_hash.hexdigest(), common_interval_us=[data.start_us, data.end_us],
                    official_roi=data.roi, event_count=int(data.total.sum()), bin_count=len(frame),
                    all_on_equivalence_passed=True, ground_truth_loaded_after_all_predictions=True,
                    predictions=predictions, evaluation=evaluation, variant_metadata=metadata))
                print(f"  {number}/{len(directories)} PASS {data.relative_name}", flush=True)
            # Checkpoint after each dataset; manifest stays running until all finish.
            write_outputs(output, rows, checks, args)
        expected = discovered_total * len(variants)
        if len(rows) != expected:
            raise AssertionError(f"Expected {expected} result rows, found {len(rows)}")
        manifest.update(status="complete", completed_sequences=discovered_total, result_rows=len(rows),
                        all_checks_passed=all(r["passed"] for r in checks))
    except Exception as exc:
        manifest.update(status="failed", failure_context=current, error=repr(exc))
        write_json(output / "failure.json", dict(context=current, error=repr(exc), traceback=traceback.format_exc()))
        # Preserve partial evidence without disguising an incomplete run as complete.
        pd.DataFrame(rows).to_csv(output / "partial_results.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(checks).to_csv(output / "equivalence_checks.csv", index=False, encoding="utf-8-sig")
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run_manifest.json", manifest)
    print(f"Complete. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
