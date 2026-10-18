#!/usr/bin/env python3
"""Audit script for development parameter selection and evaluation artifacts."""

import json
from pathlib import Path
import sys
import numpy as np

# Add src to path for metrics calculation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from orion.metrics import calculate_metrics

def main():
    root = Path(__file__).resolve().parent.parent / "orion2-run"
    selected_path = root / "selected_config.json"
    eval_path = root / "evaluation" / "results.json"
    out_path = root / "evaluation_verification.json"

    with open(selected_path, "r", encoding="utf-8") as f:
        selected = json.load(f)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 1. Seed Disjointness
    dev_seeds = set(selected["development_seeds"])
    eval_seeds = set(selected["evaluation_seeds"])
    seeds_disjoint = dev_seeds.isdisjoint(eval_seeds)

    # 2. Development Selection Audit
    max_fdp = selected.get("maximum_development_fdp", 0.01)
    selection_audit = {}
    parameter_selection_valid = True

    for method, candidates in selected.get("scores", {}).items():
        eligible = [c for c in candidates if c.get("eligible") or (c.get("mean_fdp") is not None and c.get("mean_fdp") <= max_fdp)]
        if not eligible:
            parameter_selection_valid = False
            selection_audit[method] = {"eligible_count": 0, "match": False}
            continue
        best = max(eligible, key=lambda c: c["mean_pairwise_f1"])
        chosen = selected["methods"].get(method, {})
        matches = (chosen == best["parameters"])
        selection_audit[method] = {
            "eligible_count": len(eligible),
            "best_f1": best["mean_pairwise_f1"],
            "match": matches,
        }
        if not matches:
            parameter_selection_valid = False

    # 3. Evaluation Completeness
    results = eval_data.get("results", [])
    eval_count = len(results)
    eval_complete = (eval_count == 120)

    # 4. Metric Fidelity
    metric_errors = []
    for row in results:
        pred_file = root / row["predictions_file"]
        if not pred_file.exists():
            metric_errors.append(f"Missing file: {row['predictions_file']}")
            continue
        data = np.load(pred_file)
        calc = calculate_metrics(data["labels_true"], data["labels_pred"])
        for m in ("pairwise_precision", "pairwise_recall", "pairwise_f1"):
            if abs(calc[m] - row["metrics"].get(m, 0.0)) > 1e-6:
                metric_errors.append(f"Mismatch in {row['predictions_file']} for {m}")

    metrics_valid = (len(metric_errors) == 0)

    all_verified = seeds_disjoint and parameter_selection_valid and eval_complete and metrics_valid
    status = "verified" if all_verified else "failed"

    report = {
        "status": status,
        "parameter_selection_valid": parameter_selection_valid,
        "seeds_disjoint": seeds_disjoint,
        "development_seeds": list(dev_seeds),
        "evaluation_seeds": list(eval_seeds),
        "evaluation_runs_count": eval_count,
        "metric_fidelity_valid": metrics_valid,
        "metric_errors_count": len(metric_errors),
        "selection_audit": selection_audit,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Audit completed with status: {status}")
    print(f"parameter_selection_valid: {parameter_selection_valid}")
    return 0 if all_verified else 1

if __name__ == "__main__":
    sys.exit(main())
