#!/usr/bin/env python3
"""Run the same predeclared method using aggregate framework feedback only.

This construction-evidence script is not an additional Agent input. The
candidate step reads train labels and unlabeled validation endpoints. The
selection step reads three framework result JSONs, never private labels.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

import baseline as method


def candidates(args):
    start = time.perf_counter()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train_ids, x, y = method.read_dataset(args.train, require_target=True)
    ids, xv, _ = method.read_dataset(args.validation)
    if set(train_ids) & set(ids):
        raise ValueError("Training and development IDs overlap")
    median = float(np.median(y))
    records = []
    with threadpool_limits(limits=2):
        for config in method.CONFIGURATIONS:
            target = out / config["name"]
            target.mkdir(exist_ok=True)
            model = ExtraTreesRegressor(**{k: v for k, v in config.items() if k != "name"})
            tick = time.perf_counter()
            model.fit(x, y)
            prediction = method.predict_deterministically(model, xv)
            method.write_predictions(target / "predictions.csv", ids, prediction)
            artifact = {"feature_schema": method.SCHEMA, "model": model,
                        "training_median_eV": median, "train_ids": train_ids,
                        "selected_configuration": config}
            joblib.dump(artifact, target / "model.joblib", compress=3)
            records.append({**config, "model_sha256": method.digest(target / "model.joblib"),
                            "predictions_sha256": method.digest(target / "predictions.csv"),
                            "fit_predict_seconds": time.perf_counter() - tick})
    method.write_predictions(out / "bottom" / "predictions.csv", ids,
                             np.full(len(ids), median))
    manifest = {"configurations": records, "training_median_eV": median,
                "train_count": len(train_ids), "validation_count": len(ids),
                "train_sha256": method.digest(args.train),
                "validation_sha256": method.digest(args.validation),
                "feature_schema": method.SCHEMA, "versions": method.versions(),
                "wall_seconds": time.perf_counter() - start,
                "peak_process_rss_bytes": method.peak_rss_bytes(),
                "validation_or_test_labels_read": False}
    method.write_json(out / "candidates.json", manifest)
    print(json.dumps(manifest), flush=True)


def feedback(path, expected_count):
    result = json.loads(Path(path).read_text())
    if result.get("status") != "ok":
        raise ValueError(f"Validation feedback was not accepted: {path}")
    if result.get("n_samples") != expected_count:
        raise ValueError("Feedback sample count differs from the development input")
    round_number, remaining = result["round"], result["remaining"]
    if (type(round_number) is not int or not 1 <= round_number <= 20
            or type(remaining) is not int or remaining != 20 - round_number):
        raise ValueError("Invalid validation round/remaining budget")
    value = float(result["metrics"]["mae_eV"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("Expected finite nonnegative development MAE")
    return {"name": "mae_eV", "value": value, "round": round_number,
            "remaining": remaining, "n_samples": expected_count,
            "feedback_sha256": method.digest(path)}


def select_from_feedback(leaf1_path, leaf3_path, bottom_path, expected_count):
    """Select using accepted aggregate responses; no oracle labels or model refit."""
    responses = [feedback(leaf1_path, expected_count), feedback(leaf3_path, expected_count)]
    bottom = feedback(bottom_path, expected_count)
    if len({r["round"] for r in [*responses, bottom]}) != 3:
        raise ValueError("Expected three distinct accepted development rounds")
    chosen_index = min(range(2), key=lambda i: (responses[i]["value"], i))
    if responses[chosen_index]["value"] >= bottom["value"]:
        raise ValueError("Neither selected candidate improves on the constant reference")
    return chosen_index, responses, bottom


def select(args):
    source = Path(args.candidate_dir)
    manifest = json.loads((source / "candidates.json").read_text())
    chosen_index, responses, bottom = select_from_feedback(
        args.leaf1_feedback, args.leaf3_feedback, args.bottom_feedback,
        manifest["validation_count"])
    chosen = manifest["configurations"][chosen_index]
    expected_name = method.CONFIGURATIONS[chosen_index]["name"]
    if chosen["name"] != expected_name:
        raise ValueError("Candidates differ from the predeclared order")
    model_path = source / chosen["name"] / "model.joblib"
    if method.digest(model_path) != chosen["model_sha256"]:
        raise ValueError("Candidate model changed before selection")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, out / "selected_model.joblib")
    report = {"selected": {**chosen, "validation_mae_eV": responses[chosen_index]["value"]},
              "training_median_eV": manifest["training_median_eV"],
              "bottom_validation_mae_eV": bottom["value"],
              "model_sha256": method.digest(out / "selected_model.joblib"),
              "aggregate_feedback": responses, "bottom_feedback": bottom,
              "feedback_count": 3, "validation_or_test_labels_read": False,
              "selection_rule": "Lowest validation MAE; first candidate wins exact tie"}
    method.write_json(out / "selection.json", report)
    print(json.dumps(report), flush=True)


def verify_feedback(args):
    frozen = json.loads(Path(args.frozen_selection).read_text())
    index, responses, bottom = select_from_feedback(
        args.leaf1_feedback, args.leaf3_feedback, args.bottom_feedback,
        frozen["validation_count"])
    name = method.CONFIGURATIONS[index]["name"]
    if name != frozen["selected"]["name"]:
        raise ValueError("Framework feedback selects a different candidate")
    diffs = [abs(responses[i]["value"] - frozen["runs"][i]["validation_mae_eV"])
             for i in range(2)]
    diffs.append(abs(bottom["value"] - frozen["bottom_validation_mae_eV"]))
    if max(diffs) > 1e-14:
        raise ValueError("Framework feedback disagrees with builder arithmetic")
    result = {"status": "passed", "selected_candidate": name,
              "aggregate_feedback": responses, "bottom_feedback": bottom,
              "feedback_count": 3, "private_labels_read": False,
              "model_or_predictions_modified": False,
              "builder_oracle_mae_absolute_differences": diffs,
              "frozen_selection_sha256": method.digest(args.frozen_selection),
              "staged_script_sha256": method.digest(__file__)}
    method.write_json(args.output, result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    c = sub.add_parser("candidates")
    for key in ["train", "validation", "output"]:
        c.add_argument("--" + key, type=Path, required=True)
    s = sub.add_parser("select")
    for key in ["candidate-dir", "leaf1-feedback", "leaf3-feedback", "bottom-feedback", "output"]:
        s.add_argument("--" + key, type=Path, required=True)
    v = sub.add_parser("verify-feedback")
    for key in ["frozen-selection", "leaf1-feedback", "leaf3-feedback", "bottom-feedback", "output"]:
        v.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    if args.operation == "candidates":
        candidates(args)
    elif args.operation == "select":
        select(args)
    else:
        verify_feedback(args)


if __name__ == "__main__":
    main()
