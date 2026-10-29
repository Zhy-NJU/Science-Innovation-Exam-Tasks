#!/usr/bin/env python3
"""Endpoint-only CPU reference method. This is not the task verifier.

Fit uses train labels and a small, predeclared validation search. Predict has
no label argument. Neither operation reads transition states or source IDs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits


SCHEMA = "endpoint-pair-statistics-v1"
ELEMENTS = (1, 6, 7, 8)
ELEMENT_PAIRS = tuple(itertools.combinations_with_replacement(ELEMENTS, 2))
# Fixed heuristic length scales in angstrom, not fitted to any barrier labels.
RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}
CONFIGURATIONS = (
    {"name": "et_leaf1", "n_estimators": 320, "min_samples_leaf": 1,
     "max_features": 1.0, "random_state": 20261030, "n_jobs": 2},
    {"name": "et_leaf3", "n_estimators": 320, "min_samples_leaf": 3,
     "max_features": 1.0, "random_state": 20261030, "n_jobs": 2},
)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")


def versions():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def stats(values):
    """Eight order-invariant statistics; empty element groups are zero."""
    a = np.sort(np.asarray(values, dtype=np.float64))
    if not len(a):
        return np.zeros(8, dtype=np.float64)
    return np.array([a.sum(), a.mean(), a.std(), a[0],
                     *np.quantile(a, [0.25, 0.5, 0.75]), a[-1]])


def features(row):
    """Fixed-size statistics of mapped endpoint pair distances and soft bonds."""
    z = np.asarray(row["atomic_numbers"], dtype=np.int64)
    r = np.asarray(row["reactant_positions_angstrom"], dtype=np.float64)
    p = np.asarray(row["product_positions_angstrom"], dtype=np.float64)
    if z.ndim != 1 or r.shape != (len(z), 3) or p.shape != r.shape or len(z) < 2:
        raise ValueError("Invalid atom/coordinate shapes")
    if not set(z).issubset(ELEMENTS) or not np.isfinite(r).all() or not np.isfinite(p).all():
        raise ValueError("Unsupported elements or non-finite coordinates")
    ii, jj = np.triu_indices(len(z), 1)
    dr = np.linalg.norm(r[ii] - r[jj], axis=1)
    dp = np.linalg.norm(p[ii] - p[jj], axis=1)
    scale = np.array([1.25 * (RADII[int(z[i])] + RADII[int(z[j])])
                      for i, j in zip(ii, jj)])
    # The smooth proxy avoids an abrupt bond cutoff; it is not a bond-order label.
    qr = 1.0 / (1.0 + (dr / scale) ** 8)
    qp = 1.0 / (1.0 + (dp / scale) ** 8)
    dq = qp - qr
    activity = np.zeros(len(z))
    coordination_r = np.zeros(len(z))
    coordination_p = np.zeros(len(z))
    for i, j, a, cr, cp in zip(ii, jj, abs(dq), qr, qp):
        activity[i] += a
        activity[j] += a
        coordination_r[i] += cr
        coordination_r[j] += cr
        coordination_p[i] += cp
        coordination_p[j] += cp
    low, high = np.minimum(z[ii], z[jj]), np.maximum(z[ii], z[jj])
    out = [np.array([len(z), *[np.sum(z == e) for e in ELEMENTS]], dtype=float)]
    for a, b in ELEMENT_PAIRS:
        mask = (low == a) & (high == b)
        out.append(np.array([mask.sum()], dtype=float))
        for values in (dr, dp, dp - dr, abs(dp - dr), qr, qp, dq):
            out.append(stats(values[mask]))
    for e in ELEMENTS:
        for values in (activity, coordination_r, coordination_p):
            out.append(stats(values[z == e]))
    result = np.concatenate(out)
    if result.shape != (671,) or not np.isfinite(result).all():
        raise ValueError("Invalid feature vector")
    return result


def read_dataset(path, require_target=False):
    opener = gzip.open if str(path).endswith(".gz") else open
    ids, xx, yy = [], [], []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["reaction_id"])
            xx.append(features(row))
            if require_target:
                yy.append(float(row["forward_barrier_eV"]))
            elif "forward_barrier_eV" in row:
                raise ValueError("Unlabeled prediction input contains target field")
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Missing or duplicated reaction IDs")
    x = np.asarray(xx, dtype=np.float64)
    y = np.asarray(yy, dtype=np.float64) if require_target else None
    if y is not None and not np.isfinite(y).all():
        raise ValueError("Non-finite training targets")
    return ids, x, y


def read_validation_labels(path, ids):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    values = {r["reaction_id"]: float(r["forward_barrier_eV"]) for r in rows}
    if len(values) != len(rows) or set(values) != set(ids):
        raise ValueError("Validation label IDs do not match validation inputs")
    y = np.array([values[i] for i in ids], dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("Non-finite validation targets")
    return y


def mae(pred, target):
    return math.fsum(float(abs(a - b)) for a, b in zip(pred, target)) / len(target)


def predict_deterministically(model, x):
    # Serial tree accumulation prevents scheduling-dependent reduction order.
    original_jobs = model.n_jobs
    model.n_jobs = 1
    try:
        return model.predict(x)
    finally:
        model.n_jobs = original_jobs


def write_predictions(path, ids, predictions):
    pred = np.asarray(predictions, dtype=np.float64)
    if pred.shape != (len(ids),) or not np.isfinite(pred).all():
        raise ValueError("Predictions must be complete and finite")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["reaction_id", "forward_barrier_eV"])
        writer.writerows((i, format(float(v), ".17g")) for i, v in zip(ids, pred))


def selftest(output):
    rng = np.random.default_rng(20261030)
    z = [1, 1, 6, 6, 7, 8]
    r = rng.normal(size=(len(z), 3))
    p = r + rng.normal(scale=0.5, size=r.shape)
    row = {"atomic_numbers": z, "reactant_positions_angstrom": r,
           "product_positions_angstrom": p}
    f0 = features(row)
    errors = []
    for _ in range(24):
        order = rng.permutation(len(z))
        rot_r, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rot_p, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        changed = {"atomic_numbers": np.array(z)[order],
                   "reactant_positions_angstrom": (r @ rot_r + rng.normal(size=3))[order],
                   "product_positions_angstrom": (p @ rot_p + rng.normal(size=3))[order]}
        diff = float(np.max(abs(features(changed) - f0)))
        errors.append(diff)
        np.testing.assert_allclose(features(changed), f0, rtol=1e-12, atol=1e-12)
    swapped = {**row, "reactant_positions_angstrom": p,
               "product_positions_angstrom": r}
    assert not np.allclose(features(swapped), f0)
    # Explicitly demonstrate that IDs and irrelevant/answer-like fields are ignored.
    np.testing.assert_array_equal(features({**row, "reaction_id": "unrelated",
                                           "source_id": "ignored",
                                           "forward_barrier_eV": 9876.0,
                                           "reactant_energy_eV": -1234.0,
                                           "product_energy_eV": -5678.0,
                                           "transition_state_energy_eV": 9999.0,
                                           "transition_state_positions_angstrom":
                                               np.zeros_like(r)}), f0)
    result = {"passed": True, "synthetic_cases": 24, "n_features": len(f0),
              "max_abs_invariance_error": max(errors),
              "checked": ["independent endpoint orthogonal rotations/reflections",
                          "independent endpoint translations", "common atom permutation",
                          "source IDs, target/endpoint energies and TS coordinates not used as features",
                          "reaction direction preserved"], "versions": versions()}
    write_json(output, result)
    print(json.dumps(result), flush=True)


def fit(args):
    start = time.perf_counter()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    declaration = {"feature_schema": SCHEMA, "n_features": 671,
                   "configurations": CONFIGURATIONS,
                   "selection": "Lowest validation MAE; tie goes to first configuration",
                   "training": "Train split only; no train+validation refit",
                   "bottom": "Global median of training barriers; no validation fitting",
                   "test_policy": "No test file or test label is read by fit",
                   "versions": versions(), "script_sha256": digest(__file__)}
    write_json(out / "predeclared_search.json", declaration)
    train_ids, x, y = read_dataset(args.train, require_target=True)
    val_ids, xv, _ = read_dataset(args.validation)
    if set(train_ids) & set(val_ids):
        raise ValueError("Train and validation IDs overlap")
    yv = read_validation_labels(args.validation_labels, val_ids)
    median = float(np.median(y))
    read_seconds = time.perf_counter() - start
    runs, best, best_model = [], None, None
    with threadpool_limits(limits=2):
        for conf in CONFIGURATIONS:
            settings = {k: v for k, v in conf.items() if k != "name"}
            model = ExtraTreesRegressor(**settings)
            tick = time.perf_counter()
            model.fit(x, y)
            pv = predict_deterministically(model, xv)
            metric = mae(pv, yv)
            record = {**conf, "validation_mae_eV": metric,
                      "fit_predict_seconds": time.perf_counter() - tick}
            runs.append(record)
            write_predictions(out / "validation" / conf["name"] / "predictions.csv", val_ids, pv)
            print(json.dumps(record), flush=True)
            if best is None or metric < best["validation_mae_eV"]:
                best, best_model = record, model
    bottom_mae = mae(np.full(len(yv), median), yv)
    write_predictions(out / "validation" / "bottom" / "predictions.csv", val_ids,
                      np.full(len(yv), median))
    if not best["validation_mae_eV"] < bottom_mae:
        raise RuntimeError("Predeclared main method does not beat median on validation")
    artifact = {"feature_schema": SCHEMA, "model": best_model,
                "training_median_eV": median, "selected_configuration": best,
                "train_ids": train_ids}
    model_path = out / "selected_model.joblib"
    joblib.dump(artifact, model_path, compress=3)
    report = {**declaration, "train_count": len(y), "validation_count": len(yv),
              "validation_candidate_evaluations": len(CONFIGURATIONS),
              "validation_metric_evaluations_including_bottom": len(CONFIGURATIONS) + 1,
              "runs": runs,
              "selected": best, "bottom_validation_mae_eV": bottom_mae,
              "training_median_eV": median, "read_feature_seconds": read_seconds,
              "wall_seconds": time.perf_counter() - start,
              "peak_process_rss_bytes": peak_rss_bytes(),
              "inputs": {str(p): digest(p) for p in
                         [args.train, args.validation, args.validation_labels]},
              "model_sha256": digest(model_path),
              "test_labels_read": False}
    write_json(out / "selection.json", report)
    print(json.dumps({"selected": best, "bottom_validation_mae_eV": bottom_mae,
                      "wall_seconds": report["wall_seconds"]}), flush=True)


def predict(args):
    start = time.perf_counter()
    model_dir = Path(args.model_dir)
    report = json.loads((model_dir / "selection.json").read_text())
    path = model_dir / "selected_model.joblib"
    if digest(path) != report["model_sha256"]:
        raise ValueError("Selected model hash changed after validation selection")
    artifact = joblib.load(path)
    if artifact["feature_schema"] != SCHEMA:
        raise ValueError("Wrong feature schema")
    ids, x, _ = read_dataset(args.input)
    if set(ids) & set(artifact["train_ids"]):
        raise ValueError("Prediction IDs overlap training IDs")
    with threadpool_limits(limits=2):
        pred = predict_deterministically(artifact["model"], x)
    out = Path(args.output)
    write_predictions(out / "baseline" / "predictions.csv", ids, pred)
    write_predictions(out / "bottom" / "predictions.csv", ids,
                      np.full(len(ids), artifact["training_median_eV"]))
    info = {"count": len(ids), "finite": bool(np.isfinite(pred).all()),
            "input_sha256": digest(args.input), "model_sha256": digest(path),
            "selection_sha256": digest(model_dir / "selection.json"),
            "prediction_sha256": digest(out / "baseline" / "predictions.csv"),
            "bottom_sha256": digest(out / "bottom" / "predictions.csv"),
            "wall_seconds": time.perf_counter() - start,
            "peak_process_rss_bytes": peak_rss_bytes(), "versions": versions(),
            "test_labels_read": False, "predictions_clipped": False}
    write_json(out / "prediction_provenance.json", info)
    print(json.dumps(info), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--output", type=Path, required=True)
    f = sub.add_parser("fit")
    for a in ["train", "validation", "validation-labels", "output"]:
        f.add_argument("--" + a, type=Path, required=True)
    p = sub.add_parser("predict")
    for a in ["model-dir", "input", "output"]:
        p.add_argument("--" + a, type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "selftest":
        selftest(args.output)
    elif args.operation == "fit":
        fit(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
