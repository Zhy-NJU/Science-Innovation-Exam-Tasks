#!/usr/bin/env python3
"""A fixed, deterministic, label-free-at-test-time composition ridge baseline.

Training uses only input/training_compositions.jsonl.gz. This program never reads
the Verifier, WBM reference labels, source-ID mapping or published predictions.
Features: intercept and 118 elemental atomic fractions. Objective: sum squared
formation-energy residuals plus 1.0 times the squared elemental coefficients.
The intercept is unpenalized. This predetermined baseline does not model crystal
geometry or distinguish polymorphs with the same composition.
"""
import argparse
import collections
import csv
import gzip
import json
import math
from pathlib import Path


def solve(matrix, rhs):
    """Gaussian elimination with partial pivoting, in the fixed feature order."""
    n = len(rhs)
    rows = [row[:] + [value] for row, value in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(rows[row][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        assert abs(rows[col][col]) > 1e-12, "Singular ridge system"
        for row in range(col + 1, n):
            multiplier = rows[row][col] / rows[col][col]
            if multiplier == 0:
                continue
            rows[row][col] = 0.0
            for j in range(col + 1, n + 1):
                rows[row][j] -= multiplier * rows[col][j]
    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        solution[row] = (rows[row][-1] - math.fsum(rows[row][j] * solution[j] for j in range(row + 1, n))) / rows[row][row]
    assert all(math.isfinite(x) for x in solution)
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Task input directory")
    parser.add_argument("--output", type=Path, required=True, help="Output submission.csv path")
    args = parser.parse_args()
    dimension = 119
    gram = [[0.0] * dimension for _ in range(dimension)]
    rhs = [0.0] * dimension
    for element in range(1, dimension):
        gram[element][element] = 1.0
    n_train = 0
    with gzip.open(args.input / "training_compositions.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            total = math.fsum(row["stoichiometry"])
            assert total > 0
            features = [(0, 1.0)] + [(z, count / total) for z, count in zip(row["atomic_numbers"], row["stoichiometry"])]
            target = row["e_form_eV_per_atom"]
            for i, xi in features:
                rhs[i] += xi * target
                for j, xj in features:
                    gram[i][j] += xi * xj
            n_train += 1
    assert n_train > 0
    coefficients = solve(gram, rhs)
    predictions = {}
    with gzip.open(args.input / "test_structures.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            numbers = row["atomic_numbers"]
            counts = collections.Counter(numbers)
            pred = coefficients[0] + math.fsum(coefficients[z] * count / len(numbers) for z, count in sorted(counts.items()))
            assert math.isfinite(pred)
            assert row["material_id"] not in predictions
            predictions[row["material_id"]] = pred
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["material_id", "e_form_eV_per_atom"])
        writer.writerows((mid, format(predictions[mid], ".17g")) for mid in sorted(predictions))
    report = {"algorithm": "ridge on intercept + elemental fractions", "ridge_lambda": 1.0,
              "intercept_penalized": False, "training_rows": n_train, "prediction_rows": len(predictions),
              "label_source": "Only the bundled MP training_compositions.jsonl.gz; no WBM test labels accessed",
              "features": ["intercept"] + [f"atomic_fraction_Z{z}" for z in range(1, 119)],
              "coefficients": coefficients}
    args.output.with_suffix(".model.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"features", "coefficients"}}, indent=2))


if __name__ == "__main__":
    main()
