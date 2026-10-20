#!/usr/bin/env python3
"""Public reviewer helper added after the original tests.

Rebuild baseline CSVs in a separate output directory and score them locally.
This does not run CHGNet inference, download anything, or modify task packages.
Public archives and withheld scoring labels are reviewer-only resources.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

SOURCES = {
    "thermal/chgnet.json.gz": (
        "https://ndownloader.figshare.com/files/68221282",
        "a9b23130992988c9541a62635c17ce4b8f4b8cb7da2dfbdcd67a44722d585505"),
    "chgnet-crystal.csv.gz": (
        "https://ndownloader.figshare.com/files/66646268",
        "b04ce3b031827fcd532e22058d06dc77fb4705ce8b12aea54652a14504e41f16"),
    "wbm-summary.csv.gz": (
        "https://ndownloader.figshare.com/files/64706751",
        "adbdd8b24086d4888195d8894c77f6d5fb29ce33bbb5d7d3898b4d79e964dc54"),
}
SPECS = {
    "thermal_conductivity": ("material_id", "kappa_W_mK", 103,
        "chgnet_baseline.csv",
        "74bc5657075478ba4aec20473db3bb90903d961b763ff2959612f5cf1ac6aad5",
        1.6958708753402623, 2.0, 0.0),
    "crystal_stability": ("material_id", "e_form_eV_per_atom", 215488,
        "chgnet_baseline.csv",
        "847a6984d68f69f3b8ee76a44b52889e481fe90d37285f2ef6be1f731f76b601",
        0.611111789635673, 0.2652014652014652, -1e6),
    "mof_adsorption": ("system_id", "adsorption_energy_eV", 1768,
        "baseline_submission.csv",
        "ee560463787d03bd17388ce368851642bf2695f0be268a793ba34aebd3027c5f",
        0.22528328096993902, 0.6565553732351735, 0.0),
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(root, relative):
    path = root / relative
    assert sha(path) == SOURCES[relative][1], "Source checksum mismatch: " + relative
    return path


def write_csv(path, columns, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows((key, format(value, ".17g")) for key, value in sorted(values.items()))


def thermal_values(root):
    with gzip.open(source(root, "thermal/chgnet.json.gz"), "rt") as stream:
        data = json.load(stream)
    index = {name: key for key, name in data["material_id"].items()}
    assert len(index) == len(data["material_id"]) == 103
    values = {}
    for number, name in enumerate(sorted(index), 1):
        key = index[name]
        kappa = data.get("kappa_tot_rta", {}).get(key)
        if isinstance(kappa, list) and len(kappa) == 1 and isinstance(kappa[0], list):
            kappa = kappa[0]
        invalid = bool(data.get("broken_symmetry", {}).get(key))
        invalid |= bool(data.get("has_imag_ph_modes", {}).get(key))
        invalid |= not isinstance(kappa, list) or len(kappa) != 6
        if not invalid:
            try:
                diagonal = [float(kappa[i]) for i in range(3)]
                invalid = any(not math.isfinite(value) or value < 0 for value in diagonal)
            except (TypeError, ValueError):
                invalid = True
        values[f"thermal_{number:04d}"] = 0.0 if invalid else math.fsum(diagonal) / 3
    return values


def crystal_values(root):
    with gzip.open(source(root, "wbm-summary.csv.gz"), "rt") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 256963
    selected = sorted(row["material_id"] for row in rows if row["unique_prototype"] == "True")
    assert len(selected) == len(set(selected)) == 215488
    with gzip.open(source(root, "chgnet-crystal.csv.gz"), "rt") as stream:
        archive = {row["material_id"]: float(row["e_form_per_atom"])
                   for row in csv.DictReader(stream)}
    values = {f"crystal_{number:06d}": archive[name]
              for number, name in enumerate(selected, 1)}
    assert all(math.isfinite(value) for value in values.values())
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Local official archives; names, URLs and SHA256 are listed in SOURCES")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=sorted(SPECS), default=sorted(SPECS))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    tasks = args.tasks_root.resolve()
    output = args.output_root.resolve()
    assert output != tasks and tasks not in output.parents, "Outputs must be outside task packages"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("SIE_VERIFY_CONTEXT", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    report = {"scope": "New reviewer helper; local replay of published predictions and training-only MOF median. No new full CHGNet inference or private-runtime/container test.",
              "tasks": {}, "passed": False}
    for task in args.tasks:
        identifier, prediction, count, filename, expected_sha, expected, bottom, constant = SPECS[task]
        package = tasks / task
        with (package / "input/submission.csv").open(newline="") as stream:
            ids = [row[identifier] for row in csv.DictReader(stream)]
        assert len(ids) == len(set(ids)) == count
        baseline = output / task / filename
        if task == "mof_adsorption":
            script = Path(__file__).resolve().parents[1] / "mof_adsorption/baseline.py"
            run = subprocess.run([args.python, "-B", str(script), "--input-dir",
                                  str(package / "input"), "--output", str(baseline)],
                                 env=env, capture_output=True, text=True, check=True)
            execution = json.loads(run.stdout)
            assert execution["training_count"] == 162224 and execution["fallback_count"] == 0
        else:
            values = thermal_values(args.source_root) if task == "thermal_conductivity" else crystal_values(args.source_root)
            assert set(values) == set(ids)
            write_csv(baseline, [identifier, prediction], values)
        assert sha(baseline) == expected_sha, "Rebuilt baseline bytes differ: " + task
        scored = {}
        for kind, expected_score in (("baseline", expected), ("bottom", bottom)):
            directory = output / task / "official_results" / kind
            submission = directory / "submission"
            submission.mkdir(parents=True, exist_ok=True)
            if kind == "baseline":
                (submission / "submission.csv").write_bytes(baseline.read_bytes())
            else:
                write_csv(submission / "submission.csv", [identifier, prediction],
                          {key: constant for key in ids})
            result = directory / "result.json"
            subprocess.run([args.python, "-B", str(package / "verifier/test/run/main.py"),
                            "--input", str(package / "input"), "--submission", str(submission),
                            "--result", str(result)], env=env, check=True, capture_output=True, text=True)
            value = json.loads(result.read_text())
            assert value["valid"] and value["status"] == "valid"
            assert value["primary_metric"]["value"] == expected_score
            scored[kind] = {"result": value, "sha256": sha(result)}
        report["tasks"][task] = {"count": count, "baseline_sha256": sha(baseline), "results": scored}
    report["passed"] = True
    (output / "replay_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "tasks": args.tasks, "report": str(output / "replay_results.json")}))


if __name__ == "__main__":
    main()
