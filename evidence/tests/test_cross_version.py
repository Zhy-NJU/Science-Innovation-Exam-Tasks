#!/usr/bin/env python3
"""Re-score frozen baseline CSVs through official-shape CLI with two real interpreters.
This does not mock the SDK or certify private-runtime/container integration.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import jsonschema

ROOT = next((path for path in Path(__file__).resolve().parents if (path / "official-repo").is_dir()), Path(__file__).resolve().parents[2])
BASELINES = {
    "thermal_conductivity": "thermal_conductivity/chgnet_baseline.csv",
    "crystal_stability": "crystal_stability/chgnet_baseline.csv",
    "mof_adsorption": "mof_adsorption/baseline_submission.csv",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, default=ROOT / "official-repo")
    parser.add_argument("--tasks-root", type=Path, default=ROOT / "submission_bundle/tasks")
    parser.add_argument("--evidence-root", type=Path, default=ROOT / "submission_bundle/evidence")
    parser.add_argument("--pythons", nargs="+", default=["/usr/bin/python3", "/opt/anaconda3/bin/python"])
    parser.add_argument("--report", type=Path, default=ROOT / "submission_bundle/evidence/tests/cross_version_smoke.json")
    args = parser.parse_args()
    schema = json.loads((args.official_repo / "contributor/build-sie-task-package/schemas/verifier_result.schema.json").read_text())
    runs = []
    with tempfile.TemporaryDirectory(prefix="sie-cross-version-") as directory:
        scratch = Path(directory)
        submission = scratch / "submission"
        submission.mkdir()
        for task, baseline_relative in BASELINES.items():
            package = args.tasks_root / task
            baseline = args.evidence_root / baseline_relative
            baseline_hash = digest(baseline)
            shutil.copyfile(baseline, submission / "submission.csv")
            metadata = json.loads((package / "meta.json").read_text())
            expected = metadata.get("baseline", {}).get("score")
            previous = None
            for index, python in enumerate(args.pythons):
                result_path = scratch / (task + str(index) + ".json")
                command = [python, "-B", str(package / "verifier/test/run/main.py"), "--input", str(package / "input"), "--submission", str(submission), "--result", str(result_path)]
                environment = dict(os.environ)
                environment.pop("SIE_VERIFY_CONTEXT", None)
                run = subprocess.run(command, capture_output=True, text=True, timeout=90, env=environment)
                if run.returncode != 0:
                    raise AssertionError(run.stderr)
                raw = result_path.read_bytes()
                result = json.loads(raw)
                jsonschema.validate(result, schema)
                assert result["valid"] and result["status"] == "valid"
                assert result["task_id"] == metadata["id"] == task
                score = result["primary_metric"]["value"]
                assert math.isfinite(score)
                if expected is not None:
                    assert math.isclose(score, expected, rel_tol=1e-12, abs_tol=1e-12), (task, score, expected)
                if previous is not None:
                    assert raw == previous, "Result differs between actual Python versions: " + task
                previous = raw
                runs.append({"task": task, "python": python, "python_version": subprocess.check_output([python, "--version"], text=True).strip(), "baseline_sha256": baseline_hash, "baseline_relative_path": baseline_relative, "result": result, "result_sha256": hashlib.sha256(raw).hexdigest()})
            assert digest(baseline) == baseline_hash
    report = {"scope": "Local stateless official-shape CLI re-scoring of frozen baseline CSVs; no model re-run, SDK mock, private runtime or container certification.", "official_repository_commit": subprocess.check_output(["git", "-C", str(args.official_repo), "rev-parse", "HEAD"], text=True).strip(), "contract_revision": "2026-09-17", "passed": True, "cross_version_byte_identical": True, "runs": runs}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": True, "runs": len(runs), "report": str(args.report)}))


if __name__ == "__main__":
    main()
