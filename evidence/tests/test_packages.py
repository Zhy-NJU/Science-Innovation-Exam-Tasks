#!/usr/bin/env python3
"""Independent end-to-end checks against the official SIE contract and schemas.

These tests use official contract_revision 2026-09-17 schemas and checker from
PrismaX-Team/LiveDiscoveryBench-Tasks at a pinned commit. They do not certify
container execution or the private evaluation runtime. No VerifyContext SDK
is mocked or invented; these scientific tasks are stateless.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import jsonschema


REPOSITORY = next((path for path in Path(__file__).resolve().parents if (path / "official-repo").is_dir()), Path(__file__).resolve().parents[2])
OFFICIAL_REPO = REPOSITORY / "official-repo"
OFFICIAL_SKILL = OFFICIAL_REPO / "contributor/build-sie-task-package"
OFFICIAL_SCHEMA_DIR = OFFICIAL_SKILL / "schemas"
SPECS = {
    "thermal_conductivity": {
        "id": "material_id", "prediction": "kappa_W_mK",
        "metric": "mean_sre", "direction": "minimize", "nonnegative": True,
    },
    "crystal_stability": {
        "id": "material_id", "prediction": "e_form_eV_per_atom",
        "metric": "stability_f1", "direction": "maximize", "nonnegative": False,
    },
    "mof_adsorption": {
        "id": "system_id", "prediction": "adsorption_energy_eV",
        "metric": "mean_mae", "direction": "minimize", "nonnegative": False,
    },
}


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def require_materialized_files(source):
    # SF_DATALESS is defined by Apple's SDK sys/stat.h as 0x40000000.
    # Avoid indefinitely blocking in fcopyfile on cloud-only local placeholders.
    dataless = []
    for path in source.rglob("*"):
        if path.is_file() and getattr(path.stat(), "st_flags", 0) & 0x40000000:
            dataless.append(str(path.relative_to(source)))
    if dataless:
        raise RuntimeError("Source package still has macOS dataless files; materialize before testing: " + ", ".join(dataless))


class PackageChecks(unittest.TestCase):
    """Base class: one generated subclass for each independently frozen task."""

    @classmethod
    def setUpClass(cls):
        cls.spec = SPECS[cls.task_name]
        cls.package = cls.tasks_root / cls.task_name
        cls.score_package = cls.snapshot_root / cls.task_name
        cls.labels_path = cls.package / "verifier/test/data/labels.csv"
        cls.protocol_path = cls.package / "verifier/protocol.json"
        cls.labels = load_csv(cls.labels_path)
        cls.protocol = json.loads(cls.protocol_path.read_text())
        cls.meta = json.loads((cls.package / "meta.json").read_text())
        cls.id_column = cls.spec["id"]
        cls.prediction_column = cls.spec["prediction"]
        cls.fields = [cls.id_column, cls.prediction_column]
        cls.ids = [row[cls.id_column] for row in cls.labels]
        if len(cls.ids) != len(set(cls.ids)):
            raise AssertionError("Reference IDs are not unique")
        cls.base_rows = []
        for row in cls.labels:
            if cls.task_name == "crystal_stability":
                hull = float(row["hull_ref_eV_per_atom"])
                stable = float(row["d_true_eV_per_atom"]) <= 0
                value = hull - 1.0 if stable else hull + 1.0
            else:
                value = float(row[cls.prediction_column])
            cls.base_rows.append([row[cls.id_column], format(value, ".17g")])

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="sie-package-check-")
        self.root = Path(self.scratch.name)
        self.submission = self.root / "submission"
        self.submission.mkdir()

    def tearDown(self):
        self.scratch.cleanup()

    def write_rows(self, rows=None, fields=None):
        path = self.submission / "submission.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.fields if fields is None else fields)
            writer.writerows(self.base_rows if rows is None else rows)
        return path

    def run_test(self, package=None, submission=None, result_name="result.json"):
        package = self.score_package if package is None else package
        submission = self.submission if submission is None else submission
        result_path = self.root / result_name
        command = [
            self.python, "-B", str(package / "verifier/test/run/main.py"),
            "--input", str(self.package / "input"),
            "--submission", str(submission), "--result", str(result_path),
        ]
        run = subprocess.run(command, capture_output=True, text=True, timeout=45)
        self.assertTrue(result_path.is_file(), msg=f"Missing result JSON; rc={run.returncode}; stderr={run.stderr[-3000:]}")
        raw = result_path.read_bytes()
        result = json.loads(raw)
        jsonschema.validate(result, json.loads((OFFICIAL_SCHEMA_DIR / "verifier_result.schema.json").read_text()))
        self.assertEqual(run.returncode, 0, msg=run.stderr)
        self.assertEqual(result["task_id"], self.meta["id"])
        self.assertEqual(result["status"], "valid" if result["valid"] else "invalid")
        primary = result["primary_metric"]
        self.assertEqual(primary["name"], self.spec["metric"])
        self.assertEqual(primary["direction"], self.spec["direction"])
        self.assertNotIsInstance(primary["value"], bool)
        self.assertTrue(math.isfinite(primary["value"]))
        names = [metric["name"] for metric in result["metrics"]]
        self.assertEqual(len(names), len(set(names)))
        metrics = {metric["name"]: metric["value"] for metric in result["metrics"]}
        self.assertEqual(metrics[primary["name"]], primary["value"])
        for metric in result["metrics"]:
            self.assertNotIsInstance(metric["value"], bool)
            self.assertTrue(math.isfinite(metric["value"]))
        if result["valid"]:
            self.assertIsNone(result["error"])
        else:
            self.assertIsInstance(result["error"], str)
            self.assertTrue(result["error"])
            self.assertEqual(primary["value"], self.protocol["invalid_score"])
        return result, raw

    def assert_invalid(self, rows=None):
        if rows is not None:
            self.write_rows(rows)
        result, _ = self.run_test()
        self.assertFalse(result["valid"], msg=result)

    def run_validation(self, payload=b"{}", package=None):
        package = self.score_package if package is None else package
        request = self.root / "request.json"
        response = self.root / "response.json"
        request.write_bytes(payload)
        run = subprocess.run([
            self.python, "-B", str(package / "verifier/validation/run/main.py"),
            "--input", str(self.package / "input"), "--request", str(request),
            "--response", str(response),
        ], capture_output=True, text=True, timeout=20)
        self.assertTrue(response.is_file(), msg=f"No response JSON; rc={run.returncode}; {run.stderr[-3000:]}")
        result = json.loads(response.read_text())
        self.assertEqual(run.returncode, 0, msg=run.stderr)
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("status"), "error")
        self.assertIsInstance(result.get("error"), str)
        self.assertTrue(result["error"])
        for key in ("raw_score", "score", "primary_metric", "metrics"):
            self.assertNotIn(key, result)
        return result

    def test_package_contract_files(self):
        expected = {"instruction.json", "input", "environment", "verifier", "meta.json"}
        self.assertEqual({p.name for p in self.package.iterdir()}, expected)
        for rel in ("verifier/validation/run/main.py", "verifier/test/run/main.py", "input/submission.csv"):
            self.assertTrue((self.package / rel).is_file(), rel)
        instruction = json.loads((self.package / "instruction.json").read_text())
        environment = json.loads((self.package / "environment/environment.json").read_text())
        meta = self.meta
        for value, schema_name in ((instruction, "instruction"), (environment, "environment"), (meta, "meta")):
            jsonschema.validate(value, json.loads((OFFICIAL_SCHEMA_DIR / (schema_name + ".schema.json")).read_text()))
        self.assertEqual(meta["id"], self.package.name)
        for entry in instruction["input"]:
            relative = Path(entry["path"])
            self.assertFalse(relative.is_absolute())
            self.assertNotIn("..", relative.parts)
            self.assertTrue((self.package / "input" / relative).exists(), str(relative))
        self.assertGreaterEqual(len(instruction["submission"]), 1)
        for artifact in instruction["submission"]:
            relative = Path(artifact["path"])
            self.assertFalse(relative.is_absolute())
            self.assertNotIn("..", relative.parts)
            if artifact.get("has_template", True):
                self.assertTrue((self.package / "input" / relative).is_file())
        if environment["type"] == "containerfile":
            containerfile = self.package / "environment" / environment["containerfile"]
            self.assertTrue(containerfile.is_file())
            definition = containerfile.read_text()
            self.assertIn("ARG SIE_BASE_IMAGE", definition)
            self.assertIn("FROM ${SIE_BASE_IMAGE}", definition)
        else:
            self.assertEqual(instruction["environment"], [])
            self.assertEqual({p.name for p in (self.package / "environment").iterdir()}, {"environment.json"})
        for source in meta["task_sources"]:
            self.assertTrue(source["name"])
            self.assertTrue(source["url"].startswith("https://"))
        if "baseline" in meta:
            baseline = meta["baseline"]
            for field in ("score", "bottom_line_score"):
                self.assertNotIsInstance(baseline[field], bool)
                self.assertTrue(math.isfinite(baseline[field]))
            if meta["metric"]["type"] is not None and meta["metric"]["type"].endswith("maximize"):
                self.assertGreater(baseline["score"], baseline["bottom_line_score"])
            elif meta["metric"]["type"] is not None and meta["metric"]["type"].endswith("minimize"):
                self.assertLess(baseline["score"], baseline["bottom_line_score"])

    def test_files_fit_normal_github_submission(self):
        # GitHub's documented hard block is >100 MiB for ordinary Git blobs.
        # These packages do not depend on an approved LFS/runtime-fetch contract.
        for path in self.package.rglob("*"):
            if path.is_file():
                self.assertGreater(path.stat().st_size, 0, "Empty packaged file: " + str(path))
                self.assertLessEqual(path.stat().st_size, 100 * 1024 * 1024, str(path))

    def test_official_package_checker(self):
        run = subprocess.run([
            sys.executable, "-B", str(OFFICIAL_SKILL / "scripts/check_package.py"), str(self.package),
        ], cwd=OFFICIAL_REPO, capture_output=True, text=True, timeout=90)
        self.assertEqual(run.returncode, 0, msg=run.stdout + run.stderr)
        response = json.loads(run.stdout)
        self.assertEqual(response["status"], "package_valid")

    def test_official_ci_structure_check(self):
        program = "import runpy,sys; m=runpy.run_path(sys.argv[1]); m['check_package'](m['Path'](sys.argv[2]))"
        run = subprocess.run([
            sys.executable, "-I", "-B", "-c", program,
            str(OFFICIAL_REPO / ".github/scripts/task_structure.py"), str(self.package),
        ], cwd=OFFICIAL_REPO, capture_output=True, text=True, timeout=90)
        self.assertEqual(run.returncode, 0, msg=run.stdout + run.stderr)

    def test_reference_protocol_consistency(self):
        self.assertEqual(len(self.labels), self.protocol["count"])
        self.assertEqual(sha256_file(self.labels_path).hex(), self.protocol["label_sha256"])
        self.assertEqual(self.protocol["id_column"], self.id_column)
        self.assertEqual(self.protocol["prediction_column"], self.prediction_column)
        self.assertEqual(self.protocol["metric"], self.spec["metric"])
        for row in self.labels:
            columns = ("e_form_ref_eV_per_atom", "hull_ref_eV_per_atom", "d_true_eV_per_atom") if self.task_name == "crystal_stability" else (self.prediction_column,)
            for column in columns:
                value = float(row[column])
                self.assertTrue(math.isfinite(value))
                if self.spec["nonnegative"]:
                    self.assertGreaterEqual(value, 0)
            if self.task_name == "crystal_stability":
                self.assertAlmostEqual(float(row["e_form_ref_eV_per_atom"]) - float(row["hull_ref_eV_per_atom"]), float(row["d_true_eV_per_atom"]), places=12)

    def test_template_has_exact_ids_and_empty_predictions(self):
        path = self.package / "input/submission.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, self.fields)
            rows = list(reader)
        self.assertEqual({r[self.id_column] for r in rows}, set(self.ids))
        self.assertEqual(len(rows), len(self.labels))
        self.assertTrue(all(r[self.prediction_column] == "" for r in rows))

    def test_structure_rows_and_dimensions(self):
        if self.task_name == "crystal_stability":
            path = self.package / "input/test_structures.jsonl.gz"
            expected_fields = {"material_id", "atomic_numbers", "positions_angstrom", "cell_angstrom", "pbc"}
        elif self.task_name == "mof_adsorption":
            path = self.package / "input/test.jsonl.gz"
            expected_fields = {"system_id", "framework_id", "parent_framework_id", "atomic_numbers", "positions_angstrom", "cell_angstrom", "pbc", "source_atom_tags", "nco2", "nh2o"}
        else:
            path = self.package / "input/structures.jsonl.gz"
            expected_fields = {"material_id", "atomic_numbers", "positions_angstrom", "cell_angstrom", "pbc", "temperature_K", "fc2_supercell", "fc3_supercell", "primitive_matrix", "q_point_mesh"}
        seen = set()
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                self.assertEqual(set(row), expected_fields)
                identifier = row[self.id_column]
                self.assertNotIn(identifier, seen)
                seen.add(identifier)
                numbers, positions, cell = row["atomic_numbers"], row["positions_angstrom"], row["cell_angstrom"]
                self.assertTrue(numbers)
                self.assertEqual(len(numbers), len(positions))
                self.assertTrue(all(isinstance(number, int) and not isinstance(number, bool) and 1 <= number <= 118 for number in numbers))
                self.assertEqual(len(cell), 3)
                self.assertTrue(all(len(vector) == 3 for vector in cell + positions))
                self.assertTrue(all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for vector in cell + positions for value in vector))
                self.assertEqual(len(row["pbc"]), 3)
                self.assertTrue(all(isinstance(value, bool) for value in row["pbc"]))
                if self.task_name == "thermal_conductivity":
                    self.assertEqual(row["temperature_K"], 300.0)
                    for name in ("fc2_supercell", "fc3_supercell", "primitive_matrix"):
                        matrix = row[name]
                        if name == "primitive_matrix" and matrix == "auto":
                            continue
                        self.assertEqual(len(matrix), 3)
                        self.assertTrue(all(len(vector) == 3 for vector in matrix))
                        self.assertTrue(all(math.isfinite(value) for vector in matrix for value in vector))
                        if name != "primitive_matrix":
                            self.assertTrue(all(isinstance(value, int) and not isinstance(value, bool) for vector in matrix for value in vector))
                    self.assertEqual(len(row["q_point_mesh"]), 3)
                    self.assertTrue(all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in row["q_point_mesh"]))
                if self.task_name == "mof_adsorption":
                    tags = row["source_atom_tags"]
                    self.assertEqual(len(tags), len(numbers))
                    self.assertTrue(all(isinstance(tag, int) and not isinstance(tag, bool) and tag in (0, 1, 2) for tag in tags))
                    self.assertEqual(row["pbc"], [True, True, True])
                    for count in (row["nco2"], row["nh2o"]):
                        self.assertIsInstance(count, int)
                        self.assertNotIsInstance(count, bool)
                        self.assertGreaterEqual(count, 0)
                    co2 = [number for number, tag in zip(numbers, tags) if tag == 1]
                    h2o = [number for number, tag in zip(numbers, tags) if tag == 2]
                    self.assertEqual(sorted(co2), sorted([6, 8, 8] * row["nco2"]))
                    self.assertEqual(sorted(h2o), sorted([1, 1, 8] * row["nh2o"]))
        self.assertEqual(seen, set(self.ids))

    def test_known_perfect_score(self):
        self.write_rows()
        result, _ = self.run_test()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["primary_metric"]["value"], 1.0 if self.spec["direction"] == "maximize" else 0.0)

    def test_independent_nontrivial_scientific_score(self):
        if self.task_name == "thermal_conductivity":
            rows = [[r[self.id_column], format(2 * float(r[self.prediction_column]), ".17g")] for r in self.labels]
            expected = 2.0 / 3.0
        elif self.task_name == "crystal_stability":
            rows = [[r[self.id_column], format(float(r["hull_ref_eV_per_atom"]) - 1, ".17g")] for r in self.labels]
            positive = sum(float(r["d_true_eV_per_atom"]) <= 0 for r in self.labels)
            expected = 2 * positive / (len(rows) + positive)
        elif self.spec["metric"] == "macro_group_mae":
            groups = sorted({r["group"] for r in self.labels})
            self.assertEqual(len(groups), 3)
            offsets = {group: i + 1 for i, group in enumerate(groups)}
            rows = [[r[self.id_column], format(float(r[self.prediction_column]) + offsets[r["group"]], ".17g")] for r in self.labels]
            expected = 2.0
        else:
            rows = [[r[self.id_column], format(float(r[self.prediction_column]) + 2.0, ".17g")] for r in self.labels]
            expected = 2.0
        self.write_rows(rows)
        result, _ = self.run_test()
        self.assertTrue(result["valid"], result)
        self.assertAlmostEqual(result["primary_metric"]["value"], expected, places=12)

    def test_scientific_boundary_and_absolute_error_semantics(self):
        if self.task_name == "thermal_conductivity":
            rows = [[row[0], "0"] for row in self.base_rows]
            expected = 2 * sum(float(r[self.prediction_column]) > 0 for r in self.labels) / len(self.labels)
        elif self.task_name == "crystal_stability":
            # Exactly on the frozen hull must count as stable (<=, not <).
            rows = [[r[self.id_column], r["hull_ref_eV_per_atom"]] for r in self.labels]
            positive = sum(float(r["d_true_eV_per_atom"]) <= 0 for r in self.labels)
            expected = 2 * positive / (len(rows) + positive)
        else:
            # Negative residuals must not become a signed negative MAE.
            rows = [[r[self.id_column], format(float(r[self.prediction_column]) - 2.0, ".17g")] for r in self.labels]
            expected = 2.0
        self.write_rows(rows)
        result, _ = self.run_test()
        self.assertTrue(result["valid"], result)
        self.assertAlmostEqual(result["primary_metric"]["value"], expected, places=12)

    def test_repeat_frozen_submission_is_byte_identical(self):
        self.write_rows()
        _, first = self.run_test(result_name="a.json")
        _, second = self.run_test(result_name="b.json")
        self.assertEqual(first, second)

    def test_row_order_does_not_change_score(self):
        self.write_rows()
        first, _ = self.run_test(result_name="a.json")
        self.write_rows(reversed(self.base_rows))
        second, _ = self.run_test(result_name="b.json")
        self.assertEqual(first["primary_metric"]["value"], second["primary_metric"]["value"])

    def test_missing_file_rejected(self):
        self.assert_invalid()

    def test_missing_id_rejected(self):
        self.assert_invalid(self.base_rows[1:])

    def test_duplicate_id_rejected(self):
        self.assert_invalid(self.base_rows + [self.base_rows[0]])

    def test_extra_id_rejected(self):
        self.assert_invalid(self.base_rows + [["unrecognized-test-id", "0"]])

    def test_unknown_replacement_id_rejected(self):
        self.assert_invalid([["unrecognized-test-id", "0"]] + self.base_rows[1:])

    def test_path_traversal_id_is_not_interpreted_as_a_path(self):
        sentinel = self.root / "outside-marker.txt"
        sentinel.write_text("unmodified")
        for identifier in ("../../verifier/test/data/labels.csv", str(sentinel)):
            with self.subTest(identifier=identifier):
                self.assert_invalid([[identifier, "0"]] + self.base_rows[1:])
                self.assertEqual(sentinel.read_text(), "unmodified")

    def test_extra_column_rejected(self):
        self.write_rows([row + ["extra"] for row in self.base_rows], self.fields + ["unexpected"])
        self.assert_invalid()

    def test_nonfinite_and_missing_numeric_values_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity", "", "not-a-number"):
            with self.subTest(value=value):
                self.assert_invalid([[self.base_rows[0][0], value]] + self.base_rows[1:])

    def test_sign_rule(self):
        rows = [[row[0], "-1"] for row in self.base_rows]
        self.write_rows(rows)
        result, _ = self.run_test()
        self.assertEqual(result["valid"], not self.spec["nonnegative"], result)

    def test_finite_extreme_predictions_do_not_overflow(self):
        self.write_rows([[row[0], "1e308"] for row in self.base_rows])
        result, _ = self.run_test()
        self.assertTrue(result["valid"], result)
        self.assertTrue(math.isfinite(result["primary_metric"]["value"]))
        if self.task_name == "thermal_conductivity":
            self.assertLessEqual(result["primary_metric"]["value"], 2.0)
            self.assertGreaterEqual(result["primary_metric"]["value"], 0.0)

    def test_malformed_csv_or_encoding_rejected(self):
        for contents in (b"\xff\xfe", (",".join(self.fields) + '\n"unterminated').encode()):
            with self.subTest(contents=contents):
                (self.submission / "submission.csv").write_bytes(contents)
                self.assert_invalid()

    def test_symlinked_submission_rejected(self):
        destination = self.root / "outside.csv"
        self.write_rows().rename(destination)
        (self.submission / "submission.csv").symlink_to(destination)
        self.assert_invalid()

    def test_fifo_submission_rejected_without_blocking(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("No FIFO support on this host")
        os.mkfifo(self.submission / "submission.csv")
        self.assert_invalid()

    def test_oversize_submission_rejected(self):
        # A sparse file tests the pre-read guard without writing gigabytes.
        cap = int(self.protocol["submission_max_bytes"])
        self.assertLessEqual(cap, 1024 * 1024 * 1024, "Unreasonable cap for these CSV tasks")
        with (self.submission / "submission.csv").open("wb") as stream:
            stream.truncate(cap + 1)
        self.assert_invalid()

    def test_oversize_field_rejected(self):
        self.assert_invalid([["x" * (1024 * 1024), "0"]] + self.base_rows[1:])

    def test_validation_no_scientific_feedback(self):
        for payload in (b"{}", b'{"operation":"describe"}'):
            result = self.run_validation(payload)
            self.assertEqual(result, {"status": "error", "error": "this task has no mid-run check"})

    def test_validation_rejects_malformed_or_unknown_payload(self):
        for payload in (b"[1]", b"{", b'{"operation":"score"}', b'{"operation":"describe","unexpected":1}', b'{"operation":"describe","operation":"describe"}', b'{"submission":"../../verifier/test/data/labels.csv"}', b"[" * 1200 + b"0" + b"]" * 1200):
            with self.subTest(payload=payload):
                self.run_validation(payload)

    def test_validation_works_without_hidden_reference_files(self):
        clone = self.root / "without-references"
        shutil.copytree(self.package / "verifier", clone / "verifier", ignore=shutil.ignore_patterns("data"))
        (clone / "input").mkdir()
        result = self.run_validation(package=clone)
        self.assertEqual(result, {"status": "error", "error": "this task has no mid-run check"})

    def test_declared_framework_run_requires_real_sdk(self):
        probe = subprocess.run([
            self.python, "-B", "-c", "import science_innovation_exam.verify_context",
        ], capture_output=True, text=True, timeout=20)
        if probe.returncode == 0:
            self.skipTest("The real SDK is available; missing-SDK branch does not apply")
        self.write_rows()
        result_path = self.root / "framework-result.json"
        environment = dict(os.environ)
        environment["SIE_VERIFY_CONTEXT"] = str(self.root / "framework-context-not-provided.json")
        run = subprocess.run([
            self.python, "-B", str(self.score_package / "verifier/test/run/main.py"),
            "--input", str(self.package / "input"), "--submission", str(self.submission),
            "--result", str(result_path),
        ], capture_output=True, text=True, timeout=45, env=environment)
        self.assertNotEqual(run.returncode, 0, "A framework-declared run must not silently bypass a missing SDK")
        if result_path.exists():
            self.assertIsNot(json.loads(result_path.read_text()).get("valid"), True)

    def test_reference_digest_corruption_is_infrastructure_failure(self):
        clone = self.root / "corrupted-reference"
        shutil.copytree(self.package / "verifier", clone / "verifier")
        (clone / "input").mkdir()
        damaged = clone / "verifier/test/data/labels.csv"
        with damaged.open("ab") as stream:
            stream.write(b"\n")
        self.write_rows()
        result_path = self.root / "corruption-result.json"
        run = subprocess.run([
            self.python, "-B", str(clone / "verifier/test/run/main.py"),
            "--input", str(clone / "input"), "--submission", str(self.submission),
            "--result", str(result_path),
        ], capture_output=True, text=True, timeout=45)
        self.assertNotEqual(run.returncode, 0)
        if result_path.exists():
            response = json.loads(result_path.read_text())
            self.assertIsNot(response.get("valid"), True)

    def test_agent_input_has_no_reference_file_or_private_subtree(self):
        reference_hash = sha256_file(self.labels_path)
        forbidden_names = {"labels.csv", "reference.csv", "references.csv", "protocol.json", "private"}
        for path in (self.package / "input").rglob("*"):
            self.assertFalse(path.is_symlink(), str(path))
            self.assertNotIn(path.name.lower(), forbidden_names, str(path))
            if path.is_file():
                self.assertNotEqual(sha256_file(path), reference_hash)
                if path.suffix == ".csv" and path.name != "submission.csv":
                    with path.open(newline="") as stream:
                        fields = next(csv.reader(stream), [])
                    for forbidden in ("hull_ref_eV_per_atom", "d_true_eV_per_atom", "e_form_ref_eV_per_atom"):
                        self.assertNotIn(forbidden, fields, str(path))


class JsonResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({"test": test.id(), "status": "failed", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({"test": test.id(), "status": "error", "detail": self._exc_info_to_string(err, test)})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.records.append({"test": test.id(), "status": "skipped", "reason": reason})

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err:
            self.records.append({"test": subtest.id(), "status": "failed", "detail": self._exc_info_to_string(err, test)})


def main():
    global OFFICIAL_REPO, OFFICIAL_SKILL, OFFICIAL_SCHEMA_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, default=OFFICIAL_REPO)
    parser.add_argument("--tasks-root", type=Path, default=REPOSITORY / "submission_bundle/tasks")
    parser.add_argument("--tasks", nargs="+", choices=sorted(SPECS), default=sorted(SPECS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--report", type=Path, default=REPOSITORY / "submission_bundle/evidence/tests/independent_tests.json")
    args = parser.parse_args()
    OFFICIAL_REPO = args.official_repo.resolve()
    OFFICIAL_SKILL = OFFICIAL_REPO / "contributor/build-sie-task-package"
    OFFICIAL_SCHEMA_DIR = OFFICIAL_SKILL / "schemas"
    started = time.monotonic()
    # Source packages stay read-only. Score code/reference files get independent
    # real copies, while large scientific input/environment files are audited
    # directly and protected by complete before/after digests. Every deliberate
    # corruption test makes a separate real copy under its own temporary path.
    def manifest_for_sources():
        manifest = []
        for task_name in args.tasks:
            source = args.tasks_root.resolve() / task_name
            require_materialized_files(source)
            for file in sorted(source.rglob("*")):
                if file.is_file():
                    manifest.append({"path": task_name + "/" + str(file.relative_to(source)), "bytes": file.stat().st_size, "sha256": sha256_file(file).hex()})
        return manifest
    manifest_before = manifest_for_sources()
    with tempfile.TemporaryDirectory(prefix="sie-frozen-scorers-") as temporary:
        snapshot = Path(temporary)
        for task_name in args.tasks:
            source = args.tasks_root.resolve() / task_name
            print(f"Copying isolated scoring snapshot: {task_name}", file=sys.stderr, flush=True)
            shutil.copytree(source / "verifier", snapshot / task_name / "verifier", symlinks=True)
        suite = unittest.TestSuite()
        for task_name in args.tasks:
            cls = type(task_name, (PackageChecks,), {"task_name": task_name, "tasks_root": args.tasks_root.resolve(), "snapshot_root": snapshot, "python": args.python})
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
        runner = unittest.TextTestRunner(verbosity=2, resultclass=JsonResult)
        result = runner.run(suite)
        manifest_after = manifest_for_sources()
        sources_unchanged = manifest_before == manifest_after
        report = {
            "schema_version": "scienceinnovationexam.independent-local-tests.v1",
            "scope": "Official published contract/schema and end-to-end CLI tests with isolated score/reference copies; large inputs/environment audited read-only with complete before/after SHA256 manifests. Not private SDK integration or container execution.",
            "python": args.python, "tasks": args.tasks,
            "official_contract_revision": "2026-09-17",
            "official_repository_commit": subprocess.check_output(["git", "-C", str(OFFICIAL_REPO), "rev-parse", "HEAD"], text=True).strip(),
            "driver_sha256": sha256_file(Path(__file__)).hex(),
            "source_tasks_root": str(args.tasks_root.resolve()),
            "source_files_before": manifest_before,
            "source_files_after": manifest_after,
            "source_files_unchanged": sources_unchanged,
            "tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "passed": result.wasSuccessful() and sources_unchanged, "results": result.records,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if result.wasSuccessful() and sources_unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
