# Validation evidence for three Science Innovation Exam task packages

This is a small reviewer-evidence companion to `thermal_conductivity`,
`crystal_stability` and `mof_adsorption`. It belongs on a separate evidence
branch and is not part of any task PR or agent-visible task input.

## What was tested

`final_package_manifest.json` identifies the original frozen task bytes:
**203 files, 1,403,908,064 logical bytes** across all three tasks. Paths in that
manifest are relative to a directory containing `tasks/`. The recorded
scientific results apply to those bytes, not to arbitrary later revisions.

The original check run used official repository commit
`e07dfbc595cadc42274788c8e8b69bb79eab2d64` and contract revision `2026-09-17`.
The repository was subsequently renamed from
`PrismaX-Team/LiveDiscoveryBench-Tasks` to
`PrismaX-Team/Science-Innovation-Exam-Tasks`. The published schemas, package
checker, structural checker and verifier contract did not change between that
revision and submission base `969a824ab14ce1e549577e90a442fda1c5d9fb2e`.

- All **99 independent local tests** passed, 33 per task, with zero failures,
  errors or skips and unchanged before/after task hashes.
- Official package/schema and structural checks passed. The published
  fallback checker was used; this was not a private framework run.
- Re-scoring the frozen baselines with Python 3.9.6 and 3.11.8 produced
  byte-identical result files.
- Offline wheel reconstruction, hashes, dependency closure and ELF metadata
  were checked for the thermal/crystal research environment.
- Real CPU smoke calculations exercised the supplied CHGNet checkpoint:
  one MgO force-constant/transport workflow and one bounded crystal
  relaxation/formation-energy workflow.

**The 99 tests, structure checks and static wheel audits do not establish
Docker/OCI execution or integration with the private SIE runtime.** Neither
was run. The one-structure smokes are not full-benchmark inference,
convergence studies or accuracy evaluations.

## Recorded baselines

| Task | Main reference | Raw result | Bottom reference |
|---|---|---:|---:|
| Thermal conductivity | Replay of published CHGNet-0.3.0 transport predictions; fixed zero fallback for 16 failed cases | mean SRE `1.6958708753402623`, lower better | All-zero: `2.0` |
| Crystal stability | Replay of published relaxed, MP2020-corrected CHGNet-0.3.0 formation energies on 215,488 entries | F1 `0.611111789635673`, higher better | Constant `-1e6` eV/atom, all stable: `0.2652014652014652` |
| MOF adsorption | Training-only energy medians by `(nco2, nh2o)` over all 162,224 training systems | Whole-system MAE `0.22528328096993902` eV, lower better | All-zero: `0.6565553732351735` eV |

Thermal/crystal main results are **archive replays**, not newly executed full
CHGNet runs. MOF is an actually fitted statistical baseline, not a reproduced
state-of-the-art structure model. Auxiliary composition-ridge and external
EquiformerV3 scores are retained separately and do not replace the main
baseline.

For MOF, the scientific scope is the original ODAC23 `ood_big` split:
162,224 training, 3,998 development and 1,768 test systems, with 85 test
framework names and 66 derived parent names. The task retains the original
whole-system `y_relaxed` labels and their documented empty-MOF reference-state
limitations; it does not establish corrected ground-state energies or
unseen-linker/topology generalization.

## Contents and exclusions

- `evidence/<task>/official_results/{baseline,bottom}/result.json` preserves
  the result paths used by the package metadata.
- `evidence/mof_adsorption/baseline.py` is the original training-only median
  implementation.
- `evidence/tests/` contains the original 99-test report and driver,
  cross-version report/driver, structural check results and offline audits.
- `evidence/construction/scoring_core.py` is an unchanged copy of the
  scoring helper. Task entrypoints use their own packaged copy; this file
  is provided for inspection, not as an alternative hidden-data location.
- Scientific smoke scripts and their small JSON results are included.
  Full arrays in these JSON files are actual one-structure model outputs,
  not the withheld reference labels.
- `copied_files_manifest.json` records each copied file's original
  bundle-relative path, size and SHA256. Copied files are byte-for-byte
  unchanged. Historical reports retain original interpreter/workspace
  paths and historical repository names as execution provenance; reviewers
  need not reproduce those directory names.

This selection excludes per-sample archived prediction CSVs, source-ID mapping
files, source target tables, scientific data shards, model weights, wheel
archives, the large smoke force-constant NPZ, installation logs, proposal
identity fields and account/login material. Some original manifests refer to
omitted artifacts; their inclusion here does not imply that every listed
artifact is present in this evidence branch. No original report was edited
to hide an omission.

`replay_baselines.py` and `reviewer_helper_replay.json` are explicitly new
reviewer additions. They reproduce the original baseline CSV hashes and
scores but do not replace or change the historical reports.

## Reproduce the independent tests

Prepare three local task directories matching `final_package_manifest.json`
and a separate checkout of the official repository. The task PRs remain
separate; combining their task directories in a local verification workspace
does not imply a combined PR.

Use a Python environment with `jsonschema` for the test drivers and
`packaging` for the offline-wheel audit. These are reviewer-harness
dependencies, separate from the pinned scientific environment. Set explicit
paths to override the scripts' historical build-layout defaults:

```sh
evidence_root=/absolute/path/to/evidence-checkout
tasks_root=/absolute/path/to/verification-workspace/tasks
official_root=/absolute/path/to/Science-Innovation-Exam-Tasks
run_root=/absolute/path/to/separate-review-output
mkdir -p "$run_root"

python3 "$evidence_root/evidence/tests/test_packages.py" \
  --official-repo "$official_root" --tasks-root "$tasks_root" \
  --python "$(command -v python3)" \
  --report "$run_root/independent_tests.json"

python3 "$evidence_root/evidence/tests/test_offline_environment.py" \
  --package "$tasks_root/thermal_conductivity" \
  --report "$run_root/thermal_offline_environment.json"
python3 "$evidence_root/evidence/tests/test_offline_environment.py" \
  --package "$tasks_root/crystal_stability" \
  --report "$run_root/crystal_offline_environment.json"
```

The 99-test driver makes isolated scoring/reference copies for destructive
negative tests, reads large task files directly and hashes every original
task file before and after. It does not need the omitted prediction CSVs or
source-ID mappings. The offline audit temporarily reconstructs wheel files,
so allow scratch disk space beyond the size of this evidence checkout.

## Rebuild and re-score the baselines

The new `evidence/tests/replay_baselines.py` helper makes no network requests.
For thermal/crystal replay, obtain these fixed public archives separately and
place them under a local `source_root`. They are reviewer-only resources,
forbidden as research-agent inputs.

| Relative local filename | Official download | Required SHA256 |
|---|---|---|
| `thermal/chgnet.json.gz` | [Thermal CHGNet archive](https://ndownloader.figshare.com/files/68221282) | `a9b23130992988c9541a62635c17ce4b8f4b8cb7da2dfbdcd67a44722d585505` |
| `chgnet-crystal.csv.gz` | [Crystal CHGNet archive](https://ndownloader.figshare.com/files/66646268) | `b04ce3b031827fcd532e22058d06dc77fb4705ce8b12aea54652a14504e41f16` |
| `wbm-summary.csv.gz` | [Frozen WBM subset definition](https://ndownloader.figshare.com/files/64706751) | `adbdd8b24086d4888195d8894c77f6d5fb29ce33bbb5d7d3898b4d79e964dc54` |

The helper reconstructs the original sorted source-ID ordering without
shipping an ID-mapping file, verifies all source checksums, and requires each
rebuilt baseline CSV to match the originally recorded SHA256. For MOF it
executes the unchanged training-only baseline. It scores through the packaged
test entrypoints and checks both main and bottom raw results.

```sh
source_root=/absolute/path/to/local-public-archives
python3 "$evidence_root/evidence/tests/replay_baselines.py" \
  --tasks-root "$tasks_root" --source-root "$source_root" \
  --output-root "$run_root/baseline-replay"

python3 "$evidence_root/evidence/tests/test_cross_version.py" \
  --official-repo "$official_root" --tasks-root "$tasks_root" \
  --evidence-root "$run_root/baseline-replay" \
  --pythons /absolute/path/to/python3.9 /absolute/path/to/python3.11 \
  --report "$run_root/cross_version_smoke.json"
```

To run only MOF, add `--tasks mof_adsorption` to the replay command; no public
archive file is then needed. The required `--source-root` may point to an
empty existing directory. The original cross-version driver expects all three
rebuilt baselines.

For the auxiliary crystal composition-ridge calculation:

```sh
python3 "$evidence_root/evidence/crystal_stability/run_composition_ridge_baseline.py" \
  --input "$tasks_root/crystal_stability/input" \
  --output "$run_root/composition_ridge.csv"
```

## Reproduce the scientific smokes

Use a scientific Python environment matching the package's pinned versions
and the supplied checkpoint. `research_python` below is an explicit path to
that environment; it is not a claim that a container was built. The original
host smokes ran on macOS arm64. The Linux wheel audits are separate.

```sh
research_python=/absolute/path/to/scientific-environment/bin/python
"$research_python" "$evidence_root/evidence/thermal_conductivity/run_chgnet_phonon_smoke.py" \
  --input "$tasks_root/thermal_conductivity/input" \
  --output "$run_root/thermal-smoke" --supercell 2 --qmesh 5
"$research_python" "$evidence_root/evidence/crystal_stability/run_chgnet_relaxation_smoke.py" \
  --input "$tasks_root/crystal_stability/input" \
  --output "$run_root/crystal-smoke" --steps 10 --fmax 0.01
```

The thermal smoke uses a coarse mesh and force-constant supercell; it is not
the converged benchmark calculation. The crystal smoke is a bounded
relaxation and elemental-reference subtraction on one eligible initial
structure; it makes no full-convergence or benchmark-accuracy claim. Both
scripts disable Python socket connection calls; this is not OS-level
network isolation. Full research-environment and runtime validation remains
part of technical review.
