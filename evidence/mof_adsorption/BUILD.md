# Reconstructing and checking the MOF task

The current source cache and generated payload are materialized under `/Users/daemon1/.codex/artifacts/scienceinnovationexam-2026-10-12`, outside the synchronized Documents folder. The older Documents copies became `dataless` placeholders; this package was reconstructed from a fresh download, not copied from those placeholders.

## Inputs and transformation

`build_mof.py` verifies both source SHA256 hashes and the energy table's Git blob hash before processing. It opens the official 809 MB gzip tar sequentially, extracts at most one relevant LMDB shard at a time, then deletes that shard. It keeps all official train, val and ood_big examples. No random subsampling or custom test split is used. `split_train.py` then losslessly rewrites the training stream as `input/train/part-*.jsonl.gz`, each under 50 MiB, to stay within GitHub's file limit. It checks every row hash in order, complete decompressed-stream SHA256 and exact row count before removing the generated single-file intermediate. `train_sharding_audit.json` preserves both stream hashes, the intermediate hash, per-part hashes and sizes. `train_row_sha256.txt.gz` preserves each original line's SHA256 in order.

The original test LMDBs omit `y_relaxed`. The authors' fixed-commit adsorption-energy table supplies the ood_big targets. Its values match every one of the 166,222 available train/val `y_relaxed` labels exactly. The table does not cover the original three novel-linker/topology test groups proposed earlier; those groups are not part of this constructed task.

`source_id_mapping.jsonl.gz` records the exact source split, archive member, LMDB key, source name and sid behind every exported ID. This mapping, the original energy table and the hidden target CSV must remain inaccessible to evaluated agents. Hashing IDs only makes direct lookup less convenient; it does not guarantee source anonymity or absence of contamination.

The task's test input allowlist is exactly: `system_id`, `framework_id`, `parent_framework_id`, `atomic_numbers`, `positions_angstrom`, `cell_angstrom`, `pbc`, `source_atom_tags`, `nco2`, `nh2o`. Only train and development add `adsorption_energy_eV`. The source tags retain their original meaning: 0 framework, 1 CO2, 2 H2O.

Pristine parent names are derived only for records marked defective, by stripping the final numeric defect-fraction/replicate suffix. There are no cross-split overlaps of raw framework names, derived pristine parent names or exact exported initial geometry. These checks are not a proof of crystallographic or topological uniqueness. The 85 official large-test source framework names correspond to 66 derived pristine parents; neither count should be presented as a topology count.

## Reproduction

Run `build_mof.py` with the Python environment recorded in `construction_environment.json` after placing the archive and energy table in the sibling `sources/mof` cache. The scripts resolve their package location from `__file__`, so the artifact can be moved. Use a fresh output directory when rebuilding. Only construction requires lmdb/PyTorch/PyG/NumPy. The exported input, standalone baseline and input readback audit use Python's standard library.

Run the standalone baseline with explicit `--input-dir` and `--output` paths, then run `check_mof_inputs.py`. The latter independently decodes every exported record, checks exact allowed fields, ID and parent isolation, finite geometry, element/tag/guest stoichiometry, template and hidden-label coverage, target joins to the public source table, and equality of the two separately executed baseline prediction files. Final baseline scoring must use the task's official verifier; tiny floating-point summation differences must not be mistaken for model differences.

`file_hashes.json` records payload sizes and hashes. The construction scripts and scientific audit evidence are package-external; they are not agent inputs or evaluator authorization mechanisms. The organizer must enforce filesystem and network isolation in the evaluation service.

## Scientific scope and limitations

This task freezes ODAC23's legacy full-system adsorption-energy targets. ODAC25 section 2.1.3 documents issues in the earlier empty-MOF reference calculations. We do not silently replace the reference state, claim to predict corrected energies, or claim that a lower MAE establishes a new physical discovery. This is a controlled learning/extrapolation benchmark with an executed statistical reference; further task selection review should consider whether reusing a public static benchmark meets the project's novelty and contamination criteria.

The paper's >500-atom large-MOF designation applies to its framework unit-cell selection. Exported atom counts incorporate supercell/defect/guest choices and overlap the training range. Preserve the official selection rather than filtering exported atom counts. No unseen-linker or unseen-topology claim is made.
