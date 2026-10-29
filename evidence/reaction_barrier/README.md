# Reaction-barrier construction and baseline evidence

This folder accompanies approved Proposal #12. It is reviewer evidence, outside the agent-visible task package.

The source is the complete Transition1x v4 HDF5 release (6,623,929,000 bytes), verified against the publisher MD5 and an independently recorded SHA256. Only final-state data were exported: 9,561 train, 225 development and 287 test reactions, with zero exclusions or split changes. Geometry/energy exports were checked record by record against the official loader. The package has no cross-split elemental-composition or same/reverse-reaction overlap under the documented checks.

The measured CPU reference uses 671 endpoint-only geometric statistics and ExtraTrees with two predeclared configurations. The fitting/selection constructor harness reads private development labels to obtain three predeclared aggregate measurements; it is not presented as an isolated agent run. The same three frozen prediction sets were subsequently replayed through the official validation entrypoint with the real SQLite VerifyContext. They reproduce the feedback and selected configuration within float64 rounding. The agent-facing step script supports selection from those aggregate responses without reading private labels.

The selected model and predictions were frozen before the test-label arithmetic audit. All 287 test IDs are scored. Formal MAE is 0.41108395831066713 eV; a training-global-median predictor gives 0.8001934932388185 eV. This is a reproducible CPU reference, not a claim of current state of the art. NeuralNEB's published 0.23 eV uses a different split and successful-path policy and is not a directly comparable score.

Data are used under the fixed Figshare release's MIT statement. The task preserves attribution and license notices. Geometry/energy source mappings, per-reaction prediction CSVs, raw HDF5, trained-model binaries, environments, SQLite context databases and hidden target tables are intentionally not included here. Baseline code and reports are construction-only and may not be treated as additional research inputs containing test answers.

Contract revision is 2026-09-17. Tests use official framework commit 746dc2d7c0c7c4b979c5c7371edf2e0ad15c7607. Real-SDK tests are local subprocess tests. Controller rollback unit tests replace only the container service and Linux chown operation; neither is a Docker/OCI run. Actual container execution and operator resource limits have not been verified.
