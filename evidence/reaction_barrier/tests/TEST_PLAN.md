# Independent reaction-barrier verification plan

## Evidence levels

1. **Official contract audit**: task repository `be168b96e556ea33939285b68987eb329dfe57b0`, contract revision 2026-09-17; approved Discussion 12 retained unchanged under `proposal/`.
2. **Real SDK local execution**: public framework `746dc2d7c0c7c4b979c5c7371edf2e0ad15c7607`, using its actual `science_innovation_exam.verify_context` module. Six independent SDK tests already pass; upstream SDK unit tests also pass (5/5). These do not establish container isolation or complete-run rollback.
3. **Task subprocess + real SDK integration**: the task-local `predictions_csv` encoding and scorer are frozen for testing. The harness uses new processes sharing an officially created context, not a replacement SDK. It initializes contexts with the official `VerifyContext.create`; task code only opens `VerifyContext.current()`. Synthetic boundary tests and the final full-data audit have separately named reports.
4. **Framework actions**: separately exercise actual request-ID replay/terminal behavior. Any non-container adapter used by this harness must be labeled as a local transport adapter, not an official container runtime.
5. **Controller rollback unit tests**: `test_controller_rollback.py` calls the actual official `Verifier.invoke` with the real SQLite SDK. Container service and Linux ownership setup are explicit test doubles. All five rollback/commit cases pass; this is not a test of container launch, timeout enforcement or full scientific-task execution.
6. **Docker end-to-end execution**: not available on this host. Do not infer it from levels 1–5.

## Confirmed public request boundary

The action client creates `{schema_version: "1.0", request_id: UUID, action: "validation", payload: OBJECT}`. The task's `--request` file contains OBJECT itself, not this outer envelope. `actions.py` forwards only `request["payload"]`; `verifier.py` writes it to the request mount. There is no attachment or submission-directory mount for validation.

The framework owns serialization and request-ID retries. Identical scientific predictions in a **new** action must consume one additional accepted-feedback round. Replaying the same outer request ID and identical content returns the previous action response without invoking validation again. Reusing the same request ID with different content is rejected by the framework. These are distinct cases.

The scientific schema is task-owned. The approved CSV columns and full-development-set requirement constrain its contents but do not specify a JSON field name or encoding. The final representation is recorded in `../validation_encoding_decision.md`: the exact one-key object `{"predictions_csv": "..."}`. It is a task-local encoding, not an existing official reaction schema. Do not introduce partial subsets, test-set evaluation, alternate units, filesystem references or configurable budgets.

## Independent scientific fixtures

- Final labels are frozen `E_TS - E_R`, forward direction only, in eV; verify source extraction separately from scoring.
- For isolated scoring fixtures, replace reference files only in scratch copies with small explicit values and recompute the scratch integrity digest. Never alter frozen shared packages.
- Exact/perfect predictions: MAE/RMSE/P95 zero, within-threshold fraction one.
- Constant and nonuniform known errors: independently calculate MAE, stable RMSE, the explicitly documented P95 convention and fraction at or below 0.0433641 eV.
- Boundary errors immediately below/at/above 0.0433641 eV, including count denominators.
- Negative finite predictions/references remain signed and score normally; no zero clipping.
- Very large finite predictions must receive finite mathematically valid scores when the float64 metric is representable. Use scaled calculations to avoid accidental overflow in squares or sums.
- Shuffling input rows leaves scores unchanged; frozen reference-ID order and summation order determine the result.

## Artifact and parser rejection

- Required main/auxiliary artifacts missing; malformed method/source manifests; unsafe code/model artifacts; extra files inside declared directory artifacts according to the frozen policy.
- Header mismatch, unknown/duplicate/missing IDs, duplicate columns, surplus/short cells, blank values, whitespace rules, malformed quoting, invalid UTF-8, BOM policy, embedded NUL, NaN/Inf, numeric booleans if JSON rows are chosen.
- Oversized request/file/numeric field, deeply nested or unexpected JSON, duplicate JSON keys, non-object payloads, unknown payload fields and protocol versions.
- Symlink and FIFO rejection without blocking; paths used as identifiers remain identifiers and cannot cause file reads or directory traversal.
- No unbounded read before file/type/size checks. Tests use subprocess deadlines.

## Twenty-round development budget

- Fresh experiment starts with zero accepted rounds.
- Valid complete development predictions consume exactly one round and return only the allowed development measurements.
- Repeat identical valid predictions under new requests through rounds 1–20; every call counts. The 21st valid request is explicitly rejected without new feedback or state changes.
- Interleave malformed/nonfinite/missing-ID requests; none consume accepted rounds, including before round 1, at round 19 and after exhaustion.
- Persist and reopen the real SDK in every subprocess; a fresh independent experiment remains isolated.
- Client fields cannot set/reset the count, context root, task identity or history. Inconsistent/corrupt trusted state must fail closed, not silently reset or grant new budget.
- Inject a task error after state write in a dedicated harness to distinguish SDK atomic writes from actual framework whole-invocation rollback. Only claim automatic rollback if the real `Verifier.invoke` lifecycle is exercised.

## Final test and information boundaries

- Test receives the complete frozen final artifacts and writes the official result schema; development responses never overwrite that official result.
- Repeat identical frozen test inputs: byte-identical or value-identical result, including when prior validation history exists. Framework permits this determinism check while providing no second agent submission.
- Development and final labels/IDs are disjoint. Development validation rejects test IDs and must not read final label files.
- Test input contains only allowed endpoints/fields, with no transition-state coordinates or test energies. Audit full exported records, not only sample rows or filenames.
- Responses/logs contain no per-sample hidden test answers, source mappings, test aggregates or state internals beyond explicitly allowed development feedback. Errors must not echo hidden reference values.
- Final result identity/validity/error/finite unique metrics match the official schema and independent framework validator.

## Recording

Hash every source task file before and after tests. Use real copies for negative/corruption tests; never mutate a shared original or hardlinked scoring reference. Store scripts, source commit/hash, test counts and scope under `build/tests/`. Preserve failures and distinguish local SDK/task tests, framework action tests and real Docker runs. Do not rewrite the approved proposal or label a dictionary-only unit test as SDK integration.
