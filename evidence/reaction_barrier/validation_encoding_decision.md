# Frozen validation transport encoding

## Decision and scientific scope

This construction freezes the task-specific JSON representation as exactly one
key, `predictions_csv`, whose value is a UTF-8 CSV text string:

```json
{"predictions_csv":"reaction_id,forward_barrier_eV\nDEVELOPMENT_ID_1,0.5\nDEVELOPMENT_ID_2,-0.1\n"}
```

The identifiers in this illustration are placeholders. A real request must
contain all and only the frozen development IDs, exactly once. The CSV header
is exactly `reaction_id,forward_barrier_eV`; values are finite signed float64
numbers in eV. Negative values are retained. Other JSON keys, filesystem paths,
attachments and partial ID selections are unsupported. The request JSON limit
is 4 MiB, the CSV limit is 2 MiB, and an individual numeric field is at most
128 characters. These are transport/parser bounds, not new scientific scores.
The operator's framework request limit must also accommodate the encoded
payload plus the small outer action envelope; a lower operator limit can
reject the request before the task receives it.

This is a task-local encoding chosen during the authorized package
construction. It is **not** an existing official reaction-barrier JSON schema
and does not change the approved scientific behavior. Discussion 12 already
specifies the two scientific columns, complete predictions, forward electronic
barriers in eV, an isolated development set, at most 20 accepted feedback calls,
rejections without budget consumption, repeated valid requests counting again,
and persistent experiment-local state. The chosen JSON string transports those
same CSV predictions without introducing a new candidate type or measurement.
The approved Proposal is preserved unchanged. The encoding remains visible for
technical and domain review in the task PR before evaluation.

## Basis in the real public framework

Sources are pinned to
`PrismaX-Team/Science-Innovation-Exam-Framework` commit
`746dc2d7c0c7c4b979c5c7371edf2e0ad15c7607`:

- `src/science_innovation_exam/action_client.py` wraps the task JSON in the
  framework envelope with `schema_version`, UUID `request_id`, `action` and
  `payload`.
- `src/science_innovation_exam/actions.py` sends only the envelope's `payload`
  to `Verifier.invoke`. Identical request-ID retries replay the previous
  response; new request IDs invoke the task again.
- `src/science_innovation_exam/verifier.py` writes that task payload to the
  read-only request mount. Validation does not receive a `/submission` mount
  or a generic attachment mechanism.
- `src/science_innovation_exam/verify_context.py` supplies the real private
  SQLite-backed JSON state. The framework creates it; task code uses
  `VerifyContext.current()` through the injected `SIE_VERIFY_CONTEXT`.

Thus the task validation program parses the one-key JSON object directly. It
must not expect the outer action envelope, accept a path into agent-writable
storage, or create its own context directory. No replacement SDK is bundled.

The task-repository contract snapshot is
`be168b96e556ea33939285b68987eb329dfe57b0`, contract revision `2026-09-17`,
in `contributor/build-afe-task-package/verifier-api.md`, `task-contract.md` and
`proposal-mapping.md`. Their rule against inventing an unfrozen protocol is
addressed by explicitly recording this task-local encoding before execution,
with its approved scientific contents and real framework boundary stated.

## Budget and result semantics

A complete valid development prediction consumes one round, including a repeat
of previously submitted predictions made as a new action. A transport retry
using the same framework request ID is replayed before task execution and is
not a second scientific call. Malformed requests, incomplete IDs and exhausted
budget return a nonempty error without increasing the accepted-round history.

Accepted responses have `status: "ok"`, `round`, `remaining`, `n_samples` and
the aggregate development metrics `mae_eV`, `rmse_eV`,
`p95_absolute_error_eV`, `within_1_kcal_per_mol_fraction`. They contain no
per-reaction reference values or test metrics. Rejected responses have exactly
`status: "error"` and a nonempty `error` string.

The final four-file submission is distinct from this transport encoding.
Only the final test writes the official raw score after the framework freezes
the submission. The agent receives no test score for revision/resubmission.
The framework may repeat identical frozen final input to check determinism.
