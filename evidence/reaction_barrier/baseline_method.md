# Endpoint geometry baseline

This reference method predicts the signed forward electronic barrier in eV from mapped reactant/product geometries only. It does not use endpoint energies, transition-state coordinates, source identifiers, reaction names, or test labels. It is a CPU regression baseline, not a reproduction of NeuralNEB or a claim of state-of-the-art performance.

## Representation fixed before training

The 671 features comprise:

- Total atom count and counts of H, C, N and O (5 features).
- For each of the 10 unordered element pairs: pair count and eight statistics (sum, mean, standard deviation, minimum, 25th percentile, median, 75th percentile, maximum) of seven arrays: reactant distance, product distance, signed distance change, absolute distance change, reactant soft-bond proxy, product soft-bond proxy, and signed proxy change (570 features).
- For each element: the same eight statistics of atomic reaction activity, reactant coordination proxy and product coordination proxy (96 features). Atomic activity is the sum of absolute soft-bond changes over all pairs incident on the atom; coordination is the sum of incident soft-bond proxies.

The soft-bond proxy is `1 / (1 + (distance / scale)**8)`, where `scale = 1.25 * (radius_i + radius_j)`. The heuristic radii in angstrom are fixed at H=0.31, C=0.76, N=0.71 and O=0.66. These are representation constants, not fitted bond-order labels. Empty element groups return zero statistics. There is no fitting of the feature transform and no normalization derived from validation or test data.

Distances make the representation invariant to independently translating or rotating either endpoint. Grouped pair statistics and atomic statistics make it invariant to a common permutation of mapped atoms. Reaction direction is retained. Independent permutations of the two endpoints are not permitted because they would change atom mapping. A synthetic test checks these properties before real-data training; it does not demonstrate chemical accuracy. The representation is also invariant to reflections and therefore cannot distinguish signed stereochemical changes that preserve these distance statistics. It is not a complete representation of a reaction pathway.

## Predeclared fitting and selection

Fit two `ExtraTreesRegressor` configurations using the official training split only:

| Parameter | Candidate 1 | Candidate 2 |
|---|---:|---:|
| n_estimators | 320 | 320 |
| min_samples_leaf | 1 | 3 |
| max_features | 1.0 | 1.0 |
| random_state | 20261030 | 20261030 |
| n_jobs | 2 | 2 |

Other settings retain scikit-learn 1.7.2 defaults, including squared-error splitting, no bootstrap, and unlimited depth. Training uses two tree workers; predictions temporarily use one worker so tree accumulation has a fixed order. Select the lower full-validation MAE; an exact tie selects Candidate 1. This evaluates two candidate models and the constant reference on development labels (three metric evaluations). Keep the selected train-only model without refitting on validation. The model and selection report are hashed before test predictions. No configuration change follows test scoring.

The bottom reference is the global median of training barriers, the optimal training-set constant for absolute error. It is neither fitted on validation nor set from test labels. Neither method clips negative predictions. Both must emit a finite prediction for every final test ID.

The `fit` command has no test argument, and `predict` has no label argument. Prediction input containing `forward_barrier_eV` is rejected. The selected model hash is checked at prediction time. Test predictions are generated before the final test metrics are inspected; final scoring is independent of fitting and uses the task verifier.

Validation labels are private and are not Agent inputs. `baseline.py fit` is a builder-side oracle evaluation harness: it reads those labels to record the two candidate MAEs and the constant-reference MAE. Neither candidate is trained or reconfigured using validation errors. The complete validation prediction CSVs are retained for all three methods so that the official validation entrypoint can independently reproduce the same selection with three aggregate-feedback calls, within the 20-call budget.

For an Agent-compatible execution order, `baseline_agent_steps.py candidates` trains both fixed candidates from public training labels, saves their models, and emits all three validation submissions without opening any validation or test label file. Submit those CSVs to the framework. Then `baseline_agent_steps.py select` accepts only the three aggregate response JSONs and freezes the candidate with the lower `metrics.mae_eV`. It requires `status: "ok"`, the full development sample count, and three distinct accepted rounds with consistent remaining budget. Use `baseline.py predict` with that selection directory for final test predictions. The private-label `fit` command below is solely for constructing and auditing the reference result; it is not a permission for Agents to access the oracle labels.

## Software and execution evidence

The reproducibility run uses Python 3.11 on a macOS CPU host with NumPy 2.3.2, SciPy 1.17.1, scikit-learn 1.7.2 and joblib 1.6.0, matching the documented operator-base versions of these libraries. Host-only supporting pins are threadpoolctl 3.6.0 and cloudpickle 3.1.2. Scientific fitting limits thread pools and tree parallelism to 2. This is not evidence of an executed operator Linux image.

`official_results/selection/predeclared_search.json` records the search before labels are read. `selection.json` records input hashes, software versions, selected parameters, validation MAE, elapsed time, and process peak RSS. `prediction_provenance.json` records the frozen model/selection and input/output hashes, inference time and peak RSS. Process peak RSS includes Python imports and preceding work in that process; macOS measurements are converted directly from bytes, Linux measurements from KiB.

## Commands

Builder-side audit commands, from the reaction task construction directory, using a Python environment with the pinned libraries:

```sh
python build/baseline.py selftest --output build/invariance_selftest.json
python build/baseline.py fit \
  --train task_package/input/train.jsonl.gz \
  --validation task_package/input/validation.jsonl.gz \
  --validation-labels task_package/verifier/test/data/validation_labels.csv \
  --output build/official_results/selection
python build/baseline.py predict \
  --model-dir build/official_results/selection \
  --input task_package/input/test.jsonl.gz \
  --output build/official_results
```

Baseline code and trained artifacts are construction evidence and are excluded from the Agent input directory. The actual split counts are 9,561 training, 225 development and 287 test reactions. Full results are recorded in `baseline_results.md`: the selected CPU baseline's test MAE is 0.41108395831066713 eV, compared with 0.8001934932388185 eV for the training-median reference. The real SDK and official entrypoints reproduce all final metrics. This is a measured package reference, not a state-of-the-art claim. NeuralNEB's 0.23 eV result uses different splits and converged-path filtering and cannot be treated as a comparable score.

Agent-compatible staged commands (framework feedback files are produced by three official validation calls between the two stages):

```sh
python build/baseline_agent_steps.py candidates \
  --train task_package/input/train.jsonl.gz \
  --validation task_package/input/validation.jsonl.gz \
  --output candidates
python build/baseline_agent_steps.py select \
  --candidate-dir candidates \
  --leaf1-feedback feedback/et_leaf1.json \
  --leaf3-feedback feedback/et_leaf3.json \
  --bottom-feedback feedback/bottom.json \
  --output selection
python build/baseline.py predict \
  --model-dir selection --input task_package/input/test.jsonl.gz \
  --output final_predictions
```

The staged commands produce predictions and model-selection evidence; the final submission must also include the task's required `method.md`, `research_code.zip` and `sources.json`. The construction's complete baseline and constant submissions are preserved under `build/official_verifier_results/`.

To audit the selection using the three real aggregate responses already recorded by the builder, without opening private labels or changing the model:

```sh
python build/baseline_agent_steps.py verify-feedback \
  --frozen-selection build/official_results/selection/selection.json \
  --leaf1-feedback build/official_verifier_results/et_leaf1_response.json \
  --leaf3-feedback build/official_verifier_results/et_leaf3_response.json \
  --bottom-feedback build/official_verifier_results/bottom_response.json \
  --output build/official_results/real_feedback_selection_audit.json
```

This check passed and selected leaf 3 from accepted rounds 1, 2 and 3, all with 225 reactions. The formal scorer's overflow-safe MAE calculation produces at most 1.11e-16 eV difference from the builder's direct sum on these development submissions. This numerical difference does not affect the choice. The independent full-test MAE, RMSE, 95th percentile and threshold fraction all exactly match the formal results for both methods.
