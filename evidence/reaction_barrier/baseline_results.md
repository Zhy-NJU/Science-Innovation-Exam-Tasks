# Reproduced endpoint-geometry baseline

## Result

The train-only ExtraTrees model selected on development feedback improves full-test MAE over the training-median constant reference by **48.626931%**. All **287** frozen test reactions are included. These are actual newly trained and executed predictions, not a replay of a paper's reported score.

| Method | Validation MAE (eV), 225 reactions | Test MAE (eV), 287 reactions | Test RMSE (eV) | Test 95th-percentile absolute error (eV) | Within 0.0433641 eV |
|---|---:|---:|---:|---:|---:|
| ExtraTrees, leaf 1 | 0.41622476723437213 | Not evaluated | Not evaluated | Not evaluated | Not evaluated |
| ExtraTrees, leaf 3, selected | 0.4140104770301981 | 0.41108395831066713 | 0.6048654308975404 | 1.4043683891082166 | 36/287 (12.543554%) |
| Training-median constant | 0.8102523945783389 | 0.8001934932388185 | 1.035739445532968 | 2.00276454026407 | 11/287 (3.832753%) |

The constant is **3.9035128126706695 eV**, determined only from 9,561 training barriers. Model selection compares two predeclared configurations; leaf 3 has the lower development MAE. The selected model remains fitted on training data only. The test results were calculated only after the selected model and full-test prediction files were frozen. The unselected model was not evaluated on test. No prediction clipping, dropped reaction, or test-driven retuning occurred.

The approved 15% improvement objective would correspond to MAE ≤ **0.349421364564067 eV** relative to this measured baseline; this is a research target, not a demonstrated result. The literature's NeuralNEB score is not an anchor for this protocol.

## Validation-label boundary

Private validation labels are not Agent inputs. The builder oracle harness recorded candidate scores offline. Complete development predictions for both fixed models and the constant are retained under `official_results/selection/validation/`; these support three genuine framework validation requests and reproduction of the same model choice within the 20-request limit. `baseline_agent_steps.py` provides a separate candidate-training/prediction stage with no private-label argument, followed by selection from accepted aggregate `metrics.mae_eV` feedback. Its execution and error handling were tested on synthetic data. The package builder executed the three actual development requests through the real SDK and official validation entrypoint in one context. Their accepted rounds are 1, 2 and 3, leaving 17 calls. `real_feedback_selection_audit.json` reads those exact aggregate responses and independently selects leaf 3 without private labels. The formal scorer uses overflow-safe normalization; its leaf-3 and constant development MAEs differ from the builder sum by only 5.55e-17 and 1.11e-16 eV, respectively, with no effect on selection. The table reports the formal development values.

## Execution and integrity

- Actual host: macOS-14.5-arm64-arm-64bit; Python 3.11.8; NumPy 2.3.2, SciPy 1.17.1, scikit-learn 1.7.2, joblib 1.6.0.
- Both candidate fits plus validation, feature extraction and model storage: **300.211975 s** wall time; feature reading/extraction **24.189116 s**; process peak RSS **595,738,624 bytes**.
- Final 287-row inference: **1.156685 s**; process peak RSS **405,028,864 bytes**.
- Each fit uses 2 tree workers. Prediction accumulation uses 1 worker in a fixed tree order. The timings are host observations, not measurements in an operator Linux image or a full Agent research-budget estimate.
- Selected model SHA256: `a9752f33716b06429da5e88a4f0446e2ac392cb1f4e4ce866646ae02313ef3eb`.
- Baseline test CSV SHA256: `e743e35c3bf499f77e12641326eb2b4eb79eb55fc3779a6c01604f88fc6a1fe1`.
- Constant test CSV SHA256: `2b7ceb636ca97de5e5c7dc56084bd5b3afdfe12f180ac4e7172daf7bb2cf45ad`.
- The script used for real fitting still matches `baseline_predata_plan.json`. Feature extraction ignores all endpoint energy, TS coordinate/energy and source-ID fields. Synthetic invariance checks cover translations, independent orthogonal endpoint transformations, common mapped-atom permutations and preservation of reaction direction.

The independent arithmetic audit checks exact ID sets, finite values, input/output hashes, canonical ordering, and the absence of missing predictions. MAE computed with `math.fsum`, NumPy, and a separate 60-digit Decimal calculation agrees in the final float64 value for both methods. Both actual formal test-entrypoint results match every reported independent metric exactly, cover 287 reactions, and record three prior development calls. Their submitted prediction bytes match the frozen baseline files. The comparison is recorded in `independent_test_audit.json`; the formal result files are under `official_verifier_results/`. This confirms real-SDK host execution, not an operator Linux-container run.

## Scientific limits

The representation summarizes distances and loses some local geometric and stereochemical information. It is a practical CPU reference, not a complete reaction-path model or a state-of-the-art claim. The retained tail errors and fraction within 1 kcal/mol quantify remaining headroom. The target is the released electronic barrier at the dataset's quantum-chemical level, not an experimental activation free energy or rate constant.
