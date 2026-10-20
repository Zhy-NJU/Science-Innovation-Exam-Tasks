# Executed reference model

The primary reproducible reference predicts the median training adsorption energy separately for each `(nco2, nh2o)` pair. A global training median is the fallback for an unseen pair. No development or test target is used to fit the model, choose the groups or tune any parameter. The median is a natural constant predictor under absolute error.

`baseline.py` reads only the published train and test input files. It does not read the hidden labels, source mapping or original energy table. Its prediction CSV must contain all 1,768 test IDs exactly once. The build process also independently computes the same predictor; `check_mof_inputs.py` verifies byte-identical predictions.

From this evidence directory, run:

```sh
python3 baseline.py --input-dir ../../tasks/mof_adsorption/input --output baseline/submission.csv
python3 check_mof_inputs.py
```

The baseline reads every file in `input/train/part-*.jsonl.gz` in sorted filename order and asserts exactly 162,224 training examples. `baseline_execution.json` records the actual standalone run; `input_readback_audit.json` records its independently checked output and score. The training shards are a lossless storage transformation verified in `train_sharding_audit.json`.

This is a simple statistical reference, not a state-of-the-art model or an estimate of the best attainable result. The reproducible metric, group medians, sample counts and output hashes are recorded in the execution and readback JSON files. The primary meta score must equal the final common-verifier result for this exact CSV.

The independently executed reference predicts all 1,768 systems and yields MAE **0.22528328096993902 eV** under the divide-before-`math.fsum` aggregation used by the verifier. Its CSV SHA256 is `ee560463787d03bd17388ce368851642bf2695f0be268a793ba34aebd3027c5f`. No fallback was needed because all test guest-count combinations occur in training.

Historical context only: ODAC23 Supplementary Table S9 reports 0.179 eV for Relaxation EquiformerV2-large on test-ood(b). That literature score was not rerun here and uses resources and training conditions beyond this package's statistical baseline. It is not the primary score and must not be presented as a new run, a latest global frontier, or a fair comparison under identical allowed resources. Source: https://arxiv.org/html/2311.00341v2

Legacy-label caveat: https://arxiv.org/html/2508.03162v1, section 2.1.3. This task intentionally retains the older labels; its scores should not be compared directly to corrected ODAC25 labels.
