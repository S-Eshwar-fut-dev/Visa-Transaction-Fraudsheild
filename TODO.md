# TODO: Fix XGBoost Training Issues

- [ ] Modify `_objective` method: tune `num_boost_round` instead of `n_estimators`, add `feature_names` to DMatrix, change to accept pre-created DMatrix objects for caching
- [ ] Update `tune_hyperparameters` method: create DMatrix with `feature_names`, clamp `scale_pos_weight` to max 50, pass DMatrix to `_objective`
- [ ] Update `train` method: add `feature_names` to DMatrix, remove `n_estimators` from params
- [ ] Fix `find_optimal_threshold` method: correct off-by-one error in threshold indexing
- [ ] Fix `evaluate_model` function: add check for positives in ring_mask
- [ ] Update `train_and_eval` function: correct feature importance extraction and save model using `model.save_model` in JSON format
