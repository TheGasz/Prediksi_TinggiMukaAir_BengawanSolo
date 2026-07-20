# TMA model handoff

This archive is a reproducible diagnostic handoff for the next model iteration.

Core files:
- manifest.json: run identity, timestamps, status, and stage.
- config.json / runtime_versions.json / input_inventory.json: reproducibility context (includes the exact ensemble member configs).
- data/train_features.* and test_features.*: exact model matrices when enabled. NOTE: if enable_harmonic_feature=True, the harmonic_baseline_mdpl column here is a NaN PLACEHOLDER -- real values are fold-specific and computed on the fly inside run_validation/fit_final_and_submit.
- data/train_sample_diagnostics.*: normalized target, sample weights, and spike flags.
- metrics/oof_predictions.*: leakage-safe ENSEMBLE validation predictions and residuals.
- metrics/validation_metrics.csv: global fold scores (ensemble) and baseline comparison.
- metrics/member_metrics.csv: per-member (CatBoost/LightGBM) fold scores, for diversity diagnostics.
- metrics/fold_station_metrics.csv: failure analysis by station.
- metrics/residual_slices.csv: residuals by station, month, and hour.
- metrics/feature_importance.csv: importance by fold/member; feed this back as feature_importance_source to enable pruning.
- metrics/environment_drift.csv: train-to-test feature distribution shift.
- metrics/final_feature_importance.csv: importance from the final fit, per member.
- recommended_iterations.json: median best-iteration PER ensemble member (not a single number).
- training_progress/: live CatBoost logs and snapshots for every fold/member.
- models/: fold and final CatBoost (.cbm) and LightGBM (.joblib) models.
- submission.csv: exact competition output when final fit ran.
- submission_quantiles.csv: optional CatBoost MultiQuantile prediction intervals, when enable_quantile_model=True.
- run.log / error_traceback.txt: orchestration log and failure context.

Separately, MODE="ablation" and MODE="tune" write lightweight, non-zipped diagnostic
folders (ablation_<timestamp>/, tune_<timestamp>/) next to the handoff runs -- use
those to decide use_spike_weights/prune_features/hyperparameters BEFORE committing to
a full validate/fit run.

Use OOF predictions—not training error—as the primary basis for refinements.
