"""Idempotent notebook upgrade for RMSE-aligned early-stopping ablation."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("anchor-with-mediancodex.ipynb")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} block, found {count}")
    return source.replace(old, new, 1)


METRIC_ABLATION_FUNCTION = r'''
def run_metric_ablation(
    train_features: pd.DataFrame,
    columns: list[str],
    config: ModelConfig,
    artifacts: Path,
    ablation_folds: Sequence[tuple] = FOLDS[-2:],
) -> pd.DataFrame:
    """Compare legacy early stopping against RMSE for CatBoost and LightGBM.

    The objective remains Huber. Only the validation metric used to select the
    best iteration changes. All champion data/feature/ensemble toggles are
    locked so this is a clean metric-alignment experiment.
    """
    if config.enable_anchor_feature:
        raise ValueError("metric_ablation requires ENABLE_ANCHOR_FEATURE=False")
    if config.target_mode != "raw_robust":
        raise ValueError("metric_ablation requires TARGET_MODE='raw_robust'")
    if not config.enable_harmonic_feature:
        raise ValueError("metric_ablation requires ENABLE_HARMONIC_FEATURE=True")

    columns = enforce_frozen_anchor_off_columns(train_features, columns, config)
    if columns != list(ANCHOR_OFF_FROZEN_FEATURE_COLUMNS):
        raise AssertionError("metric_ablation must use the exact frozen 84 features")

    cat_member = next(m for m in config.ensemble if m.family == "catboost")
    lgbm_member = next(m for m in config.ensemble if m.family == "lightgbm")
    members = (cat_member, lgbm_member)
    metric_variants = ("current", "rmse")
    result_parts = []

    for metric_variant in metric_variants:
        variant_config = replace(
            config,
            early_stopping_metric=metric_variant,
            target_mode="raw_robust",
            use_spike_weights=False,
            prune_features=False,
            enable_anchor_feature=False,
            enable_learned_ensemble_weights=False,
        )
        validate_early_stopping_metric(variant_config)
        for member in members:
            report, _ = run_validation(
                train_features,
                columns,
                variant_config,
                artifacts,
                folds=ablation_folds,
                members=(member,),
                write_artifacts=False,
            )
            keep = [
                c for c in report.columns
                if c.startswith("model_") or c in {"fold", "target_mode"}
            ]
            summary = report[keep].copy()
            summary["variant"] = f"early_stop_{metric_variant}"
            summary["early_stopping_metric"] = metric_variant
            summary["member"] = member.name
            summary["family"] = member.family
            summary["n_features"] = len(columns)
            summary["baseline_feature_count"] = len(columns)
            summary["anchor_off_frozen_parity"] = True
            summary["use_spike_weights"] = False
            summary["prune_features"] = False
            summary["enable_learned_ensemble_weights"] = False
            result_parts.append(summary)

    result = pd.concat(result_parts, ignore_index=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        artifacts / "ablation_early_stopping_metric.csv", index=False
    )
    write_json(artifacts / "metric_ablation_audit.json", {
        "experiment": "legacy_validation_metric_vs_rmse",
        "objective_locked": "huber",
        "members": [
            {"name": member.name, "family": member.family}
            for member in members
        ],
        "folds": [fold[0] for fold in ablation_folds],
        "feature_count_both_variants": len(columns),
        "feature_parity_passed": len(columns) == 84,
        "metric_mapping": {
            "current": {
                "catboost": "MAE",
                "lightgbm": "objective default (Huber)",
            },
            "rmse": {"catboost": "RMSE", "lightgbm": "rmse"},
        },
        "locked_settings": {
            "target_mode": "raw_robust",
            "use_spike_weights": False,
            "prune_features": False,
            "enable_anchor_feature": False,
            "enable_learned_ensemble_weights": False,
            "enable_harmonic_feature": True,
            "nino_34_included": "nino_34" in columns,
        },
        "only_intended_difference": "early_stopping_metric",
        "target_period_api_used": False,
    })
    LOGGER.info(
        "Early-stopping metric ablation:\n%s",
        result[[
            "fold", "family", "variant", "model_rmse", "model_mae",
            "model_worst_station_mae",
        ]].to_string(index=False),
    )
    return result
'''.strip()


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    # ModelConfig: champion keeps legacy selection until the ablation passes.
    source = "".join(nb["cells"][2]["source"])
    source = replace_once(
        source,
        '''    early_stopping_rounds: int = 200
    verbose_every: int = 100
''',
        '''    early_stopping_rounds: int = 200
    early_stopping_metric: str = "current"  # current | rmse
    verbose_every: int = 100
''',
        "ModelConfig early_stopping_metric",
    )
    nb["cells"][2]["source"] = source.splitlines(keepends=True)

    # Metric validation helper and family-specific metric selection.
    source = "".join(nb["cells"][12]["source"])
    if "def validate_early_stopping_metric(" not in source:
        marker = 'MODEL_TARGET_MODES = ("raw_robust", "harmonic_residual")\n'
        helper = '''EARLY_STOPPING_METRICS = ("current", "rmse")


def validate_early_stopping_metric(config: ModelConfig) -> None:
    if config.early_stopping_metric not in EARLY_STOPPING_METRICS:
        raise ValueError(
            "early_stopping_metric must be one of "
            f"{EARLY_STOPPING_METRICS}, got {config.early_stopping_metric!r}"
        )


'''
        if source.count(marker) != 1:
            raise RuntimeError("Could not place early-stopping metric helper")
        source = source.replace(marker, helper + marker, 1)
    source = replace_once(
        source,
        '''        loss_function=f"Huber:delta={config.huber_delta}",
        eval_metric="MAE",
''',
        '''        loss_function=f"Huber:delta={config.huber_delta}",
        eval_metric=(
            "RMSE" if config.early_stopping_metric == "rmse" else "MAE"
        ),
''',
        "CatBoost early-stopping metric",
    )
    source = replace_once(
        source,
        '''    if config.use_gpu:
        # gpu_device_id may use CatBoost's multi-device syntax ("0:1" for two
''',
        '''    if config.early_stopping_metric == "rmse":
        # Keep the Huber training objective; only align best-iteration selection
        # with the competition metric.
        params["metric"] = "rmse"
    if config.use_gpu:
        # gpu_device_id may use CatBoost's multi-device syntax ("0:1" for two
''',
        "LightGBM early-stopping metric",
    )
    nb["cells"][12]["source"] = source.splitlines(keepends=True)

    # Dedicated two-family metric ablation.
    source = "".join(nb["cells"][18]["source"])
    if "def run_metric_ablation(" not in source:
        marker = "\n\n# Frozen after"
        if source.count(marker) != 1:
            raise RuntimeError("Could not place metric ablation function")
        source = source.replace(
            marker, "\n\n" + METRIC_ABLATION_FUNCTION + marker, 1
        )
    nb["cells"][18]["source"] = source.splitlines(keepends=True)

    # Pipeline mode, validation guard, configuration audit, and routing.
    source = "".join(nb["cells"][26]["source"])
    source = source.replace(
        'if mode not in {"audit", "ablation", "target_ablation", "tune", "p0_experiments", "validate", "fit", "all"}:',
        'if mode not in {"audit", "ablation", "target_ablation", "metric_ablation", "tune", "p0_experiments", "validate", "fit", "all"}:',
    ).replace(
        'raise ValueError("mode must be one of: audit, ablation, target_ablation, tune, p0_experiments, validate, fit, all")',
        'raise ValueError("mode must be one of: audit, ablation, target_ablation, metric_ablation, tune, p0_experiments, validate, fit, all")',
    ).replace(
        '''    validate_model_target_config(config)
    train, test, sample, coordinates, environment = load_inputs(paths)
''',
        '''    validate_model_target_config(config)
    validate_early_stopping_metric(config)
    train, test, sample, coordinates, environment = load_inputs(paths)
''',
    ).replace(
        'if mode in {"ablation", "target_ablation", "tune", "p0_experiments"}:',
        'if mode in {"ablation", "target_ablation", "metric_ablation", "tune", "p0_experiments"}:',
    ).replace(
        '''        if mode == "target_ablation":
            table = run_target_mode_ablation(
''',
        '''        if mode == "metric_ablation":
            table = run_metric_ablation(
                train_features, columns, config, light_dir
            )
            result["metric_ablation_results"] = table
            result["metric_ablation_dir"] = str(light_dir)
        elif mode == "target_ablation":
            table = run_target_mode_ablation(
''',
    ).replace(
        '''        "target_mode": config.target_mode,
        "early_stopping_rounds": config.early_stopping_rounds,
''',
        '''        "target_mode": config.target_mode,
        "early_stopping_rounds": config.early_stopping_rounds,
        "early_stopping_metric": config.early_stopping_metric,
''',
    )
    if "metric_ablation  - legacy validation metric versus RMSE" not in source:
        source = source.replace(
            '''      target_ablation  - raw robust target versus leakage-safe harmonic residual.
                  Writes target_ablation_<timestamp>/ablation_target_residual.csv.
''',
            '''      target_ablation  - raw robust target versus leakage-safe harmonic residual.
                  Writes target_ablation_<timestamp>/ablation_target_residual.csv.
      metric_ablation  - legacy validation metric versus RMSE on CatBoost + LightGBM.
                  Writes metric_ablation_<timestamp>/ablation_early_stopping_metric.csv.
''',
            1,
        )
    nb["cells"][26]["source"] = source.splitlines(keepends=True)

    # Junior-facing flow for the current experiment.
    nb["cells"][27]["source"] = '''## 11. Flow eksperimen early-stopping metric

Champion masih menggunakan objective Huber dan target `raw_robust`. Eksperimen ini hanya menguji metric yang memilih best iteration:

- `early_stop_current`: CatBoost memakai MAE; LightGBM memakai metric default objective Huber.
- `early_stop_rmse`: CatBoost dan LightGBM memakai RMSE untuk early stopping, sementara objective training tetap Huber.

Mode `metric_ablation` otomatis menjalankan satu CatBoost (`cb_shallow`) dan satu LightGBM (`lgbm_a`) pada `sep_2024` dan `jan_2025`. Kedua variant dikunci pada frozen 84 fitur, `nino_34` ON, target raw, spike OFF, pruning OFF, anchor OFF, dan learned weighting OFF.

### Flow junior

1. Upload notebook terbaru dan pilih accelerator GPU seperti run sebelumnya.
2. Restart Session/Kernel.
3. Pastikan configuration cell menunjukkan `MODE="metric_ablation"`.
4. Klik **Run All** satu kali; jangan mengubah `EARLY_STOPPING_METRIC`, karena mode ini otomatis menguji `current` dan `rmse`.
5. Ambil folder `metric_ablation_<timestamp>`.
6. Serahkan `ablation_early_stopping_metric.csv` dan `metric_ablation_audit.json` kepada senior.
7. Jangan menjalankan full validation, fit, atau upload submission sebelum keputusan senior.

Kriteria awal: RMSE harus membaik lintas family/fold tanpa kerusakan material pada MAE, P95, atau worst-station. Jika lolos, barulah senior mengubah `EARLY_STOPPING_METRIC="rmse"` dan meminta full validation empat fold.
'''.splitlines(keepends=True)

    # Safe ready-to-run config.
    source = "".join(nb["cells"][28]["source"])
    source = source.replace(
        'MODE = "target_ablation"  # flow saat ini: raw_robust vs harmonic_residual',
        'MODE = "metric_ablation"  # otomatis menguji current vs RMSE',
    )
    if "EARLY_STOPPING_METRIC =" not in source:
        source = source.replace(
            'TARGET_MODE = "raw_robust"  # champion; target_ablation otomatis menguji residual juga\n',
            'TARGET_MODE = "raw_robust"  # champion\n'
            'EARLY_STOPPING_METRIC = "current"  # mode ablation otomatis menguji current dan rmse\n',
            1,
        )
    if "early_stopping_metric=EARLY_STOPPING_METRIC" not in source:
        source = source.replace(
            "CONFIG = ModelConfig(\n    target_mode=TARGET_MODE,\n",
            "CONFIG = ModelConfig(\n    target_mode=TARGET_MODE,\n"
            "    early_stopping_metric=EARLY_STOPPING_METRIC,\n",
            1,
        )
    source = source.replace(
        'print(f"target_mode={CONFIG.target_mode} use_spike_weights=',
        'print(f"target_mode={CONFIG.target_mode} early_stopping_metric={CONFIG.early_stopping_metric} use_spike_weights=',
        1,
    )
    nb["cells"][28]["source"] = source.splitlines(keepends=True)

    NOTEBOOK.write_text(
        json.dumps(nb, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Updated {NOTEBOOK.resolve()}")


if __name__ == "__main__":
    main()
