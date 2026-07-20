"""Idempotent notebook upgrade for LightGBM Huber-alpha ablation."""

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


ALPHA_ABLATION_FUNCTION = r'''
def run_lgbm_alpha_ablation(
    train_features: pd.DataFrame,
    columns: list[str],
    config: ModelConfig,
    artifacts: Path,
    alpha_values: Sequence[float] = (0.7, 0.9, 1.2, 1.5),
    ablation_folds: Sequence[tuple] = FOLDS[-2:],
    ablation_member: EnsembleMemberConfig | None = None,
) -> pd.DataFrame:
    """Tune only LightGBM's Huber alpha; CatBoost delta is untouched."""
    if config.enable_anchor_feature:
        raise ValueError("lgbm_alpha_ablation requires ENABLE_ANCHOR_FEATURE=False")
    if config.target_mode != "raw_robust":
        raise ValueError("lgbm_alpha_ablation requires TARGET_MODE='raw_robust'")
    if config.early_stopping_metric != "current":
        raise ValueError(
            "lgbm_alpha_ablation requires EARLY_STOPPING_METRIC='current'"
        )
    if not config.enable_harmonic_feature:
        raise ValueError("lgbm_alpha_ablation requires harmonic feature ON")

    columns = enforce_frozen_anchor_off_columns(train_features, columns, config)
    if columns != list(ANCHOR_OFF_FROZEN_FEATURE_COLUMNS):
        raise AssertionError(
            "lgbm_alpha_ablation must use the exact frozen 84 features"
        )

    member = ablation_member or next(
        m for m in config.ensemble if m.name == "lgbm_a"
    )
    if member.family != "lightgbm":
        raise ValueError("lgbm_alpha_ablation member must be LightGBM")

    alpha_values = tuple(float(value) for value in alpha_values)
    if not alpha_values or any(value <= 0 for value in alpha_values):
        raise ValueError("Every LightGBM Huber alpha must be positive")
    if len(set(alpha_values)) != len(alpha_values):
        raise ValueError("LightGBM Huber alpha grid contains duplicates")

    result_parts = []
    for alpha in alpha_values:
        variant_config = replace(
            config,
            lightgbm_huber_alpha=alpha,
            early_stopping_metric="current",
            target_mode="raw_robust",
            use_spike_weights=False,
            prune_features=False,
            enable_anchor_feature=False,
            enable_learned_ensemble_weights=False,
        )
        validate_lightgbm_huber_alpha(variant_config)
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
            if c.startswith("model_")
            or c in {"fold", "target_mode", "best_iteration"}
        ]
        summary = report[keep].copy()
        summary["variant"] = f"lgbm_huber_alpha_{alpha:g}"
        summary["lightgbm_huber_alpha"] = alpha
        summary["catboost_huber_delta_unchanged"] = config.huber_delta
        summary["early_stopping_metric"] = "current"
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
        artifacts / "ablation_lightgbm_huber_alpha.csv", index=False
    )
    write_json(artifacts / "lgbm_alpha_ablation_audit.json", {
        "experiment": "lightgbm_huber_alpha_grid",
        "objective_locked": "huber",
        "member": {"name": member.name, "family": member.family},
        "alpha_values": list(alpha_values),
        "baseline_alpha": 1.5,
        "catboost_huber_delta_unchanged": config.huber_delta,
        "folds": [fold[0] for fold in ablation_folds],
        "feature_count_all_variants": len(columns),
        "feature_parity_passed": len(columns) == 84,
        "locked_settings": {
            "target_mode": "raw_robust",
            "early_stopping_metric": "current",
            "use_spike_weights": False,
            "prune_features": False,
            "enable_anchor_feature": False,
            "enable_learned_ensemble_weights": False,
            "enable_harmonic_feature": True,
            "nino_34_included": "nino_34" in columns,
        },
        "only_intended_difference": "lightgbm_huber_alpha",
        "best_iteration_recorded_per_fold": "best_iteration" in result.columns,
        "target_period_api_used": False,
    })
    LOGGER.info(
        "LightGBM Huber-alpha ablation:\n%s",
        result[[
            "fold", "lightgbm_huber_alpha", "best_iteration",
            "model_rmse", "model_mae", "model_worst_station_mae",
        ]].to_string(index=False),
    )
    return result
'''.strip()


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    # Separate family-specific Huber parameters.
    source = "".join(nb["cells"][2]["source"])
    source = replace_once(
        source,
        '''    huber_delta: float = 1.5
    early_stopping_rounds: int = 200
''',
        '''    huber_delta: float = 1.5  # CatBoost Huber delta
    lightgbm_huber_alpha: float = 1.5
    early_stopping_rounds: int = 200
''',
        "family-specific Huber config",
    )
    nb["cells"][2]["source"] = source.splitlines(keepends=True)

    # Guard and actual LightGBM parameter wiring.
    source = "".join(nb["cells"][12]["source"])
    if "def validate_lightgbm_huber_alpha(" not in source:
        marker = 'EARLY_STOPPING_METRICS = ("current", "rmse")\n'
        helper = '''def validate_lightgbm_huber_alpha(config: ModelConfig) -> None:
    if not np.isfinite(config.lightgbm_huber_alpha):
        raise ValueError("lightgbm_huber_alpha must be finite")
    if config.lightgbm_huber_alpha <= 0:
        raise ValueError("lightgbm_huber_alpha must be positive")


'''
        if source.count(marker) != 1:
            raise RuntimeError("Could not place LightGBM alpha guard")
        source = source.replace(marker, marker + "\n\n" + helper, 1)
    source = replace_once(
        source,
        '''        objective="huber",
        alpha=config.huber_delta,  # same Huber delta as the CatBoost members
''',
        '''        objective="huber",
        alpha=config.lightgbm_huber_alpha,
''',
        "LightGBM alpha wiring",
    )
    nb["cells"][12]["source"] = source.splitlines(keepends=True)

    # Make single-member validation expose fold-level best_iteration.
    source = "".join(nb["cells"][16]["source"])
    source = replace_once(
        source,
        '''        member_preds_z: list[np.ndarray] = []
        member_preds_original: list[np.ndarray] = []
        for member in members:
''',
        '''        member_preds_z: list[np.ndarray] = []
        member_preds_original: list[np.ndarray] = []
        fold_best_iterations: dict[str, int] = {}
        for member in members:
''',
        "fold best-iteration initialization",
    )
    source = replace_once(
        source,
        '''            best_iterations[member.name].append(best_iteration)
            pred_z_member = predict_member(model, member, valid, columns, cat_columns)
''',
        '''            best_iterations[member.name].append(best_iteration)
            fold_best_iterations[member.name] = int(best_iteration)
            pred_z_member = predict_member(model, member, valid, columns, cat_columns)
''',
        "fold best-iteration capture",
    )
    source = replace_once(
        source,
        '''            "target_mode": config.target_mode,
            "ensemble_members": ",".join(member_names),
''',
        '''            "target_mode": config.target_mode,
            "best_iteration": (
                fold_best_iterations[member_names[0]]
                if len(member_names) == 1 else np.nan
            ),
            "ensemble_members": ",".join(member_names),
''',
        "fold best-iteration report",
    )
    nb["cells"][16]["source"] = source.splitlines(keepends=True)

    # Dedicated alpha-grid function.
    source = "".join(nb["cells"][18]["source"])
    if "def run_lgbm_alpha_ablation(" not in source:
        marker = "\n\n# Frozen after"
        if source.count(marker) != 1:
            raise RuntimeError("Could not place LightGBM alpha ablation")
        source = source.replace(
            marker, "\n\n" + ALPHA_ABLATION_FUNCTION + marker, 1
        )
    nb["cells"][18]["source"] = source.splitlines(keepends=True)

    # Pipeline validation, routing, and reproducibility metadata.
    source = "".join(nb["cells"][26]["source"])
    source = source.replace(
        'if mode not in {"audit", "ablation", "target_ablation", "metric_ablation", "tune", "p0_experiments", "validate", "fit", "all"}:',
        'if mode not in {"audit", "ablation", "target_ablation", "metric_ablation", "lgbm_alpha_ablation", "tune", "p0_experiments", "validate", "fit", "all"}:',
    ).replace(
        'raise ValueError("mode must be one of: audit, ablation, target_ablation, metric_ablation, tune, p0_experiments, validate, fit, all")',
        'raise ValueError("mode must be one of: audit, ablation, target_ablation, metric_ablation, lgbm_alpha_ablation, tune, p0_experiments, validate, fit, all")',
    ).replace(
        '''    validate_early_stopping_metric(config)
    train, test, sample, coordinates, environment = load_inputs(paths)
''',
        '''    validate_early_stopping_metric(config)
    validate_lightgbm_huber_alpha(config)
    train, test, sample, coordinates, environment = load_inputs(paths)
''',
    ).replace(
        'if mode in {"ablation", "target_ablation", "metric_ablation", "tune", "p0_experiments"}:',
        'if mode in {"ablation", "target_ablation", "metric_ablation", "lgbm_alpha_ablation", "tune", "p0_experiments"}:',
    ).replace(
        '''        if mode == "metric_ablation":
            table = run_metric_ablation(
''',
        '''        if mode == "lgbm_alpha_ablation":
            table = run_lgbm_alpha_ablation(
                train_features, columns, config, light_dir
            )
            result["lgbm_alpha_ablation_results"] = table
            result["lgbm_alpha_ablation_dir"] = str(light_dir)
        elif mode == "metric_ablation":
            table = run_metric_ablation(
''',
    ).replace(
        '''        "huber_delta": config.huber_delta,
        "target_mode": config.target_mode,
''',
        '''        "huber_delta": config.huber_delta,
        "lightgbm_huber_alpha": config.lightgbm_huber_alpha,
        "target_mode": config.target_mode,
''',
    )
    if "lgbm_alpha_ablation  - LightGBM Huber-alpha grid" not in source:
        source = source.replace(
            '''      metric_ablation  - legacy validation metric versus RMSE on CatBoost + LightGBM.
                  Writes metric_ablation_<timestamp>/ablation_early_stopping_metric.csv.
''',
            '''      metric_ablation  - legacy validation metric versus RMSE on CatBoost + LightGBM.
                  Writes metric_ablation_<timestamp>/ablation_early_stopping_metric.csv.
      lgbm_alpha_ablation  - LightGBM Huber-alpha grid with all champion settings locked.
                  Writes lgbm_alpha_ablation_<timestamp>/ablation_lightgbm_huber_alpha.csv.
''',
            1,
        )
    nb["cells"][26]["source"] = source.splitlines(keepends=True)

    # Current junior flow.
    nb["cells"][27]["source"] = '''## 11. Flow eksperimen LightGBM Huber alpha

CatBoost tetap memakai `Huber:delta=1.5`. Eksperimen ini hanya menguji parameter `alpha` milik LightGBM dengan grid `0.7, 0.9, 1.2, 1.5`; `1.5` adalah baseline notebook sebelumnya.

Mode `lgbm_alpha_ablation` otomatis menjalankan `lgbm_a` pada `sep_2024` dan `jan_2025`. Semua variant memakai target raw, early-stopping metric lama, frozen 84 fitur, `nino_34` ON, harmonic ON, spike OFF, pruning OFF, anchor OFF, dan learned weighting OFF.

### Flow junior

1. Upload notebook terbaru dan pilih accelerator GPU seperti run sebelumnya.
2. Restart Session/Kernel.
3. Pastikan configuration cell menunjukkan `MODE="lgbm_alpha_ablation"`.
4. Klik **Run All** satu kali. Jangan mengubah `LIGHTGBM_HUBER_ALPHA`; mode ini otomatis menjalankan seluruh grid.
5. Ambil folder `lgbm_alpha_ablation_<timestamp>`.
6. Serahkan dua file kepada senior:
   - `ablation_lightgbm_huber_alpha.csv`
   - `lgbm_alpha_ablation_audit.json`
7. Jangan menjalankan full validation, fit, atau upload submission sebelum keputusan senior.

CSV mencatat `best_iteration` untuk setiap alpha dan fold. Kandidat hanya naik tahap bila rata-rata RMSE membaik cukup berarti, `jan_2025` aman, serta MAE, P95, dan worst-station tidak memburuk material.
'''.splitlines(keepends=True)

    # Ready-to-run config.
    source = "".join(nb["cells"][28]["source"])
    source = source.replace(
        'MODE = "metric_ablation"  # otomatis menguji current vs RMSE',
        'MODE = "lgbm_alpha_ablation"  # otomatis menguji alpha 0.7, 0.9, 1.2, 1.5',
    )
    if "LIGHTGBM_HUBER_ALPHA =" not in source:
        source = source.replace(
            'EARLY_STOPPING_METRIC = "current"  # mode ablation otomatis menguji current dan rmse\n',
            'EARLY_STOPPING_METRIC = "current"  # metric champion\n'
            'LIGHTGBM_HUBER_ALPHA = 1.5  # baseline; mode alpha otomatis menguji seluruh grid\n',
            1,
        )
    if "lightgbm_huber_alpha=LIGHTGBM_HUBER_ALPHA" not in source:
        source = source.replace(
            "    early_stopping_metric=EARLY_STOPPING_METRIC,\n",
            "    early_stopping_metric=EARLY_STOPPING_METRIC,\n"
            "    lightgbm_huber_alpha=LIGHTGBM_HUBER_ALPHA,\n",
            1,
        )
    source = source.replace(
        'print(f"target_mode={CONFIG.target_mode} early_stopping_metric={CONFIG.early_stopping_metric} use_spike_weights=',
        'print(f"target_mode={CONFIG.target_mode} early_stopping_metric={CONFIG.early_stopping_metric} lightgbm_huber_alpha={CONFIG.lightgbm_huber_alpha} use_spike_weights=',
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
