"""Idempotent notebook upgrade for LightGBM alpha confirmation and OOF blend."""

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


CONFIRMATION_FUNCTION = r'''
def run_lgbm_alpha_confirmation(
    train_features: pd.DataFrame,
    columns: list[str],
    config: ModelConfig,
    artifacts: Path,
    alpha_values: tuple[float, float] = (0.7, 1.5),
    confirmation_fold_names: tuple[str, ...] = (
        "sep_2023", "sep_2024", "jan_2025"
    ),
) -> pd.DataFrame:
    """Confirm alpha 0.7 and its 50:50 blend against alpha 1.5.

    Runs lgbm_a and lgbm_b independently. Predictions from both alpha values
    are retained in memory and blended row-by-row, so the blend requires no
    additional fitted model and is evaluated on exactly aligned OOF rows.
    """
    if config.enable_anchor_feature:
        raise ValueError(
            "lgbm_alpha_confirmation requires ENABLE_ANCHOR_FEATURE=False"
        )
    if config.target_mode != "raw_robust":
        raise ValueError(
            "lgbm_alpha_confirmation requires TARGET_MODE='raw_robust'"
        )
    if config.early_stopping_metric != "current":
        raise ValueError(
            "lgbm_alpha_confirmation requires EARLY_STOPPING_METRIC='current'"
        )
    if tuple(float(value) for value in alpha_values) != (0.7, 1.5):
        raise ValueError("Confirmation alpha values must be exactly (0.7, 1.5)")

    fold_lookup = {fold[0]: fold for fold in FOLDS}
    missing_folds = sorted(set(confirmation_fold_names) - set(fold_lookup))
    if missing_folds:
        raise ValueError(f"Unknown confirmation folds: {missing_folds}")
    confirmation_folds = tuple(
        fold_lookup[name] for name in confirmation_fold_names
    )

    columns = enforce_frozen_anchor_off_columns(train_features, columns, config)
    if columns != list(ANCHOR_OFF_FROZEN_FEATURE_COLUMNS):
        raise AssertionError(
            "lgbm_alpha_confirmation must use exact frozen 84 features"
        )

    members = tuple(
        member for member in config.ensemble
        if member.name in {"lgbm_a", "lgbm_b"}
    )
    if {member.name for member in members} != {"lgbm_a", "lgbm_b"}:
        raise ValueError("Both lgbm_a and lgbm_b are required for confirmation")

    result_parts = []
    predictions_by_member_alpha: dict[tuple[str, float], pd.DataFrame] = {}
    key_columns = ["fold", *KEYS]

    for member in members:
        for alpha in alpha_values:
            variant_config = replace(
                config,
                lightgbm_huber_alpha=float(alpha),
                early_stopping_metric="current",
                target_mode="raw_robust",
                use_spike_weights=False,
                prune_features=False,
                enable_anchor_feature=False,
                enable_learned_ensemble_weights=False,
            )
            report, _ = run_validation(
                train_features,
                columns,
                variant_config,
                artifacts,
                folds=confirmation_folds,
                members=(member,),
                write_artifacts=False,
            )
            captured = report.attrs.get("captured_oof_predictions")
            if captured is None or captured.empty:
                raise RuntimeError(
                    "run_validation did not return captured OOF predictions"
                )
            prediction_frame = captured[
                key_columns + ["tma_mdpl", "prediction"]
            ].copy()
            if prediction_frame.duplicated(key_columns).any():
                raise AssertionError("Duplicate OOF keys in alpha confirmation")
            predictions_by_member_alpha[(member.name, float(alpha))] = (
                prediction_frame.rename(columns={"prediction": f"prediction_alpha_{alpha:g}"})
            )

            keep = [
                c for c in report.columns
                if c.startswith("model_")
                or c in {"fold", "target_mode", "best_iteration"}
            ]
            summary = report[keep].copy()
            summary["strategy"] = f"alpha_{alpha:g}"
            summary["variant"] = f"{member.name}_alpha_{alpha:g}"
            summary["lightgbm_huber_alpha"] = float(alpha)
            summary["member"] = member.name
            summary["family"] = member.family
            summary["n_features"] = len(columns)
            summary["anchor_off_frozen_parity"] = True
            result_parts.append(summary)

    oof_exports = []
    for member in members:
        alpha_07 = predictions_by_member_alpha[(member.name, 0.7)]
        alpha_15 = predictions_by_member_alpha[(member.name, 1.5)]
        merged = alpha_07.merge(
            alpha_15,
            on=key_columns + ["tma_mdpl"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(alpha_07) or len(merged) != len(alpha_15):
            raise AssertionError("OOF rows do not align between alpha variants")
        merged["prediction_blend_50_50"] = 0.5 * (
            merged["prediction_alpha_0.7"]
            + merged["prediction_alpha_1.5"]
        )
        merged["member"] = member.name
        oof_exports.append(merged)

        for fold_name, fold_frame in merged.groupby("fold", observed=True):
            blend_metrics = metrics_frame(
                fold_frame["tma_mdpl"].to_numpy(dtype=float),
                fold_frame["prediction_blend_50_50"].to_numpy(dtype=float),
                fold_frame["nama_pos"],
            )
            result_parts.append(pd.DataFrame([{
                "fold": fold_name,
                "target_mode": "raw_robust",
                "best_iteration": np.nan,
                **{f"model_{key}": value for key, value in blend_metrics.items()},
                "strategy": "blend_50_50",
                "variant": f"{member.name}_blend_alpha_0.7_1.5",
                "lightgbm_huber_alpha": np.nan,
                "member": member.name,
                "family": member.family,
                "n_features": len(columns),
                "anchor_off_frozen_parity": True,
            }]))

    result = pd.concat(result_parts, ignore_index=True)
    strategy_order = {"alpha_0.7": 0, "blend_50_50": 1, "alpha_1.5": 2}
    result["_strategy_order"] = result["strategy"].map(strategy_order)
    result = result.sort_values(
        ["member", "fold", "_strategy_order"]
    ).drop(columns="_strategy_order").reset_index(drop=True)

    baseline = result[result["strategy"] == "alpha_1.5"][[
        "member", "fold", "model_rmse", "model_worst_station_mae"
    ]].rename(columns={
        "model_rmse": "baseline_rmse",
        "model_worst_station_mae": "baseline_worst_station_mae",
    })

    gate_results = {}
    exact_folds = {"sep_2023", "sep_2024"}
    for strategy in ("alpha_0.7", "blend_50_50"):
        candidate = result[result["strategy"] == strategy].merge(
            baseline, on=["member", "fold"], how="left", validate="one_to_one"
        )
        candidate["rmse_delta_pct"] = 100.0 * (
            candidate["model_rmse"] / candidate["baseline_rmse"] - 1.0
        )
        candidate["worst_station_delta_pct"] = 100.0 * (
            candidate["model_worst_station_mae"]
            / candidate["baseline_worst_station_mae"] - 1.0
        )
        exact = candidate[candidate["fold"].isin(exact_folds)]
        safety = candidate[candidate["fold"] == "jan_2025"]
        exact_wins = int((exact["rmse_delta_pct"] < 0).sum())
        exact_mean_improvement_pct = float(-exact["rmse_delta_pct"].mean())
        jan_max_rmse_degradation_pct = float(safety["rmse_delta_pct"].max())
        jan_max_worst_degradation_pct = float(
            safety["worst_station_delta_pct"].max()
        )
        passed = bool(
            exact_wins >= 3
            and exact_mean_improvement_pct >= 0.5
            and jan_max_rmse_degradation_pct <= 0.3
            and jan_max_worst_degradation_pct <= 2.0
        )
        gate_results[strategy] = {
            "exact_season_wins_out_of_4": exact_wins,
            "exact_season_mean_rmse_improvement_pct": exact_mean_improvement_pct,
            "jan_2025_max_rmse_degradation_pct": jan_max_rmse_degradation_pct,
            "jan_2025_max_worst_station_degradation_pct": jan_max_worst_degradation_pct,
            "passed": passed,
        }

    passing = [
        strategy for strategy, values in gate_results.items()
        if values["passed"]
    ]
    promoted_strategy = (
        max(
            passing,
            key=lambda strategy: gate_results[strategy][
                "exact_season_mean_rmse_improvement_pct"
            ],
        )
        if passing else None
    )

    artifacts.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        artifacts / "lgbm_alpha_confirmation_results.csv", index=False
    )
    save_frame(
        pd.concat(oof_exports, ignore_index=True),
        artifacts / "lgbm_alpha_confirmation_oof",
    )
    write_json(artifacts / "lgbm_alpha_confirmation_audit.json", {
        "experiment": "alpha_0.7_vs_1.5_and_oof_blend",
        "members": [member.name for member in members],
        "folds": list(confirmation_fold_names),
        "exact_season_folds": sorted(exact_folds),
        "alpha_values": list(alpha_values),
        "blend": {"alpha_0.7": 0.5, "alpha_1.5": 0.5},
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
        "oof_alignment_keys": key_columns,
        "target_period_api_used": False,
    })
    write_json(artifacts / "lgbm_alpha_confirmation_decision.json", {
        "acceptance_criteria": {
            "exact_season_wins_minimum": 3,
            "exact_season_combinations": 4,
            "exact_season_mean_rmse_improvement_minimum_pct": 0.5,
            "jan_2025_rmse_degradation_maximum_pct": 0.3,
            "jan_2025_worst_station_degradation_maximum_pct": 2.0,
        },
        "strategies": gate_results,
        "promoted_strategy": promoted_strategy,
        "full_validation_authorized": promoted_strategy is not None,
    })
    LOGGER.info(
        "LightGBM alpha confirmation decision: %s",
        json.dumps({
            "strategies": gate_results,
            "promoted_strategy": promoted_strategy,
        }, indent=2),
    )
    return result
'''.strip()


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    # Expose fold OOF predictions through DataFrame attrs without changing the
    # long-standing two-value return signature used throughout the notebook.
    source = "".join(nb["cells"][16]["source"])
    source = replace_once(
        source,
        '''    return report, recommended
''',
        '''    report.attrs["captured_oof_predictions"] = pd.concat(
        oof_parts, ignore_index=True
    )
    return report, recommended
''',
        "validation OOF capture",
    )
    nb["cells"][16]["source"] = source.splitlines(keepends=True)

    # Confirmation function.
    source = "".join(nb["cells"][18]["source"])
    if "def run_lgbm_alpha_confirmation(" not in source:
        marker = "\n\n# Frozen after"
        if source.count(marker) != 1:
            raise RuntimeError("Could not place alpha confirmation function")
        source = source.replace(
            marker, "\n\n" + CONFIRMATION_FUNCTION + marker, 1
        )
    nb["cells"][18]["source"] = source.splitlines(keepends=True)

    # Pipeline mode and routing.
    source = "".join(nb["cells"][26]["source"])
    source = source.replace(
        'if mode not in {"audit", "ablation", "target_ablation", "metric_ablation", "lgbm_alpha_ablation", "tune", "p0_experiments", "validate", "fit", "all"}:',
        'if mode not in {"audit", "ablation", "target_ablation", "metric_ablation", "lgbm_alpha_ablation", "lgbm_alpha_confirmation", "tune", "p0_experiments", "validate", "fit", "all"}:',
    ).replace(
        'raise ValueError("mode must be one of: audit, ablation, target_ablation, metric_ablation, lgbm_alpha_ablation, tune, p0_experiments, validate, fit, all")',
        'raise ValueError("mode must be one of: audit, ablation, target_ablation, metric_ablation, lgbm_alpha_ablation, lgbm_alpha_confirmation, tune, p0_experiments, validate, fit, all")',
    ).replace(
        'if mode in {"ablation", "target_ablation", "metric_ablation", "lgbm_alpha_ablation", "tune", "p0_experiments"}:',
        'if mode in {"ablation", "target_ablation", "metric_ablation", "lgbm_alpha_ablation", "lgbm_alpha_confirmation", "tune", "p0_experiments"}:',
    ).replace(
        '''        if mode == "lgbm_alpha_ablation":
            table = run_lgbm_alpha_ablation(
''',
        '''        if mode == "lgbm_alpha_confirmation":
            table = run_lgbm_alpha_confirmation(
                train_features, columns, config, light_dir
            )
            result["lgbm_alpha_confirmation_results"] = table
            result["lgbm_alpha_confirmation_dir"] = str(light_dir)
        elif mode == "lgbm_alpha_ablation":
            table = run_lgbm_alpha_ablation(
''',
    )
    if "lgbm_alpha_confirmation  - confirm alpha 0.7" not in source:
        source = source.replace(
            '''      lgbm_alpha_ablation  - LightGBM Huber-alpha grid with all champion settings locked.
                  Writes lgbm_alpha_ablation_<timestamp>/ablation_lightgbm_huber_alpha.csv.
''',
            '''      lgbm_alpha_ablation  - LightGBM Huber-alpha grid with all champion settings locked.
                  Writes lgbm_alpha_ablation_<timestamp>/ablation_lightgbm_huber_alpha.csv.
      lgbm_alpha_confirmation  - confirm alpha 0.7, alpha 1.5, and their OOF blend.
                  Writes confirmation metrics, aligned OOF predictions, and an automatic gate.
''',
            1,
        )
    nb["cells"][26]["source"] = source.splitlines(keepends=True)

    # Junior-facing flow.
    nb["cells"][27]["source"] = '''## 11. Flow confirmation LightGBM alpha

Mode ini menguji `alpha=0.7` terhadap champion `alpha=1.5`, lalu menghitung blend OOF 50:50 tanpa melatih model tambahan.

Eksperimen otomatis menjalankan:

- Anggota: `lgbm_a` dan `lgbm_b`.
- Fold exact-season: `sep_2023` dan `sep_2024`.
- Safety fold: `jan_2025`.
- Alpha: `0.7` dan `1.5`.
- Blend: `50% prediksi alpha 0.7 + 50% prediksi alpha 1.5`.

Seluruh variant memakai target raw, early-stopping metric lama, frozen 84 fitur, `nino_34` ON, harmonic ON, spike OFF, pruning OFF, anchor OFF, dan learned weighting OFF.

### Flow junior

1. Upload notebook terbaru dan pilih accelerator GPU seperti run sebelumnya.
2. Restart Session/Kernel.
3. Pastikan configuration cell menunjukkan `MODE="lgbm_alpha_confirmation"`.
4. Klik **Run All** satu kali. Jangan mengubah `LIGHTGBM_HUBER_ALPHA`; mode mengatur kedua alpha otomatis.
5. Ambil folder `lgbm_alpha_confirmation_<timestamp>`.
6. Serahkan seluruh folder kepada senior, khususnya:
   - `lgbm_alpha_confirmation_results.csv`
   - `lgbm_alpha_confirmation_oof.parquet`
   - `lgbm_alpha_confirmation_audit.json`
   - `lgbm_alpha_confirmation_decision.json`
7. Jangan menjalankan full validation, fit, atau upload submission meskipun JSON menyatakan lolos; keputusan akhir tetap diaudit senior.

Run ini melakukan 12 fit: 2 alpha × 2 LightGBM × 3 fold. Blend tidak menambah training.
'''.splitlines(keepends=True)

    # Ready-to-run config.
    source = "".join(nb["cells"][28]["source"])
    source = source.replace(
        'MODE = "lgbm_alpha_ablation"  # otomatis menguji alpha 0.7, 0.9, 1.2, 1.5',
        'MODE = "lgbm_alpha_confirmation"  # alpha 0.7 vs 1.5 + OOF blend',
    )
    nb["cells"][28]["source"] = source.splitlines(keepends=True)

    NOTEBOOK.write_text(
        json.dumps(nb, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Updated {NOTEBOOK.resolve()}")


if __name__ == "__main__":
    main()
