"""Build the Kaggle-ready RMSE 0.81 target challenger notebook.

The generated notebook keeps the fully audited v2 pipeline and injects a v3
override layer before its configuration cell. Keeping generation deterministic
makes the large .ipynb reviewable and easy to rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "notebookfde8c8db98.ipynb"
OUTPUT = ROOT / "notebook_optimized_rmse_081.ipynb"


V3_MARKDOWN = r"""## 11. Optimasi v3 — annual-memory ensemble dan OOF stacking

Versi ini menargetkan penurunan leaderboard RMSE dari baseline 1.597 menuju 0.81.
Angka 0.81 adalah **promotion target**, bukan klaim skor sebelum submission dinilai
oleh Kaggle karena label test tidak tersedia di notebook.

Perubahan utama:

1. **Annual target memory yang deployment-safe.** Test sepanjang 242 hari hanya
   memakai label 364–366 hari sebelumnya; seluruh timestamp sumber berada sebelum
   cutoff train. Median tiga lag menahan isolated sensor spike.
2. **Diversity objective.** CatBoost RMSE dan Huber serta LightGBM L2 dan Huber
   dilatih pada target robust-normalized yang sama.
3. **Perbaikan LightGBM Huber.** `alpha` memakai 0.90 (quantile parameter valid),
   bukan Huber delta CatBoost 1.5.
4. **OOF-weighted member blend.** Bobot lima learner dioptimalkan dengan kendala
   non-negatif dan jumlah satu, di-shrink ke equal weights, lalu diuji cross-fold.
5. **Representative folds.** Model selection memakai tiga fold yang sudah memiliki
   annual history. `sep_2024` mempunyai kalender horizon yang sama dengan test.
6. **OOF stacking.** Bobot blend model–annual-memory dipelajari dari OOF dan
   di-shrink per stasiun. Metrik cross-fold disimpan agar efek stacker tidak hanya
   dinilai secara in-sample.

Guard leakage memverifikasi bahwa horizon setiap fold lebih pendek daripada annual
lag minimum. Semua feature target-derived pada baris valid/test berasal dari waktu
sebelum cutoff; tidak ada recursive prediction.
"""


V3_CODE = r'''# ================================================================
# V3 OVERRIDES: safe annual memory + objective diversity + OOF stack
# ================================================================

ANNUAL_LAG_DAYS = (364, 365, 366, 728, 729, 730, 731)
V3_FOLDS = FOLDS[1:]  # folds with at least one annual cycle available
V3_STACKING_STATE: dict = {"global_weight": 0.0, "station_weights": {}}


def _assert_annual_lag_safety(folds=V3_FOLDS) -> None:
    """Prove that every annual source timestamp precedes its fold cutoff."""
    shortest_lag = min(ANNUAL_LAG_DAYS)
    for name, train_end, _valid_start, valid_end in folds:
        horizon = (pd.Timestamp(valid_end) - pd.Timestamp(train_end)).total_seconds() / 86_400
        if horizon >= shortest_lag:
            raise ValueError(
                f"Fold {name} horizon={horizon:.1f}d is not safe for "
                f"minimum annual lag={shortest_lag}d"
            )


def add_safe_annual_target_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    target_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach only target values old enough to be known for the entire horizon.

    History is shifted forward by 364+ days and joined by exact station/timestamp.
    Consequently the value attached to row t is y(t-lag), never y(t) or a value
    inside the forecast horizon.
    """
    _assert_annual_lag_safety()
    history = target_history.loc[:, ["datetime", "nama_pos", "tma_mdpl"]].copy()
    history["datetime"] = pd.to_datetime(history["datetime"])

    def attach(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for days in ANNUAL_LAG_DAYS:
            shifted = history.rename(columns={"tma_mdpl": f"annual_level_d{days}"}).copy()
            shifted["datetime"] = shifted["datetime"] + pd.Timedelta(days=days)
            out = out.merge(shifted, on=KEYS, how="left", validate="many_to_one")

        lag1 = [f"annual_level_d{d}" for d in (364, 365, 366)]
        lag2 = [f"annual_level_d{d}" for d in (728, 729, 730, 731)]
        out["annual_median_1y"] = out[lag1].median(axis=1, skipna=True)
        out["annual_mean_1y"] = out[lag1].mean(axis=1, skipna=True)
        out["annual_spread_1y"] = out[lag1].max(axis=1) - out[lag1].min(axis=1)
        out["annual_available_1y"] = out[lag1].notna().sum(axis=1).astype("int8")
        out["annual_median_2y"] = out[lag2].median(axis=1, skipna=True)
        out["annual_available_2y"] = out[lag2].notna().sum(axis=1).astype("int8")
        out["annual_yoy_delta"] = out["annual_median_1y"] - out["annual_median_2y"]
        out["annual_disagreement"] = (
            out["annual_median_1y"] - out["harmonic_baseline_mdpl"]
            if "harmonic_baseline_mdpl" in out else np.nan
        )
        return out

    return attach(train_features), attach(test_features)


# Wrap the audited feature builder instead of duplicating HydroRIVERS logic.
_build_feature_tables_v2 = build_feature_tables


def build_feature_tables(
    train: pd.DataFrame,
    test: pd.DataFrame,
    coordinates: pd.DataFrame,
    environment: pd.DataFrame,
    paths: Paths,
    config: ModelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features, test_features, weights = _build_feature_tables_v2(
        train, test, coordinates, environment, paths, config
    )
    train_features, test_features = add_safe_annual_target_features(
        train_features, test_features, train
    )
    return train_features, test_features, weights


# Recompute annual-vs-harmonic disagreement after the leakage-safe per-fold
# harmonic model has replaced the global NaN placeholders.
_apply_station_features_v2 = apply_station_features


def apply_station_features(
    fit_frame: pd.DataFrame,
    other_frames: Sequence[pd.DataFrame],
    config: ModelConfig,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    fit_out, others_out = _apply_station_features_v2(fit_frame, other_frames, config)
    frames = [fit_out, *others_out]
    for frame in frames:
        if "annual_median_1y" in frame and "harmonic_baseline_mdpl" in frame:
            frame["annual_disagreement"] = (
                frame["annual_median_1y"] - frame["harmonic_baseline_mdpl"]
            )
    return frames[0], frames[1:]


# Heterogeneous objectives. Names deliberately encode the objective so the
# existing immutable EnsembleMemberConfig does not need a schema migration.
V3_ENSEMBLE = (
    EnsembleMemberConfig("cb_rmse_d7", "catboost", seed=17, depth=7,
                         learning_rate=0.030, l2_leaf_reg=12.0),
    EnsembleMemberConfig("cb_huber_d6", "catboost", seed=41, depth=6,
                         learning_rate=0.040, l2_leaf_reg=10.0),
    EnsembleMemberConfig("cb_huber_d8", "catboost", seed=83, depth=8,
                         learning_rate=0.025, l2_leaf_reg=16.0),
    EnsembleMemberConfig("lgbm_l2", "lightgbm", seed=7, num_leaves=63,
                         learning_rate=0.025, feature_fraction=0.82,
                         bagging_fraction=0.82, min_child_samples=35),
    EnsembleMemberConfig("lgbm_huber", "lightgbm", seed=29, num_leaves=95,
                         learning_rate=0.020, feature_fraction=0.75,
                         bagging_fraction=0.82, min_child_samples=40),
)
V3_MEMBER_WEIGHTS = {member.name: 1.0 / len(V3_ENSEMBLE) for member in V3_ENSEMBLE}


def make_catboost_model(
    member: EnsembleMemberConfig,
    config: ModelConfig,
    iterations: int,
    train_dir: Path | None = None,
):
    from catboost import CatBoostRegressor

    is_rmse = "rmse" in member.name
    params = dict(
        iterations=iterations,
        learning_rate=member.learning_rate,
        depth=member.depth,
        loss_function="RMSE" if is_rmse else f"Huber:delta={config.huber_delta}",
        eval_metric="RMSE",  # promotion metric and early stopping are aligned
        l2_leaf_reg=member.l2_leaf_reg,
        random_seed=member.seed,
        random_strength=0.7,
        bootstrap_type="Bayesian",
        bagging_temperature=0.6,
        allow_writing_files=train_dir is not None,
        thread_count=-1,
        verbose=config.verbose_every,
    )
    if train_dir is not None:
        train_dir.mkdir(parents=True, exist_ok=True)
        params.update(train_dir=str(train_dir), save_snapshot=True,
                      snapshot_file="training.snapshot",
                      snapshot_interval=config.snapshot_interval_seconds)
    return CatBoostRegressor(**params)


def make_lightgbm_model(
    member: EnsembleMemberConfig,
    config: ModelConfig,
    iterations: int,
):
    import lightgbm as lgb

    is_l2 = "l2" in member.name
    return lgb.LGBMRegressor(
        objective="regression_l2" if is_l2 else "huber",
        alpha=0.90,  # valid LightGBM Huber/quantile alpha
        n_estimators=iterations,
        learning_rate=member.learning_rate,
        num_leaves=member.num_leaves,
        max_depth=-1,
        feature_fraction=member.feature_fraction,
        bagging_fraction=member.bagging_fraction,
        bagging_freq=1,
        min_child_samples=member.min_child_samples,
        reg_lambda=20.0,
        reg_alpha=0.10,
        random_state=member.seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _read_saved_frame(stem: Path) -> pd.DataFrame:
    candidates = (stem.with_suffix(".parquet"), stem.with_suffix(".csv.gz"), stem.with_suffix(".csv"))
    for path in candidates:
        if path.exists():
            return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, parse_dates=["datetime"])
    raise FileNotFoundError(f"No saved frame found for {stem}")


def _fit_annual_blend_weights(oof: pd.DataFrame) -> dict:
    """Fit conservative model/annual weights on level-1 OOF predictions."""
    usable = oof.dropna(subset=["prediction", "annual_median_1y", "tma_mdpl"]).copy()
    grid = np.linspace(0.0, 0.60, 31)
    if usable.empty:
        return {"global_weight": 0.0, "station_weights": {}, "rows": 0}

    def best_weight(frame: pd.DataFrame) -> tuple[float, float, float]:
        y = frame["tma_mdpl"].to_numpy(dtype=float)
        model = frame["prediction"].to_numpy(dtype=float)
        annual = frame["annual_median_1y"].to_numpy(dtype=float)
        base_rmse = float(np.sqrt(np.mean((y - model) ** 2)))
        scores = [float(np.sqrt(np.mean((y - ((1-w)*model + w*annual)) ** 2))) for w in grid]
        pos = int(np.argmin(scores))
        return float(grid[pos]), base_rmse, scores[pos]

    global_w, base_rmse, blend_rmse = best_weight(usable)
    station_weights = {}
    for station, part in usable.groupby("nama_pos", observed=True):
        local_w, local_base, local_blend = best_weight(part)
        # Do not accept microscopic in-sample improvements. Shrink local weights
        # toward the global solution because each station has only a few folds.
        if local_base - local_blend < 0.005:
            local_w = global_w
        shrink = len(part) / (len(part) + 1500.0)
        station_weights[str(station)] = float(global_w + shrink * (local_w - global_w))
    return {
        "global_weight": global_w,
        "station_weights": station_weights,
        "rows": int(len(usable)),
        "model_rmse": base_rmse,
        "in_sample_stacked_rmse": blend_rmse,
    }


def _fit_member_weights(frame: pd.DataFrame, member_names: Sequence[str]) -> dict[str, float]:
    """Constrained RMSE-optimal blend with mild shrinkage toward equal weights."""
    from scipy.optimize import minimize

    cols = [f"prediction_{name}" for name in member_names]
    usable = frame.dropna(subset=["tma_mdpl", *cols])
    equal = np.full(len(member_names), 1.0 / len(member_names), dtype=float)
    if usable.empty or len(member_names) == 1:
        return {name: float(w) for name, w in zip(member_names, equal)}
    matrix = usable[cols].to_numpy(dtype=float)
    target = usable["tma_mdpl"].to_numpy(dtype=float)
    result = minimize(
        lambda w: np.mean((target - matrix @ w) ** 2),
        x0=equal,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(member_names),
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    optimum = result.x if result.success else equal
    # Ten-percent shrinkage reduces fold-to-fold weight instability.
    stable = 0.90 * optimum + 0.10 * equal
    stable /= stable.sum()
    return {name: float(w) for name, w in zip(member_names, stable)}


def _apply_member_weights(
    frame: pd.DataFrame, member_names: Sequence[str], weights: dict[str, float]
) -> np.ndarray:
    matrix = frame[[f"prediction_{name}" for name in member_names]].to_numpy(dtype=float)
    vector = np.array([weights.get(name, 0.0) for name in member_names], dtype=float)
    if vector.sum() <= 0:
        vector[:] = 1.0 / len(vector)
    else:
        vector /= vector.sum()
    return matrix @ vector


def _apply_annual_blend(frame: pd.DataFrame, state: dict) -> np.ndarray:
    weights = frame["nama_pos"].map(state.get("station_weights", {})).fillna(
        state.get("global_weight", 0.0)
    ).to_numpy(dtype=float, copy=True)
    annual = frame["annual_median_1y"].to_numpy(dtype=float)
    model = frame["prediction"].to_numpy(dtype=float)
    weights[~np.isfinite(annual)] = 0.0
    return (1.0 - weights) * model + weights * np.nan_to_num(annual, nan=0.0)


_run_validation_v2 = run_validation


def run_validation(
    train_features: pd.DataFrame,
    columns: list[str],
    config: ModelConfig,
    artifacts: Path,
    folds=V3_FOLDS,
    members=None,
    write_artifacts: bool = True,
):
    """Run audited validation, then fit and cross-check the annual stacker."""
    global V3_STACKING_STATE, V3_MEMBER_WEIGHTS
    _assert_annual_lag_safety(folds)
    report, recommended = _run_validation_v2(
        train_features, columns, config, artifacts,
        folds=folds, members=members, write_artifacts=write_artifacts,
    )
    if not write_artifacts:
        return report, recommended

    oof = _read_saved_frame(artifacts / "metrics" / "oof_predictions")
    annual_cols = ["datetime", "nama_pos", "annual_median_1y"]
    oof["datetime"] = pd.to_datetime(oof["datetime"])
    oof = oof.merge(train_features[annual_cols], on=["datetime", "nama_pos"],
                    how="left", validate="one_to_one")

    member_names = [m.name for m in (members or config.ensemble)]
    missing_member_cols = [f"prediction_{name}" for name in member_names if f"prediction_{name}" not in oof]
    if missing_member_cols:
        raise RuntimeError(f"Missing OOF member predictions: {missing_member_cols}")

    # Honest cross-fold evaluation: both member weights and annual blend weights
    # for a held fold are learned only from the other folds.
    crossfit = pd.Series(index=oof.index, dtype=float)
    crossfit_rows = []
    for held_fold, held in oof.groupby("fold", observed=True):
        level1_train = oof[oof["fold"] != held_fold].copy()
        if level1_train.empty:
            member_state = {name: 1.0 / len(member_names) for name in member_names}
        else:
            member_state = _fit_member_weights(level1_train, member_names)
        level1_train["prediction"] = _apply_member_weights(level1_train, member_names, member_state)
        annual_state = _fit_annual_blend_weights(level1_train)
        held_eval = held.copy()
        held_eval["prediction"] = _apply_member_weights(held_eval, member_names, member_state)
        pred = _apply_annual_blend(held_eval, annual_state)
        crossfit.loc[held.index] = pred
        y = held["tma_mdpl"].to_numpy(dtype=float)
        crossfit_rows.append({
            "fold": held_fold,
            "stacked_crossfit_rmse": float(np.sqrt(np.mean((y-pred)**2))),
            "stacked_crossfit_mae": float(np.mean(np.abs(y-pred))),
        })
    oof["stacked_crossfit_prediction"] = crossfit
    save_frame(oof, artifacts / "metrics" / "oof_predictions_v3")
    crossfit_table = pd.DataFrame(crossfit_rows)
    crossfit_table.to_csv(artifacts / "metrics" / "stacking_crossfit_metrics.csv", index=False)
    report = report.merge(crossfit_table, on="fold", how="left", validate="one_to_one")
    report.to_csv(artifacts / "metrics" / "validation_metrics_v3.csv", index=False)

    V3_MEMBER_WEIGHTS = _fit_member_weights(oof, member_names)
    oof["equal_weight_prediction"] = oof["prediction"]
    oof["prediction"] = _apply_member_weights(oof, member_names, V3_MEMBER_WEIGHTS)
    V3_STACKING_STATE = _fit_annual_blend_weights(oof)
    write_json(artifacts / "metrics" / "member_weights.json", V3_MEMBER_WEIGHTS)
    write_json(artifacts / "metrics" / "stacking_weights.json", V3_STACKING_STATE)
    LOGGER.info("V3 member weights: %s", V3_MEMBER_WEIGHTS)
    LOGGER.info("V3 annual stacking state: %s", V3_STACKING_STATE)
    return report, recommended


_fit_final_and_submit_v2 = fit_final_and_submit


def fit_final_and_submit(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    sample: pd.DataFrame,
    columns: list[str],
    config: ModelConfig,
    artifacts: Path,
    iterations_by_member: dict[str, int],
) -> Path:
    """Fit the full ensemble, then apply OOF-learned conservative stacking."""
    output = _fit_final_and_submit_v2(
        train_features, test_features, sample, columns, config,
        artifacts, iterations_by_member,
    )
    submission = pd.read_csv(output)
    model_only = artifacts / "submission_model_only.csv"
    submission.to_csv(model_only, index=False)

    keys = test_features[["id", "nama_pos", "annual_median_1y"]].copy()
    stacked = submission.merge(keys, on="id", how="left", validate="one_to_one")
    stacked = stacked.rename(columns={"tma_mdpl": "prediction"})
    stacked["tma_mdpl"] = _apply_annual_blend(stacked, V3_STACKING_STATE)
    if not np.isfinite(stacked["tma_mdpl"]).all():
        raise ValueError("V3 stacked submission contains non-finite predictions")
    stacked[["id", "tma_mdpl"]].to_csv(output, index=False)
    write_json(artifacts / "v3_postprocess.json", {
        "stacking_state": V3_STACKING_STATE,
        "model_only_submission": str(model_only),
        "final_submission": str(output),
    })
    return output
'''


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    insert_at = next(
        i for i, cell in enumerate(cells)
        if cell["cell_type"] == "markdown"
        and "## 11. Konfigurasi mode" in "".join(cell.get("source", []))
    )
    cells[0]["source"] = [
        "# Prediksi Tinggi Muka Air Sungai — RMSE 0.81 Target Challenger\n",
        "\n",
        "Full pipeline Kaggle: audit, feature engineering, annual-memory features, "
        "rolling-origin validation, heterogeneous ensemble, OOF stacking, final fit, dan submission.\n",
    ]
    cells[insert_at:insert_at] = [markdown_cell(V3_MARKDOWN), code_cell(V3_CODE)]

    # Retain each member's OOF prediction so v3 can learn honest blend weights.
    validation_patched = False
    final_patched = False
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def run_validation(" in source and "member_preds_z = []" in source:
            source = source.replace(
                "        member_preds_z = []\n",
                "        member_preds_z = []\n        member_pred_columns: dict[str, np.ndarray] = {}\n",
                1,
            )
            source = source.replace(
                '            member_pred = denormalize_prediction(pred_z_member, valid["nama_pos"], stats)\n',
                '            member_pred = denormalize_prediction(pred_z_member, valid["nama_pos"], stats)\n'
                '            member_pred_columns[member.name] = member_pred\n',
                1,
            )
            source = source.replace(
                '        oof["baseline_prediction"] = baseline_pred\n',
                '        oof["baseline_prediction"] = baseline_pred\n'
                '        for member_name, member_prediction in member_pred_columns.items():\n'
                '            oof[f"prediction_{member_name}"] = member_prediction\n',
                1,
            )
            cell["source"] = source.splitlines(keepends=True)
            validation_patched = True
        if "def fit_final_and_submit(" in source and "mean_z = np.mean(predictions, axis=0)" in source:
            source = source.replace(
                "    mean_z = np.mean(predictions, axis=0)\n",
                "    member_weight_vector = np.array([V3_MEMBER_WEIGHTS.get(m.name, 0.0) for m in config.ensemble], dtype=float)\n"
                "    if member_weight_vector.sum() <= 0:\n"
                "        member_weight_vector[:] = 1.0 / len(member_weight_vector)\n"
                "    else:\n"
                "        member_weight_vector /= member_weight_vector.sum()\n"
                "    mean_z = np.average(np.vstack(predictions), axis=0, weights=member_weight_vector)\n",
                1,
            )
            cell["source"] = source.splitlines(keepends=True)
            final_patched = True
    if not validation_patched or not final_patched:
        raise RuntimeError(
            f"Source notebook patch points changed: validation={validation_patched}, final={final_patched}"
        )

    # Configure the generated notebook as a standalone full Kaggle run.
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if "CONFIG = ModelConfig(" in source and "USE_SPIKE_WEIGHTS" in source:
            source = source.replace('MODE = "all"', 'MODE = "all"')
            source = source.replace("PRUNE_FEATURES = True", "PRUNE_FEATURES = False")
            source = source.replace(
                "CONFIG = ModelConfig(\n",
                "CONFIG = ModelConfig(\n    ensemble=V3_ENSEMBLE,\n",
                1,
            )
            cell["source"] = source.splitlines(keepends=True)

    # A generated deliverable must start clean and deterministic.
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    notebook.setdefault("metadata", {})["v3_optimization"] = {
        "source_notebook": SOURCE.name,
        "target_metric": "RMSE",
        "leaderboard_target": 0.81,
        "guaranteed": False,
        "annual_lag_days": list((364, 365, 366, 728, 729, 730, 731)),
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(cells)} cells")


if __name__ == "__main__":
    main()
