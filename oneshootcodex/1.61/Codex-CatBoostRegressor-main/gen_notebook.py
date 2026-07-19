"""Generate notebook-optimized.ipynb"""
import json, uuid

def uid(): return str(uuid.uuid4())[:8]

def md(text):
    return {"cell_type":"markdown","id":uid(),"metadata":{},"source":text}

def code(src):
    return {"cell_type":"code","execution_count":None,"id":uid(),"metadata":{},"outputs":[],"source":src}

# ───────────────────────────────────────────────────────────
# CELL SOURCES
# ───────────────────────────────────────────────────────────

TITLE = """\
# Prediksi Tinggi Muka Air Sungai — Optimized v2

Pipeline end-to-end dengan 5 optimasi dibanding notebook-150726:
1. **Enhanced spike detector** — one-sided extreme + physical impossible (tma=0)
2. **Dual-objective CatBoost** — blend RMSE + Huber objectives via OOF grid search
3. **Recency sample weighting** — data terbaru lebih diprioritaskan (half-life 730 hari)
4. **GPU auto-detection** — otomatis pakai GPU di Kaggle
5. **Validation anomaly flag** — spike detector juga dijalankan di validation fold

**Guard training:** default `MODE = "audit"` dan `CONFIRM_TRAIN = False`."""

SEC1 = "## 1. Import, path, dan konfigurasi"

C_CONFIG = '''\
"""Optimized TMA forecasting pipeline — v2.

Changes from notebook-150726:
  [OPT-1] Enhanced spike detector (one-sided extreme + tma==0 physical impossible)
  [OPT-2] Dual-objective CatBoost (RMSE + Huber) with OOF blend search
  [OPT-3] Recency sample weighting multiplied into training weights
  [OPT-4] GPU auto-detection for Kaggle
  [OPT-5] Spike detector also run on validation for accurate clean_RMSE
"""
from __future__ import annotations

import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import sys
import traceback
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

LOGGER = logging.getLogger("tma_forecasting_v2")
KEYS = ["datetime", "nama_pos"]
ID_PATTERN = re.compile(
    r"^(?P<datetime>\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) - (?P<nama_pos>.+)$"
)

# ── [OPT-4] GPU auto-detection ───────────────────────────────
ON_KAGGLE = os.path.exists("/kaggle/working")
TASK_TYPE = "GPU" if ON_KAGGLE else "CPU"
LOGGER.info("Running on Kaggle=%s, task_type=%s", ON_KAGGLE, TASK_TYPE)


@dataclass(frozen=True)
class Paths:
    root: Path = Path("/kaggle/input/sebelas-maret-statistics-data-science-2026") if ON_KAGGLE else Path("datasets")
    artifacts: Path = Path("/kaggle/working/artifacts") if ON_KAGGLE else Path("artifacts")

    @property
    def train(self) -> Path: return self.root / "train.csv"
    @property
    def test(self) -> Path: return self.root / "test.csv"
    @property
    def sample_submission(self) -> Path: return self.root / "sample_submission.csv"
    @property
    def coordinates(self) -> Path: return self.root / "data_pendukung" / "koordinat_pos.csv"
    @property
    def environment(self) -> Path: return self.root / "data_pendukung" / "data_lingkungan.csv"
    @property
    def rivers(self) -> Path:
        return self.root / "data_pendukung" / "HydroRIVERS_v10_au_shp" / "HydroRIVERS_v10_au.shp"


@dataclass(frozen=True)
class ModelConfig:
    seeds: tuple[int, ...] = (17, 41, 83)
    default_iterations: int = 2000
    learning_rate: float = 0.035
    depth: int = 8
    l2_leaf_reg: float = 10.0
    huber_delta: float = 1.5
    early_stopping_rounds: int = 200
    verbose_every: int = 200
    upstream_decay_km: float = 75.0
    snapshot_interval_seconds: int = 300
    # [OPT-2] dual-objective blend grid
    blend_grid: tuple[float, ...] = tuple(i / 10 for i in range(11))
    # [OPT-3] recency weighting half-life
    recency_half_life_days: float = 730.0


FOLDS = (
    ("sep_2023", "2023-09-18 18:00:00", "2023-09-19 06:00:00", "2024-05-18 18:00:00"),
    ("may_2024", "2024-05-18 18:00:00", "2024-05-19 06:00:00", "2025-01-18 18:00:00"),
    ("sep_2024", "2024-09-18 18:00:00", "2024-09-19 06:00:00", "2025-05-18 18:00:00"),
    ("jan_2025", "2025-01-18 18:00:00", "2025-01-19 06:00:00", "2025-09-18 18:00:00"),
)

RIVER_NUMERIC = [
    "LENGTH_KM","DIST_DN_KM","DIST_UP_KM","CATCH_SKM",
    "UPLAND_SKM","DIS_AV_CMS","ORD_STRA","ORD_CLAS","ORD_FLOW",
]

RAW_STATE_COLUMNS = [
    "soil_moisture_0_7cm","soil_moisture_7_28cm","soil_moisture_28_100cm",
    "soil_moisture_100_255cm","surface_pressure_hpa","pressure_msl_hpa",
    "rmm1","rmm2","mjo_phase","mjo_amplitude","mjo_active","nino_34",
]

CATEGORICAL_COLUMNS = [
    "nama_pos","landcover_name","hour_cat","month_cat","mjo_phase_cat",
    "main_riv_cat","hybas_l12_cat","landcover_class_cat","endorheic_cat",
]

NON_FEATURE_COLUMNS = {
    "datetime","id","tma_mdpl","_row_order","mjo_phase","landcover_class",
    "HYRIV_ID","NEXT_DOWN","MAIN_RIV","HYBAS_L12","ENDORHEIC","wind_direction_deg",
}

# Guard
MODE = "audit"           # audit | prepare | validate | fit | all | predict
CONFIRM_TRAIN = False    # set True sebelum training
'''

SEC2 = "## 2. Parsing ID, pemuatan data, dan audit schema"

C_DATA = '''\
def parse_submission_ids(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame["id"].str.extract(ID_PATTERN)
    if parsed.isna().any(axis=None):
        bad = frame.loc[parsed.isna().any(axis=1), "id"].head(3).tolist()
        raise ValueError(f"Invalid submission IDs: {bad}")
    result = frame.copy()
    result["datetime"] = pd.to_datetime(parsed["datetime"], errors="raise")
    result["nama_pos"] = parsed["nama_pos"]
    result["_row_order"] = np.arange(len(result), dtype=np.int64)
    return result


def load_inputs(paths: Paths) -> tuple[pd.DataFrame, ...]:
    train = pd.read_csv(paths.train, parse_dates=["datetime"])
    test_raw = pd.read_csv(paths.test)
    sample = pd.read_csv(paths.sample_submission)
    coordinates = pd.read_csv(paths.coordinates)
    environment = pd.read_csv(paths.environment, parse_dates=["datetime"])
    test = parse_submission_ids(test_raw)
    return train, test, sample, coordinates, environment


def audit_inputs(train, test, sample, coordinates, environment) -> dict:
    expected_train = {"datetime", "nama_pos", "tma_mdpl"}
    if set(train.columns) != expected_train:
        raise ValueError(f"Unexpected train schema: {train.columns.tolist()}")
    if train.duplicated(KEYS).any() or environment.duplicated(KEYS).any():
        raise ValueError("Duplicate (datetime, nama_pos) keys detected")
    if train["tma_mdpl"].isna().any():
        raise ValueError("Target contains explicit NaN values")
    if test["id"].duplicated().any():
        raise ValueError("Duplicate test IDs detected")
    if not test["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise ValueError("test.csv and sample_submission.csv IDs/order differ")
    train_stations = set(train["nama_pos"])
    for label, stations in {
        "test": set(test["nama_pos"]),
        "coordinates": set(coordinates["nama_pos"]),
        "environment": set(environment["nama_pos"]),
    }.items():
        if stations != train_stations:
            raise ValueError(f"Station mismatch for {label}")
    expected_hours = {6, 12, 18}
    if set(train["datetime"].dt.hour) - expected_hours:
        raise ValueError("Train contains target hours outside 06, 12, 18")
    if set(test["datetime"].dt.hour) != expected_hours:
        raise ValueError("Test hours are not exactly 06, 12, 18")
    env_counts = environment.groupby("nama_pos", observed=True).size()
    expected_env_rows = len(
        pd.date_range(environment["datetime"].min(), environment["datetime"].max(), freq="h")
    )
    if not (env_counts == expected_env_rows).all():
        raise ValueError("Environment is not a complete hourly grid for every station")
    return {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "environment_rows": int(len(environment)), "stations": int(len(train_stations)),
        "train_start": str(train["datetime"].min()), "train_end": str(train["datetime"].max()),
        "test_start": str(test["datetime"].min()), "test_end": str(test["datetime"].max()),
        "environment_missing_cells": int(environment.isna().sum().sum()),
        "solar_minus_999_cells": int((environment["solar_radiation_mj_m2"] <= -900).sum()),
    }
'''

SEC3 = "## 3. HydroRIVERS dan graf upstream"

C_RIVER = '''\
def _read_local_rivers(path: Path, coordinates: pd.DataFrame) -> gpd.GeoDataFrame:
    pad = 0.75
    bbox = (
        coordinates["longitude"].min() - pad, coordinates["latitude"].min() - pad,
        coordinates["longitude"].max() + pad, coordinates["latitude"].max() + pad,
    )
    rivers = gpd.read_file(path, bbox=bbox)
    required = {"HYRIV_ID","NEXT_DOWN","MAIN_RIV","HYBAS_L12","ENDORHEIC","geometry",*RIVER_NUMERIC}
    missing = required - set(rivers.columns)
    if missing:
        raise ValueError(f"HydroRIVERS is missing columns: {sorted(missing)}")
    if rivers.crs is None:
        raise ValueError("HydroRIVERS has no CRS")
    return rivers


def match_stations_to_rivers(coordinates: pd.DataFrame, rivers: gpd.GeoDataFrame) -> pd.DataFrame:
    points = gpd.GeoDataFrame(
        coordinates.copy(),
        geometry=gpd.points_from_xy(coordinates["longitude"], coordinates["latitude"]),
        crs="EPSG:4326",
    )
    points_utm = points.to_crs("EPSG:32749")
    rivers_utm = rivers.to_crs("EPSG:32749")
    nearest = gpd.sjoin_nearest(points_utm, rivers_utm, how="left", distance_col="river_distance_m")
    nearest = (
        nearest.sort_values("river_distance_m")
        .loc[lambda x: ~x.index.duplicated(keep="first")]
        .sort_index()
    )
    if nearest["HYRIV_ID"].isna().any():
        raise ValueError("At least one station could not be matched to HydroRIVERS")
    keep = ["nama_pos","latitude","longitude","river_distance_m","HYRIV_ID","NEXT_DOWN",
            "MAIN_RIV","HYBAS_L12","ENDORHEIC",*RIVER_NUMERIC]
    static = pd.DataFrame(nearest[keep]).reset_index(drop=True)
    static["river_match_confidence"] = np.exp(-static["river_distance_m"] / 1000.0)
    for col in ["CATCH_SKM","UPLAND_SKM","DIS_AV_CMS","LENGTH_KM"]:
        static[f"log1p_{col.lower()}"] = np.log1p(static[col].clip(lower=0))
    static["main_riv_cat"] = static["MAIN_RIV"].astype("Int64").astype(str)
    static["hybas_l12_cat"] = static["HYBAS_L12"].astype("Int64").astype(str)
    static["endorheic_cat"] = static["ENDORHEIC"].astype("Int64").astype(str)
    return static


def build_upstream_weights(static: pd.DataFrame, rivers: gpd.GeoDataFrame, decay_km: float) -> pd.DataFrame:
    stations = static["nama_pos"].tolist()
    reach_by_station = dict(zip(static["nama_pos"], static["HYRIV_ID"].astype(int)))
    dist_by_station = dict(zip(static["nama_pos"], static["DIST_DN_KM"].astype(float)))
    station_by_reach = {reach: station for station, reach in reach_by_station.items()}
    next_down = dict(zip(rivers["HYRIV_ID"].astype(int), rivers["NEXT_DOWN"].astype(int)))
    weights = pd.DataFrame(0.0, index=stations, columns=stations)
    for source in stations:
        weights.loc[source, source] = 1.0
        current = reach_by_station[source]
        seen = {current}
        for _ in range(len(next_down) + 1):
            current = next_down.get(current, 0)
            if current == 0 or current in seen:
                break
            seen.add(current)
            target = station_by_reach.get(current)
            if target is not None:
                distance = max(0.0, dist_by_station[source] - dist_by_station[target])
                weights.loc[target, source] = max(
                    weights.loc[target, source], math.exp(-distance / decay_km)
                )
    return weights.div(weights.sum(axis=1), axis=0)


def _add_upstream_environment(
    environment: pd.DataFrame, weights: pd.DataFrame, source_columns: Iterable[str]
) -> pd.DataFrame:
    result = environment
    station_order = weights.columns.tolist()
    for column in source_columns:
        pivot = result.pivot(index="datetime", columns="nama_pos", values=column).reindex(columns=station_order)
        upstream = pd.DataFrame(
            pivot.to_numpy(dtype=float) @ weights.to_numpy(dtype=float).T,
            index=pivot.index, columns=weights.index,
        )
        long = (
            upstream.rename_axis(columns="nama_pos").stack(future_stack=True)
            .rename(f"upstream_{column}").reset_index()
        )
        result = result.merge(long, on=KEYS, how="left", validate="one_to_one")
    return result
'''

SEC4 = "## 4. Sanitasi lingkungan dan fitur antecedent"

C_ENV = '''\
def prepare_environment(environment: pd.DataFrame, upstream_weights: pd.DataFrame) -> pd.DataFrame:
    env = environment.sort_values(KEYS).copy()
    if not env["rainfall_mm"].equals(env["rainfall_openmeteo_mm"]):
        raise ValueError("Rainfall duplicate columns have unexpectedly diverged")
    env = env.drop(columns="rainfall_openmeteo_mm")

    env["solar_radiation_missing"] = (
        env["solar_radiation_mj_m2"].isna() | (env["solar_radiation_mj_m2"] <= -900)
    ).astype("int8")
    env.loc[env["solar_radiation_mj_m2"] <= -900, "solar_radiation_mj_m2"] = np.nan

    for col in RAW_STATE_COLUMNS:
        env[f"{col}_missing"] = env[col].isna().astype("int8")

    by_station = env.groupby("nama_pos", sort=False, observed=True)
    short_state = [c for c in RAW_STATE_COLUMNS if c != "nino_34"]
    env[short_state] = by_station[short_state].ffill(limit=24)
    env[["nino_34"]] = by_station[["nino_34"]].ffill(limit=31 * 24)

    radians = np.deg2rad(env["wind_direction_deg"])
    env["wind_direction_sin"] = np.sin(radians)
    env["wind_direction_cos"] = np.cos(radians)
    env["dewpoint_depression_c"] = env["temperature_c"] - env["dew_point_c"]

    env = _add_upstream_environment(env, upstream_weights, ["rainfall_mm", "soil_moisture_0_7cm"])
    env = env.sort_values(KEYS).reset_index(drop=True)
    grouped = env.groupby("nama_pos", sort=False, observed=True)

    rolling_sum_specs = {
        "rainfall_mm": (6, 24, 72, 168, 336),
        "upstream_rainfall_mm": (6, 24, 72, 168, 336),
    }
    for col, windows in rolling_sum_specs.items():
        for hours in windows:
            env[f"{col}_sum_{hours}h"] = grouped[col].transform(
                lambda s, w=hours: s.rolling(w, min_periods=max(1, w // 4)).sum()
            )
        env[f"{col}_max_24h"] = grouped[col].transform(lambda s: s.rolling(24, min_periods=6).max())
        for half_life in (12, 48, 168):
            env[f"{col}_api_halflife_{half_life}h"] = grouped[col].transform(
                lambda s, h=half_life: s.ewm(halflife=h, adjust=False, min_periods=1).mean()
            )

    rolling_mean_specs = {
        "soil_moisture_0_7cm": (24, 168), "soil_moisture_28_100cm": (24, 168),
        "upstream_soil_moisture_0_7cm": (24, 168),
        "temperature_c": (24, 72), "humidity_pct": (24, 72),
        "surface_pressure_hpa": (24,),
    }
    for col, windows in rolling_mean_specs.items():
        for hours in windows:
            env[f"{col}_mean_{hours}h"] = grouped[col].transform(
                lambda s, w=hours: s.rolling(w, min_periods=max(1, w // 4)).mean()
            )

    env["rainfall_mm"] = env["rainfall_mm"].clip(lower=0)
    env["solar_radiation_mj_m2"] = env["solar_radiation_mj_m2"].clip(lower=0)
    for col in [c for c in env.columns if "soil_moisture" in c]:
        env[col] = env[col].clip(0, 1)
    return env
'''

SEC5 = "## 5. Kalender, kategori, dan matriks fitur"

C_FEAT = '''\
def add_calendar_and_categories(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dt = out["datetime"]
    doy = dt.dt.dayofyear.astype(float)
    hour = dt.dt.hour.astype(float)
    out["dayofyear_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["dayofyear_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["is_wet_season"] = dt.dt.month.isin([11, 12, 1, 2, 3, 4]).astype("int8")
    out["hour_cat"] = dt.dt.hour.astype(str)
    out["month_cat"] = dt.dt.month.astype(str)
    out["mjo_phase_cat"] = out["mjo_phase"].round().astype("Int64").astype(str)
    out["landcover_class_cat"] = out["landcover_class"].astype("Int64").astype(str)
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].fillna("__MISSING__").astype(str)
    return out


def build_feature_tables(
    train, test, coordinates, environment, paths: Paths, config: ModelConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rivers = _read_local_rivers(paths.rivers, coordinates)
    static = match_stations_to_rivers(coordinates, rivers)
    weights = build_upstream_weights(static, rivers, config.upstream_decay_km)
    env_features = prepare_environment(environment, weights)

    train_features = train.merge(env_features, on=KEYS, how="left", validate="many_to_one")
    test_features = test.merge(env_features, on=KEYS, how="left", validate="many_to_one")
    train_features = train_features.merge(static, on="nama_pos", how="left", validate="many_to_one")
    test_features = test_features.merge(static, on="nama_pos", how="left", validate="many_to_one")
    if train_features["rainfall_mm"].isna().any() or test_features["rainfall_mm"].isna().any():
        raise ValueError("Some target timestamps have no matching environmental row")

    train_features = add_calendar_and_categories(train_features)
    test_features = add_calendar_and_categories(test_features)
    return train_features, test_features, weights


def feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]
    forbidden = [c for c in cols if c.startswith("tma_") or "target" in c.lower()]
    if forbidden:
        raise ValueError(f"Target leakage features detected: {forbidden}")
    return cols
'''

SEC6 = """\
## 6. Target, spike weighting (enhanced), metrik, dan model

**[OPT-1]** Spike detector ditingkatkan dengan dua mekanisme baru:
- **One-sided extreme**: deteksi satu-tetangga cukup jika lompatan sangat besar (tangkap Peren 160/170m)
- **Physical impossible**: `tma == 0` untuk pos dengan median > 5m (tangkap Jurug sensor reset)

**[OPT-2]** Dua fungsi `make_model_*` untuk dual-objective CatBoost blend.

**[OPT-3]** `recency_weights` menambah prioritas data terbaru."""

C_TARGET = '''\
def fit_target_stats(train: pd.DataFrame) -> pd.DataFrame:
    grouped = train.groupby("nama_pos", observed=True)["tma_mdpl"]
    stats = grouped.agg(center="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
    stats["scale"] = ((stats["q75"] - stats["q25"]) / 1.349).clip(lower=0.05)
    return stats[["center", "scale"]]


def normalize_target(frame: pd.DataFrame, stats: pd.DataFrame) -> np.ndarray:
    center = frame["nama_pos"].map(stats["center"])
    scale = frame["nama_pos"].map(stats["scale"])
    if center.isna().any() or scale.isna().any():
        raise ValueError("Target normalization encountered an unseen station")
    return ((frame["tma_mdpl"] - center) / scale).to_numpy(dtype=float)


def denormalize_prediction(prediction: np.ndarray, stations: pd.Series, stats: pd.DataFrame) -> np.ndarray:
    center = stations.map(stats["center"]).to_numpy(dtype=float)
    scale = stations.map(stats["scale"]).to_numpy(dtype=float)
    return center + scale * np.asarray(prediction, dtype=float)


# ── [OPT-1] Enhanced spike detector ──────────────────────────────────────────
def compute_label_weights(train: pd.DataFrame, downweight: float = 0.02) -> tuple[np.ndarray, int]:
    """Enhanced spike detector with 3 tiers:

    Tier 1 (original): isolated spike — both neighbors close + agree + current extreme.
    Tier 2 (NEW): one-sided extreme — only one neighbor needed if jump is astronomically large.
                  Catches Peren tma=160/170m (normal ~91m) and similar massive single-point spikes.
    Tier 3 (NEW): physical impossible — tma==0 for stations whose median level >> 0.
                  Catches Jurug sensor reset (normal ~78–84m, recorded 0.0).
    """
    ordered = train.sort_values(KEYS).copy()
    group = ordered.groupby("nama_pos", sort=False, observed=True)
    ordered["prev_y"] = group["tma_mdpl"].shift(1)
    ordered["next_y"] = group["tma_mdpl"].shift(-1)
    ordered["prev_t"] = group["datetime"].shift(1)
    ordered["next_t"] = group["datetime"].shift(-1)

    diff_scale = group["tma_mdpl"].transform(
        lambda s: (s.diff().abs().median() * 1.4826)
    ).clip(lower=0.05)
    station_stats = fit_target_stats(ordered)
    robust_scale = ordered["nama_pos"].map(station_stats["scale"])

    midpoint = (ordered["prev_y"] + ordered["next_y"]) / 2.0

    # --- Tier 1: classic isolated spike ---
    adjacent = (
        ((ordered["datetime"] - ordered["prev_t"]) <= pd.Timedelta(hours=12))
        & ((ordered["next_t"] - ordered["datetime"]) <= pd.Timedelta(hours=12))
    )
    neighbor_gap = (ordered["prev_y"] - ordered["next_y"]).abs()
    neighbors_agree = neighbor_gap <= np.maximum(4.0 * diff_scale, 2.0 * robust_scale)
    threshold = np.maximum(12.0 * diff_scale, 8.0 * robust_scale)
    tier1 = adjacent & neighbors_agree & ((ordered["tma_mdpl"] - midpoint).abs() > threshold)

    # --- Tier 2 (NEW): one-sided extreme jump --------------------------------
    # Only one close neighbor is required. Threshold is much higher to avoid
    # flagging real flood peaks. Catches giant sensor spikes like Peren 160/170m.
    prev_close = (ordered["datetime"] - ordered["prev_t"]) <= pd.Timedelta(hours=12)
    next_close = (ordered["next_t"] - ordered["datetime"]) <= pd.Timedelta(hours=12)
    min_jump_to_either_neighbor = pd.concat(
        [
            (ordered["tma_mdpl"] - ordered["prev_y"]).abs().fillna(np.inf),
            (ordered["tma_mdpl"] - ordered["next_y"]).abs().fillna(np.inf),
        ],
        axis=1,
    ).min(axis=1)
    # Threshold: at least 25× typical step AND at least 10× station IQR-scale.
    # Real floods don't clear this because they last multiple timestamps.
    extreme_threshold = np.maximum(25.0 * diff_scale, 10.0 * robust_scale)
    tier2 = (prev_close | next_close) & (min_jump_to_either_neighbor > extreme_threshold)

    # --- Tier 3 (NEW): physical impossible -----------------------------------
    # tma == 0 is physically impossible for inland gauges with elevation >> 0.
    # Guard: only flag if station median > 5m to avoid tidal / low-lying stations.
    station_median_map = ordered.groupby("nama_pos", observed=True)["tma_mdpl"].median()
    station_median = ordered["nama_pos"].map(station_median_map)
    tier3 = (ordered["tma_mdpl"] == 0.0) & (station_median > 5.0)

    spike = tier1 | tier2 | tier3
    ordered["weight"] = np.where(spike, downweight, 1.0)
    weights = ordered["weight"].reindex(train.index).to_numpy(dtype=float)
    return weights, int(spike.sum())


# ── [OPT-3] Recency weighting ─────────────────────────────────────────────────
def recency_weights(datetime_series: pd.Series, half_life_days: float = 730.0) -> np.ndarray:
    """Exponential recency weights: recent data gets weight 1.0, old data as low as 0.25."""
    age = (datetime_series.max() - datetime_series).dt.total_seconds().to_numpy() / 86400.0
    return np.clip(np.exp(-np.log(2.0) * age / half_life_days), 0.25, 1.0)


def metrics_frame(y_true: np.ndarray, y_pred: np.ndarray, stations: pd.Series) -> dict:
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    station_mae = (
        pd.DataFrame({"station": stations.to_numpy(), "ae": errors})
        .groupby("station", observed=True)["ae"].mean()
    )
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "macro_station_mae": float(station_mae.mean()),
        "worst_station_mae": float(station_mae.max()),
        "p95_absolute_error": float(np.quantile(errors, 0.95)),
    }


def seasonal_baseline(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    reference = train.assign(month=train["datetime"].dt.month, hour=train["datetime"].dt.hour)
    lookup = reference.groupby(["nama_pos", "month", "hour"], observed=True)["tma_mdpl"].median()
    fallback = reference.groupby("nama_pos", observed=True)["tma_mdpl"].median()
    key = pd.MultiIndex.from_arrays(
        [valid["nama_pos"], valid["datetime"].dt.month, valid["datetime"].dt.hour]
    )
    pred = lookup.reindex(key).to_numpy(dtype=float)
    missing = np.isnan(pred)
    pred[missing] = valid.loc[missing, "nama_pos"].map(fallback).to_numpy(dtype=float)
    return pred


# ── [OPT-2] Dual-objective CatBoost ──────────────────────────────────────────
def _base_catboost_params(config: ModelConfig, seed: int, iterations: int) -> dict:
    return dict(
        iterations=iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        random_seed=seed,
        random_strength=0.8,
        bootstrap_type="Bayesian",
        bagging_temperature=0.7,
        task_type=TASK_TYPE,      # [OPT-4]
        thread_count=-1,
        verbose=config.verbose_every,
    )


def make_model_rmse(config: ModelConfig, seed: int, iterations: int, train_dir: Path | None = None):
    """[OPT-2] Objective A: RMSE — directly optimizes the competition metric."""
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("CatBoost required. Install requirements.txt first.") from exc
    params = _base_catboost_params(config, seed, iterations)
    params.update(loss_function="RMSE", eval_metric="RMSE",
                  allow_writing_files=train_dir is not None)
    if train_dir is not None:
        train_dir.mkdir(parents=True, exist_ok=True)
        params.update(train_dir=str(train_dir), save_snapshot=True,
                      snapshot_file="training.snapshot",
                      snapshot_interval=config.snapshot_interval_seconds)
    return CatBoostRegressor(**params)


def make_model_huber(config: ModelConfig, seed: int, iterations: int, train_dir: Path | None = None):
    """[OPT-2] Objective B: Huber — robust to residual outliers (sensor spikes)."""
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("CatBoost required. Install requirements.txt first.") from exc
    params = _base_catboost_params(config, seed, iterations)
    params.update(loss_function=f"Huber:delta={config.huber_delta}", eval_metric="MAE",
                  allow_writing_files=train_dir is not None)
    if train_dir is not None:
        train_dir.mkdir(parents=True, exist_ok=True)
        params.update(train_dir=str(train_dir), save_snapshot=True,
                      snapshot_file="training.snapshot",
                      snapshot_interval=config.snapshot_interval_seconds)
    return CatBoostRegressor(**params)


def choose_blend_weight(
    y_true: np.ndarray,
    rmse_pred: np.ndarray,
    huber_pred: np.ndarray,
    spike_mask: np.ndarray,
    grid: tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    """[OPT-2] Select Huber share using clean_RMSE + 0.25*raw_RMSE guardrail.

    Selection score mirrors Ibra's approach: prioritise clean data but keep
    raw RMSE as an anchor to prevent overfitting to the anomaly-flag mask.
    """
    rows = []
    clean_mask = ~spike_mask
    for huber_weight in grid:
        blend = (1 - huber_weight) * rmse_pred + huber_weight * huber_pred
        raw_rmse = float(mean_squared_error(y_true, blend) ** 0.5)
        clean_rmse = float(mean_squared_error(y_true[clean_mask], blend[clean_mask]) ** 0.5)
        rows.append({
            "huber_weight": huber_weight, "raw_rmse": raw_rmse,
            "clean_rmse": clean_rmse, "score": clean_rmse + 0.25 * raw_rmse,
        })
    scores = pd.DataFrame(rows)
    best = float(scores.sort_values(["score", "raw_rmse"]).iloc[0]["huber_weight"])
    return best, scores


def _xy(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.loc[:, columns].copy()
    for col in set(columns) & set(CATEGORICAL_COLUMNS):
        result[col] = result[col].fillna("__MISSING__").astype(str)
    return result
'''

SEC7 = "## 7. Artefak dan handoff"

C_HANDOFF = '''\
HANDOFF_SCHEMA_VERSION = "2.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_frame(frame: pd.DataFrame, path_without_suffix: Path) -> Path:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = path_without_suffix.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet_path, index=False, compression="zstd")
        return parquet_path
    except (ImportError, ModuleNotFoundError):
        csv_path = path_without_suffix.with_suffix(".csv.gz")
        frame.to_csv(csv_path, index=False, compression="gzip")
        return csv_path


def runtime_versions() -> dict:
    packages = ["catboost","geopandas","numpy","pandas","pyarrow","scikit-learn","shapely"]
    versions = {}
    for pkg in packages:
        try: versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: versions[pkg] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}


def create_run_directory(output_root: Path, mode: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"handoff_{stamp}_{mode}"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"handoff_{stamp}_{mode}_{suffix:02d}"
        suffix += 1
    for child in ("data", "metrics", "models", "training_progress"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def update_manifest(run_dir: Path, **changes) -> dict:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest.update(changes)
    manifest["updated_at_utc"] = utc_now_iso()
    write_json(path, manifest)
    return manifest


def create_handoff_zip(run_dir: Path, status: str) -> Path:
    if status not in {"in_progress", "complete", "failed"}:
        raise ValueError("Invalid handoff ZIP status")
    destination = run_dir.parent / f"{run_dir.name}_{status}.zip"
    temporary = destination.with_suffix(".zip.tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                already_compressed = path.suffix.lower() in {".parquet", ".gz", ".cbm", ".snapshot"}
                archive.write(
                    path, arcname=str(Path(run_dir.name) / path.relative_to(run_dir)),
                    compress_type=zipfile.ZIP_STORED if already_compressed else zipfile.ZIP_DEFLATED,
                )
    temporary.replace(destination)
    if status in {"complete", "failed"}:
        in_progress = run_dir.parent / f"{run_dir.name}_in_progress.zip"
        if in_progress.exists():
            in_progress.unlink()
    return destination
'''

SEC8 = """\
## 8. Rolling-origin validation (dual-objective + recency)

**[OPT-2]** Setiap fold melatih 2 objective × 3 seed = 6 model CatBoost.
Blend weight dicari via OOF grid search (`clean_RMSE + 0.25 × raw_RMSE`).

**[OPT-3]** Sample weight = spike_weight × recency_factor.

**[OPT-5]** Spike detector juga dijalankan pada validation fold untuk `anomaly_flag` yang akurat."""

C_VALID = '''\
def run_validation(
    train_features: pd.DataFrame,
    columns: list[str],
    config: ModelConfig,
    artifacts: Path,
) -> tuple[pd.DataFrame, dict]:
    records: list[dict] = []
    best_iters_rmse: list[int] = []
    best_iters_huber: list[int] = []
    oof_parts: list[pd.DataFrame] = []
    station_records: list[dict] = []
    importance_records: list[dict] = []
    cat_columns = [c for c in CATEGORICAL_COLUMNS if c in columns]

    for fold_name, train_end, valid_start, valid_end in FOLDS:
        fit = train_features[train_features["datetime"] <= pd.Timestamp(train_end)].copy()
        valid = train_features[
            train_features["datetime"].between(
                pd.Timestamp(valid_start), pd.Timestamp(valid_end), inclusive="both"
            )
        ].copy()
        eligible = fit.groupby("nama_pos", observed=True).size().loc[lambda s: s >= 90].index
        fit = fit[fit["nama_pos"].isin(eligible)]
        valid = valid[valid["nama_pos"].isin(eligible)]
        if fit.empty or valid.empty:
            LOGGER.warning("Skipping empty fold %s", fold_name)
            continue

        stats = fit_target_stats(fit)
        y_fit = normalize_target(fit, stats)
        y_valid_z = normalize_target(valid, stats)

        # [OPT-1] Enhanced spike detector on training set
        spike_weights, spike_count = compute_label_weights(fit)
        # [OPT-3] Recency weighting multiplied in
        sample_w = spike_weights * recency_weights(fit["datetime"], config.recency_half_life_days)

        # [OPT-5] Anomaly flag on validation set for clean_RMSE diagnostic
        valid_spike_weights, valid_spike_count = compute_label_weights(valid)
        valid_anomaly_flag = (valid_spike_weights < 1.0).astype("int8")

        progress_dir = artifacts / "training_progress" / f"fold_{fold_name}"

        # --- [OPT-2] Train both objectives, 3 seeds each ---
        rmse_preds_valid, huber_preds_valid = [], []
        fold_best_iters_rmse, fold_best_iters_huber = [], []

        for seed in config.seeds:
            # Objective A: RMSE
            model_r = make_model_rmse(
                config, seed=seed, iterations=config.default_iterations,
                train_dir=progress_dir / f"rmse_seed_{seed}",
            )
            model_r.fit(
                _xy(fit, columns), y_fit, cat_features=cat_columns,
                sample_weight=sample_w,
                eval_set=(_xy(valid, columns), y_valid_z),
                early_stopping_rounds=config.early_stopping_rounds,
                use_best_model=True,
            )
            rmse_preds_valid.append(model_r.predict(_xy(valid, columns)))
            fold_best_iters_rmse.append(int(model_r.get_best_iteration()) + 1)
            model_r.save_model(str(artifacts / "models" / f"val_{fold_name}_rmse_seed_{seed}.cbm"))
            for feat, imp in zip(columns, model_r.get_feature_importance()):
                importance_records.append({"fold": fold_name, "obj": "rmse", "seed": seed,
                                           "feature": feat, "importance": float(imp)})

            # Objective B: Huber
            model_h = make_model_huber(
                config, seed=seed, iterations=config.default_iterations,
                train_dir=progress_dir / f"huber_seed_{seed}",
            )
            model_h.fit(
                _xy(fit, columns), y_fit, cat_features=cat_columns,
                sample_weight=sample_w,
                eval_set=(_xy(valid, columns), y_valid_z),
                early_stopping_rounds=config.early_stopping_rounds,
                use_best_model=True,
            )
            huber_preds_valid.append(model_h.predict(_xy(valid, columns)))
            fold_best_iters_huber.append(int(model_h.get_best_iteration()) + 1)
            model_h.save_model(str(artifacts / "models" / f"val_{fold_name}_huber_seed_{seed}.cbm"))

        # Average across seeds per objective
        avg_rmse_z = np.mean(rmse_preds_valid, axis=0)
        avg_huber_z = np.mean(huber_preds_valid, axis=0)
        best_iters_rmse.append(int(np.median(fold_best_iters_rmse)))
        best_iters_huber.append(int(np.median(fold_best_iters_huber)))

        # [OPT-2] Find blend weight from OOF using clean_RMSE + 0.25*raw_RMSE
        pred_rmse = denormalize_prediction(avg_rmse_z, valid["nama_pos"], stats)
        pred_huber = denormalize_prediction(avg_huber_z, valid["nama_pos"], stats)
        huber_weight, blend_scores = choose_blend_weight(
            valid["tma_mdpl"].to_numpy(), pred_rmse, pred_huber,
            valid_anomaly_flag.astype(bool), config.blend_grid,
        )
        pred = (1 - huber_weight) * pred_rmse + huber_weight * pred_huber
        blend_scores.to_csv(
            artifacts / "metrics" / f"blend_search_{fold_name}.csv", index=False
        )

        # Metrics
        model_metrics = metrics_frame(valid["tma_mdpl"].to_numpy(), pred, valid["nama_pos"])
        clean_mask = valid_anomaly_flag == 0
        clean_metrics = metrics_frame(
            valid.loc[clean_mask, "tma_mdpl"].to_numpy(),
            pred[clean_mask], valid.loc[clean_mask, "nama_pos"],
        )
        baseline_pred = seasonal_baseline(fit, valid)
        baseline_metrics = metrics_frame(valid["tma_mdpl"].to_numpy(), baseline_pred, valid["nama_pos"])

        # OOF
        oof = valid[KEYS + ["tma_mdpl"]].copy()
        oof["fold"] = fold_name
        oof["prediction"] = pred
        oof["pred_rmse_obj"] = pred_rmse
        oof["pred_huber_obj"] = pred_huber
        oof["baseline_prediction"] = baseline_pred
        oof["error"] = oof["prediction"] - oof["tma_mdpl"]
        oof["absolute_error"] = oof["error"].abs()
        oof["anomaly_flag"] = valid_anomaly_flag  # [OPT-5]
        oof["huber_weight"] = huber_weight
        oof_parts.append(oof)

        for station, sg in oof.groupby("nama_pos", observed=True):
            station_records.append({
                "fold": fold_name, "nama_pos": station, "rows": len(sg),
                "mae": float(sg["absolute_error"].mean()),
                "rmse": float(np.sqrt(np.mean(sg["error"] ** 2))),
                "bias": float(sg["error"].mean()),
            })
        records.append({
            "fold": fold_name,
            "train_rows": len(fit), "valid_rows": len(valid),
            "valid_stations": valid["nama_pos"].nunique(),
            "spikes_downweighted_train": spike_count,
            "anomalies_valid": int(valid_spike_count),
            "huber_weight_selected": huber_weight,
            "iter_rmse": best_iters_rmse[-1],
            "iter_huber": best_iters_huber[-1],
            **{f"model_{k}": v for k, v in model_metrics.items()},
            **{f"clean_{k}": v for k, v in clean_metrics.items()},
            **{f"baseline_{k}": v for k, v in baseline_metrics.items()},
        })
        LOGGER.info("Fold %s: %s  huber_w=%.2f", fold_name, model_metrics, huber_weight)

        cumulative_oof = pd.concat(oof_parts, ignore_index=True)
        save_frame(cumulative_oof, artifacts / "metrics" / "oof_predictions")
        pd.DataFrame(records).to_csv(artifacts / "metrics" / "validation_metrics.csv", index=False)
        pd.DataFrame(station_records).to_csv(artifacts / "metrics" / "fold_station_metrics.csv", index=False)
        pd.DataFrame(importance_records).to_csv(artifacts / "metrics" / "feature_importance.csv", index=False)

    if not records:
        raise RuntimeError("No validation fold was runnable")
    report = pd.DataFrame(records)
    recommended = {
        "rmse_obj": int(np.median(best_iters_rmse)),
        "huber_obj": int(np.median(best_iters_huber)),
    }
    report.to_csv(artifacts / "metrics" / "validation_metrics.csv", index=False)
    oof_all = pd.concat(oof_parts, ignore_index=True)
    save_frame(oof_all, artifacts / "metrics" / "oof_predictions")
    write_json(artifacts / "recommended_iterations.json", recommended)
    return report, recommended
'''

SEC9 = "## 9. Final ensemble dan submission"

C_FIT = '''\
def fit_final_and_submit(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    sample: pd.DataFrame,
    columns: list[str],
    config: ModelConfig,
    artifacts: Path,
    iterations: dict,
    huber_weight: float,
) -> Path:
    """[OPT-2] Train 3 seeds × 2 objectives = 6 models on full training data.
    Blend with the globally selected huber_weight from OOF.
    [OPT-1] Enhanced spike detector applied to full train.
    [OPT-3] Recency weighting applied.
    """
    artifacts.mkdir(parents=True, exist_ok=True)
    stats = fit_target_stats(train_features)
    y = normalize_target(train_features, stats)
    spike_w, spike_count = compute_label_weights(train_features)
    sample_w = spike_w * recency_weights(train_features["datetime"], config.recency_half_life_days)
    cat_columns = [c for c in CATEGORICAL_COLUMNS if c in columns]

    rmse_preds_test, huber_preds_test = [], []
    importance_final = []

    for seed in config.seeds:
        # RMSE objective
        model_r = make_model_rmse(
            config, seed=seed, iterations=iterations.get("rmse_obj", config.default_iterations),
            train_dir=artifacts / "training_progress" / f"final_rmse_seed_{seed}",
        )
        model_r.fit(_xy(train_features, columns), y, cat_features=cat_columns, sample_weight=sample_w)
        model_r.save_model(str(artifacts / "models" / f"final_rmse_seed_{seed}.cbm"))
        rmse_preds_test.append(model_r.predict(_xy(test_features, columns)))
        for feat, imp in zip(columns, model_r.get_feature_importance()):
            importance_final.append({"obj": "rmse", "seed": seed, "feature": feat, "importance": float(imp)})

        # Huber objective
        model_h = make_model_huber(
            config, seed=seed, iterations=iterations.get("huber_obj", config.default_iterations),
            train_dir=artifacts / "training_progress" / f"final_huber_seed_{seed}",
        )
        model_h.fit(_xy(train_features, columns), y, cat_features=cat_columns, sample_weight=sample_w)
        model_h.save_model(str(artifacts / "models" / f"final_huber_seed_{seed}.cbm"))
        huber_preds_test.append(model_h.predict(_xy(test_features, columns)))
        for feat, imp in zip(columns, model_h.get_feature_importance()):
            importance_final.append({"obj": "huber", "seed": seed, "feature": feat, "importance": float(imp)})

    # [OPT-2] Blend with OOF-selected weight
    avg_rmse_z = np.mean(rmse_preds_test, axis=0)
    avg_huber_z = np.mean(huber_preds_test, axis=0)
    mean_z = (1 - huber_weight) * avg_rmse_z + huber_weight * avg_huber_z
    prediction = denormalize_prediction(mean_z, test_features["nama_pos"], stats)

    ordered = pd.DataFrame({
        "_row_order": test_features["_row_order"], "id": test_features["id"], "tma_mdpl": prediction,
    }).sort_values("_row_order")
    submission = sample[["id"]].merge(ordered[["id", "tma_mdpl"]], on="id", how="left", validate="one_to_one")
    if submission["tma_mdpl"].isna().any() or not np.isfinite(submission["tma_mdpl"]).all():
        raise ValueError("Submission contains missing/non-finite predictions")

    output_path = artifacts / "submission.csv"
    submission.to_csv(output_path, index=False)
    stats.to_csv(artifacts / "data" / "target_normalization.csv")
    write_json(artifacts / "feature_columns.json", {"features": columns, "categorical": cat_columns})
    pd.DataFrame(importance_final).to_csv(artifacts / "metrics" / "final_feature_importance.csv", index=False)
    write_json(artifacts / "run_metadata.json", {
        "iterations": iterations, "huber_weight": huber_weight,
        "seeds": list(config.seeds), "spikes_downweighted": spike_count,
        "train_rows": len(train_features), "test_rows": len(test_features),
        "model_config": asdict(config), "optimizations": [
            "OPT-1: enhanced spike detector (one-sided extreme + tma==0 physical impossible)",
            "OPT-2: dual-objective CatBoost (RMSE + Huber) with OOF blend",
            "OPT-3: recency sample weighting (half-life 730d)",
            "OPT-4: GPU auto-detection",
            "OPT-5: validation anomaly flag for accurate clean_RMSE",
        ],
    })
    LOGGER.info("Submission saved: %s rows, %d predictions", len(submission), submission["tma_mdpl"].notna().sum())
    return output_path
'''

SEC10 = "## 10. Runner utama"

C_RUNNER = '''\
# ── Guard ────────────────────────────────────────────────────────────────────
# MODE = "audit"     → only load data and run audit (safe, default)
# MODE = "prepare"   → build feature matrices and save, no training
# MODE = "validate"  → run rolling-origin validation (requires CONFIRM_TRAIN=True)
# MODE = "fit"       → final fit + submission (requires CONFIRM_TRAIN=True)
# MODE = "all"       → validate + fit + submission (requires CONFIRM_TRAIN=True)
# MODE = "predict"   → inference from saved bundle (not implemented in this cell)

import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

paths = Paths()
config = ModelConfig()
run_dir = create_run_directory(paths.artifacts, MODE)
attach_log = lambda rd: None  # simplified for Kaggle; full handoff ZIP optional

print(f"MODE={MODE}  |  CONFIRM_TRAIN={CONFIRM_TRAIN}  |  TASK_TYPE={TASK_TYPE}")
print(f"  data_root   : {paths.root}")
print(f"  artifacts   : {paths.artifacts}")
print(f"  run_dir     : {run_dir}")
print()


def attach_log(run_dir: Path) -> None:
    for h in list(LOGGER.handlers):
        if getattr(h, "_tma_v2_handler", False):
            LOGGER.removeHandler(h); h.close()
    h = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    h._tma_v2_handler = True
    LOGGER.addHandler(h)


try:
    attach_log(run_dir)
    write_json(run_dir / "config.json", asdict(config))
    write_json(run_dir / "runtime_versions.json", runtime_versions())
    update_manifest(run_dir, mode=MODE, status="in_progress", schema=HANDOFF_SCHEMA_VERSION)

    # ── Audit ────────────────────────────────────────────────────────────────
    LOGGER.info("Loading data...")
    train, test, sample, coordinates, environment = load_inputs(paths)
    audit = audit_inputs(train, test, sample, coordinates, environment)
    LOGGER.info("Audit OK: %s", audit)
    write_json(run_dir / "data" / "data_audit.json", audit)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    if MODE == "audit":
        print("Audit selesai. Set MODE='prepare' atau 'validate' untuk lanjut.")
        update_manifest(run_dir, status="complete", stage="audit_done")
        raise SystemExit(0)

    if MODE not in {"prepare", "validate", "fit", "all"} or not CONFIRM_TRAIN:
        if not CONFIRM_TRAIN:
            print("Set CONFIRM_TRAIN = True untuk menjalankan training.")
        raise SystemExit(0)

    # ── Feature engineering ──────────────────────────────────────────────────
    LOGGER.info("Building feature matrices (this takes a few minutes)...")
    train_features, test_features, upstream_weights = build_feature_tables(
        train, test, coordinates, environment, paths, config
    )
    cols = feature_columns(train_features)
    LOGGER.info("Feature matrix ready: %d features, %d train rows, %d test rows",
                len(cols), len(train_features), len(test_features))
    save_frame(train_features, run_dir / "data" / "train_features")
    save_frame(test_features, run_dir / "data" / "test_features")
    upstream_weights.to_csv(run_dir / "data" / "upstream_weights.csv")
    write_json(run_dir / "feature_columns.json", {
        "features": cols,
        "categorical": [c for c in CATEGORICAL_COLUMNS if c in cols],
    })
    update_manifest(run_dir, stage="features_ready", feature_count=len(cols))
    if MODE == "prepare":
        print(f"Feature matrices saved. {len(cols)} features. Set MODE='validate' untuk training.")
        update_manifest(run_dir, status="complete")
        raise SystemExit(0)

    # ── Validation ───────────────────────────────────────────────────────────
    huber_weight_global = 0.5  # will be updated from OOF
    if MODE in {"validate", "all"}:
        LOGGER.info("Starting rolling-origin validation (dual-objective, %d seeds)...", len(config.seeds))
        val_report, recommended_iters = run_validation(train_features, cols, config, run_dir)
        print()
        print("=== Validation Summary ===")
        display_cols = [c for c in [
            "fold","train_rows","valid_rows","model_rmse","model_mae",
            "clean_rmse","clean_mae","huber_weight_selected","spikes_downweighted_train","anomalies_valid",
        ] if c in val_report.columns]
        print(val_report[display_cols].to_string(index=False))
        print()
        print(f"Recommended iterations: {recommended_iters}")

        # Global huber_weight: average across folds (weighted by valid_rows)
        weights_fold = val_report["valid_rows"].to_numpy(dtype=float)
        huber_weight_global = float(
            np.average(val_report["huber_weight_selected"].to_numpy(), weights=weights_fold)
        )
        print(f"Global huber_weight (weighted average): {huber_weight_global:.3f}")
        update_manifest(run_dir, stage="validation_complete",
                        recommended_iters=recommended_iters,
                        huber_weight_global=huber_weight_global)
        create_handoff_zip(run_dir, "in_progress")

        if MODE == "validate":
            print("Validation done. Set MODE='fit' or 'all' for final submission.")
            update_manifest(run_dir, status="complete")
            raise SystemExit(0)

    # ── Final fit + submission ───────────────────────────────────────────────
    if MODE in {"fit", "all"}:
        # If coming from 'fit' without validate, use defaults
        if MODE == "fit":
            rec_path = run_dir / "recommended_iterations.json"
            if rec_path.exists():
                recommended_iters = json.loads(rec_path.read_text(encoding="utf-8"))
            else:
                recommended_iters = {"rmse_obj": config.default_iterations,
                                     "huber_obj": config.default_iterations}
                LOGGER.warning("No recommended_iterations.json found; using default %d", config.default_iterations)

        LOGGER.info("Starting final fit: iters=%s, huber_w=%.3f ...", recommended_iters, huber_weight_global)
        submission_path = fit_final_and_submit(
            train_features, test_features, sample, cols, config,
            run_dir, recommended_iters, huber_weight_global,
        )
        print(f"Submission saved to: {submission_path}")
        create_handoff_zip(run_dir, "complete")
        update_manifest(run_dir, status="complete", stage="final_fit_done",
                        submission_path=str(submission_path))

except SystemExit:
    pass
except Exception:
    tb = traceback.format_exc()
    LOGGER.error("Pipeline failed:\\n%s", tb)
    (run_dir / "error_traceback.txt").write_text(tb, encoding="utf-8")
    update_manifest(run_dir, status="failed")
    try: create_handoff_zip(run_dir, "failed")
    except Exception: pass
    raise
'''

# ───────────────────────────────────────────────────────────
# BUILD NOTEBOOK
# ───────────────────────────────────────────────────────────
cells = [
    md(TITLE),
    md(SEC1), code(C_CONFIG),
    md(SEC2), code(C_DATA),
    md(SEC3), code(C_RIVER),
    md(SEC4), code(C_ENV),
    md(SEC5), code(C_FEAT),
    md(SEC6), code(C_TARGET),
    md(SEC7), code(C_HANDOFF),
    md(SEC8), code(C_VALID),
    md(SEC9), code(C_FIT),
    md(SEC10), code(C_RUNNER),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": cells,
}

out = r"d:\Bagas\SSDS\1.61\Codex-CatBoostRegressor-main\notebook-optimized.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Notebook saved: {out}")
print(f"Total cells: {len(cells)}")
