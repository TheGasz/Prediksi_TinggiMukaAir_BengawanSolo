import pandas as pd
import numpy as np

OOF_IBRA = r"d:\Bagas\SSDS\1.61\Codex-CatBoostRegressor-main\xgboost\oof_predictions.parquet"

oof = pd.read_parquet(OOF_IBRA)

print("=== ANALISIS INTERNAL IBRA: robust_weight ===")
sample = oof.dropna(subset=["squared_prediction","robust_prediction","prediction"]).head(2000)
best_w, best_err = None, np.inf
for w in np.arange(0.0, 1.01, 0.01):
    blend = (1 - w) * sample["squared_prediction"] + w * sample["robust_prediction"]
    err = np.abs(blend - sample["prediction"]).mean()
    if err < best_err:
        best_err, best_w = err, w
print(f"  Ibra robust_weight = {best_w:.2f}  (rekonstruksi MAE={best_err:.7f})")
print()

print("=== METRIK IBRA PER FOLD ===")
fold_metrics = {}
for fold, g in oof.groupby("fold", sort=False):
    err = g["prediction"].values - g["tma_mdpl"].values
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    clean_mask = g["label_anomaly_flag"].values == 0
    clean_err = err[clean_mask]
    clean_rmse = float(np.sqrt(np.mean(clean_err**2))) if clean_mask.sum() > 0 else np.nan
    fold_metrics[fold] = {"rmse": rmse, "mae": mae, "clean_rmse": clean_rmse, "n": len(g), "anomalies": int(clean_mask.size - clean_mask.sum())}
    print(f"  {fold}: raw_rmse={rmse:.4f}  raw_mae={mae:.4f}  clean_rmse={clean_rmse:.4f}  n={len(g)}  anomalies={fold_metrics[fold]['anomalies']}")

print()
print("=== WORST STATION PER FOLD (Ibra) ===")
for fold, g in oof.groupby("fold", sort=False):
    st_rmse = g.groupby("nama_pos").apply(
        lambda x: float(np.sqrt(np.mean((x["prediction"].values - x["tma_mdpl"].values)**2)))
    ).sort_values(ascending=False)
    st_mae = g.groupby("nama_pos").apply(
        lambda x: float(np.mean(np.abs(x["prediction"].values - x["tma_mdpl"].values)))
    ).sort_values(ascending=False)
    print(f"  {fold}:")
    for i in range(min(3, len(st_rmse))):
        print(f"    #{i+1} {st_rmse.index[i]:<35} RMSE={st_rmse.iloc[i]:.4f}  MAE={st_mae[st_rmse.index[i]]:.4f}")

print()
print("=== WONOGIRI DAM DETAIL (Ibra) ===")
won = oof[oof["nama_pos"] == "Wonogiri Dam"].copy()
won["err"] = won["prediction"] - won["tma_mdpl"]
for fold, g in won.groupby("fold", sort=False):
    rmse = float(np.sqrt(np.mean(g["err"].values**2)))
    mae  = float(np.mean(np.abs(g["err"].values)))
    bias = float(np.mean(g["err"].values))
    print(f"  {fold}: rmse={rmse:.4f}  mae={mae:.4f}  bias={bias:+.4f}  n={len(g)}")

print()
print("=== PERBANDINGAN SQUARED vs ROBUST (internal Ibra blend components) ===")
for fold, g in oof.groupby("fold", sort=False):
    sq_rmse  = float(np.sqrt(np.mean((g["squared_prediction"].values - g["tma_mdpl"].values)**2)))
    rob_rmse = float(np.sqrt(np.mean((g["robust_prediction"].values - g["tma_mdpl"].values)**2)))
    fin_rmse = float(np.sqrt(np.mean((g["prediction"].values - g["tma_mdpl"].values)**2)))
    print(f"  {fold}: squared_rmse={sq_rmse:.4f}  robust_rmse={rob_rmse:.4f}  final_rmse={fin_rmse:.4f}")

print()
print("=== GRID SEARCH: KITA vs IBRA BLEND (simulasi dengan RMSE kita dari tabel) ===")
# Kita tidak punya OOF kita, tapi kita tahu clean RMSE kita per fold
# Dari tabel yang disebutkan user:
# sep_2023=1.340, may_2024=0.756, sep_2024=0.896, jan_2025=0.813
# Dan kita punya OOF Ibra, kita bisa compute delta improvement dari blend
kita_clean = {"sep_2023": 1.340, "may_2024": 0.756, "sep_2024": 0.896, "jan_2025": 0.813}
ibra_clean = {}
for fold, g in oof.groupby("fold", sort=False):
    clean = g[g["label_anomaly_flag"] == 0]
    ibra_clean[fold] = float(np.sqrt(np.mean((clean["prediction"].values - clean["tma_mdpl"].values)**2)))

print("\n  Perbandingan clean RMSE per fold:")
print(f"  {'Fold':<12} {'Kita':>8} {'Ibra':>8} {'Delta':>8}  Menang?")
print("  " + "-"*50)
for fold in ["sep_2023","may_2024","sep_2024","jan_2025"]:
    k = kita_clean[fold]
    i = ibra_clean.get(fold, float("nan"))
    delta = k - i
    winner = "KITA ✓" if k < i else "IBRA ✓"
    print(f"  {fold:<12} {k:>8.4f} {i:>8.4f} {delta:>+8.4f}  {winner}")

print()
pooled_kita_count = sum(fold_metrics[f]["n"] for f in fold_metrics)
print(f"  Total OOF rows (Ibra): {pooled_kita_count:,}")
print()
print("Catatan: Untuk blend grid search yang akurat, kita perlu OOF predictions kita sendiri.")
print("Jalankan MODE='validate' di notebook-150726.ipynb untuk menghasilkan artifacts/validation/oof_predictions.parquet")
