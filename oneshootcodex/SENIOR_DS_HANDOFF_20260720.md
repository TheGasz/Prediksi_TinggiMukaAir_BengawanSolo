# Senior Data Scientist Handoff — Prediksi TMA Sungai

**Tanggal handoff:** 20 Juli 2026  
**Notebook utama:** `anchor-with-mediancodex.ipynb`  
**Status terakhir:** champion leaderboard `1.59824`; eksperimen anchor, learned weighting, spike weighting, dan pruning sudah diputuskan.

---

## 1. Tujuan proyek

Mengembangkan model untuk memprediksi tinggi muka air (`tma_mdpl`) pada 30 pos sungai untuk periode 19 September 2025 sampai 18 Mei 2026.

Data yang tersedia:

- Koordinat 30 pos pemeriksaan TMA.
- Historis TMA 1 Januari 2023–18 September 2025 pada pukul 06.00, 12.00, dan 18.00.
- Data lingkungan per jam 1 Januari 2023–18 Mei 2026.
- Jaringan sungai HydroRIVERS dan dokumentasi `HydroRIVERS_TechDoc_v10.pdf`.
- Sample submission dengan 21.780 baris prediksi.

Metrik leaderboard adalah RMSE. Target pembanding internal adalah skor tim lain sekitar `1.54`, sedangkan baseline awal proyek ini berada di sekitar `1.602`.

---

## 2. Champion saat ini

Submission terbaru memperoleh skor leaderboard:

| Versi | Public leaderboard RMSE |
|---|---:|
| Sebelum perubahan | sekitar `1.602` |
| Champion saat ini | **`1.59824`** |

Bukti leaderboard tersedia pada `image.png`.

Konfigurasi champion:

```python
TARGET_MODE = "raw_robust"
EARLY_STOPPING_METRIC = "current"
LIGHTGBM_HUBER_ALPHA = 1.5
USE_SPIKE_WEIGHTS = False
PRUNE_FEATURES = False
ENABLE_ANCHOR_FEATURE = False
ENABLE_LEARNED_ENSEMBLE_WEIGHTS = False
ENABLE_HARMONIC_FEATURE = True
```

Champion menggunakan:

- Frozen **84 feature columns**.
- Lima anggota ensemble: tiga CatBoost dan dua LightGBM.
- Equal-weight ensemble.
- Target robust-normalized per station.
- Harmonic seasonal baseline sebagai fitur.
- Anchor feature tidak digunakan.

> **Peringatan konfigurasi:** jangan mengandalkan nilai toggle yang mungkin tersimpan dari eksekusi lama di configuration cell. Sebelum menjalankan notebook, tetapkan kembali seluruh konfigurasi champion secara eksplisit seperti blok di atas.

---

## 3. Ringkasan keputusan eksperimen

| Eksperimen | Keputusan | Alasan utama |
|---|---|---|
| Anchor ON/OFF | **OFF** | Anchor merusak generalisasi, terutama fold terbaru dan tail/worst-station. |
| Pseudo-origin anchor | **Ditolak** | Tidak memperbaiki mismatch secara konsisten; RMSE dan stabilitas memburuk. |
| Solar missing robustness | **Native missing dipertahankan** | Missing test 57,44% sudah relatif stabil; drop/imputation tidak memberi kemenangan konsisten. |
| Learned ensemble weights | **OFF** | Hanya membaik tipis di `sep_2024`, tetapi memburuk di `may_2024` dan `jan_2025`; bobot terlalu terkonsentrasi. |
| Spike weighting | **OFF** | Setelah parity diperbaiki, spike-ON kalah pada kedua fold terbaru. |
| Feature pruning | **OFF** | Pruning 84→72 fitur memperburuk RMSE pada kedua fold terbaru dan merusak worst-station `jan_2025`. |
| Equal-weight ensemble | **Champion** | Lebih stabil daripada learned weighting dan sudah meningkatkan leaderboard ke `1.59824`. |

---

## 4. Anchor confirmation

Tiga anchor features yang telah dihapus dari model serving:

```text
horizon_days
recent_anomaly_mdpl
anchor_basis_mdpl
```

Notebook sekarang mempunyai daftar frozen `ANCHOR_OFF_FROZEN_FEATURE_COLUMNS` berisi tepat 84 fitur unik. Fungsi `enforce_frozen_anchor_off_columns` memastikan ketiga anchor columns tidak masuk kembali ketika anchor dinonaktifkan.

Artefak terkait:

- `Anchor_confirm/`
- `p0_experiments/p0_anchor_summary.csv`
- `p0_experiments/p0_solar_summary.csv`
- `anchor_off_frozen_feature_report.json` di handoff validasi terbaru.

---

## 5. Validasi learned ensemble weighting

Handoff validasi:

```text
handoff_20260720T095756Z_validate/
```

Perbandingan learned-weight terhadap equal-weight:

| Fold | Equal RMSE | Learned RMSE | Perubahan learned vs equal |
|---|---:|---:|---:|
| `sep_2023` | 2.632796 | 2.632796 | 0,000% |
| `may_2024` | 0.759435 | 0.762421 | **+0,393% lebih buruk** |
| `sep_2024` | 1.209040 | 1.206952 | −0,173% lebih baik |
| `jan_2025` | 1.178612 | 1.189124 | **+0,892% lebih buruk** |

Pada `jan_2025`, learned weighting juga menyebabkan:

- MAE memburuk sekitar `1,381%`.
- Worst-station MAE memburuk sekitar `22,08%`.
- P95 absolute error memburuk sekitar `3,49%`.

Bobot final yang dipelajari:

```text
cb_shallow = 0.0000
cb_base    = 0.0000
cb_deep    = 0.4441
lgbm_a     = 0.0000
lgbm_b     = 0.5559
```

Residual anggota ensemble memiliki korelasi tinggi, sekitar `0,95–0,996`. Optimizer karena itu tidak mendapatkan diversitas yang cukup dan cenderung memilih dua anggota secara berlebihan.

**Keputusan:** `ENABLE_LEARNED_ENSEMBLE_WEIGHTS=False` untuk champion.

Artefak audit utama:

- `handoff_20260720T095756Z_validate/metrics/validation_metrics.csv`
- `handoff_20260720T095756Z_validate/metrics/member_metrics.csv`
- `handoff_20260720T095756Z_validate/metrics/member_oof_predictions.parquet`
- `handoff_20260720T095756Z_validate/metrics/fold_ensemble_weights.csv`
- `handoff_20260720T095756Z_validate/metrics/ensemble_weights.json`

---

## 6. Temuan kualitas target

RMSE validasi sangat dipengaruhi beberapa target yang tampak seperti isolated sensor/data errors:

| Waktu | Pos | Target | Prediksi equal-weight (sekitar) |
|---|---|---:|---:|
| 28 Februari 2025 06.00 | Jurug | 0,0 | 81,0 |
| 29 April 2025 06.00 | Peren | 170,1 | 91,3 |
| 15 Juni 2025 18.00 | Floodway Bridge C | 47,0 | 3,49 |

Dua titik Jurug dan Peren muncul pada `sep_2024` dan `jan_2025` karena validation folds overlap. Keduanya menyumbang hampir separuh SSE pada fold terbaru.

Sebagai diagnostik saja—bukan metrik seleksi resmi—mengeluarkan dua error terbesar akan mengubah:

- `sep_2024`: RMSE sekitar `1.209 → 0.897`.
- `jan_2025`: RMSE sekitar `1.179 → 0.859`.

Label validasi **tidak boleh dihapus** dari scoring resmi. Temuan ini hanya menjelaskan tingginya noise RMSE dan alasan optimizer ensemble mudah overfit.

---

## 7. Bug feature parity pada mode ablation

### Penyebab

Jalur `validate/fit` memanggil `enforce_frozen_anchor_off_columns`, tetapi jalur cepat `MODE="ablation"` sebelumnya langsung memberikan seluruh 99 kandidat fitur kepada `run_ablation_study`.

Akibatnya, ablation awal benar-benar melatih model dengan 99 fitur. Ini bukan sekadar kesalahan label `n_features`.

### Perbaikan

Notebook sudah diperbarui sehingga `run_ablation_study`:

1. Memanggil `enforce_frozen_anchor_off_columns` di awal fungsi.
2. Melakukan hard assertion jika baseline anchor-OFF tidak identik dengan frozen 84 features.
3. Menambahkan kolom berikut ke `ablation_results.csv`:
   - `baseline_feature_count`
   - `anchor_off_frozen_parity`
4. Menulis `ablation_feature_parity_report.json`.

Smoke test aktual:

```text
incoming_feature_count: 99
baseline_feature_count: 84
expected_frozen_feature_count: 84
parity_passed: true
```

Semua code cells pada notebook telah lolos compile check setelah perbaikan.

Updater yang digunakan tersedia di `fix_ablation_parity.py`. Updater telah dibuat idempotent dan diuji dijalankan dua kali tanpa menduplikasi patch.

---

## 8. Hasil spike weighting dan pruning setelah parity fix

Sumber:

```text
ablation_resultsafterfixedbug.csv
```

Semua baseline menggunakan 84 fitur dan menunjukkan `anchor_off_frozen_parity=True`. Variant pruned menggunakan 72 fitur; angka 72 tersebut disengaja dan bukan parity failure.

### Spike ON versus OFF pada frozen 84 fitur

| Fold | Dampak RMSE spike-ON | Dampak MAE | Dampak worst-station MAE |
|---|---:|---:|---:|
| `sep_2024` | +0,062% lebih buruk | +0,388% | +0,919% |
| `jan_2025` | +0,583% lebih buruk | +0,945% | +5,617% |

**Keputusan:** spike weighting OFF. Tidak perlu full validation atau submission spike-ON.

### Pruned 72 versus full 84, keduanya spike-OFF

| Fold | Dampak RMSE pruning | Dampak MAE | Dampak worst-station MAE |
|---|---:|---:|---:|
| `sep_2024` | +0,538% lebih buruk | +0,046% | −3,514% lebih baik |
| `jan_2025` | +0,906% lebih buruk | +1,846% | +12,026% lebih buruk |

**Keputusan:** pruning OFF. Tidak perlu full validation atau submission pruned.

---

## 9. Prosedur menjalankan champion

Sebelum final fit, isi configuration cell dengan:

```python
MODE = "fit"

TARGET_MODE = "raw_robust"
EARLY_STOPPING_METRIC = "current"
LIGHTGBM_HUBER_ALPHA = 1.5

USE_SPIKE_WEIGHTS = False
PRUNE_FEATURES = False
FEATURE_IMPORTANCE_SOURCE = None

ENABLE_ANCHOR_FEATURE = False
ENABLE_LEARNED_ENSEMBLE_WEIGHTS = False
ENABLE_HARMONIC_FEATURE = True

ENABLE_QUANTILE_MODEL = False
ITERATIONS_BY_MEMBER = None
SAVE_FEATURE_MATRICES = True
```

Kemudian:

1. Restart session/kernel jika sebelumnya telah menjalankan konfigurasi eksperimen lain.
2. Jalankan **Run All**.
3. Pastikan `anchor_off_frozen_feature_report.json` menunjukkan 84 fitur.
4. Pastikan metadata final menunjukkan:
   - `use_spike_weights=false`
   - equal ensemble weights
   - tidak ada anchor columns.
5. Periksa schema, jumlah baris, urutan `id`, NaN, dan nilai non-finite pada `submission.csv` sebelum upload.

Champion sudah dibuat dan memperoleh `1.59824`; prosedur ini hanya digunakan jika perlu mereproduksi submission tersebut.

---

## 10. Langkah berikutnya untuk sesi baru

### Prioritas P0 — Ablation `nino_34` ON/OFF

Alasan:

- `nino_34` mengalami standardized mean shift train→test sekitar `-0,818`.
- Feature importance pada fold terbaru mencapai sekitar `12,7%` untuk beberapa anggota.
- Kombinasi drift besar dan importance tinggi membuatnya kandidat spurious/non-stationary feature.

Desain eksperimen:

- Baseline: frozen 84 features.
- Challenger: frozen 84 dikurangi hanya `nino_34`, sehingga 83 features.
- Anchor OFF.
- Spike weighting OFF.
- Pruning OFF.
- Learned ensemble weighting OFF.
- Pertama gunakan ablation murah pada `sep_2024` dan `jan_2025` dengan anggota model yang sama untuk kedua variant.
- Jangan memakai target periode test/API untuk pemilihan model.

Kriteria kenaikan ke full validation:

1. Challenger tidak kalah RMSE pada `jan_2025`.
2. Rata-rata RMSE dua fold terbaru membaik secara cukup berarti; target awal minimal sekitar `0,5%` agar tidak mengejar noise.
3. MAE, P95 absolute error, dan worst-station tidak memburuk material.
4. Feature parity report menunjukkan baseline 84 dan challenger 83.

Jika lolos:

1. Jalankan full ensemble, 4-fold validation.
2. Bandingkan dengan equal-weight champion.
3. Hanya buat challenger submission bila menang minimal 3/4 fold dan aman pada `jan_2025`.

Jika gagal:

- Pertahankan `nino_34`.
- Lanjutkan ke eksperimen leakage-safe seasonal residual target: model memprediksi residual terhadap harmonic station baseline, kemudian baseline ditambahkan kembali saat serving.

---

## 11. Hal yang tidak perlu diulang

- Jangan mengulang pseudo-origin anchor tanpa hipotesis baru.
- Jangan mengaktifkan learned weights dari hasil OOF saat ini.
- Jangan menjalankan spike-ON full validation.
- Jangan menjalankan pruned 72-feature full validation.
- Jangan menghapus anomali dari validation labels demi mempercantik RMSE.
- Jangan menggunakan data target test/API sebagai fitur, koreksi, atau pemilih model.
- Jangan membandingkan angka absolut dari ablation single-member langsung dengan full ensemble; gunakan delta ON/OFF di dalam run yang sama.

---

## 12. Indeks artefak penting

| Artefak | Isi |
|---|---|
| `anchor-with-mediancodex.ipynb` | Notebook pipeline utama yang sudah memiliki frozen anchor-OFF dan parity fix. |
| `handoff_20260720T095756Z_validate/` | Full validation learned-weight versus equal-weight. |
| `handoff_20260720T095756Z_validate/metrics/validation_metrics.csv` | Metrik per fold. |
| `handoff_20260720T095756Z_validate/metrics/member_oof_predictions.parquet` | OOF seluruh anggota ensemble. |
| `handoff_20260720T095756Z_validate/metrics/ensemble_weights.json` | Learned weights yang ditolak. |
| `Anchor_confirm/` | Hasil confirmation anchor ON/OFF. |
| `p0_experiments/` | Pseudo-origin anchor dan solar missing experiments. |
| `ablation_results spike_weight.csv` | Ablation awal dengan bug parity; jangan dipakai untuk keputusan final. |
| `ablation_resultsafterfixedbug.csv` | Ablation spike/pruning yang valid setelah parity fix. |
| `image.png` | Bukti leaderboard champion `1.59824`. |
| `fix_ablation_parity.py` | Updater idempotent untuk perbaikan parity notebook. |
| `upgrade_weighted_ensemble.py` | Updater learned weighting; fitur tetap tersedia tetapi toggle harus OFF pada champion. |

---

## 13. Keputusan beku saat handoff

```text
Champion leaderboard      : 1.59824
Anchor                    : OFF
Learned ensemble weighting: OFF
Spike weighting           : OFF
Feature pruning           : OFF
Harmonic baseline feature : ON
Feature count             : 84
Ensemble aggregation      : equal weight
Target mode               : raw_robust
Early stopping metric     : current
LightGBM Huber alpha      : 1.5 (champion baseline)
Next experiment           : lgbm_alpha_confirmation
```

Dokumen ini merupakan sumber keputusan utama untuk memulai sesi selanjutnya. Bila notebook outputs atau configuration cell bertentangan dengan keputusan beku di atas, gunakan keputusan dalam bagian ini dan verifikasi kembali artefak run sebelum training.

---

## 14. Addendum sesi lanjutan (20 Juli 2026)

Bagian ini menggantikan prioritas lama `nino_34` pada bagian 10 dan menjadi keputusan terbaru.

### `nino_34` ON/OFF: pertahankan ON

Sumber: `ablation_results_nino_34.csv`. Parity valid: baseline 84 fitur, challenger 83 fitur, anchor OFF, dan semua toggle lain terkunci.

| Fold | Dampak saat `nino_34` OFF vs ON (RMSE) | Dampak MAE | Dampak worst-station |
|---|---:|---:|---:|
| `sep_2024` | **+3,440% lebih buruk** | +3,590% | +34,015% |
| `jan_2025` | -0,530% lebih baik | -1,328% | +0,320% |
| Rata-rata dua fold | **+1,489% lebih buruk** | +1,301% | +14,938% |

Keputusan: `nino_34` tetap ON pada frozen 84 fitur. Tidak perlu full validation/submission untuk variant OFF.

### Harmonic residual target: ditolak

Sumber: `ablation_target_residual.csv`. Variant residual memprediksi residual terhadap harmonic baseline lalu menambahkannya kembali saat inference.

| Fold | Dampak residual vs `raw_robust` (RMSE) | Dampak MAE | Dampak worst-station |
|---|---:|---:|---:|
| `sep_2024` | +6,288% | +8,651% | +43,120% |
| `jan_2025` | +5,217% | +13,153% | -23,701% |
| Rata-rata dua fold | **+5,762%** | **+10,746%** | +5,318% |

Keputusan: gunakan `TARGET_MODE = "raw_robust"`. Harmonic baseline tetap ON sebagai fitur, tetapi bukan sebagai transformasi target. Audit transform tidak ikut tersalin saat review, tetapi CSV ablation dan parity sudah cukup untuk menolak variant residual.

### Early stopping RMSE: ditolak

Sumber: `ablation/ablation_early_stopping_metric.csv` dan `metric_ablation_audit.json`. Objective Huber tidak berubah; variant hanya mengganti metrik evaluasi/early stopping menjadi RMSE.

| Model | Rata-rata dampak RMSE variant RMSE | Rata-rata dampak MAE | Dampak worst-station |
|---|---:|---:|---:|
| CatBoost | +0,717% | +2,347% | +10,203% |
| LightGBM | +0,398% | +0,395% | +1,827% |
| Gabungan | **+0,556%** | **+1,354%** | **+5,613%** |

Keputusan: gunakan `EARLY_STOPPING_METRIC = "current"`; jangan jalankan full validation/submission untuk variant RMSE.

### Grid LightGBM Huber alpha: kandidat confirmation terbatas

Sumber: `ablation/ablation_lightgbm_huber_alpha.csv` dan `lgbm_alpha_ablation_audit.json`. Nilai `1.5` sebelumnya dipakai untuk CatBoost `Huber:delta` dan LightGBM `alpha`; keduanya adalah parameter berbeda dan tidak boleh dianggap ekuivalen. Lihat [dokumentasi CatBoost Huber](https://catboost.ai/docs/en/concepts/loss-functions-regression) dan [parameter LightGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

| Alpha LightGBM | Rata-rata RMSE | Delta RMSE vs 1,5 | Catatan |
|---|---:|---:|---|
| `0.7` | **1,215704** | **-0,482%** | Bagus di `sep_2024` (-1,454%), tetapi `jan_2025` +0,523% dan worst-station +6,779%. |
| `0.9` | 1,217859 | -0,306% | Tidak cukup kuat; worst-station `jan_2025` +11,251%. |
| `1.2` | 1,224210 | +0,214% | Ditolak. |
| `1.5` | 1,221597 | baseline | Champion saat ini. |

Keputusan: alpha `0.7` belum boleh menggantikan champion atau dibuat submission. Ia hanya masuk confirmation karena rata-rata improvement masih di bawah ambang praktis 0,5% dan merusak `jan_2025`. Alpha `0.9` dan `1.2` dihentikan.

### Langkah berikutnya satu-satunya: `lgbm_alpha_confirmation`

Notebook sudah diperbarui. Mode ini membandingkan alpha `0.7` dan `1.5` pada `lgbm_a` dan `lgbm_b` untuk fold `sep_2023`, `sep_2024`, dan `jan_2025`, lalu menghitung blend OOF 50:50 tanpa training tambahan.

```python
MODE = "lgbm_alpha_confirmation"
TARGET_MODE = "raw_robust"
EARLY_STOPPING_METRIC = "current"
LIGHTGBM_HUBER_ALPHA = 1.5  # baseline; mode menguji 0.7 dan 1.5 secara internal
USE_SPIKE_WEIGHTS = False
PRUNE_FEATURES = False
FEATURE_IMPORTANCE_SOURCE = None
ENABLE_ANCHOR_FEATURE = False
ENABLE_LEARNED_ENSEMBLE_WEIGHTS = False
ENABLE_HARMONIC_FEATURE = True
```

Flow junior: restart kernel/session, set konfigurasi di atas, lalu Run All. Setelah selesai, periksa `lgbm_alpha_confirmation_decision.json` dan berhenti; jangan menjalankan `validate` atau `fit` challenger tanpa review senior.

Gate otomatis: alpha `0.7` atau blend harus menang minimal 3 dari 4 kombinasi exact-season (`2 model x 2 fold`), mean RMSE exact-season membaik minimal 0,5%, degradasi RMSE `jan_2025` maksimum 0,3%, dan degradasi worst-station `jan_2025` maksimum 2%. Output: `lgbm_alpha_confirmation_results.csv`, `lgbm_alpha_confirmation_oof.parquet`, `lgbm_alpha_confirmation_audit.json`, dan `lgbm_alpha_confirmation_decision.json`.
