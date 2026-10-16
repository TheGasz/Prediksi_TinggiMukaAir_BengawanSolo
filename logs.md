# Log Pengembangan

## 2026-07-13 – EDA & Pipeline Preprocessing Awal
- Menyusun `EDA.ipynb` untuk menganalisis distribusi `tma_mdpl` di 30 pos, mendeteksi spike sensor & drop ke nol, memetakan pola musiman/diurnal, serta memvalidasi keterhubungan hulu-hilir menggunakan shapefile `HydroRIVERS_v10_au_shp`.
- Menyusun pipeline preprocessing sekuensial (`Preprocessing.ipynb`) yang mencakup pembersihan target berbasis IQR & drop-to-zero, resampling data lingkungan `1H -> 6H` (`label='right'`), feature engineering waktu siklikal & lag/rolling, normalisasi `StandardScaler` per pos tanpa *data leakage*, dan pembentukan *windowed sequences*.

## 2026-07-15 – Audit Data, Pipeline Eksogen v1, dan Dukungan Handoff
- Menginventarisasi seluruh data kompetisi (`train.csv` 84.396 baris, `test.csv` 21.780 baris, `data_lingkungan.csv` 888.480 baris, `koordinat_pos.csv` 30 pos, dan `HydroRIVERS` 836.472 reach).
- Menemukan duplikasi identik antara `rainfall_mm` dan `rainfall_openmeteo_mm` (100% sama) serta 100.080 nilai sentinel `-999` pada `solar_radiation_mj_m2` di periode prediksi yang harus dikonversi menjadi `NaN`.
- Membangun fitur agregasi hulu-hilir (`NEXT_DOWN`) dengan peluruhan jarak hidraulik dan validasi *rolling-origin* 4 fold berhorizon panjang.
- Mencapai skor Leaderboard awal **RMSE 1.61079** dengan ensemble 3 seed CatBoostRegressor.

## 2026-07-21 s/d 2026-07-24 – Pengembangan v2 (`anchor-with-mediancodex.ipynb`)
- **Ensemble Terdiversifikasi (3 CatBoost + 2 LightGBM):** Mengganti 3 CatBoost identik menjadi 5 anggota lintas keluarga (`cb_shallow`, `cb_base`, `cb_deep`, `lgbm_a`, `lgbm_b`) dengan rekomendasi iterasi per member dari validasi 4-fold.
- **Harmonic Baseline per Pos (`harmonic_baseline_mdpl`):** Menambahkan fitur kurva musiman/diurnal berbasis Ridge Fourier yang di-fit secara kausal pada prefix training tiap fold untuk membantu ekstrapolasi level dasar pos (terutama pada pos dengan dinamika bendungan seperti Wonogiri Dam).
- **Studi Ablasi Terisolasi & Eksperimen P0:** Menambahkan mode ablasi untuk menguji *spike downweighting*, *feature pruning* (`ANCHOR_OFF_FROZEN_FEATURE_COLUMNS` 84 fitur), *pseudo-origin anchor*, *short-memory state*, dan sensitivitas *missing solar radiation*.
- **Peningkatan Skor:** Konfigurasi champion v2 (`target_mode="raw_robust"`, `enable_harmonic_feature=True`, 5-member ensemble) menurunkan RMSE validasi dan memperbaiki skor Leaderboard menjadi **1.60232**.

## 2026-10-16 – Perapian Workspace & Upgrade Arsitektur v3 Champion
- **Perapian Workspace:** Merapikan repositori hingga menyisakan komponen inti (`datasets/`, `EDA.ipynb`, `Preprocessing.ipynb`, `anchor-with-mediancodex.ipynb`, dan dokumentasi utama).
- **Diagnosis & Feature Engineering Hidrologi Fisik v3 (~178 Fitur):**
  - Membuka kunci `ANCHOR_OFF_FROZEN_FEATURE_COLUMNS` (84 fitur) dan mengecualikan `solar_radiation_mj_m2` yang hilang 57,5% di periode test.
  - Menambahkan fitur fisik neraca air FAO-56 (`vpd_kpa`, `et0_proxy_mm`, `net_water_mm`), kelembapan tanah kolom penuh (`0–255 cm` termasuk lapisan dalam `100–255 cm`), gradien vertikal kelembapan tanah, dan indeks limpasan permukaan non-linear (`rainfall * sm^2`).
  - Menambahkan memori penyimpanan jangka panjang (`720h` / 30d, `1440h` / 60d, `2160h` / 90d) yang menaikkan korelasi fitur terhadap TMA di **Wonogiri Dam** dari $r=0.274$ menjadi **$r=0.841$**.
  - Menambahkan agregasi spasial **Regional Inverse-Distance Weighting (IDW 35 km)** (`regional_*`) untuk melengkapi matriks `upstream_*` pada 14 pos hulu (*headwaters*).
- **Ensemble 4-Keluarga + Post-Processing Kausal:**
  - Memperluas ensemble menjadi 7 anggota lintas 4 keluarga: **CatBoost** (`cb_shallow`, `cb_base`, `cb_deep`), **LightGBM** (`lgbm_a`, `lgbm_b`), **XGBoost** (`xgb_a` dengan `reg:pseudohubererror`), dan **`StationPhysicsExpertRegressor`** (`station_expert` — gabungan Huber-weighted Ridge + ExtraTrees spesialis per pos).
  - Menambahkan koreksi **Causal Cutoff-Residual Decay Anchor** di ruang meter (`window=21d`, `tau=45d`, `shrink=0.50`) dan **3-Point Hydrograph Smoothing** (`[0.20, 0.60, 0.20]`).
  - Mengonfigurasi deteksi path otomatis (`resolve_competition_dirs`) dan akselerasi GPU otomatis untuk **Google Colab**, **Kaggle**, dan **Lokal**.
- **Hasil Benchmark Validasi 4-Fold Rolling-Origin:**
  - Mean 4-Fold RMSE turun drastis dari **`1.44331` (v2)** menjadi **`1.39427` (v3 Champion)** (**-0.04904 RMSE**).
  - Pooled OOF RMSE turun dari **`1.61778`** menjadi **`1.57813`** (**-0.03965 RMSE**).
  - Mean 4-Fold MAE turun dari **`0.58403`** menjadi **`0.53597`** (**-0.04806 MAE / -8.23%**).
