# Prediksi Tinggi Muka Air Sungai (TMA Forecasting)

Prediksi `tma_mdpl` (tinggi muka air, meter di atas permukaan laut) untuk **30 pos pemantauan sungai** dengan horizon **242 hari** (19 September 2025 – 18 Mei 2026, pukul 06.00 / 12.00 / 18.00 = **21.780 baris**). Metrik kompetisi: RMSE.

Pipeline ini bersifat **eksogen-kausal**: tidak ada TMA aktual setelah 18 September 2025, sehingga fitur hanya berasal dari data lingkungan per jam, topologi sungai, dan kalender. Pendekatan non-rekursif dipilih agar error tidak menumpuk selama 8 bulan.

## Ringkasan Hasil

| Versi | Deskripsi | Metrik |
|---|---|---|
| v1 | CatBoost 3 seed | LB RMSE `1.61079` |
| v2 | Ensemble 3 CatBoost + 2 LightGBM + harmonic baseline | LB RMSE `1.60232` |
| **v3** | **Ensemble 4 keluarga (LGBM + XGB + CatBoost + StationExpert) + fitur hidrologi fisik + cutoff anchor** | **CV 4-fold Mean RMSE `1.39427`** (v2: `1.44331`), MAE `0.53597` (v2: `0.58403`) |

> Skor LB v3 belum tersedia saat dokumen ini ditulis. Angka v3 di atas berasal dari *rolling-origin CV* (eksperimen di luar notebook), bukan dari leaderboard.

Perbandingan per fold (RMSE):

| Varian | Mean 4-Fold | `sep_2023` | `may_2024` | `sep_2024` | `jan_2025` |
|---|:---:|:---:|:---:|:---:|:---:|
| V0 baseline 84 fitur (v2) | 1.44331 | 2.63814 | 0.71016 | 1.22388 | 1.20106 |
| V1 + fitur hidrologi fisik | 1.40866 | 2.63013 | 0.67350 | 1.17946 | 1.15154 |
| V4 + multi-family ensemble | 1.39835 | 2.61022 | 0.66792 | 1.17124 | 1.14403 |
| **V5 v3 champion (+ anchor + smoothing)** | **1.39427** | **2.60529** | **0.66073** | **1.16762** | **1.14345** |

Perbaikan konsisten di keempat fold. Fold `sep_2023` tetap paling sulit karena data train baru ~9 bulan.

## Struktur Repositori

```text
.
├── datasets/                         # TIDAK di-commit (lihat .gitignore), siapkan manual
│   ├── train.csv                     # datetime, nama_pos, tma_mdpl (84.396 baris, 30 pos)
│   ├── test.csv                      # id = "<datetime> - <nama_pos>" (21.780 baris)
│   ├── sample_submission.csv
│   └── data_pendukung/
│       ├── data_lingkungan.csv       # 888.480 baris, per jam, 27 kolom
│       ├── koordinat_pos.csv
│       └── HydroRIVERS_v10_au_shp/   # jaringan sungai HydroRIVERS v10
├── EDA.ipynb                         # 1. eksplorasi data
├── Preprocessing.ipynb               # 2. preprocessing sekuensial (eksplorasi/alternatif DL)
├── anchor-with-mediancodex.ipynb     # 3. training + validasi + submission (model final v3)
├── architecture.md                   # detail arsitektur & alasan desain
├── logs.md                           # catatan pengembangan
└── requirements.txt
```

## Alur Kerja: dari EDA sampai Model

### 1. EDA (`EDA.ipynb`)
Temuan yang memengaruhi desain model:
- **Target:** distribusi dan outlier per pos (boxplot, time series mentah); ada spike sensor satu titik (mis. Napel `34,95 → 325,83 → 34,55`).
- **Musiman:** pola musim hujan vs kemarau → fitur kalender siklik dan baseline harmonik.
- **Autokorelasi (ACF/PACF):** menentukan skala memori tiap pos; sungai besar/waduk lambat, hulu cepat.
- **Korelasi antar pos:** pos hulu–hilir berkorelasi → fitur *upstream*.
- **Fitur eksogen:** `rainfall_mm` identik dengan `rainfall_openmeteo_mm` (hanya satu dipakai); korelasi 4 lapis soil moisture; respon TMA terhadap hujan (lag/lead); `nino_34` dan indikator MJO sangat lambat berubah.
- **Leakage & konsistensi temporal:** pengecekan window `rainfall_max_24h_mm` (backward-looking), tidak ada overlap tanggal train vs test, granularitas indeks iklim.
- **Spasial:** peta 30 pos, struktur hulu–hilir HydroRIVERS (`NEXT_DOWN`, `MAIN_RIV`), korelasi TMA vs jarak.
- **Data statis:** `built_surface_m2` dan `landcover_*` konstan per pos → fitur statis per pos.

Temuan data penting lain (dari audit): 100.080 nilai sentinel `-999` pada `solar_radiation_mj_m2` di periode test, dan 4.884 timestamp target hilang dari grid teoritis (terutama 4–28 Februari 2025).

### 2. Preprocessing (`Preprocessing.ipynb`)
Pipeline sekuensial: pembersihan target (IQR, drop-to-zero), resampling lingkungan `1H → 6H`, penggabungan, fitur waktu siklikal/lag/rolling, normalisasi target per pos dan fitur global, label encoding, *windowed sequences*, lalu simpan artefak ke `artifacts/preprocessing/`.

> Catatan: notebook ini adalah jalur preprocessing untuk model sekuensial. Model final (v3) membangun fiturnya sendiri di dalam `anchor-with-mediancodex.ipynb` dan tidak membaca artefak ini.

### 3. Modeling (`anchor-with-mediancodex.ipynb`)

**Target:** diskalakan robust per pos, `z = (TMA − median) / max(IQR/1.349, 0.05)`, di-fit hanya pada data train tiap fold.

**Fitur (~187, bebas leakage):**
- Hidrologi fisik: VPD FAO-56 (`vpd_kpa`), proksi evapotranspirasi, neraca air `P − ET0`, kelembapan tanah kolom penuh 0–255 cm, indeks limpasan jenuh (`hujan × sm²`).
- Memori jangka panjang 30/60/90 hari (`720h/1440h/2160h`) untuk waduk dan sungai besar. Contoh Wonogiri Dam: korelasi hujan 14 hari `r=0.274` vs kelembapan tanah kolom 30 hari `r=0.841`.
- Agregasi spasial ganda: bobot hulu HydroRIVERS (`upstream_*`) + IDW regional 35 km (`regional_*`), karena 14 dari 30 pos tidak punya hulu di jaringan 30 pos.
- Baseline harmonik per pos (Ridge Fourier, di-fit kausal per fold), kalender siklik, fitur statis.
- `solar_radiation_mj_m2` **dibuang** (0% missing di train vs 57,5% `-999` di test → *covariate shift*).

**Ensemble 7 anggota, 4 keluarga:** `lgbm_a`, `lgbm_b` (LightGBM Huber), `xgb_a`, `xgb_b` (XGBoost `reg:pseudohubererror`), `cb_shallow`, `cb_base` (CatBoost Huber), `station_ridge_expert` (Ridge Huber-weighted per pos). Bobot dioptimasi pada prediksi OOF.

**Post-processing kausal:** koreksi residual cutoff yang meluruh (`window=21 hari`, `tau=45 hari`, `shrink=0.50`) di ruang meter, lalu *3-point smoothing* `[0.20, 0.60, 0.20]` per pos.

**Validasi:** *rolling-origin* 4 fold, tiap fold ~8 bulan:

| Fold | Train sampai | Validasi |
|---|---|---|
| `sep_2023` | 18 Sep 2023 | 19 Sep 2023 – 18 Mei 2024 |
| `may_2024` | 18 Mei 2024 | 19 Mei 2024 – 18 Jan 2025 |
| `sep_2024` | 18 Sep 2024 | 19 Sep 2024 – 18 Mei 2025 |
| `jan_2025` | 18 Jan 2025 | 19 Jan 2025 – 18 Sep 2025 |

Detail alasan tiap keputusan (termasuk yang ditolak) ada di [architecture.md](architecture.md).

## Cara Menjalankan

### Data
Dataset tidak ikut repo. Letakkan folder `datasets/` dengan struktur di atas (data kompetisi *Sebelas Maret Statistics Data Science 2026*).

### Google Colab (GPU)
1. Upload `anchor-with-mediancodex.ipynb`, pilih runtime GPU.
2. Letakkan `datasets/` di `/content/datasets` atau `MyDrive/SSDS/datasets`. Atau set `CUSTOM_DATA_DIR` di cell konfigurasi.
3. **Run all**. Path divalidasi otomatis (header `train.csv` + keberadaan `data_pendukung/data_lingkungan.csv`); jalur terpilih dicetak sebagai `Resolved DATA_DIR`.

### Kaggle / Lokal
```powershell
python -m pip install -r requirements.txt
```
Lalu buka notebook; path Kaggle dan `datasets/` lokal terdeteksi otomatis.

### Mode eksekusi
Atur `MODE` di cell konfigurasi: `audit` (cek data saja), `validate` (CV 4-fold), `fit` (latih final + `submission.csv`), `all` (validate + fit), serta `ablation` dan `tune`.

## Keterbatasan yang Perlu Diketahui
- Skor LB v3 belum diukur; CV dan LB bisa berbeda (fold `sep_2023` sangat berpengaruh pada rata-rata).
- Angka CV v3 dihasilkan dari skrip eksperimen terpisah. Setelah itu encoding kategorikal XGBoost diganti ke kode numerik (menghindari crash kategori yang tidak ada di train); dampaknya pada skor diperkirakan kecil tetapi belum diukur ulang.
- Eksekusi penuh di GPU Colab belum diverifikasi; smoke test end-to-end hanya dijalankan di CPU lokal dengan iterasi kecil.
- Horizon 8 bulan tanpa observasi membuat prediksi pos waduk/bendung (aturan operasi pintu air) tetap berisiko.
