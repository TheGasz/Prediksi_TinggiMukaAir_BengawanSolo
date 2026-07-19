# Project Overview — Prediksi Tinggi Muka Air Sungai

## Ringkasan

Project ini membangun pipeline machine learning untuk memprediksi tinggi muka air sungai (`tma_mdpl`) pada 30 pos pengamatan. Prediksi dibuat untuk periode 19 September 2025 sampai 18 Mei 2026 pada pukul 06.00, 12.00, dan 18.00, sehingga total keluaran test adalah 21.780 baris.

Pendekatan utama saat ini adalah ensemble non-autoregresif yang menggabungkan CatBoost dan LightGBM. Model memakai data cuaca dan lingkungan, kalender, karakteristik stasiun, serta topologi jaringan sungai HydroRIVERS. Model sengaja tidak memakai lag target TMA karena nilai TMA aktual tidak tersedia selama horizon test delapan bulan; penggunaan prediksi sebelumnya secara rekursif berisiko mengakumulasi error.

Notebook utama terkini adalah `notebookfde8c8db98.ipynb`. Folder `1.61/Codex-CatBoostRegressor-main/` berisi riwayat baseline, eksperimen optimasi, hasil ablation, dan challenger XGBoost.

## Tujuan

- Menghasilkan prediksi `tma_mdpl` untuk setiap ID pada `test.csv`.
- Menjaga kondisi validasi sedekat mungkin dengan deployment jangka panjang.
- Memanfaatkan hubungan spasial hulu-hilir dan kondisi lingkungan secara kausal.
- Menyediakan artefak eksperimen yang dapat diaudit: konfigurasi, metrik OOF, feature importance, model, log, dan submission.
- Membandingkan beberapa keluarga model tanpa mengubah kontrak data dan evaluasi.

## Cakupan data

| Sumber | Ukuran | Periode/isi | Peran |
|---|---:|---|---|
| `train.csv` | 84.396 × 3 | 1 Jan 2023 06.00–18 Sep 2025 18.00 | Label `tma_mdpl` untuk 30 pos |
| `test.csv` | 21.780 × 1 | 19 Sep 2025 06.00–18 Mei 2026 18.00 | ID prediksi; berisi timestamp dan nama pos |
| `data_pendukung/data_lingkungan.csv` | 888.480 × 27 | Grid per jam, 1 Jan 2023–18 Mei 2026 | Cuaca, soil moisture, tekanan, MJO, Nino 3.4, dan data statis |
| `data_pendukung/koordinat_pos.csv` | 30 × 3 | Nama pos, latitude, longitude | Lokasi stasiun |
| `data_pendukung/HydroRIVERS_v10_au_shp/` | Shapefile HydroRIVERS | Reach sungai Australia/Asia region bundle | Atribut sungai dan graf hulu-hilir |
| `EDA/pos_topologi_sungai.csv` | 30 × 13 | Hasil pencocokan stasiun–reach | Artefak inspeksi topologi |

Data train tidak memiliki nilai target eksplisit yang hilang, tetapi tidak membentuk grid waktu-pos yang sepenuhnya lengkap. Pipeline tidak menginterpolasi label yang tidak tersedia.

## Pendekatan model

### Pipeline utama

`notebookfde8c8db98.ipynb` menjalankan ensemble lima anggota:

- tiga CatBoost: `cb_shallow`, `cb_base`, dan `cb_deep`;
- dua LightGBM: `lgbm_a` dan `lgbm_b`.

Semua anggota memprediksi target yang dinormalisasi secara robust per stasiun. Prediksi pada skala normalisasi dirata-ratakan, kemudian dikembalikan ke satuan `tma_mdpl`. Pipeline juga memiliki fitur baseline harmonik per stasiun dan anchor anomali terbaru yang meluruh terhadap horizon prediksi.

Mode yang tersedia adalah `audit`, `ablation`, `tune`, `validate`, `fit`, dan `all`.

### Eksperimen pendukung

| Implementasi | Lokasi | Status/peran |
|---|---|---|
| CatBoost baseline | `1.61/Codex-CatBoostRegressor-main/notebook-150726.ipynb` | Baseline historis dan sumber desain awal |
| CatBoost optimized v2 | `1.61/Codex-CatBoostRegressor-main/notebook-optimized.ipynb` | Eksperimen dual-objective, recency weighting, dan GPU |
| XGBoost residual | `1.61/Codex-CatBoostRegressor-main/xgboost/xgboostibra.ipynb` | Challenger berbasis residual terhadap baseline harmonik |
| EDA | `EDA/EDA.ipynb` | Audit distribusi, temporal, leakage, spasial, dan data statis |
| Generator notebook | `1.61/Codex-CatBoostRegressor-main/gen_notebook.py` | Membangun notebook optimized secara terprogram |
| Analisis OOF XGBoost | `1.61/Codex-CatBoostRegressor-main/xgboost/analyze_oof.py` | Ringkasan error OOF dan metrik per stasiun |

## Status hasil

README eksperimen CatBoost mencatat skor historis RMSE `1,61079`.

Output yang tersimpan di notebook utama menunjukkan run Kaggle `all` berhasil dengan 87 fitur dan rekomendasi iterasi per anggota. Metrik rolling-origin yang tercatat adalah:

| Fold | RMSE ensemble | MAE ensemble |
|---|---:|---:|
| `sep_2023` | 2,606707 | 0,657148 |
| `may_2024` | 0,674649 | 0,424886 |
| `sep_2024` | 1,251559 | 0,604438 |
| `jan_2025` | 1,186086 | 0,513005 |

Run tersebut menghasilkan submission di lingkungan Kaggle, tetapi folder artefaknya tidak tersimpan di workspace lokal. Karena itu, metrik di atas dapat diaudit dari output notebook, sedangkan model final dan submission run tersebut tidak dapat direproduksi hanya dari artefak lokal yang tersedia sekarang.

Folder XGBoost memiliki OOF dan metrik lokal. Hasilnya menunjukkan performa sangat bervariasi antar-fold, khususnya `sep_2023`, sehingga belum cukup bukti untuk menggantikan ensemble utama tanpa perbandingan OOF yang konsisten.

## Struktur workspace

```text
oneshootcodex/
├── notebookfde8c8db98.ipynb          # pipeline utama terkini
├── train.csv                          # label training
├── test.csv                           # ID yang harus diprediksi
├── data_pendukung/
│   ├── data_lingkungan.csv            # data lingkungan per jam
│   ├── koordinat_pos.csv              # koordinat 30 pos
│   ├── HydroRIVERS_TechDoc_v10.pdf
│   └── HydroRIVERS_v10_au_shp/        # shapefile dan sidecar
├── EDA/
│   ├── EDA.ipynb
│   ├── pos_topologi_sungai.csv
│   └── output_eda/                    # enam plot hasil EDA
├── 1.61/Codex-CatBoostRegressor-main/
│   ├── notebook-150726.ipynb
│   ├── notebook-optimized.ipynb
│   ├── notebook-template.ipynb
│   ├── compare_submission.ipynb
│   ├── ablation_results.csv
│   ├── oof_predictions (1).parquet
│   ├── gen_notebook.py
│   ├── requirements.txt
│   └── xgboost/
│       ├── xgboostibra.ipynb
│       ├── analyze_oof.py
│       ├── oof_predictions.parquet
│       ├── metrics.csv
│       └── station_metrics.csv
├── project_overview.md
└── architecture.md
```

## Menjalankan project

### Kaggle

Notebook utama mendeteksi dataset kompetisi pada:

```text
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026
```

Output ditulis ke `/kaggle/working`. Atur `MODE` pada cell konfigurasi sesuai tahap yang dibutuhkan:

1. `audit` untuk pemeriksaan input tanpa training.
2. `tune` untuk sweep ringan CatBoost pada satu fold.
3. `validate` untuk rolling-origin validation penuh.
4. `ablation` untuk membandingkan spike weighting dan feature pruning.
5. `fit` untuk final fit memakai iterasi eksplisit atau rekomendasi run terdahulu.
6. `all` untuk validasi, final fit, dan submission dalam satu run.

### Lokal

Dependensi dasar tercatat di `1.61/Codex-CatBoostRegressor-main/requirements.txt`. Pipeline utama juga memerlukan `lightgbm`, `joblib`, Jupyter, dan engine Parquet seperti `pyarrow`; challenger memerlukan `xgboost`.

Workspace lokal belum langsung siap untuk `Run All` karena:

- notebook memakai `DATA_DIR = Path("datasets")`, sedangkan data saat ini berada di root workspace;
- `sample_submission.csv` yang diwajibkan oleh audit dan penulisan submission tidak tersedia;
- daftar dependensi belum mencakup seluruh library pipeline utama dan eksperimen.

Untuk eksekusi lokal, sediakan `sample_submission.csv`, lalu pilih salah satu layout berikut:

- ubah `DATA_DIR` menjadi `Path(".")`; atau
- pindahkan/salin seluruh input ke folder `datasets/` dengan struktur yang sama.

Mulai dari `MODE = "audit"`. Cell konfigurasi notebook saat ini tersimpan dengan `MODE = "all"`, sehingga harus diubah lebih dahulu bila training penuh tidak diinginkan.

## Risiko dan batasan

- Hanya sekitar 2,7 tahun label tersedia untuk horizon prediksi delapan bulan.
- Operasi bendungan atau perubahan datum sensor mungkin tidak terwakili oleh fitur lingkungan.
- Target mengandung beberapa spike/reset sensor; keputusan pembobotannya perlu berdasarkan ablation, bukan asumsi.
- Ketersediaan beberapa fitur berubah pada periode test, khususnya radiasi surya dan indeks iklim.
- Banyak logika produksi masih berada dalam notebook monolitik, belum menjadi package Python yang teruji otomatis.
- Artefak run utama terbaru hanya direferensikan dalam output notebook Kaggle, tidak ada di workspace lokal.
- Belum ada lockfile, data versioning, experiment tracker terpusat, atau CI untuk validasi pipeline.

## Rekomendasi pengembangan

1. Jadikan `MODE = "audit"` sebagai default aman dan pindahkan konfigurasi ke file YAML/TOML.
2. Ekstrak fungsi dari notebook menjadi package `src/` dengan modul data, features, models, validation, dan artifacts.
3. Lengkapi `requirements.txt` atau gunakan environment lock yang mencakup CatBoost, LightGBM, XGBoost, PyArrow, dan library EDA.
4. Tambahkan test untuk parsing ID, cardinality join, causal rolling window, traversal upstream, normalisasi target, dan urutan submission.
5. Simpan artefak final dan metadata dataset di storage berversi agar hasil Kaggle dapat direproduksi lokal.
6. Bandingkan model hanya dari OOF rolling-origin yang sama, lalu dokumentasikan alasan promosi model.

