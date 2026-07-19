# Arsitektur Prediksi Tinggi Muka Air

## 1. Tujuan dan batasan

Pipeline memprediksi `tma_mdpl` untuk 30 pos pada 19 September 2025 sampai 18 Mei 2026, pukul 06.00, 12.00, dan 18.00. Horizon ini terdiri dari 242 hari atau 21.780 prediksi. Kode berada di `notebook-150726.ipynb`; training tidak dijalankan pada saat pengembangan ini.

Prinsip desain utamanya adalah kesetaraan antara kondisi validasi dan deployment. Karena tidak ada TMA aktual setelah 18 September 2025, model utama tidak memakai lag TMA sebagai input. Lag autoregresif terlihat kuat pada validasi satu langkah, tetapi untuk horizon delapan bulan harus memakai prediksinya sendiri dan akan mengakumulasi error serta gagal saat rezim hidrologi berubah.

## 2. Ringkasan audit data

| Sumber | Ukuran/coverage | Temuan penting |
|---|---:|---|
| Target train | 84.396 baris, 30 pos | 1 Januari 2023 06.00 - 18 September 2025 18.00; tidak ada `NaN` atau key duplikat |
| Grid target teoritis | 89.280 baris | 4.884 timestamp tidak tercatat sebagai baris, terutama 4-28 Februari 2025; Gunungsari baru tersedia sejak 13 Agustus 2024 |
| Test/submission | 21.780 baris | ID dan urutan `test.csv` identik dengan sample submission |
| Lingkungan | 888.480 baris, 27 kolom | Grid per jam lengkap untuk 30 pos dari 1 Januari 2023 sampai 18 Mei 2026 |
| Koordinat | 30 baris | Set nama pos persis sama dengan train/test/lingkungan |
| HydroRIVERS | 836.472 reach pada tile AU | CRS WGS84; 15 atribut sesuai dokumentasi |

Temuan kualitas khusus:

- `rainfall_mm` dan `rainfall_openmeteo_mm` identik pada seluruh 888.480 baris. Hanya satu yang digunakan agar model tidak memberi bobot ganda pada sinyal yang sama.
- `solar_radiation_mj_m2` memiliki 100.080 nilai sentinel `-999`, seluruhnya pada masa mendatang. Sentinel diubah menjadi missing dan disertai flag; nilai negatif tidak diperlakukan sebagai radiasi nyata.
- Pada 18 Mei 2026, enam variabel soil/pressure dan lima variabel MJO kosong selama 24 jam. Nilainya di-forward-fill secara kausal maksimum 24 jam dan flag missing dipertahankan.
- `nino_34` kosong pada 1-18 Mei 2026. Karena indeks ini berubah lambat/bulanan, digunakan causal forward-fill maksimal 31 hari plus flag missing.
- Ada lonjakan TMA satu-titik yang tidak fisik, misalnya Napel 34,95 -> 325,83 -> 34,55. Titik seperti ini tidak dihapus secara kasar; detektor lokal yang konservatif memberi bobot 0,05 hanya bila dua tetangga temporal saling konsisten. Kejadian tinggi yang berlangsung beberapa timestamp tetap berbobot penuh.

## 3. Model yang dipilih

Model utama adalah ensemble tiga `CatBoostRegressor` global dengan seed 17, 41, dan 83. Target ditransformasi per pos:

```text
z(pos, t) = (TMA(pos, t) - median_train(pos)) / max(IQR_train(pos) / 1,349; 0,05)
```

Setiap model menggunakan loss Huber (`delta=1,5`). Prediksi tiga seed dirata-ratakan pada skala `z`, lalu dikembalikan ke skala meter tiap pos.

CatBoost dipilih karena jumlah label sedang, terdapat interaksi nonlinier kuat, fitur kategorikal/statis, missing value, dan distribusi yang tidak Gaussian. Normalisasi robust mencegah pos dengan elevasi absolut 140 mdpl mendominasi pos dengan TMA sekitar 1-10 mdpl. Huber dan ensemble seed mengurangi sensitivitas terhadap error sensor dan variance tanpa memotong kejadian banjir yang valid.

LSTM/Transformer temporal tidak dipilih sebagai arsitektur utama: histori efektif hanya sekitar 2,7 tahun, satu pos memiliki histori jauh lebih pendek, dan rollout 242 hari menimbulkan distribution shift besar. GNN end-to-end juga belum sebanding dengan hanya 30 node dan kurang dari tiga siklus tahunan. Informasi graf tetap digunakan secara eksplisit melalui fitur upstream yang lebih mudah diaudit.

## 4. Aliran data

```text
train/test ID -----> validasi key dan waktu ----------------------------+
                                                                      |
lingkungan per jam -> sanitasi sentinel/missing -> fitur rolling ------+--> CatBoost global
                                      |                               |       x 3 seed
HydroRIVERS + koordinat -> nearest reach -> graf NEXT_DOWN -> upstream-+          |
                                      |                               |          v
                                      +-> atribut reach statis --------+    inverse robust scale
                                                                      |          |
kalender siklik -------------------------------------------------------+          v
                                                                            submission.csv
```

Tidak ada fitur yang diturunkan dari target pada matriks `X`. Pipeline notebook juga menolak nama fitur yang diawali `tma_` atau mengandung kata `target`.

## 5. Preprocessing

1. ID submission diparse dengan regex timestamp tetap. String tidak di-split secara bebas karena beberapa nama pos mengandung ` - `.
2. Seluruh sumber diperiksa untuk duplikasi `(datetime, nama_pos)`, kesamaan set pos, jam target, urutan sample submission, dan kelengkapan grid lingkungan.
3. Missing baris target tidak diinterpolasi dan tidak dibuat sebagai pseudo-label. Baris berlabel yang benar saja masuk training.
4. Sentinel radiasi `-999` diubah ke `NaN`. CatBoost dapat memakai missing numerik secara native; flag missing tetap menjadi fitur.
5. State lingkungan kosong diisi hanya dengan nilai masa lalu dalam limit yang sesuai. Tidak ada backward-fill, sehingga validasi tidak membaca masa depan.
6. Arah angin diubah menjadi pasangan sinus/kosinus agar 359 derajat dekat dengan 0 derajat.
7. Hujan/radiasi dibatasi pada minimum fisik nol dan soil moisture pada rentang 0-1. Tidak ada clipping berdasarkan statistik test.

## 6. Feature engineering

### 6.1 Dinamis lokal

- Nilai lingkungan saat ini: hujan, kelembapan, suhu, dew point, cloud, angin, soil moisture berlapis, pressure, MJO, Nino 3.4, built surface, dan land cover.
- Dew-point depression dan arah angin sinus/kosinus.
- Akumulasi hujan trailing 6, 24, 72, 168, dan 336 jam.
- Maksimum intensitas 24 jam.
- Antecedent precipitation index berbasis EWM dengan half-life 12, 48, dan 168 jam.
- Rata-rata trailing soil moisture, suhu, kelembapan, dan pressure. Semua window bersifat right-closed/kausal.

### 6.2 Jaringan sungai

Setiap pos dicocokkan dengan reach terdekat di EPSG:32749. Jarak pencocokan disimpan sebagai fitur, bersama confidence `exp(-distance_m / 1000)`. Jarak terbesar yang ditemukan sekitar 1,24 km (Bengkelolor dan Jarum), sehingga confidence penting untuk membatasi ketergantungan buta pada match tersebut.

Dokumentasi HydroRIVERS menjelaskan:

- `NEXT_DOWN`: reach berikutnya di hilir; digunakan untuk traversal graf yang eksak.
- `MAIN_RIV`: reach paling hilir pada jaringan DAS; dipakai sebagai kategori basin.
- `LENGTH_KM`, `DIST_DN_KM`, `DIST_UP_KM`: posisi dan panjang hidraulik.
- `CATCH_SKM`, `UPLAND_SKM`: luas catchment lokal dan total hulu.
- `DIS_AV_CMS`: estimasi debit jangka panjang; bukan pengukuran saat ini.
- `ORD_STRA`, `ORD_CLAS`, `ORD_FLOW`: tiga definisi orde/ukuran sungai.
- `HYBAS_L12`: identitas sub-basin HydroBASINS level 12.

Untuk setiap target pos, kode menelusuri `NEXT_DOWN` dari pos sumber. Jika reach target ditemukan di hilir, kondisi lingkungan sumber menjadi anggota agregat upstream. Bobotnya `exp(-jarak_hidraulik / 75 km)`, kemudian dinormalisasi. Dari sini dibentuk hujan dan soil moisture upstream beserta window/API yang sama dengan fitur lokal.

### 6.3 Kalender dan statis

- Siklus day-of-year dan jam dengan sinus/kosinus.
- Kategori jam dan bulan serta flag musim hujan November-April.
- Latitude/longitude, built surface, land cover.
- Atribut HydroRIVERS mentah dan `log1p` untuk luas catchment, upstream area, debit, serta panjang reach.

## 7. Validasi untuk generalisasi

Validasi menggunakan rolling-origin dengan horizon panjang. Setiap fold hanya melatih data sebelum cutoff dan menguji seluruh rentang berikutnya:

| Fold | Train sampai | Validasi |
|---|---|---|
| sep_2023 | 18 Sep 2023 | 19 Sep 2023 - 18 Mei 2024 |
| may_2024 | 18 Mei 2024 | 19 Mei 2024 - 18 Jan 2025 |
| sep_2024 | 18 Sep 2024 | 19 Sep 2024 - 18 Mei 2025 |
| jan_2025 | 18 Jan 2025 | 19 Jan 2025 - 18 Sep 2025 |

Pos dengan kurang dari 90 label sebelum cutoff tidak dinilai pada fold tersebut. Hal ini mencegah statistik target dari pos yang belum pernah terlihat. Early stopping dilakukan pada target ternormalisasi; jumlah iterasi final adalah median best iteration lintas fold.

Metrik yang dicatat:

- MAE dan RMSE global;
- macro-MAE antar-pos agar semua pos mendapat bobot setara;
- worst-station MAE untuk mendeteksi kegagalan lokal;
- persentil ke-95 absolute error;
- pembanding median historis per pos-bulan-jam.

Model layak dipakai hanya jika perbaikannya konsisten lintas fold, bukan hanya rata-rata pada satu split. Analisis residual sebaiknya dilanjutkan per pos, bulan, intensitas hujan, dan kuantil TMA setelah training benar-benar dijalankan.

## 8. Artefak dan cara menjalankan

Instalasi:

```powershell
python -m pip install -r requirements.txt
```

Setelah membuka `notebook-150726.ipynb`, jalankan semua cell dengan mode default berikut untuk audit tanpa training:

```python
MODE = "audit"
```

Untuk validasi saja, ubah cell konfigurasi mode menjadi:

```python
MODE = "validate"
```

Untuk validasi, final fit, dan submission:

```python
MODE = "all"
```

Final fit dapat dijalankan terpisah memakai rekomendasi iterasi dari validasi:

```python
MODE = "fit"
```

Pada runtime lokal, output ditulis ke `artifacts/`. Pada Kaggle, output ditulis ke `/kaggle/working/`. Setiap eksekusi training memiliki subdirektori dan ZIP tersendiri berdasarkan waktu mulai run.

## 9. Guardrail operasional

- `MODE = "audit"` adalah default agar Run All tidak melatih model tanpa sengaja.
- CatBoost di-import secara lazy hanya pada mode training.
- Setiap merge memakai validasi cardinality dan submission akhir harus lengkap serta finite.
- Seed, feature list, target scaling, bobot upstream, dan konfigurasi disimpan sebagai artefak.
- Prediksi tidak menggunakan recursive TMA, interpolasi label, backward-fill lingkungan, atau statistik test untuk fit preprocessing.

## 10. Konfigurasi Kaggle

Notebook mendeteksi Kaggle dari keberadaan direktori kompetisi. Path input yang digunakan adalah:

```text
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/train.csv
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/test.csv
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/sample_submission.csv
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/data_pendukung/koordinat_pos.csv
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/data_pendukung/data_lingkungan.csv
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/data_pendukung/HydroRIVERS_v10_au_shp/...
```

Seluruh file hasil ditulis ke `/kaggle/working/`. Jika path kompetisi tidak ditemukan, notebook otomatis kembali memakai `datasets/` dan `artifacts/` untuk eksekusi lokal.

Cell konfigurasi yang disarankan untuk training Kaggle penuh:

```python
MODE = "all"
DATA_DIR = KAGGLE_DATA_DIR
OUTPUT_DIR = Path("/kaggle/working")
ITERATIONS = None
SAVE_FEATURE_MATRICES = True
```

## 11. Handoff training untuk penyempurnaan

Nama run memakai waktu UTC saat pipeline training dimulai:

```text
handoff_YYYYMMDDTHHMMSSZ_<mode>/
handoff_YYYYMMDDTHHMMSSZ_<mode>_in_progress.zip
handoff_YYYYMMDDTHHMMSSZ_<mode>_complete.zip
handoff_YYYYMMDDTHHMMSSZ_<mode>_failed.zip
```

ZIP `in_progress` dibuat setelah feature preparation dan diperbarui setelah setiap fold validasi serta setiap seed final selesai. Setelah sukses, ZIP tersebut diganti oleh `complete`. Jika terjadi exception, artefak parsial, snapshot CatBoost, log, dan traceback dikemas sebagai `failed`.

### 11.1 File inti yang wajib ada

| File | Kegunaan penyempurnaan |
|---|---|
| `manifest.json` | Identitas run, waktu UTC, mode, status, stage terakhir, dan schema handoff |
| `config.json` | Hyperparameter, seed, definisi fold, iteration request, dan toggle feature matrix |
| `runtime_versions.json` | Versi Python, CatBoost, pandas, NumPy, GeoPandas, dan dependency utama |
| `input_inventory.json` | Path dan ukuran seluruh input termasuk sidecar shapefile |
| `data_audit.json` | Coverage data dan jumlah missing/sentinel sebelum preprocessing |
| `feature_columns.json` | Urutan fitur aktual serta daftar fitur kategorikal |
| `data/target_normalization.csv` | Center dan robust scale setiap pos |
| `data/upstream_station_weights.csv` | Bobot graf sungai yang benar-benar digunakan |
| `data/train_sample_diagnostics.parquet` | Target asli/normalisasi, bobot sampel, dan flag spike |
| `metrics/environment_drift.csv` | Pergeseran distribusi setiap fitur antara train dan test |
| `run.log` | Urutan stage dan pesan pipeline |
| `HANDOFF_README.md` | Indeks singkat isi ZIP |

Jika `SAVE_FEATURE_MATRICES=True`, handoff juga memuat `data/train_features.parquet` dan `data/test_features.parquet`. Kedua file ini sangat disarankan karena memungkinkan Codex menganalisis fitur persis yang masuk model tanpa membangun ulang dataset. Bila Parquet tidak tersedia, pipeline otomatis memakai `csv.gz`.

### 11.2 File dari validasi

- `metrics/oof_predictions.parquet`: datetime, pos, fold, target, prediksi, baseline, target/prediksi ternormalisasi, error, dan absolute error.
- `metrics/validation_metrics.csv`: MAE/RMSE global, macro/worst-station MAE, p95 error, baseline, dan best iteration.
- `metrics/fold_station_metrics.csv`: MAE, RMSE, dan bias setiap pos pada setiap fold.
- `metrics/residual_slices.csv`: residual per pos, bulan, dan jam.
- `metrics/feature_importance.csv`: feature importance setiap fold.
- `models/validation_<fold>.cbm`: model fold untuk inspeksi/reproduksi.
- `data/target_stats_<fold>.csv`: normalisasi target yang hanya di-fit pada training fold.

### 11.3 File dari final fit

- `models/catboost_seed_<seed>.cbm`: tiga anggota ensemble final.
- `metrics/final_feature_importance.csv`: importance setiap seed final.
- `training_progress/final_seed_<seed>/`: log iterasi dan snapshot CatBoost.
- `submission.csv`: output kompetisi dengan urutan ID asli.
- `run_metadata.json` dan `result_summary.json`: ringkasan final iterasi, sumber iterasi, jumlah baris, serta lokasi handoff.

Folder `training_progress/` ditulis langsung oleh CatBoost selama fitting (`learn_error.tsv`, `test_error.tsv`, `catboost_training.json`, `time_left.tsv`, dan snapshot bila tersedia). Penggunaan `train_dir`, `allow_writing_files`, dan snapshot mengikuti [dokumentasi resmi output CatBoost](https://catboost.ai/docs/en/references/training-parameters/output). Untuk penyempurnaan berikutnya, ZIP `complete` atau `failed` inilah yang perlu diberikan kembali kepada Codex. OOF prediction harus menjadi dasar utama keputusan model, bukan training loss.
