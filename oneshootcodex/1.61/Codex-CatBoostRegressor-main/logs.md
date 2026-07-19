# Log Pengembangan

## 2026-07-15 - Audit dan implementasi awal

Status akhir sesi: kode selesai ditulis dan diverifikasi secara statis/audit-only. Training model belum dijalankan.

### Aktivitas

- Menginventarisasi seluruh file proyek. `architecture.md` dan `logs.md` semula kosong; pipeline baru memakai sumber data resmi di bawah `datasets/` dan tidak mengubah notebook yang sudah ada.
- Membaca teks dan merender visual 7 halaman `HydroRIVERS_TechDoc_v10.pdf`. Halaman atribut dan proyeksi diperiksa secara visual.
- Membaca shapefile HydroRIVERS: 836.472 reach, CRS EPSG:4326, 15 kolom termasuk geometry.
- Memeriksa schema, coverage, key duplikat, missing value, distribusi, dan anomali train/test/lingkungan/koordinat.
- Mencocokkan 30 koordinat pos ke HydroRIVERS di UTM 49S dan memverifikasi relasi upstream-downstream dari `NEXT_DOWN`.
- Menulis pipeline training, validation, final ensemble, dan pembuatan submission; implementasi awal berada di `train.py` sebelum diminta dipindahkan ke notebook.
- Menulis dependensi runtime di `requirements.txt` dan keputusan desain lengkap di `architecture.md`.
- Menjalankan `python -m py_compile train.py`: berhasil.
- Menjalankan `python train.py --mode audit`: berhasil. Perintah ini tidak mengimpor CatBoost dan tidak melakukan fitting.
- Menjalankan builder preprocessing/feature engineering secara langsung tanpa fitting: berhasil membentuk 98 fitur model, 84.396 baris train, dan 21.780 baris test. Tidak ditemukan key duplikat, infinity, atau target timestamp tanpa pasangan hujan.
- Memverifikasi matriks upstream: 180 edge berbobot termasuk self-link dan jumlah bobot setiap target pos sama dengan 1 (dalam toleransi floating point).

### Temuan data

- Train: 84.396 baris, 30 pos, 1 Januari 2023 06.00 sampai 18 September 2025 18.00.
- Test/submission: 21.780 baris, 19 September 2025 06.00 sampai 18 Mei 2026 18.00; ID dan urutannya identik.
- Lingkungan: 888.480 baris atau 29.616 jam x 30 pos; tidak ada key yang hilang atau duplikat.
- Grid target teoritis memiliki 89.280 baris; 4.884 observasi target tidak tercatat. Missing paling umum terjadi pada 4-28 Februari 2025. Gunungsari tidak memiliki data sebelum 13 Agustus 2024.
- Tidak ada `NaN` eksplisit pada `tma_mdpl`, tetapi ada beberapa spike satu-titik yang sangat besar dan segera kembali ke baseline.
- Detektor spike konservatif menandai 37 dari 84.396 label (0,044%) untuk bobot 0,05; semua label tetap berada di dataset.
- Missing eksplisit lingkungan berjumlah 20.880 sel: 720 per variabel pada 18 Mei 2026 untuk soil/pressure/MJO dan 12.960 pada `nino_34` untuk 1-18 Mei 2026.
- Selain `NaN`, `solar_radiation_mj_m2` memuat 100.080 sentinel `-999`. Sentinel tersebut menyebabkan mean masa prediksi tampak negatif dan harus diubah menjadi missing.
- `rainfall_mm` dan `rainfall_openmeteo_mm` identik 100%.
- Nearest-river distance berada pada sekitar 2,7 m sampai 1.237 m; jarak disimpan sebagai fitur agar match yang lemah dapat dikenali model.

### Keputusan model

- Ensemble 3 seed CatBoost global, loss Huber, target robust-normalized per pos.
- Fitur eksogen non-rekursif agar validasi dan prediksi 242 hari mempunyai informasi yang sama.
- Rolling-origin validation dengan empat horizon panjang dan baseline median pos-bulan-jam.
- Missing label tidak diimputasi; isolated spike hanya diturunkan bobotnya, bukan dihapus.
- Agregat hujan/soil upstream dibangun dari traversal `NEXT_DOWN`, dengan decay jarak hidraulik.

### Verifikasi yang belum dijalankan

- CatBoost cross-validation, early stopping, dan final ensemble sengaja belum dijalankan sesuai instruksi.
- Karena model belum dilatih, belum ada angka MAE/RMSE, best iteration, feature importance, model `.cbm`, atau submission prediktif.

## 2026-07-15 - Pemindahan pipeline ke notebook

- Seluruh definisi dari `train.py` dipindahkan ke `notebook-150726.ipynb` dan dibagi menjadi 11 bagian yang dapat dibaca/diuji secara bertahap.
- Ditambahkan fungsi `run_pipeline()` sebagai pengganti CLI serta cell `MODE`, `DATA_DIR`, `ARTIFACTS_DIR`, dan `ITERATIONS`.
- Mode default notebook adalah `audit`; Run All tidak memanggil fitting CatBoost kecuali `MODE` diubah eksplisit menjadi `validate`, `fit`, atau `all`.
- `train.py` dihapus setelah pemindahan sehingga tidak ada dua implementasi yang dapat menyimpang.
- Notebook berhasil diparse sebagai nbformat 4.5 dengan 23 cell; seluruh 11 code cell lolos pemeriksaan AST Python.
- Seluruh cell kemudian diverifikasi dalam mode default `audit`: audit menghasilkan jumlah baris/pos yang sama seperti implementasi awal dan memastikan modul CatBoost tidak dimuat.
- Training tetap belum dijalankan.

## 2026-07-15 - Dukungan Kaggle dan handoff training

- Menambahkan deteksi otomatis input kompetisi di `/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026` beserta seluruh isi `data_pendukung/`.
- Menetapkan `/kaggle/working/` sebagai output root pada Kaggle, dengan fallback lokal `datasets/` dan `artifacts/`.
- Menambahkan run ID bertimestamp UTC berbentuk `handoff_YYYYMMDDTHHMMSSZ_<mode>`.
- Menambahkan manifest, konfigurasi, versi runtime, inventory input, data audit, matriks fitur opsional, diagnostic sample weight, drift report, dan README handoff.
- Menambahkan log CatBoost per fold/seed, snapshot berkala 300 detik, OOF prediction, metrik per pos, residual slice, feature importance, model fold, dan model ensemble final.
- Menambahkan ZIP `in_progress` setelah feature preparation dan setiap fold/seed; run sukses menghasilkan ZIP `complete`, sedangkan exception menghasilkan ZIP `failed` beserta traceback.
- Memperbaiki export feature matrix agar nama key yang juga menjadi fitur kategorikal tidak menghasilkan kolom duplikat.
- Seluruh 12 code cell lolos AST. Notebook berhasil dijalankan dalam mode `audit` dengan path lokal fallback dan tanpa memuat CatBoost.
- Mekanisme handoff diuji memakai frame dummy: run ID waktu, Parquet, manifest, ZIP in-progress, penggantian menjadi ZIP complete, serta struktur anggota ZIP semuanya valid.
- Handoff pra-training juga diuji dengan seluruh 98 fitur nyata dan `SAVE_FEATURE_MATRICES=False`: diagnostic sample, target normalization, upstream weights, drift report, manifest, dan ZIP in-progress berhasil dibuat; hasil uji kemudian dibersihkan.
- Parameter `train_dir`, `allow_writing_files`, dan snapshot diverifikasi terhadap dokumentasi resmi CatBoost.
- Training CatBoost tetap belum dijalankan.
