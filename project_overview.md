# Project Overview: Prediksi Tinggi Muka Air (TMA) - SSDS

Dokumen ini berisi rangkuman informasi tentang proyek prediksi Tinggi Muka Air (TMA), perkembangan terkini (progress), dan rencana ke depannya berdasarkan tahapan yang telah dikerjakan.

---

## 1. Deskripsi Proyek
Proyek ini bertujuan untuk memprediksi tinggi muka air sungai (`tma_mdpl`) di berbagai pos pengamatan. Prediksi ini kemungkinan ditujukan sebagai peringatan dini (*early warning system*) banjir atau manajemen sumber daya air. 
Data yang tersedia meliputi:
- **Data Target**: `train.csv` (TMA per pos dengan resolusi 6 jam).
- **Data Eksogen**: `data_lingkungan.csv` (fitur cuaca dan iklim global dengan resolusi 1 jam).
- **Data Spasial**: `koordinat_pos.csv` dan topologi sungai (`HydroRIVERS`).

---

## 2. Progress Saat Ini (Tahap Eksplorasi Data - EDA)

Saat ini, kita sedang berada di tahap **Exploratory Data Analysis (EDA)**. Berbagai eksperimen telah dilakukan di dalam *notebook* `EDA/EDA.ipynb`. Berikut adalah detail pencapaian yang telah diselesaikan:

### A. Eksplorasi Target (`tma_mdpl`)
1. **Identifikasi Distribusi & Outlier**: 
   - Mengecek *boxplot* dan deret waktu (*time-series*) mentah TMA di masing-masing pos stasiun.
   - **Tujuan**: Memisahkan lonjakan liar akibat *noise* dari sensor dengan pola pergerakan banjir yang asli.
2. **Trend Musiman (Seasonal)**: 
   - Mengekstrak informasi bulan dan menganalisis rata-rata perbedaan TMA (musim hujan vs kemarau).
   - **Tujuan**: Memutuskan apakah penambahan fitur siklikal (seperti bulan atau *day-of-year*) diperlukan.
3. **Autokorelasi (ACF/PACF)**:
   - Membuat plot *Autocorrelation Function* (ACF) dan *Partial Autocorrelation Function* (PACF) untuk setiap pos sungai.
   - **Tujuan**: Mengukur tingkat kelambatan (*response time*) sungai dalam merespons kenaikan air dan menentukan rentang waktu (*window size* / *lag*) yang optimal untuk *feature engineering*.
4. **Perbandingan Pos Searah (Hulu-Hilir)**:
   - Membangun matriks korelasi (*heatmap*) antar pos stasiun.
   - **Tujuan**: Memvalidasi fitur aliran silang (*upstream*). Jika dua stasiun berada di sungai yang sama, korelasi TMA-nya dapat digunakan sebagai dasar pergerakan air hulu ke hilir.

### B. Eksplorasi Fitur Eksogen (`data_lingkungan.csv`)
1. **Perbandingan Fitur Curah Hujan**:
   - Membandingkan dan menghitung korelasi Pearson antara `rainfall_mm` (data observasi stasiun) dan `rainfall_openmeteo_mm` (data reanalisis).
   - Mengevaluasi persentase kemunculan hujan bernilai 0 dari kedua sumber.
2. **Pola *Missing Value* & Distribusi Fitur Cuaca**:
   - Melakukan pengecekan kelengkapan data cuaca (`humidity`, `temperature`, `wind speed`, `solar radiation`).
   - Memvisualisasikan distribusi khusus pada nilai curah hujan dan radiasi di atas 0 (`>0`).
3. **Korelasi 4 Layer *Soil Moisture***:
   - Membuat *heatmap* korelasi kelembaban tanah di empat kedalaman berbeda (`0-7cm`, `7-28cm`, `28-100cm`, `100-255cm`).
   - **Tujuan**: Mencari redundansi antar fitur. Jika korelasi tinggi, lapisan ini dapat digabung/dirata-rata.
4. **Agregasi Resolusi Temporal & Hubungan Lag-Lead (Hujan vs TMA)**:
   - Mengatasi perbedaan granularitas waktu dengan menyamakan fitur lingkungan (yang semula per jam) menjadi jeda 6 jam (sesuai target TMA) menggunakan skema `sum` untuk hujan dan `mean` untuk suhu/kelembaban.
   - Menganalisis korelasi silang (*cross-correlation*) dan *hydrograph* untuk mengetahui *response time* aktual kenaikan TMA akibat akumulasi curah hujan.
5. **Cek Granularitas Fitur Makro-Iklim (RMM1, RMM2, MJO, Nino3.4)**:
   - Mem-plot pergerakan harian indeks iklim jangka panjang.
   - **Kesimpulan Temuan**: Variabel seperti ENSO (`nino_34`) bergerak sangat lambat (*slow-varying*). Menggunakannya sebagai nilai numerik titik-per-titik sangat rawan memicu *overfitting*.

---

## 3. Struktur Direktori Proyek

```text
.
|   .gitignore
|   project_overview.md
|   sample_submission.csv
|   test.csv
|   train.csv
|   
+---data_pendukung
|   |   data_lingkungan.csv
|   |   HydroRIVERS_TechDoc_v10.pdf
|   |   koordinat_pos.csv
|   |   
|   \---HydroRIVERS_v10_au_shp
|           HydroRIVERS_v10_au.dbf
|           HydroRIVERS_v10_au.prj
|           HydroRIVERS_v10_au.sbn
|           HydroRIVERS_v10_au.sbx
|           HydroRIVERS_v10_au.shp
|           HydroRIVERS_v10_au.shx
|           
\---EDA
        EDA.ipynb
```
