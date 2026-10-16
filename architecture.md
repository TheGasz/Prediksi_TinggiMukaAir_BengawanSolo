# Arsitektur Prediksi Tinggi Muka Air (v3 Champion Pipeline)

## 1. Tujuan dan Batasan

Pipeline memprediksi `tma_mdpl` untuk 30 pos pemantauan sungai di DAS Bengawan Solo pada 19 September 2025 sampai 18 Mei 2026, pukul 06.00, 12.00, dan 18.00 (242 hari / 21.780 prediksi). Implementasi utama berada di `anchor-with-mediancodex.ipynb`.

Karena tidak tersedia observasi aktual `tma_mdpl` setelah 18 September 2025 pukul 18.00, model utama bersifat **eksogen-kausal** terhadap variabel hidrometeorologi dan topologi sungai, ditambah koreksi memori kausal dari anomali TMA pada titik potong terakhir (*cutoff residual decay*).

## 2. Diagnosis Kelemahan Pipeline v2 & Solusi v3

Melalui audit mendalam terhadap data dan kode v2 (`anchor-with-mediancodex.ipynb`), ditemukan 6 bottleneck utama yang membatasi akurasi v2:

1. **Kunci Fitur `ANCHOR_OFF_FROZEN_FEATURE_COLUMNS` (84 Fitur) yang Terlalu Sempit:**
   - Pada v2, fungsi `enforce_frozen_anchor_off_columns()` membuang semua fitur baru di luar daftar beku 84 kolom.
   - **Solusi v3:** Membuka kunci daftar fitur (`enforce_frozen_84_features=False`) dan memperkaya ruang fitur menjadi **~178 fitur fisik hidrologi** yang bebas *leakage*.
2. **Kurangnya Memori Penyimpanan Jangka Panjang (30–90 Hari) untuk Waduk & Sungai Besar:**
   - Di v2, jendela *rolling* terpanjang hanya `336 jam` (14 hari). Padahal pada pos waduk/bendung seperti **Wonogiri Dam**, korelasi hujan 14 hari terhadap TMA hanya $r = 0.274$, sedangkan kelembapan tanah kolom penuh (`0–255 cm`) *rolling* 720 jam (30 hari) mencapai **$r = 0.841$** dan neraca air bersih (*net water balance*) *rolling* 2160 jam (90 hari) mencapai **$r = 0.808$**.
   - **Solusi v3:** Menambahkan memori hidrologi jangka panjang (`720h` / 30 hari, `1440h` / 60 hari, dan `2160h` / 90 hari) serta EWM API `336h` dan `720h`.
3. **14 dari 30 Pos Tidak Memiliki Hulu di Jaringan 30 Pos (`upstream_* = 0`):**
   - Matriks `upstream_*` berbasis HydroRIVERS `NEXT_DOWN` hanya memberi sinyal pada 16 pos hilir, sedangkan 14 pos hulu (*headwaters* seperti Wonogiri Dam, Tangen, Ketonggo, Sekayu, dll.) bernilai nol pada seluruh fitur `upstream_*`.
   - **Solusi v3:** Menambahkan **Spatial Inverse-Distance Weighting (IDW) Regional** (`regional_*` dengan $\tau = 35\text{ km}$, mengecualikan diri sendiri) sehingga semua 30 pos memperoleh konteks curah hujan, neraca air, dan kelembapan tanah sub-DAS di sekitarnya.
4. **Pergeseran Distribusi (*Covariate Shift*) pada `solar_radiation_mj_m2`:**
   - `solar_radiation_mj_m2` memiliki 0% missing di data train, tetapi **57,5% bernilai sentinel `-999` (`NaN`) di periode test** (3 Desember 2025 – 18 Mei 2026).
   - **Solusi v3:** Mengecualikan `solar_radiation_mj_m2` dan `solar_radiation_missing` dari daftar fitur model, serta menggantikan sinyal penguapan atmosfer dengan **FAO-56 Vapor Pressure Deficit (`vpd_kpa`)**, **Evapotranspiration Proxy (`et0_proxy_mm`)**, dan **Net Water Balance (`net_water_mm`)** yang 100% tersedia tanpa missing di test set.
5. **Keterbatasan Model Global Tunggal untuk Pos dengan Rezim Unik:**
   - GBDT global harus membagi *split* pohon untuk 30 pos sekaligus, sehingga kurang presisi menangkap kurva histeresis musiman & respon neraca air spesifik per pos (seperti aturan operasi waduk di Wonogiri Dam atau backwater di Babat Barrage).
   - **Solusi v3:** Membangun **`StationPhysicsExpertRegressor`** (kombinasi Ridge ter-regularisasi per pos dengan bobot Huber + *ExtraTrees residual learner* per pos pada ~36 fitur fisik inti) sebagai anggota keluarga ensemble ke-4 yang independen.
6. **Train–Serving Skew pada Fitur Anchor Lama:**
   - Di v2, fitur `anchor_cutoff_Anomaly` memiliki `horizon_days <= 0` di seluruh baris train dan `horizon_days > 0` di seluruh baris test, sehingga pohon keputusan tidak pernah belajar peluruhan horizon positif.
   - **Solusi v3:** Mengganti fitur anchor di dalam pohon menjadi **koreksi post-processing kausal di ruang meter (`apply_cutoff_residual_anchor`)** yang mengestimasi bias level pos pada 21 hari terakhir sebelum cutoff ($\tau = 45\text{ hari}$, shrinkage $0.50$), dikombinasikan dengan **3-point causal hydrograph smoothing (`[0.20, 0.60, 0.20]`)**.

## 3. Rekayasa Fitur Fisik Hidrologi (v3 — ~178 Fitur)

### 3.1 Neraca Air & Permintaan Evaporasi Atmosfer (FAO-56)
- Tekanan uap jenuh $e_s(T)$, tekanan uap aktual $e_a(T_d)$, dan **Vapor Pressure Deficit (`vpd_kpa`)**.
- **Evapotranspiration Proxy (`et0_proxy_mm`)** berbasis kombinasi defisit tekanan uap, suhu, dan kecepatan angin.
- **Net Water Balance (`net_water_mm`)** = `rainfall_mm - et0_proxy_mm`.
- **Non-Linear Saturation Excess Runoff (`runoff_Index_surf`, `runoff_Index_col`)**: Interaksi non-linear hujan dengan kuadrat kelembapan tanah permukaan (`rainfall_mm * soil_moisture_0_7cm^2`) dan kolom penuh.

### 3.2 Fisika Kolom Tanah Lengkap (`0–255 cm`)
- Integrasi kelembapan tanah tertimbang kedalaman (`sm_column_total`) mencakup lapisan `0–7 cm`, `7–28 cm`, `28–100 cm`, dan lapisan dalam **`100–255 cm`** (yang sebelumnya terlewat dari fitur *rolling*).
- Gradien vertikal kelembapan tanah (`sm_vertical_grad = sm_0_7 - sm_100_255`) dan rasio penyimpanan tanah dalam (`sm_deep_ratio = sm_100_255 / sm_column_total`) sebagai indikator *baseflow* air tanah.

### 3.3 Memori Multi-Skala (6 Jam hingga 90 Hari) & Perubahan Hidrograf
- **Jangka Pendek–Menengah (Sungai Cepat/Flashy):** Akumulasi & rata-rata `6h, 12h, 24h, 72h, 168h, 336h`, intensitas hujan maksimum `6h, 24h, 72h`, serta *rate-of-change* (`diff_6h`, `diff_24h`, `diff_72h`, `diff_168h`) untuk membedakan fase naik (*rising limb*) dan fase surut (*recession limb*).
- **Jangka Panjang (Waduk & Sungai Hilir Besar):** Akumulasi & rata-rata `720h` (30 hari), `1440h` (60 hari), dan `2160h` (90 hari) untuk `rainfall_mm`, `net_water_mm`, `sm_column_total`, `soil_moisture_100_255cm`, dan `vpd_kpa`, ditambah EWM API dengan *half-life* `12h, 48h, 168h, 336h, 720h`.
- **Anomali Z-Score Per Pos:** Normalisasi *expanding/station-level* pada fitur memori utama (`rain_sum_168h`, `rain_sum_720h`, `net_water_sum_720h`, `sm_col_mean_168h`, `sm_col_mean_720h`).

### 3.4 Agregasi Spasial Ganda: HydroRIVERS Upstream + Regional IDW
- **HydroRIVERS `upstream_*`:** Bobot jaringan sungai searah aliran (`NEXT_DOWN`, decay 75 km) dengan *travel-time lags* (`6h, 12h, 18h, 24h, 48h`) dan memori hingga `1440h`.
- **Regional IDW `regional_*`:** Rata-rata tertimbang jarak invers ($\exp(-d / 35\text{ km})$) dari pos-pos di sekitarnya untuk menangkap hujan dan kebasahan DAS secara merata baik di pos hulu maupun hilir.

## 4. Arsitektur Ensemble 4-Keluarga (7 Anggota)

Setiap pohon GBDT dilatih pada skala target robust per pos (`z = (TMA - median) / max(IQR / 1.349, 0.05)`) dengan *sample weights* yang mereduksi pengaruh spike sensor (`0.15` pada spike terdeteksi):

1. **`cb_shallow`**: CatBoost (`depth=6`, `lr=0.045`, `l2=8.0`, Huber $\delta=1.5$)
2. **`cb_base`**: CatBoost (`depth=8`, `lr=0.020`, `l2=10.0`, Huber $\delta=1.5$)
3. **`cb_deep`**: CatBoost (`depth=8`, `lr=0.045`, `l2=6.0`, Huber $\delta=1.5$)
4. **`lgbm_a`**: LightGBM (`depth=8`, `num_leaves=63`, `lr=0.030`, `colsample=0.80`, `subsample=0.80`, Huber $\alpha=1.5$)
5. **`lgbm_b`**: LightGBM (`depth=6`, `num_leaves=31`, `lr=0.025`, `colsample=0.65`, `subsample=0.80`, `min_child_samples=80`, Huber $\alpha=1.5$)
6. **`xgb_a`**: XGBoost (`max_depth=6`, `lr=0.025`, `colsample=0.70`, `subsample=0.80`, `reg_lambda=8.0`, `reg_alpha=1.0`, `reg:pseudohubererror`, `huber_slope=1.5`)
7. **`station_expert`**: `StationPhysicsExpertRegressor` (Model spesialis per-pos: 65% Huber-weighted Ridge + 35% per-station ExtraTrees pada ~36 fitur fisik inti + *harmonic seasonal interaction*).

### Post-Processing Kausal
1. **Simplex Blend Optimization (`learned_oof`):** Bobot ketujuh anggota dioptimalkan pada prediksi OOF 4-fold menggunakan proyeksi simpleks (`min_weight >= 0.02`).
2. **Causal Cutoff-Residual Decay Anchor (`apply_cutoff_residual_anchor`):** Menghitung median terbobot eksponensial dari residual $(\text{TMA}_{\text{aktual}} - \hat{y})$ pada 21 hari terakhir sebelum cutoff, lalu meluruhkannya ke masa depan sejauh $\exp(-\Delta t / 45\text{ hari})$ dengan faktor shrinkage $0.50$.
3. **3-Point Hydrograph Smoothing (`apply_station_prediction_smoothing`):** Penghalusan deret waktu per pos (`60%` nilai saat ini + `20%` langkah $-6\text{h}$ + `20%` langkah $+6\text{h}$) untuk meredam osilasi *step-to-step* pohon keputusan.
