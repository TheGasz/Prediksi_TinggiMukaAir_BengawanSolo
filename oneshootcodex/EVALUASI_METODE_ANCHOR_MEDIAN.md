# Evaluasi metode prediksi TMA `anchor-with-median`

Tanggal evaluasi: 20 Juli 2026  
Run yang dievaluasi: `handoff_20260720T063605Z_all`

## Ringkasan keputusan

Metode ini **layak sebagai candidate model**, tetapi belum cukup kuat untuk dinyatakan sebagai model final tanpa eksperimen tambahan. Desain dasarnya tepat untuk horizon panjang: model bersifat non-recursive, memakai lingkungan masa depan yang disediakan, normalisasi robust per pos, validasi rolling-origin delapan bulan, serta fitur statis dan topologi HydroRIVERS.

Backtest menunjukkan perbaikan konsisten terhadap median historis per pos-bulan-jam. Rata-rata empat fold, MAE turun dari 0,7793 menjadi 0,5513 (29,3%) dan RMSE turun dari 1,7289 menjadi 1,4326 (17,1%). Pada fold paling relevan, `jan_2025`, MAE turun 26,3% dan RMSE turun 14,7%.

Namun, ada empat risiko utama:

1. fitur anchor adalah fitur paling dominan; ablasi murah menunjukkan hasil campuran (kalah pada `sep_2024`, menang pada `jan_2025`) dan belum mencakup empat fold/full ensemble;
2. konstruksi anchor untuk baris training memakai anomali 30 hari di ujung cutoff yang ditempelkan juga ke baris training lama, sehingga tidak row-wise causal dan menciptakan train-serving mismatch;
3. pruning menghapus flag missing radiasi karena flag tersebut konstan di train, padahal 57,44% nilai radiasi pada test missing;
4. ensemble equal-weight tidak terbukti optimal dan sedikit kalah dari anggota terbaik pada tiga dari empat fold.

## 1. Data dan reproduksibilitas

Artefak mencakup 84.396 label train, 21.780 baris test, 888.480 observasi lingkungan per jam, dan 30 pos. Target train berjalan dari 1 Januari 2023 06.00 sampai 18 September 2025 18.00; test dari 19 September 2025 06.00 sampai 18 Mei 2026 18.00.

Grid target teoritis berisi 89.280 baris, sehingga terdapat 4.884 kombinasi waktu-pos tanpa label. Missing terbesar berasal dari Gunungsari (1.850), Floodway Bridge C (599), dan Bojonegoro - Kali Kethek (288). Bulan Februari 2025 sendiri menyumbang 2.240 baris missing. Pipeline benar karena tidak menginterpolasi target atau membuat pseudo-label, tetapi evaluasi per pos perlu selalu melaporkan coverage.

Reproduksibilitas run tergolong baik: konfigurasi, versi runtime, daftar fitur, matriks train/test, model final, prediksi OOF, normalisasi target, bobot upstream, dan feature importance tersedia.

## 2. Kekuatan metode

### 2.1 Formulasi horizon panjang

Tidak digunakannya lag TMA recursive adalah keputusan yang defensible untuk horizon 242 hari. Model tidak perlu melakukan rollout prediksi ke prediksi yang berpotensi mengakumulasi error. Sinyal persistensi tetap masuk melalui median/harmonic baseline dan anomali terakhir.

Catatan penggunaan: lingkungan September 2025-Mei 2026 tersedia dalam dataset kompetisi, sehingga sah untuk task ini. Untuk sistem operasional yang membuat prediksi pada 18 September 2025, variabel lingkungan aktual hingga Mei 2026 belum tersedia; deployment nyata memerlukan weather forecast/scenario dan evaluasi uncertainty yang berbeda.

### 2.2 Validasi temporal

Empat fold menggunakan cutoff dan horizon panjang. Fold `jan_2025` paling menyerupai deployment karena sama-sama memprediksi delapan bulan ke depan dan berakhir tepat pada batas train final. Ini jauh lebih representatif daripada random split atau validasi one-step-ahead.

| Fold | Model MAE | Median MAE | Perbaikan MAE | Model RMSE | Median RMSE | Perbaikan RMSE |
|---|---:|---:|---:|---:|---:|---:|
| sep_2023 | 0,6464 | 0,8660 | 25,4% | 2,5982 | 2,7209 | 4,5% |
| may_2024 | 0,4268 | 0,6573 | 35,1% | 0,6766 | 1,1612 | 41,7% |
| sep_2024 | 0,6155 | 0,8924 | 31,0% | 1,2643 | 1,6372 | 22,8% |
| jan_2025 | 0,5166 | 0,7013 | 26,3% | 1,1914 | 1,3961 | 14,7% |

Model menang atas median pada seluruh fold untuk MAE dan RMSE. Meski demikian, pada `jan_2025` worst-station MAE model adalah 1,6527, lebih buruk dari baseline 1,5120. Artinya peningkatan rata-rata masih menyembunyikan regresi pada pos tertentu.

### 2.3 Normalisasi dan loss

Median dan robust scale `(IQR / 1,349)` per pos mencegah pos berelevasi tinggi mendominasi loss. Huber loss juga masuk akal karena ditemukan spike sensor terisolasi dan RMSE jauh lebih sensitif terhadap outlier daripada MAE.

### 2.4 Lingkungan dan HydroRIVERS

Rolling feature lingkungan bersifat trailing/right-closed dan forward-fill bersifat kausal. Sentinel radiasi `-999` sudah diubah menjadi missing. Topologi sungai menggunakan `NEXT_DOWN`, sesuai dokumentasi HydroRIVERS untuk navigasi jaringan. Selisih `DIST_DN_KM` juga tepat sebagai pendekatan jarak sepanjang jaringan dari pos upstream ke downstream.

Matching spasial umumnya baik, tetapi Bengkelolor dan Jarum berjarak sekitar 1,24 km dari reach terdekat; Kali Pepe - Tugu Boto dan Peren sekitar 0,76-0,79 km. Pos-pos ini perlu audit peta manual karena nearest reach belum tentu sungai yang benar pada jaringan paralel.

## 3. Temuan kritis

### 3.1 Ablasi anchor tersedia, tetapi buktinya belum konklusif

Tiga fitur anchor (`horizon_days`, `recent_anomaly_mdpl`, `anchor_basis_mdpl`) berada di matriks final. Rata-rata importance lintas lima model menempatkan:

- `horizon_days`: peringkat 1 (7,10);
- `anchor_basis_mdpl`: peringkat 3 (6,53);
- `recent_anomaly_mdpl`: peringkat 5 (4,29).

File `ablation_results (3).csv` di workspace root memang mengisolasi `anchor_on` dan `anchor_off`. Pengujiannya memakai satu member cepat dan dua fold terbaru, dengan 102 fitur saat anchor aktif dan 99 fitur saat anchor nonaktif.

| Fold | Metrik | Anchor ON | Anchor OFF | Dampak anchor |
|---|---|---:|---:|---:|
| sep_2024 | MAE | 0,6274 | 0,6134 | 2,28% lebih buruk |
| sep_2024 | RMSE | 1,2705 | 1,2330 | 3,04% lebih buruk |
| sep_2024 | worst-station MAE | 1,4301 | 1,3106 | 9,12% lebih buruk |
| jan_2025 | MAE | 0,5103 | 0,5484 | 6,96% lebih baik |
| jan_2025 | RMSE | 1,1918 | 1,2049 | 1,09% lebih baik |
| jan_2025 | worst-station MAE | 1,7256 | 1,8761 | 8,02% lebih baik |

Jadi anchor menang pada fold paling relevan, `jan_2025`, tetapi kalah pada `sep_2024`. Kesimpulannya bukan “anchor tidak terbukti sama sekali”, melainkan **efek anchor bergantung forecast origin** dan bukti dua-fold/satu-member belum memenuhi kriteria menang minimal tiga dari empat fold. Ablasi ini juga memakai full feature set 102/99, sedangkan model final memakai 87 fitur hasil pruning; interaksi anchor dengan konfigurasi final belum diuji secara murni.

### 3.2 Konstruksi anchor mengalami train-serving mismatch

Untuk setiap fold, kode menghitung satu `recent_anomaly` dari 30 hari terakhir sebelum cutoff, lalu menempelkan nilai tersebut ke **semua** baris fit. Pada baris lama, fitur itu berisi informasi target yang terjadi setelah timestamp baris tersebut. Target validasi tidak bocor, jadi skor fold tidak langsung membaca label validasi; tetapi training bukan lagi row-wise causal.

Masalah kedua, `horizon_days` pada fit selalu kurang dari atau sama dengan nol, sedangkan pada valid/test selalu positif. Tree tidak dapat mengekstrapolasi hubungan kontinu di luar rentang train. `anchor_basis_mdpl` memakai nilai absolut horizon, tetapi relasi yang dipelajari pada fit adalah hubungan mundur dari cutoff, bukan evolusi anomali ke masa depan.

Perbaikan yang disarankan adalah membuat pseudo-origin training. Untuk setiap pos dan beberapa cutoff historis, hitung recent anomaly hanya dari 30 hari sebelum cutoff tersebut, kemudian bentuk contoh target pada horizon +1 sampai +242 hari. Dengan begitu, distribusi `horizon_days` train sama dengan valid/test dan seluruh fitur target-derived tetap causal terhadap forecast origin.

### 3.3 Risiko missing radiasi

`solar_radiation_mj_m2` tetap menjadi fitur model, tetapi `solar_radiation_missing` dibuang oleh pruning. Pada train, missing radiasi 0%; pada test, 57,44%. Feature importance tidak mungkin menilai flag yang konstan di train, sehingga keputusan membuangnya bukan bukti bahwa missingness aman.

Eksperimen wajib:

1. drop nilai radiasi dan flag sekaligus;
2. imputasi climatology pos-bulan-jam + flag missing;
3. masked-feature augmentation: sembunyikan blok radiasi pada train dengan pola/durasi serupa test, lalu pertahankan flag.

Pilih berdasarkan rolling-origin RMSE dan robustness pada slice dengan radiasi sengaja dimasking, bukan berdasarkan importance train.

### 3.4 Equal-weight ensemble belum efisien

Rata-rata RMSE anggota model menunjukkan `lgbm_b` terbaik (1,4262), diikuti `lgbm_a` (1,4348), `cb_shallow` (1,4566), `cb_deep` (1,4620), dan `cb_base` (1,4751). Ensemble hanya mengalahkan anggota terbaik sebesar 0,00006 RMSE pada `sep_2024`; pada tiga fold lain ensemble lebih buruk 0,0069-0,0130.

Simpan prediksi OOF setiap member, hitung korelasi residual, lalu pelajari bobot non-negatif yang jumlahnya satu. Bobot harus dipelajari pada fold terdahulu dan dinilai pada fold berikutnya agar tidak overfit. Kandidat sederhana yang layak diuji: bobot lebih besar untuk dua LightGBM dan `cb_shallow`, atau median ensemble sebagai alternatif yang robust.

### 3.5 Pruning dan pemilihan model memakai fold yang sama

Feature importance dari run validasi sebelumnya dipakai untuk memilih 85% cumulative importance, lalu konfigurasi hasil pruning dinilai kembali pada empat fold yang sama. Ini merupakan selection bias ringan. Gunakan nested rolling evaluation atau tetapkan `jan_2025` sebagai untouched final model-selection fold setelah keputusan fitur dibuat memakai fold sebelumnya.

### 3.6 File OOF berisi forecast yang overlap

`oof_predictions.parquet` memiliki 81.102 baris tetapi hanya 61.954 key unik. Sebanyak 19.148 key muncul dua kali karena fold saling overlap; 47,22% baris berada pada key yang diduplikasi. Ini bukan leakage, karena setiap prediksi berasal dari forecast origin berbeda, tetapi file tersebut bukan OOF satu-prediksi-per-row.

`residual_slices.csv` saat ini menggabungkan semua baris tersebut, sehingga periode overlap mendapat bobot lebih besar. Tambahkan `forecast_origin`, `horizon_day`, dan dua ringkasan terpisah: per-origin metrics serta deduplicated metrics dengan aturan origin terdekat atau terjauh.

### 3.7 Topologi upstream belum memakai travel time

Bobot upstream menggabungkan hujan pada timestamp yang sama dengan peluruhan jarak 75 km. Untuk respon sungai, kontribusi upstream semestinya bergeser menurut travel time. Saat ini 17 dari 30 pos hanya memiliki self-weight, sedangkan Karanggeneng memiliki 25 sumber dan Babat 23; normalisasi dapat mendilusi event lokal pada node hilir.

Tambahkan fitur upstream dengan lag kandidat 6, 12, 24, dan 48 jam, serta travel-time proxy dari selisih `DIST_DN_KM` dengan beberapa asumsi kecepatan aliran. Ablasi fitur graf harus dibandingkan dengan model local-environment-only. Jika graf tidak konsisten memperbaiki `sep_2024` dan `jan_2025`, jangan mempertahankannya hanya karena masuk akal secara fisik.

## 4. Evaluasi retrospektif API

`API_TEST/api_manual_reference_clean.csv` berisi observasi pada periode submission. Karena horizon telah berlalu, file ini berguna sebagai retrospective holdout. File ini **tidak boleh** dipakai untuk fit, memilih bobot blend, mengoreksi bias per pos, atau tuning hyperparameter apabila submission dinilai seolah dibuat pada 18 September 2025.

Evaluator yang tersedia melaporkan coverage 100%, API RMSE 0,8569, MAE 0,5315, dan bias -0,0450. Estimasi skor kompetisi 1,5991 berasal dari regresi hanya tujuh submission, sehingga harus dianggap indikator kasar, bukan ground truth kompetisi.

Error membesar terhadap horizon: RMSE sekitar 0,39 pada sisa September, 0,90 pada Januari-Februari, dan mencapai sekitar 1,01 pada Maret. Pos retrospektif terlemah adalah Babat, Wonogiri Dam, Floodway Bridge C, Bojonegoro - Kali Kethek, dan Kedungupit. Babat memiliki bias positif besar, sedangkan beberapa pos lain cenderung underpredict.

Pola ini mendukung dua prioritas: pseudo-origin horizon training untuk mengurangi degradation terhadap lead time, dan calibration per-pos yang dipelajari hanya dari OOF historis.

## 5. Rencana eksperimen prioritas

Urutan berikut memaksimalkan informasi per biaya komputasi.

| Prioritas | Eksperimen | Tujuan | Kriteria lolos |
|---:|---|---|---|
| P0 | Konfirmasi anchor on/off empat fold/full ensemble pada shared pruned feature set | Melanjutkan ablasi dua-fold/satu-member yang hasilnya 1 menang dan 1 kalah | Menang pada minimal 3/4 fold dan pada `jan_2025`; RMSE, MAE, dan worst-station tidak memburuk material |
| P0 | Pseudo-origin anchor | Menghapus train-serving mismatch | Menurunkan RMSE `jan_2025` dan slope error terhadap horizon |
| P0 | Audit radiasi missing | Menguji robustness terhadap 57,44% missing test | Stabil pada masked validation dan tidak merusak fold normal |
| P1 | Simpan member OOF + constrained blend | Mengganti equal weight | Bobot dipilih walk-forward; menang terhadap equal-weight pada fold sesudahnya |
| P1 | Bias calibration per pos | Mengurangi bias persisten Babat/Wonogiri/dll. | Shrinkage OOF, tanpa API target; macro-MAE dan worst-station membaik |
| P1 | Upstream lag/travel time | Memakai topologi secara hidrologis | Ablasi menang konsisten pada pos downstream dan dua fold terbaru |
| P2 | Model residual terhadap harmonic/median | Memisahkan level musiman dan respons cuaca | Lebih stabil antar-pos dan antar-horizon daripada direct normalized target |
| P2 | Quantile model P10/P50/P90 | Mengukur uncertainty horizon panjang | Coverage interval dinilai per horizon dan pos, bukan hanya pinball loss global |

## 6. Desain validasi yang direkomendasikan

1. Pertahankan empat forecast origin delapan bulan.
2. Untuk keputusan fitur/hyperparameter, gunakan fold sampai `sep_2024`.
3. Bekukan konfigurasi lalu gunakan `jan_2025` satu kali sebagai final selection check.
4. Laporkan MAE, RMSE, macro-station MAE, worst-station MAE, p95 absolute error, dan bias.
5. Tambahkan metrik per horizon bucket: 0-30, 31-90, 91-180, dan 181-242 hari.
6. Tambahkan slice hujan ekstrem, TMA tinggi, pos hilir, match sungai berjarak >500 m, dan periode radiasi missing.
7. Bandingkan selalu dengan: station median, station-month-hour median, harmonic, year-over-year, serta harmonic + causal recent anomaly.
8. Jangan gunakan API periode target sampai semua keputusan model dibekukan; setelah itu pakai hanya untuk laporan generalisasi retrospektif.

## 7. Rekomendasi final

Pertahankan ensemble exogenous global sebagai fondasi. Perubahan pertama bukan menambah arsitektur yang lebih kompleks, melainkan membenahi anchor menjadi pseudo-origin causal, menguji missing radiasi secara eksplisit, dan mengganti equal weighting dengan blend berbasis member OOF. Setelah tiga hal ini lolos validasi, tambahkan travel-time upstream dan calibration per pos dengan shrinkage.

Status saat ini: **promising, but conditional**. Model sudah jelas lebih baik dari baseline median pada backtest. Ablasi awal menunjukkan anchor membantu forecast origin `jan_2025` tetapi merugikan `sep_2024`, sehingga manfaatnya belum stabil lintas origin dan belum menjamin robustness terhadap missing regime serta error yang meningkat sepanjang horizon.
