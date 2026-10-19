# PRD: Sistem Prediksi Probabilitas Pertandingan Sepak Bola (Machine Learning)

| Item | Detail |
|---|---|
| Nama proyek | Football Match Probability Predictor (kode: `FMPP`) |
| Versi dokumen | 1.0 (Draft) |
| Tanggal | 14 Oktober 2026 |
| Peran penulis | Project Manager IT |
| Pemilik proyek | Faris |
| Status | Draft, menunggu review pemilik proyek |

> Tanda **[ASUMSI]** artinya poin itu belum dikonfirmasi dan diisi pakai perkiraan wajar. Bisa dikoreksi kapan saja.

---

## 1. Ringkasan Eksekutif

FMPP adalah sistem machine learning yang memprediksi **probabilitas** hasil pertandingan sepak bola yang akan datang, bukan menebak satu skor pasti. Untuk setiap laga, sistem mengeluarkan:

1. Probabilitas **Menang / Seri / Kalah** (1X2).
2. Probabilitas **jumlah gol besar atau kecil** (Over/Under), dengan acuan utama 2.5 dan garis lain yang ikut dihitung.
3. Probabilitas **BTTS** (Both Teams To Score, kedua tim mencetak gol).
4. **Ekspektasi gol** tiap tim (misal Tim A 1.8, Tim B 0.9).

Sistem bisa dipakai lewat **CLI** dan **antarmuka web**. User tinggal memilih laga yang main **malam ini, besok, atau lusa**. Prediksi dibuat dari performa **5 sampai 10 laga terakhir** tiap tim.

## 2. Latar Belakang dan Masalah

- Prediksi sepak bola biasanya cuma berupa tebakan atau opini, tanpa angka probabilitas yang bisa diuji.
- Proyek ini jadi sarana **analisis data dan machine learning** dari dataset nyata: mengumpulkan data, membersihkan, membuat fitur, melatih model, dan mengevaluasi hasilnya secara ilmiah.
- Tujuan proyek: **analisis dan pembelajaran**, bukan layanan taruhan. Lihat bagian Disclaimer.

## 3. Tujuan dan Metrik Keberhasilan

### 3.1 Tujuan
| ID | Tujuan |
|---|---|
| G1 | Menghasilkan probabilitas 1X2 yang **terkalibrasi** (angka 60% benar-benar terjadi sekitar 60% dari waktu) |
| G2 | Menghasilkan probabilitas Over/Under dan BTTS dari model jumlah gol |
| G3 | Menyediakan CLI dan web UI untuk laga hari ini, besok, dan lusa |
| G4 | Pipeline data yang bisa dijalankan ulang (reproducible) dan terdokumentasi |
| G5 | Hasil evaluasi yang jujur, dibandingkan dengan baseline sederhana |

### 3.2 Metrik (target awal **[ASUMSI]**, disesuaikan setelah data nyata terlihat)
| Metrik | Dipakai untuk | Target |
|---|---|---|
| Log Loss | 1X2, Over/Under, BTTS | Lebih baik dari baseline "tebak frekuensi historis" |
| Brier Score | Semua probabilitas | Lebih baik dari baseline |
| RPS (Ranked Probability Score) | 1X2 | Lebih baik dari baseline; mendekati odds bandar bila datanya ada |
| Accuracy 1X2 | Pembanding kasar | Sekitar 50 sampai 55% di liga top (sepak bola memang sulit diprediksi) |
| Accuracy Over/Under 2.5 dan BTTS | Pembanding kasar | Sekitar 55 sampai 62% |
| Calibration curve | Semua probabilitas | Kurva mendekati diagonal |
| MAE gol per tim | Ekspektasi gol | Lebih baik dari rata-rata liga |

Catatan: akurasi bukan metrik utama. Karena outputnya probabilitas, yang dinilai adalah **kualitas probabilitasnya** (Log Loss, Brier, RPS, kalibrasi).

## 4. Ruang Lingkup (Scope)

### 4.1 In Scope
- Cabang olahraga: **sepak bola** saja.
- Kompetisi (9):
  1. UEFA Champions League (UCL)
  2. AFC Champions League (ACL)
  3. Premier League (Inggris)
  4. Liga 1 (Indonesia)
  5. Serie A (Italia)
  6. La Liga (Spanyol)
  7. Bundesliga (Jerman)
  8. Ligue 1 (Prancis)
  9. Eredivisie (Belanda)
- Prediksi: 1X2, Over/Under (garis 2.5 sebagai acuan + garis lain), BTTS, ekspektasi gol per tim.
- Fitur dari **5 sampai 10 laga terakhir** tiap tim.
- Antarmuka: CLI dan web.
- Pilihan laga: malam ini, besok, lusa.

### 4.2 Out of Scope (versi 1.0)
- Prediksi live/in-play.
- Prediksi skor pasti sebagai output utama (hanya ditampilkan sebagai distribusi/skor paling mungkin).
- Data cedera, skuad, dan lineup real-time (masuk backlog).
- Fitur taruhan, saran bet, atau perhitungan taruhan.
- Aplikasi mobile native.
- Cabang olahraga selain sepak bola.

## 5. Pengguna dan User Stories

**Persona utama:** Faris, mahasiswa Informatika yang membangun proyek analisis data dan ML.

| ID | User story |
|---|---|
| US1 | Sebagai user, saya ingin melihat daftar laga yang main malam ini/besok/lusa supaya bisa memilih satu laga. |
| US2 | Sebagai user, saya ingin melihat probabilitas menang/seri/kalah kedua tim. |
| US3 | Sebagai user, saya ingin melihat probabilitas Over/Under dan BTTS. |
| US4 | Sebagai user, saya ingin tahu **kenapa** model memprediksi begitu (form, rata-rata gol, kekuatan tim). |
| US5 | Sebagai user, saya ingin menjalankan semuanya lewat CLI untuk otomatisasi dan scripting. |
| US6 | Sebagai user, saya ingin melihat seberapa akurat model ini di masa lalu (backtest). |
| US7 | Sebagai user, saya ingin mengatur jumlah laga terakhir (N = 5 sampai 10) yang dipakai. |

## 6. Kebutuhan Fungsional

### 6.1 Data
| ID | Kebutuhan |
|---|---|
| FR-D1 | Mengambil data historis hasil laga dari sumber data (lihat bagian 7). |
| FR-D2 | Mengambil jadwal laga mendatang (hari ini, besok, lusa) untuk 9 kompetisi. |
| FR-D3 | Menyimpan data ke database lokal (SQLite) dan file mentah (CSV) supaya tidak bergantung koneksi terus-menerus. |
| FR-D4 | Menormalisasi nama tim antar sumber data (contoh: "Man United" = "Manchester United"). |
| FR-D5 | Menangani zona waktu: jadwal ditampilkan dalam **WIB**. |
| FR-D6 | Data tim ditarik dari **semua kompetisi**, bukan hanya satu liga. Contoh: form Real Madrid di UCL ikut dihitung bersama La Liga. |

### 6.2 Feature Engineering
Untuk setiap tim sebelum laga, dihitung dari N laga terakhir (N = 5 sampai 10, default 5 **[ASUMSI]**):

| Kelompok | Fitur |
|---|---|
| Form | Poin per laga, jumlah menang/seri/kalah, streak |
| Gol | Rata-rata gol dicetak, rata-rata kebobolan, selisih gol |
| Over/BTTS historis | Persentase laga Over 2.5, persentase BTTS, persentase clean sheet, persentase gagal mencetak gol |
| Konteks | Performa kandang vs tandang, hari istirahat sejak laga terakhir |
| Kekuatan tim | Rating **Elo** yang diperbarui tiap laga |
| Lanjutan (jika data tersedia) | Tembakan, tembakan on target, xG |
| Head-to-head | Opsional: hasil pertemuan terakhir |

Aturan wajib: fitur hanya boleh memakai data **sebelum** tanggal laga (cegah data leakage).

### 6.3 Model Machine Learning
Pendekatan dua jalur, hasilnya digabung konsisten.

**Jalur A, model jumlah gol (inti Over/Under dan BTTS)**
1. Prediksi ekspektasi gol tim kandang (λ_home) dan tim tandang (λ_away) memakai **Poisson Regression** atau gradient boosting dengan objective Poisson.
2. Bangun **matriks skor** P(gol_home = i, gol_away = j) dari distribusi Poisson. Opsional: koreksi **Dixon-Coles** untuk skor rendah (0-0, 1-0, 0-1, 1-1).
3. Dari matriks skor diturunkan:
   - P(Menang), P(Seri), P(Kalah)
   - P(Over x.5) dan P(Under x.5) untuk beberapa garis
   - P(BTTS Ya) = P(home ≥ 1 dan away ≥ 1)
   - Total gol yang diharapkan = λ_home + λ_away

**Jalur B, model klasifikasi 1X2 langsung (pembanding dan ensemble)**
- Baseline: Logistic Regression multinomial.
- Pembanding: XGBoost atau LightGBM, lalu dikalibrasi (Platt atau isotonic).
- Opsi akhir: ensemble rata-rata terbobot Jalur A dan B, bobot ditentukan dari hasil validasi.

**Aturan garis Over/Under (sesuai permintaan):**
- **2.5 selalu ditampilkan** sebagai acuan utama.
- Garis lain (1.5, 3.5, 4.5) juga dihitung dari matriks skor.
- Sistem juga menampilkan **"garis seimbang model"**, yaitu garis di mana probabilitas Over mendekati 50%. Jadi kalau analisis model condong ke laga banyak gol, outputnya bisa menunjukkan misalnya "Over 3.5 masih 52%", tidak terkunci di 2.5.

Model baseline pembanding (wajib ada): frekuensi historis liga, dan model "rata-rata gol tim saja".

### 6.4 Output Prediksi
Tiap prediksi menampilkan:
- Nama laga, kompetisi, waktu (WIB).
- Probabilitas Menang/Seri/Kalah dalam persen.
- Probabilitas Over/Under untuk 1.5, 2.5, 3.5, 4.5, plus garis seimbang model.
- Probabilitas BTTS Ya/Tidak.
- Ekspektasi gol per tim dan total.
- 3 skor paling mungkin beserta probabilitasnya.
- Ringkasan alasan (fitur paling berpengaruh) dalam bahasa sederhana.
- Indikator **tingkat keyakinan data**: peringatan jika tim punya kurang dari N laga historis.

Contoh tampilan CLI:
```
Arsenal vs Chelsea | Premier League | 15 Okt 2026 02:00 WIB
-------------------------------------------------------
Hasil     : Arsenal 48.2% | Seri 27.1% | Chelsea 24.7%
Gol       : xG model Arsenal 1.72 | Chelsea 1.15 | Total 2.87
Over/Under: O1.5 82.0% | O2.5 56.3% | O3.5 31.4% | O4.5 14.8%
Garis seimbang model: 2.5 (Over 56.3%)
BTTS      : Ya 58.9% | Tidak 41.1%
Skor top  : 1-1 (11.9%) | 2-1 (10.4%) | 1-0 (9.3%)
Data      : 8 laga terakhir per tim | keyakinan data: Tinggi
```
(Angka di atas hanya contoh format, bukan hasil model.)

### 6.5 Antarmuka CLI
| Perintah (rencana) | Fungsi |
|---|---|
| `fmpp update-data` | Perbarui data historis dan jadwal |
| `fmpp matches --day today\|tomorrow\|dayafter [--league <kode>]` | Daftar laga |
| `fmpp predict --match-id <id>` | Prediksi satu laga |
| `fmpp predict --home "Arsenal" --away "Chelsea" --last-n 8` | Prediksi manual dengan N laga terakhir |
| `fmpp predict-day --day today` | Prediksi semua laga di hari itu |
| `fmpp train` | Latih ulang model |
| `fmpp evaluate` | Jalankan backtest dan tampilkan metrik |
| `--format table\|json\|csv` | Format output |

### 6.6 Antarmuka Web
- Halaman utama: tab **Malam Ini / Besok / Lusa**, filter kompetisi.
- Kartu laga: tim, jam WIB, ringkasan probabilitas.
- Halaman detail laga: bar probabilitas 1X2, tabel Over/Under, BTTS, heatmap matriks skor, form 5 sampai 10 laga terakhir, penjelasan singkat.
- Halaman **Performa Model**: metrik backtest dan kurva kalibrasi per kompetisi.
- Pilihan N laga terakhir (5 sampai 10).
- Tampilan responsif (HP dan laptop), bahasa Indonesia.
- Disclaimer selalu tampil di footer.

## 7. Rekomendasi Sumber Data

> Penting: cakupan liga, batas request gratis, dan syarat penggunaan bisa berubah. **Verifikasi di Fase 0 sebelum membangun apa pun.**

| Sumber | Cocok untuk | Catatan |
|---|---|---|
| **football-data.co.uk** (CSV gratis) | Data historis Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Eredivisie | Berisi hasil, statistik tembakan, kartu, dan odds penutupan. Sangat cocok untuk **training dan benchmark**. Tidak mencakup UCL, ACL, dan Liga 1 Indonesia. |
| **football-data.org** (API, ada tier gratis) | Jadwal dan hasil beberapa kompetisi top Eropa termasuk UCL | Ada batas request per menit di tier gratis. Cocok untuk jadwal laga mendatang. |
| **API-Football (api-sports.io)** | Cakupan luas: Liga 1 Indonesia, ACL, dan liga lain; jadwal, hasil, statistik | Tier gratis dibatasi per hari. Cek apakah 9 kompetisi tercakup dan seberapa dalam datanya. |
| **Kaggle** (dataset publik) | Data historis tambahan atau cadangan | Cek lisensi dan kelengkapan. Berguna untuk Liga 1 dan ACL jika API terbatas. |
| **Understat / FBref** | xG dan statistik lanjutan | Hanya untuk fitur opsional. Cek kebijakan scraping dan ketersediaan sebelum dipakai. |

**Rekomendasi arsitektur data (kombinasi):**
1. **Training top-5 liga + Eredivisie:** football-data.co.uk (gratis, bersih, ada odds untuk benchmark).
2. **Jadwal mendatang + UCL/ACL/Liga 1:** football-data.org dan/atau API-Football.
3. **Cadangan:** Kaggle untuk Liga 1 dan ACL bila API tidak cukup.

**Risiko data yang perlu disadari:** Liga 1 Indonesia dan ACL biasanya datanya lebih sedikit dan kurang detail dibanding liga top Eropa, sehingga model untuk keduanya kemungkinan kurang akurat. Sistem harus menampilkan peringatan jelas untuk kompetisi dengan data terbatas.

## 8. Kebutuhan Non-Fungsional

| ID | Kebutuhan |
|---|---|
| NFR-1 | Prediksi satu laga di CLI selesai dalam kurang dari 3 detik (data sudah ter-cache). |
| NFR-2 | Pipeline reproducible: random seed tetap, versi library dikunci (`requirements.txt`). |
| NFR-3 | Validasi **berbasis waktu** (train di masa lalu, tes di masa depan), bukan random split. |
| NFR-4 | Kode modular: `data/`, `features/`, `models/`, `cli/`, `web/`. |
| NFR-5 | Logging dan penanganan error yang jelas saat API gagal atau data kurang. |
| NFR-6 | Kunci API disimpan di file `.env`, tidak di-commit ke Git. |
| NFR-7 | Mematuhi syarat penggunaan masing-masing sumber data. |

## 9. Arsitektur Teknis (Usulan)

```
[Sumber Data] -> [Data Ingestion] -> [SQLite + CSV]
                                          |
                                   [Feature Engineering]
                                          |
                          [Model Gol (Poisson)] + [Model 1X2]
                                          |
                                   [Prediction Engine]
                                    /              \
                              [CLI (Typer)]   [API (FastAPI)] -> [Web UI (HTML/JS)]
```

| Lapisan | Teknologi |
|---|---|
| Bahasa | Python 3.11+ |
| Data | pandas, numpy, requests, SQLite |
| ML | scikit-learn, statsmodels/scipy, XGBoost atau LightGBM |
| Evaluasi | scikit-learn metrics, matplotlib/seaborn |
| CLI | Typer (atau argparse) |
| Backend web | FastAPI |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Eksperimen | Jupyter Notebook |
| Versi kode | Git |

## 10. Struktur Folder (Usulan)

```
fmpp/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/        # EDA dan eksperimen
├── src/
│   ├── ingestion/
│   ├── features/
│   ├── models/
│   ├── prediction/
│   ├── cli/
│   └── api/
├── web/              # HTML, CSS, JS
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## 11. Rencana Evaluasi

1. **Walk-forward backtesting:** latih dengan data sampai musim/bulan X, uji ke periode berikutnya, geser maju, ulangi.
2. Bandingkan model dengan: (a) baseline frekuensi liga, (b) baseline rata-rata gol, (c) odds penutupan bandar (benchmark, bila tersedia).
3. Laporkan metrik **per kompetisi**, karena kualitas data tiap liga berbeda.
4. Analisis kalibrasi dan analisis kesalahan (laga seperti apa yang sering meleset).
5. Uji sensitivitas: bandingkan hasil dengan N = 5, 7, 10 laga terakhir.

## 12. Rencana Pengerjaan dan Timeline

**[ASUMSI]**: dikerjakan 1 orang, sekitar 8 minggu. Jika tenggat atau jumlah anggota berbeda, timeline disesuaikan.

| Fase | Minggu | Isi | Deliverable |
|---|---|---|---|
| 0. Riset data | 1 | Verifikasi cakupan 9 kompetisi, batas API, lisensi | Daftar sumber data final |
| 1. Data pipeline | 2 | Ingestion, SQLite, normalisasi nama tim | Dataset bersih + skrip update |
| 2. EDA dan fitur | 3 | Eksplorasi, feature engineering, Elo | Notebook EDA, tabel fitur |
| 3. Modeling | 4 sampai 5 | Poisson, 1X2, kalibrasi, ensemble | Model terlatih + evaluasi awal |
| 4. CLI | 6 | Semua perintah CLI | CLI berfungsi |
| 5. Web UI | 7 | FastAPI + halaman web | Web berjalan lokal |
| 6. Pengujian dan dokumentasi | 8 | Backtest akhir, laporan, README | Laporan analisis + dokumentasi |

## 13. Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Data Liga 1/ACL minim | Model kurang akurat | Peringatan keyakinan data; Kaggle sebagai cadangan; model khusus atau rating Elo saja |
| Batas request API gratis | Jadwal/data terlambat | Cache lokal, update terjadwal, kombinasi beberapa sumber |
| Data leakage | Evaluasi menyesatkan | Fitur hanya dari data sebelum laga; split berbasis waktu |
| Sepak bola sangat acak | Akurasi mentok | Fokus pada kalibrasi probabilitas, bukan tebakan benar/salah |
| 5 sampai 10 laga terlalu sedikit | Fitur berisik | Tambahkan Elo (memori panjang) dan uji sensitivitas N |
| Beda nama tim antar sumber | Data tergabung salah | Tabel mapping nama tim + validasi manual |
| Tim baru promosi / awal musim | Data minim | Fallback ke musim sebelumnya, tampilkan peringatan |
| Cedera/rotasi pemain tidak terdata | Prediksi meleset | Dicatat sebagai keterbatasan; masuk backlog v2 |
| Syarat penggunaan data | Masalah legal | Baca ToS tiap sumber; gunakan untuk tujuan akademik |

## 14. Asumsi

1. **[ASUMSI]** Dikerjakan solo, sekitar 8 minggu.
2. **[ASUMSI]** Default N = 5 laga terakhir, bisa diubah ke 10.
3. **[ASUMSI]** Prediksi dijalankan lokal (laptop), belum di-deploy ke internet.
4. **[ASUMSI]** Update data dijalankan manual lewat CLI, belum otomatis terjadwal.
5. **[ASUMSI]** Target metrik di bagian 3.2 bersifat awal dan akan disesuaikan setelah melihat data.

## 15. Pertanyaan Terbuka

1. Berapa tenggat proyek dan berapa orang yang mengerjakan? (menentukan timeline)
2. Apakah web akan hanya dijalankan lokal, atau perlu di-deploy online?
3. Apakah perlu data lanjutan seperti xG, atau cukup hasil dan skor?
4. Apakah ada format laporan akhir yang diwajibkan (jika proyek ini untuk tugas/dosen)?

## 16. Backlog Versi Berikutnya (v2)

- Data cedera, skuad, dan lineup.
- Pembaruan data otomatis terjadwal.
- Model khusus per liga.
- Prediksi in-play.
- Deploy online.
- Notifikasi laga baru.

## 17. Disclaimer

Sistem ini adalah alat **analisis dan pembelajaran**. Probabilitas yang dihasilkan adalah perkiraan statistik, bukan jaminan hasil. Hasil pertandingan sepak bola dipengaruhi banyak faktor yang tidak tercakup model. Sistem ini tidak dirancang dan tidak boleh digunakan sebagai dasar keputusan perjudian.
