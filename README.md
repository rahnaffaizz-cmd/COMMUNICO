# Gesture Voice — Pengenal Huruf BISINDO Bersuara

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-00A67E?logo=google&logoColor=white)
![Lisensi](https://img.shields.io/badge/Lisensi-MIT-blue)

Program yang membaca huruf **BISINDO** (Bahasa Isyarat Indonesia) lewat kamera,
membacakan hasilnya dengan suara, dan menjadi jembatan percakapan dua arah antara
penyandang tuli dan lawan bicara yang tidak tuli.

```
kamera ──▶ MediaPipe cari titik tangan ──▶ 126 angka koordinat
       ──▶ model dua cabang menebak huruf A-Z (satu tangan / dua tangan)
       ──▶ disaring lewat voting ──▶ huruf ditumpuk jadi kata
       ──▶ dibunyikan pakai Google TTS (bahasa Indonesia)

mikrofon ──▶ Whisper bahasa Indonesia (CPU saja) ──▶ teks di percakapan
```

Dibuat untuk membantu komunikasi sehari-hari yang singkat: mengeja nama, nama
jalan, atau nama obat, ketika juru bahasa isyarat tidak tersedia.

## Yang bisa dilakukan sekarang

- **Membaca huruf A sampai Z** lewat kamera, baik huruf satu tangan maupun dua
  tangan. Model `rf_bisindo_v7.pkl` punya dua cabang: kalau satu tangan terlihat
  hanya C, E, I, J, L, O, R, U, V, Z yang bisa dijawab, kalau dua tangan terlihat
  hanya 16 huruf sisanya. Diukur per foto asli pada dataset publik, pada sampel
  yang mengikuti aturan itu, ketepatannya 86,4% (model satu cabang sebelumnya
  71,2%). Lihat `VERSI_MODEL.md`.
- **Percakapan dua sisi** (`.\run.ps1 -Percakapan`): kiri untuk Bahasa Isyarat
  (kamera dan teks hasil bacaan), kanan untuk lawan bicara yang mengetik **atau
  berbicara**. Tiap baris diberi tanda siapa yang bicara.
- **Suara ke teks bahasa Indonesia** yang jalan di CPU saja, tanpa kartu grafis
  (WER 13,9% pada ucapan bacaan nyata, lihat bagian percakapan di bawah).
- **Disetel dengan tangan dan kamera Anda sendiri**: peragakan satu huruf dengan
  `--ukur`, masukkan hasilnya jadi data latih dengan `impor_ukur.py`, lalu latih
  ulang dengan `latih_cabang.py`.
- **Tahan terhadap hal-hal yang biasa mengganggu**: urutan tangan dari MediaPipe
  yang berubah-ubah (tangan disusun kiri ke kanan dulu), satu tangan yang hilang
  sesaat dari deteksi, dan suara pembacaan yang tertangkap mikrofon.

---

## Isi folder

| File | Gunanya |
|---|---|
| `gesture_voice.py` | program utama, pakai kamera |
| `percakapan.py` | percakapan dua sisi: isyarat di kiri, lawan bicara mengetik atau berbicara di kanan |
| `suara_ke_teks.py` | suara ke teks bahasa Indonesia (Whisper di CPU), dipakai `percakapan.py` |
| `siapkan_model_suara.py` | menyiapkan model suara bahasa Indonesia, sekali per komputer |
| `uji_gambar.py` | uji model pakai file gambar, tanpa kamera |
| `cabang.py` | model dua cabang (satu tangan / dua tangan) yang dipakai v7 |
| `latih_cabang.py` | melatih model dua cabang dan menulis ambang hurufnya |
| `optimasi.py` | mengoptimalkan pembacaan huruf otomatis: ukur, impor, latih ulang, bandingkan |
| `rekam.py` | merekam peragaan huruf Anda sendiri sebagai data latih |
| `impor_ukur.py` | mengubah hasil `--ukur` jadi data latih berlabel |
| `lihat_huruf.py` | menggambar bentuk tangan tiap huruf dari dataset |
| `pengukur.py` | mengukur seberapa baik satu huruf terbaca (`--ukur`) |
| `fitur.py` | mengubah titik tangan jadi angka yang dipelajari model |
| `catatan.py` | mencatat jalannya program di tiap tahap |
| `diagnosa.py` | memeriksa tiap huruf dan mencari ambang keyakinan terbaiknya |
| `ambang_huruf.json` | ambang keyakinan per huruf, dipakai otomatis |
| `AKURASI.md` | catatan ketepatan tiap huruf beserta cara mengukurnya |
| `VERSI_MODEL.md` | daftar versi model dan cara kembali ke versi lama |
| `putar_ulang.py` | bandingkan beberapa model pada rekaman sesi yang sama |
| `latih.py` | melatih ulang model dan membandingkannya dengan yang lama |
| `evaluasi.py` | ukur ketepatan model secara jujur, jadi tolok ukur perbaikan |
| `rf_bisindo_v7.pkl` | **model yang dipakai sekarang**, dua cabang (7 MB) |
| `ambang_rf_bisindo_v7.json` | ambang keyakinan per huruf khusus v7, dipakai otomatis |
| `rf_bisindo_v2.pkl` ... `v6` | versi sebelumnya, disimpan untuk pembanding (lihat `VERSI_MODEL.md`) |
| `rf_bisindo_99.pkl` | model paling awal, disimpan untuk pembanding (30 MB) |
| `contoh/` | 4 foto isyarat untuk bahan uji coba |
| `run.ps1` / `run.bat` | tinggal dijalankan, tidak perlu ketik path Python |
| `setup.ps1` | pemasangan otomatis di komputer baru |
| `requirements.txt` | daftar library yang dibutuhkan |
| `.vscode/` | konfigurasi VS Code, sudah siap pakai |

---

## Memasang di komputer baru

Butuh **Python 3.11 atau 3.12**. MediaPipe belum menyediakan versi untuk
Python 3.13 ke atas, jadi versi yang lebih baru tidak akan bisa dipakai.
Kalau di komputer itu belum ada Python sama sekali, pasang `uv` saja
(`winget install astral-sh.uv`) — nanti Python 3.12-nya diunduh otomatis.

```powershell
git clone https://github.com/IOH-AIRAN-Laboratory/gesture-voice.git
cd gesture-voice
.\setup.ps1
```

`setup.ps1` mengerjakan semuanya sendiri: mencari Python yang cocok, membuat
`.venv`, memasang seluruh library, lalu menguji hasilnya pakai gambar contoh.
Sekitar 600 MB yang diunduh, jadi siapkan koneksi yang lumayan.

Kalau PowerShell menolak menjalankan skrip, jalankan dulu satu baris ini:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

> **Catatan penting soal versi.** `protobuf` wajib versi **4.25.3**. Mulai
> versi 4.25.4, MediaPipe 0.10.14 langsung error dengan pesan
> `'FieldDescriptor' object has no attribute 'label'`. Versi ini sudah dikunci
> di `requirements.txt`, jangan dinaikkan sendiri.

---

## Cara menjalankan

### Lewat VS Code

Buka folder ini di VS Code. Konfigurasinya sudah ikut di dalam repo, jadi:

- **F5** → pilih **"Jalankan program (pakai kamera)"**
- Pilihan lain di panel Run and Debug: kamera 1, mode tanpa suara, atau uji
  tanpa kamera
- Lewat menu **Terminal → Run Task** ada tiga langkah berurutan:
  1. Pasang (siapkan .venv + library)
  2. Uji tanpa kamera
  3. Jalankan program

VS Code otomatis memakai Python dari `.venv` di folder ini, dan akan menawarkan
pemasangan ekstensi Python kalau belum ada.

### Lewat terminal

```powershell
.\run.ps1
```

Atau klik dua kali file `run.bat`.

### Percakapan dua sisi

```powershell
.\run.ps1 -Percakapan
```

Jendela dibagi dua. **Kiri** untuk penyandang tuli: video kamera, lalu di
bawahnya huruf yang sedang disusun dari isyarat dan riwayat percakapan. **Kanan**
untuk lawan bicara yang tidak tuli: riwayat percakapan, dan kotak untuk mengetik
(Enter untuk kirim). Tiap baris diberi tanda siapa yang bicara: biru dan rata
kiri untuk penyandang tuli, hijau dan rata kanan untuk lawan bicara, lengkap
dengan jam. Pesan isyarat terkirim sendiri kalau tidak ada huruf baru selama
2,5 detik, lalu diucapkan (matikan dengan `--tanpa-suara` atau tombol F6).
Riwayatnya disimpan di `log/percakapan_<waktu>.txt`. Tombol **Bersihkan
percakapan** (F8) mengosongkan riwayat di layar; catatan di berkas tetap
lengkap, hanya diberi penanda kapan layar dibersihkan. Semua pilihan
`gesture_voice.py` yang menyangkut pembacaan huruf (`--model`, `--kamera`,
`--min-keyakinan`, dan seterusnya) berlaku juga di sini.

**Lawan bicara bisa berbicara, bukan mengetik.** Di kanan atas ada pilihan
*masukan*: **Ketik** atau **Suara** (tombol F7). Pada mode Suara, ucapan lewat
mikrofon diubah jadi teks dan muncul sebagai satu pesan, sama seperti ketikan,
dengan tanda "(suara)" di namanya. Pengenalnya model Whisper (`faster-whisper`)
yang jalan di **CPU saja**, tanpa kartu grafis. Selama suara hasil pembacaan
isyarat sedang berbunyi, mikrofon diabaikan supaya tidak terbaca ulang.

**Dioptimalkan untuk bahasa Indonesia.** Whisper bawaan lemah di bahasa
Indonesia, jadi dipakai Whisper yang disetel khusus untuk bahasa Indonesia
(`id-base`, dari maleo-ai, lisensi Apache-2.0). Tingkat kesalahan kata (WER),
diukur pada 64 ucapan manusia nyata dari FLEURS bahasa Indonesia, panjang sampai
15 detik:

| Model | WER | Waktu per ucapan (sekitar 5 detik) |
|---|---|---|
| Whisper `base` biasa (sebelumnya) | 45,6% | 0,9 detik |
| Whisper `small` biasa | 19,2% | sekitar 3 detik |
| **`id-base`** (bawaan, beam 5) | **13,9%** | **1,1 detik** |
| `id-small` | 10,8% | sekitar 3 detik |

Ini ucapan bacaan yang cukup bersih. Percakapan spontan, derau ruangan, dan
mikrofon laptop akan menghasilkan angka yang lebih buruk.

Modelnya disiapkan **sekali** (diunduh atau dikonversi, sekitar 2 menit, butuh
`uv` dan internet), otomatis saat mode suara pertama dinyalakan, atau lewat:

```powershell
.\.venv\Scripts\python.exe siapkan_model_suara.py            # id-base
.\.venv\Scripts\python.exe siapkan_model_suara.py --semua     # termasuk id-small
```

Hasilnya di folder `model_suara/` dan tidak ikut ke GitHub.

```powershell
.\run.ps1 -Percakapan --mulai-dengan-suara     # langsung mode suara
.\run.ps1 -Percakapan --model-ucapan id-small   # paling teliti, sekitar 3x lebih lambat
.\run.ps1 -Percakapan --mikrofon 1             # pilih mikrofon lain
```

### Mengoptimalkan pembacaan satu huruf

Kalau ada huruf yang sering salah, satu perintah mengerjakan semuanya: mengukur
huruf itu di depan kamera Anda, memasukkan hasilnya jadi data latih, menyalin
model lama sebagai cadangan, lalu melatih ulang.

```powershell
.\run.ps1 -Optimasi K                      # dua sesi ukur untuk K, lalu latih ulang
.\run.ps1 -Optimasi K H P --verifikasi     # beberapa huruf, lalu ukur lagi dan bandingkan
.\run.ps1 -Optimasi --lewati-ukur          # latih dari sesi ukur yang sudah ada
```

Anda hanya memperagakan hurufnya (tekan Q sesudah 200 sampai 300 frame). Ubah
posisi dan jarak tangan di tiap sesi, kalau tidak model hanya menghafal jarak
tangan. Model lama disalin ke `log/cadangan_<waktu>/`.

Untuk mencoba tanpa menyalakan kamera, pakai gambar di folder `contoh/`:

```powershell
.\run.ps1 -Uji
```

Hasil yang seharusnya keluar:

```
letter_A.jpg  -> A  keyakinan 0.63  BENAR
letter_I.jpg  -> I  keyakinan 0.91  BENAR
letter_P.jpg  -> P  keyakinan 0.79  BENAR
letter_S.jpg  -> S  keyakinan 0.27  BENAR

Ketepatan: 4/4 = 100%
```

Launcher akan mencari Python secara berurutan: folder `.venv` di dalam repo ini,
lalu `.venv` di folder induk, terakhir `python` yang ada di PATH.

---

## Ambang keyakinan yang berbeda tiap huruf

> **Catatan.** Angka dan tabel di bagian ini berasal dari model lama (v2) dan
> `ambang_huruf.json`; bagian ini dibiarkan sebagai catatan bagaimana cara
> kerjanya. Model bawaan sekarang (v7) punya ambang sendiri di
> `ambang_rf_bisindo_v7.json`, dihitung `latih_cabang.py` dari tebakan yang
> belum pernah dilihat modelnya, dan **tidak ada huruf yang ditolak**. Program
> memilih berkas ambang yang cocok dengan modelnya, dan memuat ulang saat model
> diganti.

Ambang tunggal untuk semua huruf itu kasar, karena perilaku tiap huruf jauh
berbeda. `diagnosa.py` memeriksanya satu per satu, lalu menulis ambang yang pas
ke `ambang_huruf.json` yang dipakai program secara otomatis.

Yang dibandingkan adalah **keyakinan model saat menebak benar** dengan
**keyakinan saat menebak salah**. Kalau keduanya berjauhan, ambang bisa
memisahkan. Kalau berdekatan, ambang setinggi apa pun percuma.

Hasilnya untuk huruf yang tertolong:

| Huruf | Akurasi apa adanya | Presisi setelah ambang | Huruf tersisa |
|---|---|---|---|
| H | 37,1% | **92,2%** | 97,8% |
| O | 42,2% | **88,8%** | 100% |
| T | 54,2% | 80,7% | 97,0% |
| V | 93,9% | **99,8%** | 100% |
| R | 99,2% | **99,3%** | 100% |

Huruf **H** melonjak dari 37% ke 92% tanpa kehilangan apa pun, karena saat benar
model yakin 0,54 sedangkan saat salah cuma 0,26.

### Enam huruf yang ditolak: B, D, E, I, M, N

Untuk huruf ini model justru **lebih yakin waktu menebak salah**:

| Huruf | Yakin saat benar | Yakin saat salah |
|---|---|---|
| E | 0,28 | **0,86** |
| N | 0,63 | **0,87** |
| B | 0,52 | **0,83** |
| M | 0,68 | **0,79** |

Menaikkan ambang malah menyaring keluar jawaban yang benar dan menyisakan yang
salah. Karena itu keenam huruf ini **tidak pernah dimasukkan ke teks**. Untuk
mengeja, huruf yang salah lebih merugikan daripada huruf yang tidak muncul:
kalau tidak muncul tinggal diulang, kalau salah masuk tanpa disadari, kata yang
diucapkan jadi keliru.

Konsekuensinya nyata: kata seperti **ITU** tidak bisa dieja, karena I ditolak.
Kalau Anda tetap ingin mencobanya:

```powershell
.\run.ps1 --ambang-seragam
```

Semua huruf jadi ikut masuk, tapi ketepatannya turun drastis.

### Mengukur ulang atau mengubah target

```powershell
.\.venv\Scripts\python diagnosa.py                      # target presisi 80%
.\.venv\Scripts\python diagnosa.py --target-presisi 90  # lebih ketat
.\.venv\Scripts\python diagnosa.py --huruf I J T        # periksa huruf tertentu
```

Makin tinggi targetnya, makin sedikit huruf yang lolos tapi makin jarang salah.

---

## Melihat apa yang terjadi di dalam program

Setiap kali dijalankan, program mencatat jalannya proses ke layar sekaligus ke
berkas di folder `log/`. Tanpa ini, kalau hurufnya salah kita cuma tahu
"hasilnya keliru" tanpa tahu di mana macetnya.

Tampilan biasa hanya menampilkan kejadian penting:

```
10:30:21 [MULAI   ] model rf_bisindo_v7.pkl, butuh 197 angka masukan
10:30:21 [HURUF   ] 'I' diterima  ->  teks sekarang: "I"
10:30:21 [HURUF   ] 'T' diterima  ->  teks sekarang: "IT"
10:30:21 [HURUF   ] 'U' diterima  ->  teks sekarang: "ITU"
10:30:21 [SUARA   ] minta dibunyikan: "ITU"
10:30:23 [SUARA   ] selesai membunyikan "ITU" (1.52 detik)
10:30:23 [SELESAI ] 42 frame dalam 6.4 detik (6.5 frame per detik)
```

Kalau sebuah huruf tidak pernah mau diterima, jalankan dengan `--catat-detail`.
Di situ kelihatan tebakan model beserta pesaingnya, dan kemajuan voting:

```
10:29:50 [TEBAK   ] I (0.75)   pesaing: J 0.15
10:29:50 [VOTING  ] I: menang 5/8 frame, rata-rata keyakinan 0.75
10:29:50 [VOTING  ] I belum diterima, keyakinan 0.31 di bawah batas 0.45
```

Di layar kamera juga muncul **tiga tebakan teratas** di pojok kanan atas. Kalau
huruf yang Anda maksud muncul di urutan kedua atau ketiga, berarti model
sebenarnya mengenalinya tapi kalah bersaing — biasanya bisa ditolong dengan
menurunkan `--min-keyakinan` atau memperbaiki posisi tangan.

Pakai `--tanpa-catatan` kalau ingin layar bersih.

### Ringkasan sepanjang sesi

Selagi kamera menyala, tiap 20 detik dicatat ringkasan berjalan:

```
12:04:11 [PANTAU  ] 167 frame | tersering: A 90x/0.37, I 77x/0.42
```

Saat keluar, muncul tabel lengkap yang menjawab pertanyaan paling sering muncul,
yaitu kenapa sebuah huruf tidak pernah masuk ke teks:

```
  huruf | frame | keyakinan          | ambang | masuk | keterangan
  ------+-------+--------------------+--------+-------+------------------
    A   |    90 | 0.37  0.36  0.53   |  0.79  |     0 | keyakinan kurang, maks 0.53
    I   |    80 | 0.42  0.42  0.54   |  0.30  |     1 | lolos
    E   |    60 | 0.32  0.31  0.48   |  0.99  |     0 | DITOLAK di ambang_huruf.json
    G   |    50 | 0.71  0.71  0.81   |  0.20  |     2 | lolos
```

Rincian tiap frame juga ditulis ke `log/sesi_*.csv` supaya bisa dibuka di Excel.

> Angka di tabel ini **bukan ketepatan**, karena program tidak tahu huruf apa
> yang Anda maksud. Untuk mengukur ketepatan satu huruf, pakai `--ukur HURUF`.

---

## Tombol saat program jalan

| Tombol | Fungsinya |
|---|---|
| Q / ESC | keluar dari program |
| BACKSPACE | hapus huruf terakhir |
| C | hapus semua teks |
| SPASI | tambah spasi |
| ENTER | ucapkan teksnya sekarang juga |
| P | nyalakan / matikan ucap otomatis |

---

## Pengaturan tambahan

| Argumen | Bawaan | Gunanya |
|---|---|---|
| `--kamera` | 0 | ganti kalau kamera yang kebuka salah |
| `--frame-konfirmasi` / `--jendela-voting` | 8 / 12 | makin besar makin stabil, tapi makin lambat menerima huruf |
| `--min-keyakinan` | 0.45 | naikkan kalau sering salah huruf |
| `--jeda` | 1.2 detik | jarak minimal antar huruf |
| `--ucap-setelah` | 2.5 detik | lama diam sebelum kata dibunyikan |
| `--spasi-setelah` | 2.0 detik | lama tangan hilang sebelum spasi ditambah |
| `--bahasa` | id | `id` Indonesia, `en` Inggris |
| `--pelan` | mati | suara dibaca lebih pelan |
| `--tanpa-eja-huruf` | mati | cuma kata yang dibunyikan |
| `--tanpa-auto-ucap` | mati | tidak bunyi otomatis, tetap bisa lewat ENTER |

---

## Cara kerja penyaring voting

Kalau setiap tebakan langsung dimasukkan, satu pose tangan yang ditahan sebentar
bisa berubah jadi sepuluh huruf. Karena itu ada tiga syarat sebelum sebuah huruf
diterima:

1. Huruf itu harus menang minimal **8 dari 12** frame terakhir.
2. Rata-rata keyakinan model untuk huruf itu minimal **0,45**.
3. Sudah lewat **1,2 detik** sejak huruf sebelumnya diterima.

Kalau tangan hilang dari layar lebih dari 2 detik, itu dianggap satu kata sudah
selesai, dan program menambahkan spasi sendiri.

---

## Soal suara

Google TTS mengubah teks jadi file mp3, lalu diputar lewat pemutar bawaan
Windows (`winmm`), jadi tidak butuh library audio tambahan.

**Butuh internet waktu pertama kali** sebuah huruf atau kata dibunyikan. Setelah
itu file suaranya disimpan di `%TEMP%\gesture_voice_tts`, jadi berikutnya bisa
jalan tanpa internet. Kalau sedang tidak ada internet dan suaranya belum pernah
tersimpan, kata itu dilewati dan muncul peringatan di terminal — programnya
tidak ikut berhenti.

---

## Kebutuhan komputer

Diukur langsung, bukan perkiraan:

| Bagian | Terpakai | Minimal |
|---|---|---|
| RAM | ±570 MB, puncak 650–700 MB | 4 GB |
| CPU | ±1 inti penuh, dapat ±17 frame/detik | 2 inti, 2 GHz |
| GPU | tidak dipakai | — |
| Penyimpanan | model 30 MB + library ±600 MB | 1 GB kosong |
| Kamera | webcam biasa, 480p cukup | 720p |
| Internet | hanya saat pertama membunyikan sebuah kata | — |

---

## Seberapa tepat sebenarnya model ini?

**Sekitar 46%, bukan 99%.** Nama berkas modelnya memang `rf_bisindo_99.pkl`,
tapi angka 99 itu menyesatkan, dan ini penjelasannya.

Dataset aslinya cuma **312 foto**, lalu diperbanyak lewat augmentasi jadi
**14.493 sampel** — sekitar 46 salinan per foto. Pembagian data latih dan data
uji dilakukan **setelah** augmentasi, secara acak. Akibatnya tiap sampel uji
punya belasan sampai puluhan "saudara kembar" di data latih. Model tidak diuji
kemampuannya mengenali, tapi kemampuannya mengingat.

Buktinya, metode paling sederhana sekalipun (mencari tetangga terdekat) sudah
dapat 99,31% di data uji itu — jarak sebuah sampel uji ke kembarannya cuma
**3,1%** dari jarak antar huruf yang berbeda.

Setelah salinan-salinan itu dikelompokkan kembali ke foto asalnya dan pembagian
dilakukan per foto, angkanya jatuh jauh.

### Perbaikan yang sudah dilakukan

Ternyata penyebab utamanya bukan cuma datanya sedikit, tapi juga **bentuk
fiturnya**. Model lama disuapi koordinat mentah posisi tangan di layar (nilai
0 sampai 1), jadi yang dipelajari adalah "tangan ada di posisi ini", bukan
"tangan berbentuk ini". Begitu orangnya bergeser atau berdiri lebih dekat,
angkanya berubah semua.

Perbaikannya ada di `fitur.py`: pergelangan digeser ke titik nol lalu
diskalakan, kemudian ditambah ukuran geometris jari (jarak antar ujung jari,
sudut tekukan, sudut renggang).

| Cara mengukur | Akurasi | Macro F1 |
|---|---|---|
| Pembagian acak, fitur mentah (klaim asli) | 99,55% | 99,53% |
| Pembagian per foto, fitur mentah (model lama) | 46,52% | 38,22% |
| **Pembagian per foto, fitur olahan (model sekarang)** | **61,84%** | **55,96%** |

Naik **15,3 poin** akurasi dan **17,7 poin** macro F1, tanpa menambah data sama
sekali. Huruf **Z** yang tadinya tidak pernah benar sekarang 94%. Huruf **G**
100%, **J** 99,7%, **R** 99,2%.

Yang masih bermasalah: **I** dan **E** hampir tidak pernah benar, dan **M**,
**C**, **N**, **D**, **A**, **H**, **B** masih di bawah 40%.

> **Angka lengkap tiap huruf ada di [AKURASI.md](AKURASI.md)** — ketepatan per
> huruf, huruf apa yang tertukar dengan apa, sebaran keyakinan saat benar dan
> saat salah, serta ambang yang dipakai masing-masing.

Angka 62% itu pun masih terlalu optimis untuk pemakaian nyata, karena semua foto
berasal dari orang, ruangan, dan pencahayaan yang sama. Untuk memperbaikinya
lebih jauh, yang dibutuhkan adalah **data dari lebih banyak orang**, bukan lebih
banyak foto dari orang yang sama.

### Mengukur ulang sendiri

```powershell
.\.venv\Scripts\python latih.py               # latih ulang + bandingkan, lalu simpan
.\.venv\Scripts\python evaluasi.py            # tolok ukur fitur lama, beberapa menit
.\.venv\Scripts\python evaluasi.py --cepat    # gambaran kasar, jauh lebih cepat
.\.venv\Scripts\python evaluasi.py --model lain.pkl   # nilai berkas model lain
```

Di VS Code ada di **Terminal → Run Task → "4. Ukur ketepatan model"**.

Data ujinya (23 MB) diunduh otomatis sekali saja ke folder `data/`.

**Gunakan ini sebagai pembanding setiap kali ada model baru.** `latih.py` bahkan
menolak menyimpan model kalau hasilnya lebih buruk dari yang lama. Kalau model
baru tidak mengalahkan **61,84%** pada pembagian per foto, berarti belum ada
perbaikan nyata — seberapa pun bagusnya angka yang keluar dari pembagian acak.

> **Kalau `fitur.py` diubah, modelnya wajib dilatih ulang.** Program utama dan
> `latih.py` memakai berkas yang sama persis supaya perhitungannya dijamin
> sama. Kalau keduanya sampai berbeda, model akan menebak ngawur walaupun angka
> latihannya bagus.

---

## Batasan yang perlu diketahui

- **Ketepatannya sekitar 62% pada foto baru.** Sudah naik dari 46%, tapi masih
  jauh dari sempurna. Lihat bagian di atas.
- Huruf **I** dan **E** hampir tidak pernah dikenali dengan benar.
- Model ini hanya mengenali **26 huruf diam**. Huruf **J, R, dan Y** di BISINDO
  dibuat dengan gerakan, jadi hasilnya tidak bisa diandalkan.
- Paling akurat kalau latar belakangnya polos dan tangan terlihat penuh,
  kira-kira 50–70 cm dari kamera.
- Model dilatih dari kumpulan gambar Kaggle yang kecil (312 foto, lalu
  diperbanyak lewat augmentasi), jadi ketepatannya di dunia nyata masih
  sedang-sedang saja. Kalau sering salah, naikkan `--min-keyakinan`.
- Model disimpan sebagai file pickle scikit-learn. Kalau versi scikit-learn
  jauh berbeda dari 1.5.2, file ini bisa gagal dibuka.

---

## Kalau ada masalah

| Gejala | Penyebab & solusinya |
|---|---|
| `.\run.ps1 tidak dikenali` | belum `cd` ke folder repo, atau lupa awalan `.\` |
| PowerShell menolak menjalankan skrip | jalankan sekali: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `kamera 0 tidak bisa dibuka` | coba `--kamera 1` atau `--kamera 2`, dan tutup aplikasi lain yang sedang memakai kamera |
| `'FieldDescriptor' object has no attribute 'label'` | versi protobuf terlalu baru, kembalikan ke 4.25.3 |
| Tidak ada suara, muncul `Google TTS tidak bisa dihubungi` | tidak ada internet, dan kata itu belum pernah tersimpan |
| Program terasa lambat | wajar, MediaPipe memang berat di CPU. ±17 frame/detik sudah cukup karena tiap huruf ditahan sekitar 1 detik |

---

## Ucapan terima kasih

- **Abravia Mouzahra Ramadhani** dan **Shabri Sebastian Siregar** (SMK Telkom Malang)
  — proyek [gestura-bisindo](https://github.com/ShabriSebastian/gestura-bisindo) (MIT),
  sistem pengenalan BISINDO real-time dengan MediaPipe, TensorFlow, dan aplikasi
  Android. Gesture Voice dikembangkan di dalam proyek induk ini.
- **Krisna Santosa** — model `rf_bisindo_99.pkl` dan foto contoh, dari proyek
  [realtime-bisindo-classification](https://github.com/KrisnaSantosa15/realtime-bisindo-classification) (MIT).
- **Google MediaPipe** — pendeteksi titik tangan.
- **Google TTS (gTTS)** — pembuat suara bahasa Indonesia.
- Dataset asal model: [Alfabet BISINDO](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo) di Kaggle.

Rincian lisensi tiap komponen ada di berkas [LICENSE](LICENSE).

---

## Lisensi

MIT — lihat [LICENSE](LICENSE).
