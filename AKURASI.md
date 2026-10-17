# Ketepatan Model per Huruf

Catatan hasil pengukuran, supaya perbaikan berikutnya punya pembanding yang
jelas dan tidak mengandalkan kesan belaka.

Semua angka di sini diukur dengan **pembagian per foto asli**, bukan pembagian
acak. Penjelasan kenapa itu penting ada di bagian
[Cara mengukurnya](#cara-mengukurnya) di bawah.

> **Peringatan yang perlu dibaca lebih dulu.** Seluruh angka di halaman ini
> berasal dari dataset Kaggle 312 foto. Pemakaian nyata sudah membuktikan angka
> itu **bisa keliru** — lihat [Ketika dataset ternyata salah](#ketika-dataset-ternyata-salah).
> Pakailah tabel ini sebagai pembanding antar percobaan, bukan sebagai ramalan
> hasil di lapangan.

---

## Ringkasan

| Cara mengukur | Akurasi | Macro F1 |
|---|---|---|
| Pembagian acak, fitur mentah (klaim asli "99%") | 99,55% | 99,53% |
| Pembagian per foto, fitur mentah (model lama) | 46,52% | 38,22% |
| **Pembagian per foto, fitur olahan (model sekarang)** | **61,84%** | **55,96%** |

Model sekarang: `rf_bisindo_v2.pkl`, Random Forest 300 pohon, 162 fitur dari
`fitur.py`.

---

## Ketepatan tiap huruf

Kolom **tertukar jadi** memperlihatkan huruf apa yang muncul saat model salah,
beserta berapa kali. Ini yang paling berguna untuk mencari akar masalah.

| Huruf | Model lama | Model sekarang | Selisih | Paling sering tertukar jadi |
|---|---|---|---|---|
| **Z** | 0,0% | **94,1%** | **+94,1** | E (5), B (4) |
| **R** | 7,0% | **99,2%** | **+92,2** | P (3), Y (2) |
| **X** | 46,0% | 93,4% | +47,4 | Y (29), T (6) |
| **V** | 59,2% | 93,9% | +34,7 | Y (30) |
| **U** | 46,7% | 79,7% | +33,0 | Y (67), L (31), W (20) |
| **B** | 7,0% | 38,7% | +31,8 | **E (308)** |
| **W** | 62,3% | 90,1% | +27,8 | U (21), Y (12) |
| **G** | 72,9% | **100,0%** | +27,1 | — |
| **F** | 68,8% | 92,7% | +23,8 | Y (12), S (11) |
| **C** | 5,8% | 29,3% | +23,5 | D (149), P (53), U (50) |
| **Y** | 39,2% | 60,1% | +20,9 | T (41), W (39), U (29) |
| **S** | 79,2% | 94,7% | +15,5 | D (14) |
| **D** | 20,6% | 35,6% | +15,0 | A (220), C (57), P (51) |
| **L** | 33,2% | 47,7% | +14,5 | Y (202), A (41) |
| **P** | 67,2% | 81,5% | +14,3 | A (27), D (26), L (15) |
| **A** | 24,4% | 36,7% | +12,4 | W (187), D (152), S (22) |
| **K** | 75,7% | 83,2% | +7,5 | P (47), X (27) |
| **J** | 94,8% | 99,7% | +4,8 | G (2) |
| **H** | 35,5% | 37,1% | +1,7 | **Y (307)** |
| **Q** | 93,8% | 94,0% | +0,2 | O (32) |
| **I** | 0,0% | **0,0%** | 0,0 | **J (436)**, U (119), A (45) |
| **M** | 27,0% | 24,9% | −2,2 | **N (445)** |
| **N** | 39,7% | 34,2% | −5,5 | **M (394)** |
| **T** | 61,1% | 54,2% | −7,0 | Y (176), F (36) |
| **O** | 74,0% | **42,2%** | **−31,8** | **Q (347)** |
| **E** | 37,5% | **0,3%** | **−37,2** | **B (598)** |

### Perbaikan fitur juga memakan korban

Naiknya 15,3 poin secara keseluruhan tidak gratis. Normalisasi membuang
informasi posisi dan orientasi mutlak. Untuk sebagian besar huruf itu justru
bagus, karena bentuk tangan jadi yang menentukan. Tapi untuk huruf yang
**dibedakan oleh orientasi**, informasi itu ikut hilang:

- **E** hampir seluruhnya tertelan **B** (598 dari 600), padahal sebelumnya 37,5%
- **O** tertelan **Q**, padahal sebelumnya 74%

Ini perlu diakui, bukan disembunyikan di balik rata-rata.

### Pasangan yang saling menelan

| Pasangan | Keterangan |
|---|---|
| **I ↔ J** | Di BISINDO, **J adalah I yang digerakkan**. Dari satu frame diam keduanya identik. Tidak ada fitur apa pun yang bisa membedakannya tanpa melihat gerakan. |
| **M ↔ N** | Saling tukar dua arah (445 dan 394), bentuknya memang mirip. |
| **B → E** | Muncul setelah normalisasi. Keduanya kemungkinan dibedakan oleh orientasi telapak. |
| **O → Q** | Sama, orientasi ikut hilang saat dinormalkan. |

---

## Ambang keyakinan per huruf

Akurasi saja tidak cukup untuk mengeja. Yang lebih penting **presisi**: dari
sekian kali model bilang "ini huruf X", berapa persen yang benar. Huruf salah
yang masuk lebih merugikan daripada huruf yang tidak muncul — tidak muncul
tinggal diulang, salah masuk tanpa disadari membuat kata yang diucapkan keliru.

Kuncinya membandingkan keyakinan model **saat benar** dengan **saat salah**.
Kalau berjauhan, ambang bisa memisahkan. Kalau berdekatan atau terbalik, ambang
setinggi apa pun percuma.

| Huruf | Akurasi | Yakin saat benar | Yakin saat salah | Ambang | Presisi setelah ambang | Huruf tersisa |
|---|---|---|---|---|---|---|
| G | 100,0% | 0,76 | 0,43 | 0,20 | **98,5%** | 100,0% |
| R | 99,2% | 0,65 | 0,26 | 0,20 | **99,3%** | 100,0% |
| V | 93,9% | 0,60 | 0,22 | 0,20 | **99,8%** | 100,0% |
| Z | 94,1% | 0,45 | — | 0,20 | **100,0%** | 100,0% |
| F | 92,7% | 0,85 | 0,22 | 0,20 | 95,0% | 96,7% |
| K | 83,2% | 0,58 | 0,27 | 0,20 | 96,1% | 99,2% |
| S | 94,7% | 0,57 | 0,27 | 0,20 | 90,7% | 98,1% |
| **H** | 37,1% | 0,54 | 0,26 | 0,20 | **92,2%** | 97,8% |
| **O** | 42,2% | 0,57 | 0,54 | 0,20 | **88,8%** | 100,0% |
| X | 93,4% | 0,48 | 0,38 | 0,20 | 86,3% | 99,7% |
| J | 99,7% | 0,72 | 0,26 | 0,28 | 82,4% | 70,2% |
| **T** | 54,2% | 0,55 | 0,38 | 0,23 | 80,7% | 97,0% |
| U | 79,7% | 0,52 | 0,28 | 0,25 | 80,7% | 79,4% |
| P | 81,5% | 0,54 | 0,29 | 0,26 | 80,2% | 81,4% |
| L | 47,7% | 0,45 | 0,30 | 0,28 | 81,2% | 78,6% |
| W | 90,1% | 0,73 | 0,30 | 0,30 | 80,1% | 70,7% |
| Q | 94,0% | 0,75 | 0,57 | 0,59 | 80,7% | 59,1% |
| Y | 60,1% | 0,53 | 0,35 | 0,60 | 81,3% | 12,2% |
| C | 29,3% | 0,37 | 0,35 | 0,51 | 84,6% | 9,7% |
| A | 36,7% | 0,40 | 0,38 | 0,79 | 100,0% | 0,2% |
| **I** | 0,0% | — | 0,38 | *0,30* | — | *disetel manual* |
| **B** | 38,7% | 0,52 | **0,83** | 0,99 | 25,2% | **ditolak** |
| **D** | 35,6% | 0,45 | 0,35 | 0,99 | 35,5% | **ditolak** |
| **E** | 0,3% | 0,28 | **0,86** | 0,99 | 0,6% | **ditolak** |
| **M** | 24,9% | 0,68 | **0,79** | 0,99 | 27,3% | **ditolak** |
| **N** | 34,2% | 0,63 | **0,87** | 0,99 | 31,4% | **ditolak** |

Angka-angka ini tersimpan di [`ambang_huruf.json`](ambang_huruf.json) dan dipakai
program secara otomatis. Dibuat ulang dengan `python diagnosa.py`.

### Huruf yang paling tertolong ambang

**H** melonjak dari akurasi 37,1% menjadi presisi **92,2%** hampir tanpa
kehilangan apa pun, karena saat benar model yakin 0,54 sedangkan saat salah cuma
0,26. Selisih itu yang bisa dipisahkan. **O** dan **T** juga begitu.

### Lima huruf yang ditolak: B, D, E, M, N

Untuk huruf ini model justru **lebih yakin waktu menebak salah**:

| Huruf | Yakin saat benar | Yakin saat salah |
|---|---|---|
| E | 0,28 | **0,86** |
| N | 0,63 | **0,87** |
| B | 0,52 | **0,83** |
| M | 0,68 | **0,79** |

Menaikkan ambang malah menyaring keluar jawaban yang benar dan menyisakan yang
salah. Ini bukan masalah penyetelan, melainkan masalah fitur.

Kalau tetap ingin mencoba huruf-huruf ini:

```powershell
.\run.ps1 --ambang-seragam
```

---

## Ketika dataset ternyata salah

**Huruf I awalnya ikut ditolak**, karena pada dataset tidak pernah sekali pun
benar dan selalu kalah oleh J (436 dari 600 kali). Alasannya masuk akal secara
teori: J adalah I yang digerakkan.

Saat dicoba dengan kamera dan tangan pengguna sebenarnya, hasilnya berbeda:

```
I 0.40     <- tebakan teratas, voting penuh
U 0.19
O 0.08
```

**J tidak muncul sama ssekali di tiga besar.** Ambang I kemudian diturunkan ke
**0,30** berdasarkan bukti itu, dan alasannya dicatat di dalam
`ambang_huruf.json`.

Pelajaran yang perlu diingat:

> Angka dari dataset **tidak otomatis berlaku** untuk kamera, tangan, dan
> ruangan yang berbeda. Kalau pengamatan langsung bertentangan dengan tabel di
> halaman ini, **percayai pengamatan langsung**.

Lima huruf yang masih ditolak (B, D, E, M, N) belum pernah diuji seperti itu.
Bisa jadi sebagiannya sebenarnya terbaca bagus. Cara mengeceknya:

```powershell
.\run.ps1 --ukur E
```

Peragakan huruf itu 10–20 detik, tekan Q, lalu baca laporannya.

---

## Cara mengukurnya

### Kenapa angka 99% itu palsu

Dataset aslinya **312 foto**, diperbanyak lewat augmentasi jadi **14.493
sampel** — sekitar 46 salinan per foto. Pembagian latih/uji dilakukan **setelah**
augmentasi, secara acak. Akibatnya tiap sampel uji punya belasan sampai puluhan
"saudara kembar" di data latih.

Buktinya:

| | |
|---|---|
| Jarak sampel uji ke kembarannya di data latih | 0,08 |
| Jarak antar huruf yang berbeda | 2,67 |
| **Rasio** | **3,1%** |
| Akurasi metode paling sederhana (1 tetangga terdekat) | **99,31%** |
| Akurasi Random Forest | 99,45% |

Kalau mencari tetangga terdekat saja sudah dapat 99%, soal ujiannya memang sudah
bocor. Model tidak diuji kemampuannya mengenali, melainkan mengingat.

### Pembagian yang benar

Salinan augmentasi dikelompokkan kembali ke foto asalnya (269 kelompok,
mendekati 312 yang sebenarnya), lalu pembagian dilakukan per kelompok. Satu foto
beserta seluruh salinannya hanya boleh ada di satu sisi.

### Mengulang pengukuran

```powershell
.\.venv\Scripts\python latih.py      # latih ulang + bandingkan fitur lama vs baru
.\.venv\Scripts\python diagnosa.py   # ambang per huruf
.\.venv\Scripts\python evaluasi.py   # tolok ukur fitur lama
```

`latih.py` **menolak menyimpan model** kalau hasilnya lebih buruk dari yang lama.

---

## Yang dibutuhkan untuk perbaikan berikutnya

Perbaikan fitur sudah mentok. Yang tersisa:

1. **Data dari lebih banyak orang.** Ini yang paling menentukan. Menambah 10.000
   foto dari orang yang sama tidak banyak menolong; 500 foto dari 20 orang
   berbeda jauh lebih berharga.
2. **Kembalikan sebagian informasi orientasi** — misalnya arah telapak tangan
   sebagai fitur tersendiri, tanpa mengembalikan ketergantungan pada posisi
   layar. Kemungkinan menyelamatkan E dan O.
3. **Deteksi gerakan** untuk memisahkan I dari J. Pendekatan satu-frame tidak
   akan pernah bisa. Proyek gestura menyimpan 20 frame berturut-turut untuk ini.
