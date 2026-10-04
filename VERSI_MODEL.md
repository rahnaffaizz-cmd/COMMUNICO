# Versi Model dan Cara Kembali ke Versi Lama

Model berganti versi setiap kali bentuk fiturnya berubah. Halaman ini mencatat
apa bedanya, dan bagaimana kembali ke versi lama kalau versi baru ternyata
lebih buruk.

## Daftar versi

| Berkas | Fitur | Apa yang ditambahkan |
|---|---|---|
| `rf_bisindo_99.pkl` | 126 | koordinat mentah apa adanya, model asli dari pembuatnya |
| `rf_bisindo_v2.pkl` | 162 | titik dinormalkan + ukuran geometris jari |
| `rf_bisindo_v3.pkl` | 166 | + letak relatif antar tangan (dx, dy, dz, jarak) |
| `rf_bisindo_v4.pkl` | 169 | + sudut arah, ketegakan, dan selisih arah hadap |
| `rf_bisindo_v5.pkl` | 172 | + jarak ujung jempol antar tangan dan jarak jempol-telunjuk silang |
| `rf_bisindo_v6.pkl` | 197 | + jarak 5x5 ujung jari tangan kiri ke kanan (dua tangan dipandang sebagai satu bentuk) |
| `rf_bisindo_v7.pkl` | 197 | fitur sama dengan v6, tapi dua cabang: satu tangan -> C E I J L O R U V Z, dua tangan -> sisanya (`cabang.py`, dilatih `latih_cabang.py`) |

## Cara berpindah versi

```powershell
.\run.ps1                                  # pakai yang jadi bawaan
.\run.ps1 --model rf_bisindo_v3.pkl        # coba versi lain
.\run.ps1 --model rf_bisindo_v2.pkl        # kembali ke versi sebelumnya
```

Tidak perlu melatih ulang atau menukar berkas apa pun. Kalau nama berkasnya
salah ketik, program menampilkan daftar model yang tersedia.

Untuk mengubah bawaannya secara tetap, ganti satu baris di `gesture_voice.py`:

```python
PATH_MODEL = "rf_bisindo_v2.pkl"
```

## Kenapa semua versi tetap bisa dipakai

`fitur.py` selalu menghitung fitur versi **terbaru dan terlengkap**, lalu
`sesuaikan_untuk()` memotongnya sesuai yang diminta model. Ini bisa karena
fitur baru selalu **ditambahkan di belakang**, tidak pernah menyisip di tengah:

```
0-125    titik yang sudah dinormalkan
126-161  ukuran geometris jari          -> berhenti di sini = 162 (v2)
162-165  letak relatif dasar            -> berhenti di sini = 166 (v3)
166-168  sudut, ketegakan, arah hadap   -> berhenti di sini = 169 (v4)
169-171  titik sentuh ujung jari silang -> berhenti di sini = 172 (v5)
172-196  jarak ujung jari kiri-kanan 5x5 -> berhenti di sini = 197 (v6)
```

Jadi satu berkas `fitur.py` melayani semua versi model sekaligus, dan keduanya
bisa dibandingkan langsung tanpa bertukar berkas.

**Aturan yang harus dipegang:** kalau menambah fitur baru, taruh di belakang.
Kalau menyisip di tengah atau mengubah urutan, semua model lama langsung rusak
tanpa pesan error, cuma tebakannya jadi ngawur.

## Apa yang diperbaiki di v4

Versi v3 memberi model `dx` dan `dy` terpisah. Untuk mengenali "sejajar atau
bertumpuk", model harus menggabungkan keduanya sendiri. Masalahnya pohon
keputusan memisahkan data dengan garis tegak lurus sumbu, jadi menggabungkan
dua sumbu butuh banyak percabangan bertingkat.

Versi v4 memberikan hasil gabungannya langsung:

| Angka | Gunanya |
|---|---|
| sudut arah | mendekati 0 kalau sejajar, mendekati 1,57 kalau bertumpuk |
| ketegakan | 0 berarti benar-benar sejajar, 1 berarti benar-benar bertumpuk |
| selisih arah hadap | membedakan tangan yang menghadap sama atau berlawanan |

Pengujian dengan dua susunan buatan:

```
              dx      dy    jarak   sudut  ketegakan
sejajar     1.250  -0.098   1.259  -0.079      0.073
bertumpuk   0.178   0.973   0.996   1.389      0.845
```

Angka **ketegakan** memisahkan keduanya dengan sangat jelas, 0,07 berbanding
0,85, dan hanya butuh satu percabangan untuk dipelajari.

## Apa yang diperbaiki di v5

v4 (dan v3 sebelumnya) melihat letak relatif tangan lewat **pergelangan**:
sejajar atau bertumpuk. Tapi ada huruf yang bentuk jari dan susunan tangannya
mirip, bedanya cuma **titik sentuhnya**: A adalah ujung jempol ketemu ujung
jempol, S adalah ujung jempol satu tangan ketemu ujung telunjuk tangan
lainnya. Pergelangan tidak melihat sampai sedetail itu.

v5 menambahkan jarak ujung jempol-ke-jempol dan jarak jempol-ke-telunjuk
silang (yang terkecil dari dua kemungkinan pasangan, karena urutan tangan
dari MediaPipe tidak selalu sama antar frame), dihitung dari
`_titik_sentuh_silang()` di `fitur.py`.

Diuji dengan memutar ulang rekaman sesi nyata (`putar_ulang.py`) lewat v2 dan
v5 dari koordinat identik:

```
v2 : ...SSSSSSASSSSSSSSSSSGSSSSSSSZ...   tersering: S 889x
v5 : ...SSASFSAAAASSASSAAAAGSSSAAASSZZ...   tersering: A 581x
```

Deretan panjang S di v2 banyak yang berubah jadi A di v5. Pada foto contoh
yang jawabannya sudah pasti benar, keyakinannya ikut naik untuk kedua huruf,
bukan cuma tertukar:

| Foto | Keyakinan v2 | Keyakinan v5 |
|---|---|---|
| `letter_A.jpg` | 0,81 | 0,86 |
| `letter_S.jpg` | 0,37 | 0,60 |

**Belum tentu ini perbaikan bersih.** Perubahan sebesar ini pada sesi bebas
(bukan sesi terkontrol) bisa juga berarti A jadi kebablasan menelan S yang
sebenarnya benar. Sama seperti peringatan di atas: perbandingan hanya berarti
kalau ada jawaban yang benar-benar diketahui.

**Sebelum mengganti versi bawaan ke v5**, uji secara terkontrol dulu:

```powershell
.\run.ps1 --model rf_bisindo_v5.pkl --ukur A    # peragakan A terus-menerus
.\run.ps1 --model rf_bisindo_v5.pkl --ukur S    # lalu peragakan S terus-menerus
```

Atau rekam sesi baru lalu putar ulang dengan `--benar`:

```powershell
.\.venv\Scripts\python putar_ulang.py log\sesi_*.csv --benar A `
    --model rf_bisindo_v2.pkl rf_bisindo_v5.pkl
```

## Cara membandingkan dua versi secara adil

Mencoba dua model bergantian di depan kamera **tidak adil**, karena peragaan
tangan mustahil sama persis dua kali. Perbedaan hasilnya bisa jadi karena
peragaannya, bukan karena modelnya.

Karena itu log sesi ikut menyimpan koordinat mentah, sehingga satu rekaman yang
sama bisa disuapkan ke semua versi:

```powershell
.\run.ps1                                    # rekam sekali, peragakan satu huruf
.\.venv\Scripts\python putar_ulang.py log\sesi_*.csv --benar A `
    --model rf_bisindo_v2.pkl rf_bisindo_v3.pkl rf_bisindo_v4.pkl rf_bisindo_v5.pkl
```

Keluarannya menampilkan teks yang dihasilkan tiap model dari rekaman yang
identik, beserta berapa huruf yang benar dan berapa yang salah.

## Peringatan soal angka dataset

Angka ketepatan pada dataset **sudah tiga kali terbukti menyesatkan** untuk
pemakaian nyata:

1. Huruf **I** ditolak karena di dataset tidak pernah benar, padahal saat dicoba
   langsung terbaca 0,40 dan pesaingnya J bahkan tidak muncul.
2. Fitur letak relatif dinilai tidak berguna karena dataset tidak menunjukkan
   perbaikan, padahal A tertukar jadi S cuma 22 kali di sana sementara di kamera
   pengguna itu masalah utamanya.
3. Ambang huruf **A** dua kali salah disetel dari angka dataset, 0,79 lalu 0,42,
   keduanya membuat A tidak pernah lolos sama sekali.

**Pilih versi berdasarkan pengujian dengan kamera Anda sendiri, bukan
berdasarkan angka dataset.** Itulah sebabnya semua versi disimpan, bukan
ditimpa.

## v7: dua cabang menurut jumlah tangan

Jumlah tangan yang terlihat dijadikan aturan tetap, bukan ditebak model:
satu tangan hanya bisa menjawab C E I J L O R U V Z, dua tangan hanya bisa
menjawab 16 huruf sisanya. Pengujian per foto asli:

| | v6 | v7 |
|---|---|---|
| semua sampel | 63,8% | 56,8% |
| sampel yang sesuai aturan | 71,2% | 86,4% |

Skor semua sampel lebih rendah karena 34% dataset melanggar aturan ini (huruf
dua tangan yang direkam dengan satu tangan, misalnya A 54%, B 64%, Y 91%).
Huruf-huruf itu tidak mungkin benar di v7. Y paling parah (4%) karena hampir
semua sampelnya satu tangan; rekam Y dengan dua tangan (`rekam.py`) kalau
mau memakainya. Kalau MediaPipe kehilangan satu tangan di tengah peragaan
huruf dua tangan, v7 berpindah cabang dan bisa menjawab huruf satu tangan.

Memuat v7 butuh `cabang.py` di folder yang sama.

### v7 dilatih dengan rekaman sendiri, dan jadi model bawaan

v7 kini dilatih bersama sesi `--ukur` milik pengguna untuk A sampai Z (5578
frame, dimasukkan lewat `impor_ukur.py`, bobot 3). v7 punya berkas ambang
sendiri, `ambang_rf_bisindo_v7.json`, yang dibuat `latih_cabang.py` dan dimuat
otomatis oleh `gesture_voice.py` untuk model ini. Model lain tetap memakai
`ambang_huruf.json`. Angka pada frame yang ikut melatih terlalu optimis; angka
per foto asli: 84,1% pada sampel yang sesuai aturan (v6 72,6%).

Kelemahan yang masih ada: kalau MediaPipe kehilangan satu tangan di tengah
huruf dua tangan (H, K, P, X, Y paling sering), v7 pindah ke cabang satu
tangan dan menjawab U, J, Z, atau L. Kembali ke v6 dengan
`.un.ps1 --model rf_bisindo_v6.pkl`.

