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

## Cara membandingkan dua versi secara adil

Mencoba dua model bergantian di depan kamera **tidak adil**, karena peragaan
tangan mustahil sama persis dua kali. Perbedaan hasilnya bisa jadi karena
peragaannya, bukan karena modelnya.

Karena itu log sesi ikut menyimpan koordinat mentah, sehingga satu rekaman yang
sama bisa disuapkan ke semua versi:

```powershell
.\run.ps1                                    # rekam sekali, peragakan satu huruf
.\.venv\Scripts\python putar_ulang.py log\sesi_*.csv --benar A `
    --model rf_bisindo_v2.pkl rf_bisindo_v3.pkl rf_bisindo_v4.pkl
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
