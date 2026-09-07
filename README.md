# Gesture Voice — Pengenal Huruf BISINDO Bersuara

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-00A67E?logo=google&logoColor=white)
![Lisensi](https://img.shields.io/badge/Lisensi-MIT-blue)

Program yang membaca huruf **BISINDO** (Bahasa Isyarat Indonesia) lewat kamera,
lalu langsung membacakan hasilnya dengan suara.

```
kamera ──▶ MediaPipe cari titik tangan ──▶ 126 angka koordinat
       ──▶ model Random Forest menebak huruf (rf_bisindo_99.pkl)
       ──▶ disaring lewat voting ──▶ huruf ditumpuk jadi kata
       ──▶ dibunyikan pakai Google TTS (bahasa Indonesia)
```

Dibuat untuk membantu komunikasi sehari-hari yang singkat: mengeja nama, nama
jalan, atau nama obat, ketika juru bahasa isyarat tidak tersedia.

---

## Isi folder

| File | Gunanya |
|---|---|
| `gesture_voice.py` | program utama, pakai kamera |
| `uji_gambar.py` | uji model pakai file gambar, tanpa kamera |
| `evaluasi.py` | ukur ketepatan model secara jujur, jadi tolok ukur perbaikan |
| `rf_bisindo_99.pkl` | model yang sudah dilatih (30 MB) |
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
dilakukan per foto:

| Cara mengukur | Akurasi | Macro F1 |
|---|---|---|
| Pembagian acak (cara pembuat model) | 99,55% | 99,53% |
| **Pembagian per foto asli (jujur)** | **45,84%** | **37,46%** |

Sembilan huruf di bawah 30%. Huruf **I** dan **Z** tidak pernah sekali pun benar
pada foto yang belum pernah dilihat.

Angka 46% itu pun masih terlalu optimis untuk pemakaian nyata, karena semua foto
berasal dari orang, ruangan, dan pencahayaan yang sama.

### Mengukur ulang sendiri

```powershell
.\.venv\Scripts\python evaluasi.py            # tolok ukur lengkap, beberapa menit
.\.venv\Scripts\python evaluasi.py --cepat    # gambaran kasar, jauh lebih cepat
.\.venv\Scripts\python evaluasi.py --model model_baru.pkl   # nilai model lain
```

Di VS Code ada di **Terminal → Run Task → "4. Ukur ketepatan model"**.

Data ujinya (23 MB) diunduh otomatis sekali saja ke folder `data/`.

**Gunakan ini sebagai pembanding setiap kali ada model baru.** Kalau model baru
tidak mengalahkan 45,84% pada pembagian per foto, berarti belum ada perbaikan
nyata — seberapa pun bagusnya angka yang keluar dari pembagian acak.

---

## Batasan yang perlu diketahui

- **Ketepatannya sekitar 46% pada foto baru**, jauh di bawah kesan yang
  diberikan nama berkasnya. Lihat bagian di atas.
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

- **Krisna Santosa** — model `rf_bisindo_99.pkl` dan foto contoh, dari proyek
  [realtime-bisindo-classification](https://github.com/KrisnaSantosa15/realtime-bisindo-classification) (MIT).
- **Google MediaPipe** — pendeteksi titik tangan.
- **Google TTS (gTTS)** — pembuat suara bahasa Indonesia.
- Dataset asal model: [Alfabet BISINDO](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo) di Kaggle.

Rincian lisensi tiap komponen ada di berkas [LICENSE](LICENSE).

---

## Lisensi

MIT — lihat [LICENSE](LICENSE).
