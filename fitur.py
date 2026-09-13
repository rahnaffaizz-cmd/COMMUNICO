"""Mengubah titik-titik tangan dari MediaPipe menjadi angka yang dipelajari model.

KENAPA TIDAK LANGSUNG DIPAKAI SAJA
----------------------------------
MediaPipe memberi 21 titik per tangan, masing-masing berupa koordinat x, y, z
di layar. Kalau angka itu langsung disuapkan ke model, yang dipelajari model
adalah "tangan berada di posisi ini di layar", bukan "tangan berbentuk ini".
Begitu orangnya bergeser sedikit, atau berdiri lebih dekat ke kamera, angkanya
berubah semua dan model langsung bingung.

Makanya di sini dikerjakan dua hal:

1. Dinormalkan. Pergelangan tangan digeser ke titik nol, lalu semua titik
   diskalakan memakai jarak pergelangan ke pangkal jari tengah. Hasilnya tidak
   lagi bergantung pada posisi tangan di layar maupun jaraknya dari kamera.

2. Ditambah ukuran geometris. Jarak tiap ujung jari ke pergelangan, jarak antar
   ujung jari yang bersebelahan, sudut tekukan tiap jari, dan sudut renggang
   antar jari. Ini yang sebenarnya membedakan satu huruf dengan huruf lain.

Pengujian menunjukkan langkah 1 saja justru memperburuk (46,69% turun ke
34,44%), karena informasi posisi dibuang tanpa ada penggantinya. Baru setelah
ditambah langkah 2 hasilnya naik ke 62,07%.

SUSUNAN ANGKANYA
----------------
Masukan  : 126 angka, susunannya x,y,z per titik, titik 0-20 tangan pertama
           lalu titik 21-41 tangan kedua. Tangan yang tidak terlihat diisi nol.
Keluaran : 162 angka, yaitu 126 titik yang sudah dinormalkan ditambah 36 ukuran
           geometris (18 per tangan).

Berkas ini dipakai bersama oleh latih.py, gesture_voice.py, dan uji_gambar.py,
supaya perhitungannya dijamin sama persis. Kalau berkas ini diubah, modelnya
WAJIB dilatih ulang.
"""

from __future__ import annotations

import numpy as np

JUMLAH_TITIK = 21
PANJANG_MASUKAN = 126
PANJANG_KELUARAN = 166

# Pasangan (pangkal, ujung) tiap jari: jempol, telunjuk, tengah, manis, kelingking
JARI = [(2, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
UJUNG_JARI = [4, 8, 12, 16, 20]
UJUNG_BERSEBELAHAN = [(4, 8), (8, 12), (12, 16), (16, 20)]
PANGKAL_JARI = [2, 5, 9, 13, 17]


def _sudut(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sudut antara dua arah, dalam radian. Nol kalau salah satu arahnya kosong."""
    panjang_a = np.linalg.norm(a, axis=-1)
    panjang_b = np.linalg.norm(b, axis=-1)
    layak = (panjang_a > 1e-8) & (panjang_b > 1e-8)

    kosinus = np.zeros(len(a))
    kosinus[layak] = (a[layak] * b[layak]).sum(-1) / (panjang_a[layak] * panjang_b[layak])
    return np.arccos(np.clip(kosinus, -1.0, 1.0))


def _normalkan(titik: np.ndarray) -> np.ndarray:
    """Geser pergelangan ke titik nol, lalu skalakan. Bentuk masukan (n, 2, 21, 3)."""
    hasil = titik.copy()
    for tangan in range(2):
        t = hasil[:, tangan]                                  # (n, 21, 3)

        # Tangan yang tidak terlihat isinya nol semua, jangan diutak-atik.
        terlihat = np.abs(t).sum(axis=(1, 2)) > 1e-9

        t = t - t[:, 0:1, :]                                  # pergelangan jadi titik nol

        # Jarak pergelangan ke pangkal jari tengah dipakai sebagai ukuran tangan.
        ukuran = np.linalg.norm(t[:, 9, :], axis=1, keepdims=True)
        ukuran[ukuran < 1e-6] = 1.0                           # hindari bagi nol
        t = t / ukuran[:, :, None]

        t[~terlihat] = 0.0
        hasil[:, tangan] = t
    return hasil


def _ukuran_geometris(titik: np.ndarray) -> list[np.ndarray]:
    """18 angka per tangan: 5 jarak + 4 jarak + 5 sudut tekuk + 4 sudut renggang."""
    kumpulan = []
    for tangan in range(2):
        t = titik[:, tangan]
        terlihat = np.abs(t).sum(axis=(1, 2)) > 1e-9

        bagian = [
            # Jarak tiap ujung jari ke pergelangan. Mengecil kalau jarinya ditekuk.
            np.stack([np.linalg.norm(t[:, u] - t[:, 0], axis=-1)
                      for u in UJUNG_JARI], axis=1),

            # Jarak antar ujung jari bersebelahan. Membedakan jari rapat dan renggang.
            np.stack([np.linalg.norm(t[:, a] - t[:, b], axis=-1)
                      for a, b in UJUNG_BERSEBELAHAN], axis=1),

            # Sudut tekukan: antara arah pergelangan-pangkal dan pangkal-ujung.
            # Mendekati nol kalau jarinya lurus, membesar kalau ditekuk.
            np.stack([_sudut(t[:, p] - t[:, 0], t[:, u] - t[:, p])
                      for p, u in JARI], axis=1),

            # Sudut renggang antar pangkal jari yang bersebelahan.
            np.stack([_sudut(t[:, a] - t[:, 0], t[:, b] - t[:, 0])
                      for a, b in zip(PANGKAL_JARI[:-1], PANGKAL_JARI[1:])], axis=1),
        ]

        # Tangan yang tidak terlihat harus menghasilkan nol semua. Tanpa ini,
        # perhitungan sudut mengembalikan pi/2 untuk tangan kosong, padahal
        # nilai itu juga bisa keluar dari tangan yang benar-benar ada.
        for b in bagian:
            b[~terlihat] = 0.0
        kumpulan.extend(bagian)
    return kumpulan


def _posisi_relatif(mentah_titik: np.ndarray) -> np.ndarray:
    """Letak tangan kedua terhadap tangan pertama: 4 angka.

    Ini WAJIB dihitung dari koordinat MENTAH, sebelum tiap tangan digeser ke
    pergelangannya masing-masing. Setelah digeser, kedua tangan sama-sama
    berpusat di titik nol dan letak relatifnya hilang sama sekali.

    Padahal untuk huruf dua tangan, letak relatif itulah yang membedakan.
    Contohnya A dan S: keduanya dua tangan dengan bentuk jari mirip, bedanya
    A kedua tangannya sejajar kiri-kanan, sedangkan S tersusun atas-bawah.
    Tanpa angka ini, model mustahil membedakan keduanya.

    Isinya: selisih letak pergelangan (x, y, z) dan jaraknya, semuanya
    diskalakan dengan ukuran tangan pertama supaya tidak bergantung pada
    seberapa dekat orangnya ke kamera.
    """
    banyak = len(mentah_titik)
    hasil = np.zeros((banyak, 4), dtype=np.float32)

    tangan1 = mentah_titik[:, 0]
    tangan2 = mentah_titik[:, 1]
    dua_tangan = (np.abs(tangan1).sum(axis=(1, 2)) > 1e-9) & \
                 (np.abs(tangan2).sum(axis=(1, 2)) > 1e-9)
    if not dua_tangan.any():
        return hasil

    # Ukuran tangan pertama dipakai sebagai satuan, sama seperti saat normalisasi.
    ukuran = np.linalg.norm(tangan1[:, 9] - tangan1[:, 0], axis=1)
    ukuran[ukuran < 1e-6] = 1.0

    selisih = (tangan2[:, 0] - tangan1[:, 0]) / ukuran[:, None]
    hasil[dua_tangan, :3] = selisih[dua_tangan]
    hasil[dua_tangan, 3] = np.linalg.norm(selisih[dua_tangan], axis=1)
    return hasil


def ambil_fitur(mentah: np.ndarray) -> np.ndarray:
    """Ubah titik mentah jadi 162 angka siap pakai.

    Menerima satu sampel (126 angka) maupun banyak sampel sekaligus (n, 126).
    Keluarannya selalu berbentuk (n, 162).
    """
    mentah = np.asarray(mentah, dtype=np.float32)
    if mentah.ndim == 1:
        mentah = mentah.reshape(1, -1)

    if mentah.shape[1] != PANJANG_MASUKAN:
        raise ValueError(
            f"butuh {PANJANG_MASUKAN} angka per sampel, yang diberikan {mentah.shape[1]}"
        )

    # Nilai kosong dianggap tangan tidak terlihat.
    mentah = np.nan_to_num(mentah, nan=0.0)

    mentah_titik = mentah.reshape(-1, 2, JUMLAH_TITIK, 3)

    # Letak relatif dihitung LEBIH DULU, selagi koordinatnya masih mentah.
    # Setelah dinormalkan, keterangan itu sudah hilang dan tidak bisa dipulihkan.
    relatif = _posisi_relatif(mentah_titik)

    titik = _normalkan(mentah_titik)
    bagian = [titik.reshape(len(mentah), -1)] + _ukuran_geometris(titik) + [relatif]

    hasil = np.concatenate(bagian, axis=1).astype(np.float32)
    assert hasil.shape[1] == PANJANG_KELUARAN, hasil.shape
    return hasil


def sesuaikan_untuk(fitur_lengkap: np.ndarray, jumlah_diminta: int) -> np.ndarray:
    """Potong fitur agar cocok dengan yang diminta model.

    Ada dua versi model yang beredar:
      162 angka  tanpa letak relatif antar tangan (versi lama)
      166 angka  dengan letak relatif (versi baru)

    Karena 4 angka letak relatif ditaruh paling belakang, versi lama cukup
    diperoleh dengan membuang empat angka terakhir. Dengan begitu satu berkas
    fitur.py bisa melayani kedua model, sehingga keduanya bisa dibandingkan
    langsung tanpa perlu bertukar berkas.
    """
    tersedia = fitur_lengkap.shape[1]
    if jumlah_diminta == tersedia:
        return fitur_lengkap
    if jumlah_diminta < tersedia:
        return fitur_lengkap[:, :jumlah_diminta]
    raise ValueError(
        f"model meminta {jumlah_diminta} angka, fitur.py cuma menyediakan {tersedia}"
    )
