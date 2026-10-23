"""Membuat salinan sintetis "tangan anak" dari data tangan orang dewasa.

KENAPA PERLU
------------
Dataset bawaan dan rekaman data_pengguna.csv semuanya tangan orang dewasa.
Uji coba dengan anak-anak (2026-10-23) memperlihatkan huruf sering salah masuk,
terutama J, Z, dan U yang muncul padahal tidak diperagakan. Ukuran tangan
sendiri sudah tidak berpengaruh, karena fitur.py menskalakan semua titik dengan
jarak pergelangan ke pangkal jari tengah. Yang masih berpengaruh adalah:

1. Perbandingan jari dan telapak. Jari anak relatif lebih pendek dibanding
   telapaknya, jadi jarak ujung jari ke pergelangan (setelah dinormalkan) lebih
   kecil daripada milik orang dewasa dengan bentuk huruf yang sama.
2. Bentuk huruf yang kurang presisi. Anak belum bisa menekuk dan merenggangkan
   jari setepat orang dewasa, jadi arah tiap jari sedikit melenceng.
3. Titik yang lebih goyah. Tangan anak lebih kecil di layar, sehingga
   kesalahan titik MediaPipe yang besarnya sama terhitung lebih besar setelah
   dinormalkan. Ditambah tangan anak jarang benar-benar diam.

Salinan dibuat dari koordinat MENTAH (126 angka), sebelum fitur.py, supaya
semua fitur turunan (sudut, jarak silang, titik sentuh) ikut berubah dengan
konsisten.

BATASANNYA
----------
Ini tiruan, bukan rekaman anak sungguhan. Angka ketepatan pada data tiruan
hanya menunjukkan model lebih tahan terhadap perubahan yang DITIRU di sini.
Rekaman anak asli lewat `--ukur` tetap jauh lebih berharga.
"""

from __future__ import annotations

import numpy as np

JUMLAH_TITIK = 21

# Rantai tiap jari: titik pangkal (tidak digeser) lalu ruas-ruas berikutnya.
RANTAI_JARI = [
    (1, 2, 3, 4),       # jempol, pangkalnya di dekat pergelangan
    (5, 6, 7, 8),       # telunjuk
    (9, 10, 11, 12),    # tengah
    (13, 14, 15, 16),   # manis
    (17, 18, 19, 20),   # kelingking
]

# Panjang jari anak dibanding telapaknya, relatif terhadap orang dewasa.
# Anak usia sekolah dasar kira-kira 85-97% (jari tumbuh belakangan).
PANJANG_JARI = (0.85, 0.97)
SELISIH_ANTAR_JARI = 0.04      # tiap jari boleh beda sedikit dari jari lainnya
MELENCENG_DERAJAT = 6.0        # simpangan baku arah jari, bentuk kurang presisi
GOYAH = 0.015                  # simpangan baku titik, dalam satuan ukuran tangan


def _putar_xy(v: np.ndarray, sudut: np.ndarray) -> np.ndarray:
    """Putar vektor (n, 3) pada bidang layar sebesar `sudut` radian (n,)."""
    c, s = np.cos(sudut), np.sin(sudut)
    hasil = v.copy()
    hasil[:, 0] = c * v[:, 0] - s * v[:, 1]
    hasil[:, 1] = s * v[:, 0] + c * v[:, 1]
    return hasil


def tiru_tangan_anak(mentah: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ubah (n, 126) titik mentah jadi tiruan tangan anak dengan bentuk huruf sama.

    Tangan yang tidak terlihat (nol semua) dibiarkan nol.
    """
    mentah = np.nan_to_num(np.asarray(mentah, dtype=np.float32))
    banyak = len(mentah)
    titik = mentah.reshape(banyak, 2, JUMLAH_TITIK, 3).copy()

    for tangan in range(2):
        t = titik[:, tangan]                                    # (n, 21, 3)
        terlihat = np.abs(t).sum(axis=(1, 2)) > 1e-9
        ukuran = np.linalg.norm(t[:, 9] - t[:, 0], axis=1)      # (n,)

        asli = t.copy()
        dasar = rng.uniform(*PANJANG_JARI, size=banyak)
        for rantai in RANTAI_JARI:
            skala = dasar * (1 + rng.uniform(-SELISIH_ANTAR_JARI, SELISIH_ANTAR_JARI, banyak))
            sudut = np.deg2rad(rng.normal(0, MELENCENG_DERAJAT, banyak))
            # Ruas demi ruas dari pangkal, jadi seluruh jari ikut memendek dan
            # berbelok dari pangkalnya, bukan tiap titik bergeser sendiri.
            for sebelum, sekarang in zip(rantai[:-1], rantai[1:]):
                ruas = asli[:, sekarang] - asli[:, sebelum]
                t[:, sekarang] = t[:, sebelum] + _putar_xy(ruas, sudut) * skala[:, None]

        goyah = rng.normal(0, GOYAH, size=t.shape) * ukuran[:, None, None]
        goyah[:, 0] = 0.0                                       # pergelangan jadi patokan
        t += goyah

        t[~terlihat] = 0.0
        titik[:, tangan] = t

    return titik.reshape(banyak, -1)
