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
Keluaran : 197 angka, disusun bertingkat supaya model versi lama tetap bisa
           dipakai dengan memotong bagian belakangnya:
             0-125    titik yang sudah dinormalkan
             126-161  ukuran geometris jari, 18 per tangan   -> sampai sini 162
             162-168  letak relatif antar tangan             -> sampai sini 169
             169-171  titik sentuh ujung jari antar tangan   -> sampai sini 172
             172-196  jarak 5x5 ujung jari tangan kiri ke kanan -> sampai sini 197

Berkas ini dipakai bersama oleh latih.py, gesture_voice.py, dan uji_gambar.py,
supaya perhitungannya dijamin sama persis. Kalau berkas ini diubah, modelnya
WAJIB dilatih ulang.
"""

from __future__ import annotations

import numpy as np

JUMLAH_TITIK = 21
PANJANG_MASUKAN = 126
PANJANG_KELUARAN = 197

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
    """Letak tangan kedua terhadap tangan pertama: 7 angka.

    Ini WAJIB dihitung dari koordinat MENTAH, sebelum tiap tangan digeser ke
    pergelangannya masing-masing. Setelah digeser, kedua tangan sama-sama
    berpusat di titik nol dan letak relatifnya hilang sama sekali.

    Padahal untuk huruf dua tangan, letak relatif itulah yang membedakan.
    Contohnya A dan S: keduanya dua tangan dengan bentuk jari mirip, bedanya
    A kedua tangannya SEJAJAR kiri-kanan, sedangkan S tersusun ATAS-BAWAH.
    Tanpa angka ini, model mustahil membedakan keduanya.

    Isi ketujuh angkanya:
      0-2  selisih letak pergelangan (x, y, z)
      3    jarak antar pergelangan
      4    sudut arah selisih itu pada bidang layar
      5    seberapa tegak susunannya, 0 berarti sejajar dan 1 berarti bertumpuk
      6    selisih arah hadap kedua tangan

    Angka 4 dan 5 sebenarnya bisa dihitung sendiri oleh model dari angka 0
    dan 1. Tetapi pohon keputusan memisahkan data dengan garis tegak lurus
    sumbu, jadi untuk mengenali "sejajar atau bertumpuk" ia harus menyusun
    banyak percabangan x dan y sekaligus. Kalau sudutnya diberikan langsung,
    satu percabangan saja sudah cukup. Inilah inti perbaikan versi ini.

    Semuanya diskalakan dengan ukuran tangan pertama supaya tidak bergantung
    pada seberapa dekat orangnya ke kamera.
    """
    banyak = len(mentah_titik)
    hasil = np.zeros((banyak, 7), dtype=np.float32)

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
    dx, dy = selisih[:, 0], selisih[:, 1]

    hasil[dua_tangan, :3] = selisih[dua_tangan]
    hasil[dua_tangan, 3] = np.linalg.norm(selisih[dua_tangan], axis=1)

    # Sudut arah selisih. Mendekati 0 atau pi kalau sejajar kiri-kanan,
    # mendekati setengah pi kalau bertumpuk atas-bawah.
    hasil[dua_tangan, 4] = np.arctan2(dy[dua_tangan], dx[dua_tangan])

    # Ukuran ketegakan yang tidak terpengaruh arah kiri atau kanan. Nilai 0
    # berarti benar-benar sejajar, 1 berarti benar-benar bertumpuk.
    mendatar = np.abs(dx) + np.abs(dy) + 1e-8
    hasil[dua_tangan, 5] = (np.abs(dy) / mendatar)[dua_tangan]

    # Selisih arah hadap kedua tangan, diukur dari pergelangan ke pangkal jari
    # tengah. Membedakan tangan yang menghadap sama dengan yang berlawanan.
    arah1 = tangan1[:, 9] - tangan1[:, 0]
    arah2 = tangan2[:, 9] - tangan2[:, 0]
    hasil[dua_tangan, 6] = _sudut(arah1, arah2)[dua_tangan]
    return hasil


def _titik_sentuh_silang(mentah_titik: np.ndarray) -> np.ndarray:
    """3 angka: titik sentuh ujung jari antar tangan.

    Sebagian huruf dua tangan bentuk jarinya nyaris identik (mengepal), tapi
    beda titik sentuhnya. Contoh nyata: A (ujung jempol ketemu ujung jempol)
    lawan S (ujung jempol satu tangan ketemu ujung telunjuk tangan lainnya).
    `_posisi_relatif` di atas hanya melihat pergelangan, jadi buta terhadap
    perbedaan ini.

    Sama seperti `_posisi_relatif`, WAJIB dihitung dari koordinat MENTAH
    sebelum dinormalkan, karena setelah digeser tiap tangan berpusat di
    pergelangannya sendiri-sendiri dan jarak antar tangan hilang.

    Isi ketiga angkanya, diskalakan dengan ukuran tangan pertama:
      0  jarak ujung jempol tangan pertama ke ujung jempol tangan kedua
      1  jarak jempol-telunjuk silang yang paling dekat (dua kemungkinan
         pasangannya, diambil yang terkecil karena urutan tangan dari
         MediaPipe tidak selalu sama antar frame)
      2  selisih angka 0 dan 1 -- negatif berarti condong ke pola A (jempol
         ketemu jempol), positif condong ke pola S (jempol ketemu telunjuk).
         Pohon keputusan butuh banyak percabangan untuk menyimpulkan sendiri
         mana yang lebih kecil dari dua angka, jadi selisihnya diberi langsung.
    """
    banyak = len(mentah_titik)
    hasil = np.zeros((banyak, 3), dtype=np.float32)

    tangan1 = mentah_titik[:, 0]
    tangan2 = mentah_titik[:, 1]
    dua_tangan = (np.abs(tangan1).sum(axis=(1, 2)) > 1e-9) & \
                 (np.abs(tangan2).sum(axis=(1, 2)) > 1e-9)
    if not dua_tangan.any():
        return hasil

    ukuran = np.linalg.norm(tangan1[:, 9] - tangan1[:, 0], axis=1)
    ukuran[ukuran < 1e-6] = 1.0

    jempol1, jempol2 = tangan1[:, 4], tangan2[:, 4]
    telunjuk1, telunjuk2 = tangan1[:, 8], tangan2[:, 8]

    d_jempol = np.linalg.norm(jempol1 - jempol2, axis=-1) / ukuran
    d_silang = np.minimum(
        np.linalg.norm(jempol1 - telunjuk2, axis=-1),
        np.linalg.norm(jempol2 - telunjuk1, axis=-1),
    ) / ukuran

    hasil[dua_tangan, 0] = d_jempol[dua_tangan]
    hasil[dua_tangan, 1] = d_silang[dua_tangan]
    hasil[dua_tangan, 2] = (d_jempol - d_silang)[dua_tangan]
    return hasil


def _jarak_ujung_silang(mentah_titik: np.ndarray) -> np.ndarray:
    """25 angka: jarak tiap ujung jari tangan kiri ke tiap ujung jari tangan kanan.

    Dua tangan dipandang sebagai SATU bentuk, bukan dua bentuk yang berdiri
    sendiri-sendiri. Di huruf dua tangan, yang membedakan huruf adalah ujung
    jari mana bertemu ujung jari mana. A: jempol ketemu jempol. S: jempol
    ketemu telunjuk tangan satunya. Pengujian per foto asli memperlihatkan
    pasangan A dan S naik dari 87,1% ke 95,3% dengan 25 angka ini, sementara
    ketepatan 26 huruf hampir tidak berubah (63,5% ke 62,7%).

    Urutan tangan dari MediaPipe tidak tetap antar frame, jadi tangan diurutkan
    dulu dari kiri ke kanan layar menurut letak pergelangannya. Kalau tidak,
    kolom "tangan kiri" kadang berisi tangan kanan dan model dibuat bingung.

    Tetap dihitung dari koordinat MENTAH (sebelum dinormalkan), diskalakan
    dengan ukuran tangan kiri, atau tangan yang terlihat kalau cuma satu.
    Kalau cuma satu tangan yang terlihat, hasilnya nol semua.

    Urutan isinya: kiri[jempol] ke kanan[jempol..kelingking], lalu
    kiri[telunjuk] ke kanan[jempol..kelingking], dan seterusnya.
    """
    banyak = len(mentah_titik)
    hasil = np.zeros((banyak, len(UJUNG_JARI) ** 2), dtype=np.float32)

    ada = np.abs(mentah_titik).sum(axis=(2, 3)) > 1e-9          # (n, 2)
    dua_tangan = ada[:, 0] & ada[:, 1]
    if not dua_tangan.any():
        return hasil

    # Urutkan kiri-kanan. Hanya baris yang memang dua tangan yang ditukar.
    tukar = dua_tangan & (mentah_titik[:, 0, 0, 0] > mentah_titik[:, 1, 0, 0])
    urut = mentah_titik.copy()
    urut[tukar] = mentah_titik[tukar][:, ::-1]
    kiri, kanan = urut[:, 0], urut[:, 1]

    ukuran = np.linalg.norm(kiri[:, 9] - kiri[:, 0], axis=1)
    ukuran[ukuran < 1e-6] = 1.0

    kolom = [np.linalg.norm(kiri[:, a] - kanan[:, b], axis=-1) / ukuran
             for a in UJUNG_JARI for b in UJUNG_JARI]
    hasil[dua_tangan] = np.stack(kolom, axis=1)[dua_tangan]
    return hasil


def ambil_fitur(mentah: np.ndarray) -> np.ndarray:
    """Ubah titik mentah jadi 197 angka siap pakai.

    Menerima satu sampel (126 angka) maupun banyak sampel sekaligus (n, 126).
    Keluarannya selalu berbentuk (n, 197).
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

    # Letak relatif dan titik sentuh dihitung LEBIH DULU, selagi koordinatnya
    # masih mentah. Setelah dinormalkan, keterangan itu sudah hilang dan tidak
    # bisa dipulihkan.
    relatif = _posisi_relatif(mentah_titik)
    sentuh = _titik_sentuh_silang(mentah_titik)
    silang = _jarak_ujung_silang(mentah_titik)

    titik = _normalkan(mentah_titik)
    bagian = [titik.reshape(len(mentah), -1)] + _ukuran_geometris(titik) + [relatif, sentuh, silang]

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
