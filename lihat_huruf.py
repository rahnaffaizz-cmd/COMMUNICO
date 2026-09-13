"""Menggambar ulang bentuk tangan tiap huruf dari dataset.

KENAPA PERLU
------------
Dataset yang kita punya isinya cuma angka koordinat titik tangan, bukan foto.
Jadi kalau sebuah huruf sering salah terbaca, kita tidak tahu bentuk seperti
apa sebenarnya yang dipelajari model, dan tidak tahu apakah peragaan kita sudah
mirip atau belum.

Padahal koordinat itu bisa digambar ulang jadi kerangka tangan. Hasilnya
memperlihatkan bentuk yang model harapkan, sehingga peragaan bisa disesuaikan.

    python lihat_huruf.py A            gambar huruf A
    python lihat_huruf.py A S           bandingkan A dengan S
    python lihat_huruf.py --semua       semua 26 huruf sekaligus

Gambarnya disimpan sebagai berkas png di folder bentuk/.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

FOLDER = Path(__file__).resolve().parent
FOLDER_GAMBAR = FOLDER / "bentuk"

# Sambungan antar titik, sama seperti yang dipakai MediaPipe untuk menggambar.
SAMBUNGAN = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # jempol
    (0, 5), (5, 6), (6, 7), (7, 8),            # telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),       # jari tengah
    (9, 13), (13, 14), (14, 15), (15, 16),     # jari manis
    (13, 17), (17, 18), (18, 19), (19, 20),    # kelingking
    (0, 17),                                   # telapak
]


def ambil_tangan(baris: np.ndarray, nomor: int) -> np.ndarray:
    """Potong 126 angka jadi satu tangan berbentuk (21, 3)."""
    return baris[nomor * 63:(nomor + 1) * 63].reshape(21, 3)


def tangan_ada(titik: np.ndarray) -> bool:
    return bool(np.abs(titik).sum() > 1e-9)


def contoh_paling_khas(X: np.ndarray, jumlah: int = 3) -> list[int]:
    """Cari sampel yang paling mewakili, yaitu yang terdekat ke rata-rata."""
    tengah = X.mean(axis=0)
    jarak = np.linalg.norm(X - tengah, axis=1)
    return list(np.argsort(jarak)[:jumlah])


def gambar_satu(sumbu, baris: np.ndarray, judul: str) -> None:
    """Gambar kerangka satu sampel (bisa satu atau dua tangan) di satu kotak."""
    warna = ["#1f77b4", "#d62728"]        # tangan pertama biru, kedua merah

    for nomor in range(2):
        titik = ambil_tangan(baris, nomor)
        if not tangan_ada(titik):
            continue

        # Sumbu y di koordinat gambar menghadap ke bawah, jadi dibalik supaya
        # tangannya tampak tegak seperti aslinya.
        x, y = titik[:, 0], -titik[:, 1]

        for a, b in SAMBUNGAN:
            sumbu.plot([x[a], x[b]], [y[a], y[b]],
                       color=warna[nomor], linewidth=1.8, zorder=1)
        sumbu.scatter(x, y, s=14, color=warna[nomor], zorder=2)
        # Pergelangan ditandai lebih besar supaya arah tangan jelas.
        sumbu.scatter(x[0], y[0], s=55, facecolors="none",
                      edgecolors=warna[nomor], linewidths=1.5, zorder=3)

    sumbu.set_title(judul, fontsize=10)
    sumbu.set_aspect("equal")
    sumbu.axis("off")


def gambar_huruf(huruf: str, X: np.ndarray, y: np.ndarray, per_baris: int = 3):
    """Buat satu berkas gambar untuk satu huruf, dipisah menurut jumlah tangan."""
    import matplotlib
    matplotlib.use("Agg")                 # tidak perlu jendela, langsung ke berkas
    import matplotlib.pyplot as plt

    pilih = X[y == huruf]
    if len(pilih) == 0:
        print(f"  {huruf}: tidak ada sampelnya di dataset")
        return None

    # Pisahkan yang satu tangan dan yang dua tangan, karena beberapa huruf di
    # dataset ini diperagakan dengan dua cara yang berbeda.
    dua_tangan = np.array([tangan_ada(ambil_tangan(b, 1)) for b in pilih])
    kelompok = []
    if (~dua_tangan).sum():
        kelompok.append(("satu tangan", pilih[~dua_tangan]))
    if dua_tangan.sum():
        kelompok.append(("dua tangan", pilih[dua_tangan]))

    total_kotak = sum(min(per_baris, len(k)) for _, k in kelompok)
    gambar, sumbu_semua = plt.subplots(
        len(kelompok), per_baris,
        figsize=(per_baris * 2.6, len(kelompok) * 3.0), squeeze=False)

    for nomor_baris, (nama, data) in enumerate(kelompok):
        bagian = len(data) / len(pilih) * 100
        for kolom, idx in enumerate(contoh_paling_khas(data, per_baris)):
            gambar_satu(sumbu_semua[nomor_baris][kolom], data[idx],
                        f"{nama}  ({bagian:.0f}% sampel)" if kolom == 0 else "")
        # Kotak sisa dikosongkan kalau contohnya kurang dari per_baris.
        for kolom in range(len(contoh_paling_khas(data, per_baris)), per_baris):
            sumbu_semua[nomor_baris][kolom].axis("off")

    gambar.suptitle(f"Huruf {huruf}  —  {len(pilih)} sampel di dataset",
                    fontsize=13, fontweight="bold")
    gambar.tight_layout()

    FOLDER_GAMBAR.mkdir(exist_ok=True)
    path = FOLDER_GAMBAR / f"huruf_{huruf}.png"
    gambar.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(gambar)

    rincian = ", ".join(f"{nama} {len(d) / len(pilih) * 100:.0f}%"
                        for nama, d in kelompok)
    print(f"  {huruf}: {len(pilih)} sampel ({rincian})  ->  bentuk/{path.name}")
    return path


def utama() -> int:
    p = argparse.ArgumentParser(
        description="Menggambar ulang bentuk tangan tiap huruf dari dataset")
    p.add_argument("huruf", nargs="*", help="huruf yang ingin dilihat, misalnya A S")
    p.add_argument("--semua", action="store_true", help="gambar semua 26 huruf")
    p.add_argument("--contoh", type=int, default=3,
                   help="berapa contoh ditampilkan tiap jenis")
    arg = p.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Butuh matplotlib. Pasang dengan:")
        print("  .venv\\Scripts\\pip install matplotlib")
        return 1

    from evaluasi import siapkan_data

    X, y = siapkan_data()
    print(f"Data: {len(X)} sampel, {len(np.unique(y))} huruf\n")

    daftar = sorted(np.unique(y)) if arg.semua else [h.upper() for h in arg.huruf]
    if not daftar:
        print("Sebutkan hurufnya, misalnya:  python lihat_huruf.py A")
        print("Atau pakai --semua untuk menggambar semuanya.")
        return 1

    print("Menggambar:")
    for huruf in daftar:
        gambar_huruf(huruf, X, y, per_baris=arg.contoh)

    print(f"\nSemua gambar ada di folder {FOLDER_GAMBAR.name}/")
    print("Titik berlubang menandai pergelangan tangan.")
    print("Tangan pertama digambar biru, tangan kedua merah.")
    return 0


if __name__ == "__main__":
    sys.exit(utama())
