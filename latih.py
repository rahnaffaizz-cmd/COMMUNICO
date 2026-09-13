"""Melatih ulang model pengenal huruf, lalu membandingkannya dengan model lama.

Model tidak akan ditulis ke berkas kalau ternyata lebih buruk dari model lama.
Pembandingnya memakai pembagian per foto asli, bukan pembagian acak, supaya
angkanya jujur. Penjelasan lengkap soal ini ada di evaluasi.py.

    python latih.py                     latih, bandingkan, simpan kalau lebih baik
    python latih.py --paksa             tetap simpan walaupun lebih buruk
    python latih.py --keluaran lain.pkl simpan ke nama berkas lain

Fitur yang dipakai diambil dari fitur.py, berkas yang sama dengan yang dipakai
program utama saat berjalan. Ini penting: kalau keduanya berbeda hitungan,
model akan menebak ngawur walaupun angka latihannya bagus.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from evaluasi import kelompokkan_per_foto, siapkan_data  # noqa: E402
from fitur import ambil_fitur  # noqa: E402

FOLDER = Path(__file__).resolve().parent
MODEL_LAMA = FOLDER / "rf_bisindo_99.pkl"
MODEL_BARU = FOLDER / "rf_bisindo_v2.pkl"

# Hasil pengukuran model lama, dipakai sebagai ambang kelulusan.
ACUAN_LAMA = 46.69


def judul(teks: str) -> None:
    print(f"\n{'=' * 62}\n{teks}\n{'=' * 62}")


def buat_model():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)


def ukur(X, y, grup, nama: str):
    """Latih dan uji 5 kali dengan pembagian per foto, lalu rata-ratakan."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold

    akurasi, f1, benar, total = [], [], {}, {}
    for nomor, (latih, uji) in enumerate(GroupKFold(5).split(X, y, groups=grup), 1):
        model = buat_model().fit(X[latih], y[latih])
        tebak = model.predict(X[uji])

        akurasi.append((tebak == y[uji]).mean())
        f1.append(f1_score(y[uji], tebak, average="macro"))
        print(f"    lipatan {nomor}: {akurasi[-1] * 100:6.2f}%")

        for huruf in np.unique(y[uji]):
            pilih = y[uji] == huruf
            benar[huruf] = benar.get(huruf, 0) + (tebak[pilih] == huruf).sum()
            total[huruf] = total.get(huruf, 0) + pilih.sum()

    rata = float(np.mean(akurasi))
    print(f"  {nama}: {rata * 100:6.2f}% (simpangan {np.std(akurasi) * 100:.2f}), "
          f"macro F1 {np.mean(f1) * 100:.2f}%")
    return rata, float(np.mean(f1)), benar, total


def utama() -> int:
    p = argparse.ArgumentParser(description="Melatih ulang model pengenal huruf BISINDO")
    p.add_argument("--keluaran", type=Path, default=MODEL_BARU,
                   help="nama berkas untuk menyimpan model")
    p.add_argument("--paksa", action="store_true",
                   help="tetap simpan walaupun hasilnya lebih buruk dari model lama")
    arg = p.parse_args()

    import joblib

    X_mentah, y = siapkan_data()
    grup = kelompokkan_per_foto(X_mentah, y)
    print(f"Data: {len(X_mentah)} sampel, {len(np.unique(y))} huruf, "
          f"{len(set(grup))} kelompok foto")

    X_baru = ambil_fitur(X_mentah)
    print(f"Fitur: {X_mentah.shape[1]} angka mentah -> {X_baru.shape[1]} angka siap pakai")

    judul("MENGUKUR FITUR LAMA (koordinat mentah)")
    lama, f1_lama, _, _ = ukur(X_mentah, y, grup, "rata-rata")

    judul("MENGUKUR FITUR BARU (dinormalkan + ukuran jari)")
    baru, f1_baru, benar, total = ukur(X_baru, y, grup, "rata-rata")

    judul("PERBANDINGAN")
    print(f"  fitur lama : {lama * 100:6.2f}%   macro F1 {f1_lama * 100:6.2f}%")
    print(f"  fitur baru : {baru * 100:6.2f}%   macro F1 {f1_baru * 100:6.2f}%")
    selisih = (baru - lama) * 100
    tanda = "naik" if selisih > 0 else "TURUN"
    print(f"  selisih    : {selisih:+6.2f} poin persen  ({tanda})")

    print("\n  Ketepatan tiap huruf dengan fitur baru:")
    for nilai, huruf, jumlah in sorted((benar[h] / total[h], h, total[h]) for h in total):
        batang = "#" * int(nilai * 30)
        print(f"    {huruf}: {nilai * 100:6.2f}%  {batang:<30} ({jumlah})")

    if baru <= lama and not arg.paksa:
        judul("MODEL TIDAK DISIMPAN")
        print("  Fitur baru tidak mengalahkan yang lama, jadi tidak ada gunanya diganti.")
        print("  Pakai --paksa kalau tetap ingin menyimpannya.")
        return 1

    judul("MELATIH MODEL AKHIR")
    print("  Kali ini memakai seluruh data, bukan dibagi-bagi lagi.")
    model = buat_model().fit(X_baru, y)
    joblib.dump(model, arg.keluaran, compress=3)
    ukuran = arg.keluaran.stat().st_size / 1048576
    print(f"  Disimpan: {arg.keluaran.name} ({ukuran:.1f} MB)")

    judul("LANGKAH BERIKUTNYA")
    print(f"  1. Program utama sudah otomatis memakai {arg.keluaran.name}")
    print("     karena keduanya sama-sama memanggil fitur.py.")
    print("  2. Coba dulu tanpa kamera:  python uji_gambar.py")
    print("  3. Kalau sudah cocok, perbarui angka acuan di evaluasi.py")
    print(f"     menjadi {baru * 100:.2f}%, lalu commit.")
    return 0


if __name__ == "__main__":
    sys.exit(utama())
