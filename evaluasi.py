"""Mengukur ketepatan model secara jujur, bukan asal angka besar.

KENAPA SKRIP INI ADA
--------------------
Model bawaan program ini (rf_bisindo_99.pkl) namanya mengandung "99" karena
pembuatnya mengukur ketepatan 99%. Angka itu benar secara hitungan, tapi
menyesatkan, dan penyebabnya begini:

Dataset aslinya cuma 312 foto. Foto-foto itu diperbanyak lewat augmentasi
(diputar sedikit, digeser, diubah terangnya) sampai jadi 14.493 sampel, jadi
ada sekitar 46 salinan untuk tiap foto asli. Barulah setelah itu data dibagi
secara acak menjadi bagian latih dan bagian uji.

Masalahnya, pembagian acak membuat salinan-salinan dari satu foto yang sama
tersebar ke dua sisi. Sebuah sampel uji hampir selalu punya belasan sampai
puluhan "saudara kembar" di data latih. Jadi waktu diuji, model tidak sedang
diuji kemampuannya mengenali isyarat baru, melainkan kemampuannya mengingat
foto yang sudah pernah dilihat. Ujian yang soalnya sudah bocor.

Skrip ini mengukur dengan cara yang benar: salinan-salinan dikembalikan dulu
ke kelompok foto asalnya, lalu pembagian dilakukan per kelompok. Satu foto
beserta seluruh salinannya hanya boleh ada di satu sisi. Dengan begitu model
benar-benar diuji pada foto yang belum pernah dilihat.

Selisih hasilnya besar sekali: 99% turun menjadi sekitar 46%.

CARA PAKAI
----------
    python evaluasi.py                     tolok ukur lengkap, dua cara dibandingkan
    python evaluasi.py --cepat             1 lipatan saja, untuk cek sekilas
    python evaluasi.py --model baru.pkl    nilai satu berkas model pada ujian jujur

Data uji diunduh otomatis dari repositori asal (sekitar 23 MB) dan disimpan di
folder data/, jadi pengunduhan hanya terjadi sekali.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

FOLDER = Path(__file__).resolve().parent
FOLDER_DATA = FOLDER / "data"
BERKAS_FITUR = FOLDER_DATA / "bisindo_features.csv"
SIMPANAN_GRUP = FOLDER_DATA / "kelompok_foto.npz"

ALAMAT_FITUR = (
    "https://raw.githubusercontent.com/KrisnaSantosa15/"
    "realtime-bisindo-classification/main/data/bisindo_features.csv"
)

# Salinan augmentasi dari satu foto jaraknya sekitar 0,08 satu sama lain,
# sedangkan huruf yang berbeda berjarak sekitar 2,7. Ambang 1,0 memisahkan
# keduanya dengan aman dan menghasilkan ~269 kelompok, mendekati 312 foto asli.
AMBANG_KELOMPOK = 1.0

# Angka acuan hasil pengukuran sebelumnya, dipakai sebagai pembanding.
ACUAN_ACAK = 99.55
ACUAN_JUJUR = 45.84


def judul(teks: str) -> None:
    print(f"\n{'=' * 62}\n{teks}\n{'=' * 62}")


# ------------------------------------------------------------------------ data
def siapkan_data():
    """Unduh berkas fitur kalau belum ada, lalu baca isinya."""
    import pandas as pd

    if not BERKAS_FITUR.exists():
        FOLDER_DATA.mkdir(exist_ok=True)
        print(f"Mengunduh data uji (~23 MB) ke {BERKAS_FITUR.name} ...")
        sementara = BERKAS_FITUR.with_suffix(".sedang-diunduh")
        try:
            urllib.request.urlretrieve(ALAMAT_FITUR, sementara)
            sementara.replace(BERKAS_FITUR)
        except Exception as e:
            sementara.unlink(missing_ok=True)
            print(f"Gagal mengunduh: {e}")
            print("Periksa koneksi internet, lalu coba lagi.")
            sys.exit(1)
        print("Selesai diunduh.\n")

    df = pd.read_csv(BERKAS_FITUR)
    y = df["label"].values
    # Tangan kedua yang tidak terlihat ditandai kosong di berkas ini, sedangkan
    # program utama mengisinya dengan nol. Disamakan supaya adil.
    X = df.drop(columns=["label"]).fillna(0.0).values
    return X, y


def kelompokkan_per_foto(X, y):
    """Kembalikan salinan-salinan augmentasi ke kelompok foto asalnya.

    Hasilnya disimpan supaya pemanggilan berikutnya tidak perlu menghitung ulang.
    """
    if SIMPANAN_GRUP.exists():
        simpanan = np.load(SIMPANAN_GRUP, allow_pickle=True)
        if len(simpanan["grup"]) == len(y):
            return simpanan["grup"]

    from sklearn.cluster import AgglomerativeClustering

    print("Mengelompokkan salinan augmentasi ke foto asalnya (sekali saja) ...")
    grup = np.empty(len(y), dtype=object)
    for huruf in sorted(np.unique(y)):
        pilih = y == huruf
        label = AgglomerativeClustering(
            n_clusters=None, distance_threshold=AMBANG_KELOMPOK, linkage="average"
        ).fit_predict(X[pilih])
        grup[pilih] = [f"{huruf}_{i}" for i in label]

    FOLDER_DATA.mkdir(exist_ok=True)
    np.savez(SIMPANAN_GRUP, grup=grup)
    print(f"Ketemu {len(set(grup))} kelompok foto (aslinya 312).\n")
    return grup


# -------------------------------------------------------------------- penilaian
def buat_model():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


def nilai_dengan_pembagian(X, y, pembagi, judul_cara: str, cepat: bool):
    """Latih dan uji berulang kali, lalu rata-ratakan hasilnya."""
    from sklearn.metrics import f1_score

    daftar_akurasi, daftar_f1 = [], []
    kumpulan_benar, kumpulan_total = {}, {}

    for nomor, (latih, uji) in enumerate(pembagi, start=1):
        model = buat_model().fit(X[latih], y[latih])
        tebakan = model.predict(X[uji])

        akurasi = (tebakan == y[uji]).mean()
        daftar_akurasi.append(akurasi)
        daftar_f1.append(f1_score(y[uji], tebakan, average="macro"))
        print(f"    lipatan {nomor}: {akurasi * 100:6.2f}%")

        for huruf in np.unique(y[uji]):
            pilih = y[uji] == huruf
            kumpulan_benar[huruf] = kumpulan_benar.get(huruf, 0) + (tebakan[pilih] == huruf).sum()
            kumpulan_total[huruf] = kumpulan_total.get(huruf, 0) + pilih.sum()

        if cepat:
            break

    rata = float(np.mean(daftar_akurasi))
    print(f"  {judul_cara}: {rata * 100:6.2f}%"
          f"  (simpangan {np.std(daftar_akurasi) * 100:.2f}), "
          f"macro F1 {np.mean(daftar_f1) * 100:.2f}%")
    return rata, kumpulan_benar, kumpulan_total


def tampilkan_per_huruf(benar: dict, total: dict) -> None:
    print("\nKetepatan tiap huruf, diurutkan dari yang paling parah:")
    baris = sorted((benar[h] / total[h], h, total[h]) for h in total)
    for nilai, huruf, jumlah in baris:
        batang = "#" * int(nilai * 30)
        catatan = "  <- tidak pernah benar" if nilai == 0 else ""
        print(f"  {huruf}: {nilai * 100:6.2f}%  {batang:<30} ({jumlah} sampel){catatan}")

    parah = [h for n, h, _ in baris if n < 0.30]
    if parah:
        print(f"\n  {len(parah)} huruf di bawah 30%: {' '.join(sorted(parah))}")


# ------------------------------------------------------------------- dua mode
def tolok_ukur(X, y, grup, cepat: bool) -> None:
    """Bandingkan cara pengukuran yang keliru dengan cara yang benar."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    lipatan = 5

    judul("CARA 1 - pembagian acak (cara yang dipakai pembuat model)")
    print("  Salinan dari satu foto tersebar ke data latih dan data uji.")
    print("  Model tinggal mengingat, tidak perlu benar-benar mengenali.\n")
    acak, _, _ = nilai_dengan_pembagian(
        X, y,
        StratifiedKFold(lipatan, shuffle=True, random_state=42).split(X, y),
        "rata-rata", cepat,
    )

    judul("CARA 2 - pembagian per foto asli (cara yang jujur)")
    print("  Satu foto beserta seluruh salinannya hanya ada di satu sisi.")
    print("  Model diuji pada foto yang benar-benar belum pernah dilihat.\n")
    jujur, benar, total = nilai_dengan_pembagian(
        X, y,
        GroupKFold(lipatan).split(X, y, groups=grup),
        "rata-rata", cepat,
    )

    judul("KESIMPULAN")
    print(f"  pembagian acak      : {acak * 100:6.2f}%   <- angka yang menyesatkan")
    print(f"  pembagian per foto  : {jujur * 100:6.2f}%   <- ketepatan sebenarnya")
    print(f"  selisih             : {(acak - jujur) * 100:6.2f} poin persen")
    print(f"\n  Acuan pengukuran sebelumnya: {ACUAN_ACAK}% dan {ACUAN_JUJUR}%.")
    print("  Kalau angka di atas jauh berbeda, berarti ada yang berubah.")

    tampilkan_per_huruf(benar, total)

    judul("CATATAN KEJUJURAN")
    print("  Angka jujur di atas pun masih terlalu optimis untuk pemakaian nyata,")
    print("  karena semua foto berasal dari orang, ruangan, dan pencahayaan yang")
    print("  sama. Untuk pengguna baru di tempat lain, hasilnya kemungkinan besar")
    print("  lebih rendah lagi.")
    print()
    print("  Yang paling menentukan perbaikan bukan jumlah gambarnya, melainkan")
    print("  jumlah orang dan variasi kondisinya. Menambah 10.000 foto dari orang")
    print("  yang sama tidak akan banyak menolong; 500 foto dari 20 orang berbeda")
    print("  jauh lebih berharga.")


def nilai_satu_model(path_model: Path, X, y, grup) -> None:
    """Uji sebuah berkas model pada foto yang belum pernah dilihatnya."""
    import joblib
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold

    if not path_model.exists():
        print(f"Berkas model tidak ditemukan: {path_model}")
        sys.exit(1)

    model = joblib.load(path_model)
    judul(f"MENILAI {path_model.name}")
    print(f"  jenis  : {type(model).__name__}")
    print(f"  fitur  : {getattr(model, 'n_features_in_', '?')}")
    print(f"  huruf  : {''.join(getattr(model, 'classes_', []))}\n")

    print("  PERINGATAN PENTING")
    print("  Angka di bawah hanya berarti kalau model ini dilatih dengan data LAIN.")
    print("  Kalau model ini dilatih memakai dataset yang sama, seluruh data di")
    print("  sini adalah data latihnya sendiri, jadi hasilnya pasti tinggi dan")
    print("  tidak bermakna apa-apa.\n")

    # Dipisah per kelompok foto supaya sebanding dengan tolok ukur di atas.
    daftar, benar, total = [], {}, {}
    for _, uji in GroupKFold(5).split(X, y, groups=grup):
        tebakan = model.predict(X[uji])
        daftar.append((tebakan == y[uji]).mean())
        for huruf in np.unique(y[uji]):
            pilih = y[uji] == huruf
            benar[huruf] = benar.get(huruf, 0) + (tebakan[pilih] == huruf).sum()
            total[huruf] = total.get(huruf, 0) + pilih.sum()

    seluruh = model.predict(X)
    print(f"  ketepatan seluruh data : {(seluruh == y).mean() * 100:6.2f}%")
    print(f"  macro F1               : {f1_score(y, seluruh, average='macro') * 100:6.2f}%")
    print(f"\n  Pembanding, tolok ukur jujur sebelumnya: {ACUAN_JUJUR}%")

    tampilkan_per_huruf(benar, total)


# ------------------------------------------------------------------ program utama
def utama() -> int:
    p = argparse.ArgumentParser(
        description="Mengukur ketepatan model pengenal huruf BISINDO secara jujur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("CARA PAKAI")[1] if "CARA PAKAI" in __doc__ else None,
    )
    p.add_argument("--model", type=Path,
                   help="nilai satu berkas model .pkl, bukan menjalankan tolok ukur")
    p.add_argument("--cepat", action="store_true",
                   help="cukup 1 lipatan, hasilnya kasar tapi jauh lebih cepat")
    arg = p.parse_args()

    try:
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        print("Butuh pandas dan scikit-learn. Pasang dulu dengan:")
        print("  .venv\\Scripts\\pip install pandas scikit-learn")
        return 1

    X, y = siapkan_data()
    print(f"Data: {len(X)} sampel, {len(np.unique(y))} huruf, {X.shape[1]} fitur")
    grup = kelompokkan_per_foto(X, y)

    if arg.model:
        nilai_satu_model(arg.model, X, y, grup)
    else:
        if arg.cepat:
            print("\nMode cepat: hanya 1 lipatan per cara, hasilnya kasar.")
        else:
            print("\nIni butuh beberapa menit (10 kali pelatihan Random Forest).")
            print("Pakai --cepat kalau cuma ingin gambaran kasar.")
        tolok_ukur(X, y, grup, arg.cepat)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(utama())
