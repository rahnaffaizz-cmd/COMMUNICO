"""Memeriksa tiap huruf satu per satu, lalu mencari ambang keyakinan terbaiknya.

KENAPA PERLU AMBANG PER HURUF
-----------------------------
Program memakai satu ambang keyakinan untuk semua huruf (--min-keyakinan).
Padahal tiap huruf perilakunya jauh berbeda:

  G  hampir selalu benar dan yakin        -> ambang rendah pun aman
  I  tidak pernah benar, selalu kalah J   -> ambang berapa pun percuma
  T  kadang benar kadang tidak            -> perlu ambang tinggi biar aman

Kalau ambangnya disamakan, huruf yang mudah jadi terlalu sulit diterima,
sementara huruf yang sulit tetap lolos padahal salah.

APA YANG DICARI
---------------
Untuk mengeja, huruf yang SALAH lebih merugikan daripada huruf yang tidak
muncul. Kalau tidak muncul, pengguna tinggal mengulang. Kalau salah masuk,
dia harus menghapus dulu, dan kalau tidak sadar, kata yang diucapkan jadi
keliru. Jadi yang dikejar di sini ketepatan (presisi), bukan banyak-banyakan
huruf diterima.

Untuk tiap huruf dicari ambang terendah yang membuat tebakan huruf itu benar
minimal sekian persen (--target-presisi, bawaannya 80%). Kalau tidak ada
ambang yang sanggup, huruf itu ditandai untuk ditolak saja.

CARA PAKAI
----------
    python diagnosa.py                     periksa semua huruf, tulis ambang
    python diagnosa.py --target-presisi 90 minta lebih ketat
    python diagnosa.py --huruf I J T       periksa huruf tertentu saja

Hasilnya ditulis ke ambang_huruf.json, yang langsung dipakai gesture_voice.py.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from evaluasi import kelompokkan_per_foto, siapkan_data  # noqa: E402
from fitur import ambil_fitur  # noqa: E402

FOLDER = Path(__file__).resolve().parent
BERKAS_AMBANG = FOLDER / "ambang_huruf.json"

# Kalau bahkan pada ambang tertinggi pun presisinya tidak tercapai, huruf itu
# lebih baik ditolak daripada menyesatkan.
AMBANG_MENYERAH = 0.99


def judul(teks: str) -> None:
    print(f"\n{'=' * 70}\n{teks}\n{'=' * 70}")


def kumpulkan_tebakan(X, y, grup):
    """Latih 5 kali dengan pembagian per foto, kumpulkan semua tebakan apa adanya.

    Yang dikumpulkan: untuk tiap sampel, huruf tebakan model, keyakinannya, dan
    huruf yang sebenarnya. Semua sampel pasti kebagian jadi data uji sekali.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    tebakan = np.empty(len(y), dtype=object)
    keyakinan = np.zeros(len(y))

    for nomor, (latih, uji) in enumerate(GroupKFold(5).split(X, y, groups=grup), 1):
        print(f"  melatih lipatan {nomor} dari 5 ...")
        model = RandomForestClassifier(n_estimators=300, random_state=42,
                                       n_jobs=-1).fit(X[latih], y[latih])
        peluang = model.predict_proba(X[uji])
        pilih = peluang.argmax(axis=1)
        tebakan[uji] = model.classes_[pilih]
        keyakinan[uji] = peluang[np.arange(len(pilih)), pilih]

    return tebakan, keyakinan


def cari_ambang(benar_semua: np.ndarray, yakin: np.ndarray,
                target: float) -> tuple[float, float, float]:
    """Cari ambang terendah yang presisinya mencapai target.

    benar_semua: untuk tiap tebakan huruf ini, apakah tebakannya benar
    yakin      : keyakinan model pada tebakan itu

    Balik: (ambang, presisi yang dicapai, berapa bagian tebakan yang tersisa)
    """
    if len(yakin) == 0:
        return AMBANG_MENYERAH, 0.0, 0.0

    for ambang in np.round(np.arange(0.20, 1.00, 0.01), 2):
        lolos = yakin >= ambang
        if lolos.sum() == 0:
            break
        presisi = benar_semua[lolos].mean()
        if presisi >= target:
            return round(float(ambang), 2), float(presisi), float(lolos.mean())

    # Tidak ada ambang yang sanggup mencapai target.
    return AMBANG_MENYERAH, float(benar_semua.mean()), 0.0


def utama() -> int:
    p = argparse.ArgumentParser(
        description="Memeriksa tiap huruf dan mencari ambang keyakinan terbaiknya")
    p.add_argument("--target-presisi", type=float, default=80.0,
                   help="berapa persen tebakan huruf harus benar (bawaan 80)")
    p.add_argument("--huruf", nargs="*", default=None,
                   help="periksa huruf tertentu saja, misalnya --huruf I J T")
    p.add_argument("--tanpa-tulis", action="store_true",
                   help="cuma tampilkan, jangan tulis ambang_huruf.json")
    arg = p.parse_args()
    target = arg.target_presisi / 100

    X_mentah, y = siapkan_data()
    grup = kelompokkan_per_foto(X_mentah, y)
    X = ambil_fitur(X_mentah)
    print(f"Data: {len(X)} sampel, {len(np.unique(y))} huruf, "
          f"{len(set(grup))} kelompok foto\n")

    judul("MENGUMPULKAN TEBAKAN")
    print("  Tiap sampel diuji memakai model yang belum pernah melihat fotonya.")
    tebakan, yakin = kumpulkan_tebakan(X, y, grup)

    daftar_huruf = sorted(np.unique(y))
    if arg.huruf:
        daftar_huruf = [h.upper() for h in arg.huruf]

    judul(f"HASIL PER HURUF  (target presisi {arg.target_presisi:.0f}%)")
    print("huruf | akurasi | yakin saat  | yakin saat | ambang | presisi | huruf")
    print("      |  apa    |   BENAR     |   SALAH    | usulan | setelah | tersisa")
    print("      |  adanya | rata  teng  | rata  teng |        | ambang  |")
    print("------+---------+-------------+------------+--------+---------+--------")

    ambang_akhir = {}
    catatan_khusus = []

    for huruf in daftar_huruf:
        asli = y == huruf                       # sampel yang sebenarnya huruf ini
        ditebak = tebakan == huruf              # sampel yang DITEBAK huruf ini

        akurasi = (tebakan[asli] == huruf).mean() if asli.sum() else 0.0

        # Keyakinan saat model menebak huruf ini dan ternyata benar / salah.
        benar_ditebak = (y[ditebak] == huruf) if ditebak.sum() else np.array([])
        yakin_ditebak = yakin[ditebak] if ditebak.sum() else np.array([])

        yb = yakin_ditebak[benar_ditebak] if len(benar_ditebak) else np.array([])
        ys = yakin_ditebak[~benar_ditebak] if len(benar_ditebak) else np.array([])

        ambang, presisi, sisa = cari_ambang(benar_ditebak, yakin_ditebak, target)
        ambang_akhir[huruf] = ambang

        def ringkas(a):
            return f"{a.mean():.2f} {np.median(a):.2f}" if len(a) else " -    - "

        tanda = ""
        if ambang >= AMBANG_MENYERAH:
            tanda = "  <- TOLAK SAJA"
            catatan_khusus.append((huruf, "tidak ada ambang yang sanggup"))
        elif sisa < 0.3:
            tanda = "  <- jarang lolos"

        print(f"  {huruf}   | {akurasi * 100:6.1f}% | {ringkas(yb):11s} | "
              f"{ringkas(ys):10s} |  {ambang:.2f}  | {presisi * 100:5.1f}%  | "
              f"{sisa * 100:5.1f}%{tanda}")

    judul("CATATAN")
    print("  'yakin saat BENAR' vs 'yakin saat SALAH' adalah kuncinya.")
    print("  Kalau kedua angka itu berjauhan, ambang bisa memisahkan keduanya.")
    print("  Kalau berdekatan, model sendiri tidak bisa membedakan benar dan")
    print("  salah, jadi ambang setinggi apa pun tidak menolong.\n")

    if catatan_khusus:
        print("  Huruf yang sebaiknya ditolak saja:")
        for huruf, sebab in catatan_khusus:
            print(f"    {huruf}: {sebab}")
        print()
        print("  Huruf ini tidak akan pernah masuk ke teks. Lebih baik begitu")
        print("  daripada memasukkan huruf yang salah tanpa disadari pengguna.")

    if not arg.tanpa_tulis:
        isi = {
            "target_presisi": arg.target_presisi,
            "ambang": ambang_akhir,
            "keterangan": ("Ambang keyakinan per huruf, dibuat oleh diagnosa.py. "
                           f"Nilai {AMBANG_MENYERAH} berarti huruf itu ditolak."),
        }
        BERKAS_AMBANG.write_text(json.dumps(isi, indent=2), encoding="utf-8")
        print(f"\n  Ambang ditulis ke {BERKAS_AMBANG.name}.")
        print("  Program utama akan otomatis memakainya.")
    return 0


if __name__ == "__main__":
    raise SystemExit(utama())
