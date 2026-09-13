"""Memutar ulang rekaman sesi dengan model lain, untuk membandingkan secara adil.

KENAPA PERLU
------------
Membandingkan dua model dengan cara mencobanya bergantian di depan kamera itu
tidak adil. Peragaan tangan mustahil sama persis dua kali, jadi perbedaan
hasilnya bisa jadi karena peragaannya yang berbeda, bukan karena modelnya.

Sejak log sesi ikut menyimpan koordinat mentah, satu rekaman yang sama bisa
disuapkan ke model mana pun. Peragaannya identik sampai ke angka terakhir,
sehingga perbedaan hasilnya murni berasal dari modelnya.

CARA PAKAI
----------
    python putar_ulang.py log/sesi_20260913_122849.csv
    python putar_ulang.py log/sesi_*.csv --model rf_bisindo_v3.pkl
    python putar_ulang.py log/sesi_*.csv --benar A

Pakai --benar kalau Anda tahu huruf apa yang sedang diperagakan, supaya
ketepatannya ikut dihitung.

Berkas sesi dari versi lama yang belum menyimpan koordinat tidak bisa diputar
ulang, hanya bisa dibandingkan tebakannya saja.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

FOLDER = Path(__file__).resolve().parent


def baca_sesi(path: str):
    """Ambil koordinat mentah dari berkas sesi. None kalau tidak tersimpan."""
    import pandas as pd

    df = pd.read_csv(path)

    # Harus huruf k diikuti angka saja. Kalau cuma dicek awalannya, kolom
    # "keyakinan" ikut terjaring dan jumlahnya jadi meleset.
    kolom = [k for k in df.columns if re.fullmatch(r"k\d+", k)]
    if len(kolom) != 126:
        return df, None

    # Urutkan menurut angkanya, bukan menurut abjad, supaya k9 tidak
    # mendahului k10.
    kolom.sort(key=lambda k: int(k[1:]))
    return df, df[kolom].values.astype(np.float32)


def jalankan(model, titik, df, ambang, arg):
    """Suapkan tiap frame ke model, saring lewat voting, kumpulkan hurufnya."""
    import gesture_voice as gv
    from fitur import ambil_fitur, sesuaikan_untuk

    fitur = sesuaikan_untuk(ambil_fitur(titik), model.n_features_in_)
    peluang = model.predict_proba(fitur)
    pilih = peluang.argmax(axis=1)
    tebakan = model.classes_[pilih]
    keyakinan = peluang[np.arange(len(pilih)), pilih]

    voting = gv.VotingHuruf(arg.jendela_voting, arg.frame_konfirmasi,
                            arg.min_keyakinan, arg.jeda, ambang_huruf=ambang)
    huruf = []
    for t, k, detik in zip(tebakan, keyakinan, df["detik"].values):
        hasil = voting.masukkan(str(t), float(k), float(detik))
        if hasil:
            huruf.append(hasil)
    return tebakan, keyakinan, "".join(huruf)


def laporkan(nama, tebakan, keyakinan, teks, benar: str | None):
    print(f"\n--- {nama} ---")
    print(f"  teks yang keluar : \"{teks}\"")

    if benar:
        tepat = int((tebakan == benar).sum())
        print(f"  frame tebakannya {benar} : {tepat}/{len(tebakan)} "
              f"({tepat / len(tebakan) * 100:.1f}%)")
        if teks:
            cocok = teks.count(benar)
            print(f"  huruf yang masuk : {cocok} benar, {len(teks) - cocok} salah")
        else:
            print("  huruf yang masuk : tidak ada satu pun")

    hitung: dict[str, int] = {}
    for t in tebakan:
        hitung[str(t)] = hitung.get(str(t), 0) + 1
    urut = sorted(hitung.items(), key=lambda x: -x[1])[:4]
    ringkas = ", ".join(f"{h} {n}x" for h, n in urut)
    print(f"  tebakan tersering : {ringkas}")
    print(f"  keyakinan         : rata {keyakinan.mean():.3f}, "
          f"maks {keyakinan.max():.3f}")


def utama() -> int:
    p = argparse.ArgumentParser(
        description="Memutar ulang rekaman sesi dengan model lain")
    p.add_argument("sesi", nargs="+", help="berkas log/sesi_*.csv")
    p.add_argument("--model", nargs="*", default=None,
                   help="berkas model yang dibandingkan, bawaannya v2 dan v3")
    p.add_argument("--benar", default=None,
                   help="huruf yang sebenarnya diperagakan, untuk menghitung ketepatan")
    p.add_argument("--jendela-voting", type=int, default=12)
    p.add_argument("--frame-konfirmasi", type=int, default=8)
    p.add_argument("--min-keyakinan", type=float, default=0.45)
    p.add_argument("--jeda", type=float, default=1.2)
    arg = p.parse_args()

    import joblib

    daftar_sesi = []
    for pola in arg.sesi:
        daftar_sesi.extend(sorted(glob.glob(pola)))
    if not daftar_sesi:
        print("Berkas sesi tidak ditemukan.")
        return 1

    daftar_model = arg.model or ["rf_bisindo_v2.pkl", "rf_bisindo_v3.pkl"]
    daftar_model = [m for m in daftar_model if (FOLDER / m).exists()]
    if not daftar_model:
        print("Tidak ada berkas model yang bisa dipakai.")
        return 1

    berkas_ambang = FOLDER / "ambang_huruf.json"
    ambang = json.loads(berkas_ambang.read_text(encoding="utf-8"))["ambang"] \
        if berkas_ambang.exists() else {}

    for path in daftar_sesi:
        df, titik = baca_sesi(path)
        print(f"\n{'=' * 66}")
        print(f"{Path(path).name}  -  {len(df)} frame")
        print("=" * 66)

        if titik is None:
            print("  Koordinat mentah tidak tersimpan di berkas ini, jadi tidak")
            print("  bisa diputar ulang. Rekam sesi baru dengan versi terbaru.")
            continue

        for nama_model in daftar_model:
            model = joblib.load(FOLDER / nama_model)
            tebakan, keyakinan, teks = jalankan(model, titik, df, ambang, arg)
            laporkan(f"{nama_model} ({model.n_features_in_} fitur)",
                     tebakan, keyakinan, teks, arg.benar)

    if arg.benar is None:
        print("\nPakai --benar HURUF kalau Anda tahu huruf apa yang diperagakan,")
        print("supaya ketepatannya ikut dihitung.")
    return 0


if __name__ == "__main__":
    sys.exit(utama())
