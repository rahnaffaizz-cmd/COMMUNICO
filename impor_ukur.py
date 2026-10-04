"""Mengubah hasil `--ukur` jadi data latih milik Anda sendiri.

Tiap kali `python gesture_voice.py --ukur B` dijalankan, program menyimpan dua
berkas di log/ dengan cap waktu yang sama:
    ukur_B_<waktu>.csv    tebakan model tiap frame, termasuk huruf targetnya
    sesi_<waktu>.csv      koordinat mentah tangan tiap frame

Huruf yang diperagakan sudah pasti (targetnya), jadi pasangan berkas itu bisa
langsung dipakai sebagai data berlabel, tanpa perlu merekam ulang dengan
rekam.py. Hasilnya ditambahkan ke data/data_pengguna.csv, berkas yang sama
yang dibaca latih.py dan latih_cabang.py.

    python impor_ukur.py                 impor semua pasangan yang belum diimpor
    python impor_ukur.py --huruf A B C   hanya huruf tertentu

Aman dijalankan berulang: sesi yang sudah masuk dilewati.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
FOLDER_LOG = FOLDER / "log"
BERKAS_PENGGUNA = FOLDER / "data" / "data_pengguna.csv"
KOLOM = [f"k{i}" for i in range(126)]


def sesi_sudah_ada() -> set[str]:
    if not BERKAS_PENGGUNA.exists():
        return set()
    return set(pd.read_csv(BERKAS_PENGGUNA, usecols=["sesi"])["sesi"].astype(str))


def utama() -> int:
    p = argparse.ArgumentParser(description="Impor hasil --ukur sebagai data latih")
    p.add_argument("--huruf", nargs="*", default=None,
                   help="hanya impor huruf ini, misalnya --huruf A B C")
    arg = p.parse_args()
    pilih = {h.upper() for h in arg.huruf} if arg.huruf else None

    sudah = sesi_sudah_ada()
    baru, dilewati = [], 0
    for berkas in sorted(FOLDER_LOG.glob("ukur_*_*.csv")):
        cocok = re.fullmatch(r"ukur_([A-Z])_(\d{8}_\d{6})\.csv", berkas.name)
        if not cocok:
            continue
        huruf, waktu = cocok.groups()
        if pilih and huruf not in pilih:
            continue

        sesi_id = f"ukur_{waktu}"
        if sesi_id in sudah:
            dilewati += 1
            continue

        pasangan = FOLDER_LOG / f"sesi_{waktu}.csv"
        if not pasangan.exists():
            print(f"  {berkas.name}: tidak ada {pasangan.name}, dilewati")
            continue

        ukur = pd.read_csv(berkas)
        sesi = pd.read_csv(pasangan)
        if len(ukur) != len(sesi) or not (ukur["tebakan"].values == sesi["tebakan"].values).all():
            print(f"  {berkas.name}: baris tidak cocok dengan {pasangan.name}, dilewati")
            continue

        angka = np.nan_to_num(sesi[KOLOM].values.astype(np.float32))
        ada = np.abs(angka).sum(axis=1) > 0
        baru.append((huruf, sesi_id, angka[ada]))
        print(f"  {huruf}: {int(ada.sum())} frame dari {berkas.name} "
              f"({int((sesi['tangan'] == 2).sum())} frame dua tangan)")

    if not baru:
        print(f"Tidak ada sesi baru ({dilewati} sudah pernah diimpor).")
        return 0

    BERKAS_PENGGUNA.parent.mkdir(exist_ok=True)
    perlu_judul = not BERKAS_PENGGUNA.exists()
    with open(BERKAS_PENGGUNA, "a", newline="", encoding="utf-8") as f:
        tulis = csv.writer(f)
        if perlu_judul:
            tulis.writerow(["label", "sesi"] + KOLOM)
        for huruf, sesi_id, angka in baru:
            for baris in angka:
                tulis.writerow([huruf, sesi_id] + [f"{v:.6f}" for v in baris])

    total = sum(len(a) for _, _, a in baru)
    print(f"\nDitambahkan {total} frame dari {len(baru)} sesi ke "
          f"{BERKAS_PENGGUNA.relative_to(FOLDER)}.")
    return 0


if __name__ == "__main__":
    sys.exit(utama())
