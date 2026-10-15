"""Merekam peragaan huruf Anda sendiri, untuk melatih model dengan tangan dan kamera Anda.

KENAPA PERLU
------------
Dataset bawaan dibuat orang lain dengan kamera lain. Pengujian langsung
memperlihatkan hasilnya tidak selalu cocok dengan peragaan Anda. Rekaman ini
memakai kamera yang sama, jarak yang sama, dan cara bertangan yang sama dengan
saat program dipakai, jadi paling mendekati keadaan sebenarnya.

CARA PAKAI
----------
    python rekam.py A                  rekam huruf A (300 frame)
    python rekam.py A --jumlah 500     rekam lebih banyak
    python rekam.py A --wajib-dua-tangan
                                       frame yang cuma terlihat satu tangan
                                       dilewati, cocok untuk huruf dua tangan

Di jendela kamera:
    SPASI   mulai / jeda merekam
    Q       selesai

Rekam beberapa kali (sesi), tiap kali dengan posisi, jarak, dan sudut yang
sedikit berbeda. Tiap sesi dihitung sebagai satu kelompok saat pengujian,
supaya angka ketepatannya jujur. Hasilnya ditambahkan ke
data/data_pengguna.csv dan langsung dipakai oleh latih.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

warnings.filterwarnings("ignore", message=".*GetPrototype.*")

FOLDER = Path(__file__).resolve().parent
BERKAS_PENGGUNA = FOLDER / "data" / "data_pengguna.csv"
JUMLAH_ANGKA = 126
HURUF_VALID = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

HIJAU, MERAH, PUTIH, KUNING, GELAP = (
    (80, 220, 80), (60, 60, 230), (255, 255, 255), (60, 200, 255), (30, 30, 30),
)


def vektor_titik_tangan(hasil) -> tuple[np.ndarray | None, int]:
    """Sama persis dengan gesture_voice.py: 126 angka, tangan kosong diisi nol."""
    if not hasil.multi_hand_landmarks:
        return None, 0
    angka: list[float] = []
    for tangan in hasil.multi_hand_landmarks[:2]:
        for titik in tangan.landmark:
            angka.extend((titik.x, titik.y, titik.z))
    jumlah = len(hasil.multi_hand_landmarks[:2])
    if jumlah == 1:
        angka.extend([0.0] * 63)
    return np.asarray(angka, dtype=np.float32), jumlah


def simpan(baris: list[list[float]], huruf: str, sesi: str, berkas: Path) -> None:
    berkas.parent.mkdir(exist_ok=True)
    baru = not berkas.exists()
    with open(berkas, "a", newline="", encoding="utf-8") as f:
        tulis = csv.writer(f)
        if baru:
            tulis.writerow(["label", "sesi"] + [f"k{i}" for i in range(JUMLAH_ANGKA)])
        for angka in baris:
            tulis.writerow([huruf, sesi] + [f"{v:.6f}" for v in angka])


def hitung_tersimpan(berkas: Path, huruf: str) -> int:
    if not berkas.exists():
        return 0
    with open(berkas, encoding="utf-8") as f:
        return sum(1 for b in f if b.startswith(huruf + ","))


def teks(frame, isi: str, y: int, warna=PUTIH) -> None:
    cv2.putText(frame, isi, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GELAP, 4)
    cv2.putText(frame, isi, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna, 2)


def utama() -> int:
    from gesture_voice import muat_pengaturan_deteksi

    pengaturan_deteksi = muat_pengaturan_deteksi()
    p = argparse.ArgumentParser(description="Merekam peragaan huruf BISINDO Anda sendiri")
    p.add_argument("huruf", help="huruf yang direkam, A sampai Z")
    p.add_argument("--jumlah", type=int, default=300, help="jumlah frame yang direkam")
    p.add_argument("--kamera", type=int, default=0)
    p.add_argument("--keyakinan-deteksi", type=float,
                   default=pengaturan_deteksi["min_detection_confidence"],
                   help="ambang MediaPipe untuk mulai mendeteksi tangan")
    p.add_argument("--keyakinan-tracking", type=float,
                   default=pengaturan_deteksi["min_tracking_confidence"],
                   help="ambang MediaPipe untuk terus mengikuti tangan")
    p.add_argument("--wajib-dua-tangan", action="store_true",
                   help="lewati frame yang cuma terlihat satu tangan")
    p.add_argument("--keluaran", type=Path, default=BERKAS_PENGGUNA)
    arg = p.parse_args()

    huruf = arg.huruf.strip().upper()
    if len(huruf) != 1 or huruf not in HURUF_VALID:
        print(f"Huruf harus satu karakter A sampai Z, bukan '{arg.huruf}'.")
        return 1

    sesi = datetime.now().strftime("%Y%m%d_%H%M%S")
    sebelumnya = hitung_tersimpan(arg.keluaran, huruf)

    kamera = (cv2.VideoCapture(arg.kamera, cv2.CAP_DSHOW) if sys.platform == "win32"
              else cv2.VideoCapture(arg.kamera))
    if not kamera.isOpened():
        print(f"Kamera {arg.kamera} tidak bisa dibuka.")
        return 1

    mp_tangan, mp_gambar = mp.solutions.hands, mp.solutions.drawing_utils
    pendeteksi = mp_tangan.Hands(
        max_num_hands=2,
        min_detection_confidence=arg.keyakinan_deteksi,
        min_tracking_confidence=arg.keyakinan_tracking,
    )

    baris: list[list[float]] = []
    merekam = False
    terlewat = 0
    print(f"Merekam huruf {huruf}, target {arg.jumlah} frame. "
          f"Tekan SPASI untuk mulai, Q untuk selesai.")

    try:
        while True:
            ok, frame = kamera.read()
            if not ok:
                print("Gagal membaca kamera.")
                break
            frame = cv2.flip(frame, 1)
            hasil = pendeteksi.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            angka, jumlah_tangan = vektor_titik_tangan(hasil)

            if hasil.multi_hand_landmarks:
                for tangan in hasil.multi_hand_landmarks:
                    mp_gambar.draw_landmarks(frame, tangan, mp_tangan.HAND_CONNECTIONS)

            if merekam and angka is not None:
                if arg.wajib_dua_tangan and jumlah_tangan < 2:
                    terlewat += 1
                else:
                    baris.append(angka.tolist())

            teks(frame, f"Huruf {huruf}   tangan terlihat: {jumlah_tangan}", 30,
                 HIJAU if jumlah_tangan else MERAH)
            teks(frame, f"{len(baris)} / {arg.jumlah} frame"
                        + (f"   (dilewati {terlewat})" if terlewat else ""), 62, KUNING)
            teks(frame, "MEREKAM" if merekam else "JEDA - tekan SPASI", 94,
                 MERAH if merekam else PUTIH)
            cv2.imshow("Rekam huruf", frame)

            tombol = cv2.waitKey(1) & 0xFF
            if tombol in (ord("q"), ord("Q"), 27):
                break
            if tombol == ord(" "):
                merekam = not merekam
            if len(baris) >= arg.jumlah:
                break
    finally:
        kamera.release()
        pendeteksi.close()
        cv2.destroyAllWindows()

    if not baris:
        print("Tidak ada frame yang tersimpan.")
        return 1

    simpan(baris, huruf, sesi, arg.keluaran)
    print(f"Tersimpan {len(baris)} frame huruf {huruf} (sesi {sesi}) "
          f"di {arg.keluaran.relative_to(FOLDER)}.")
    print(f"Total huruf {huruf} milik Anda sekarang: {sebelumnya + len(baris)} frame.")
    print("Rekam lagi dengan posisi berbeda, atau jalankan  python latih.py  untuk melatih.")
    return 0


if __name__ == "__main__":
    sys.exit(utama())
