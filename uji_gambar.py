"""Uji model pakai file gambar, tanpa perlu menyalakan kamera.

Berguna untuk memastikan model dan MediaPipe masih jalan normal, atau untuk
mengukur seberapa tepat model menebak foto isyarat buatan sendiri.

Cara pakai:
    python uji_gambar.py                    # pakai gambar di folder contoh/
    python uji_gambar.py folder/foto        # semua gambar dalam satu folder
    python uji_gambar.py a.jpg b.png        # file tertentu saja

Kalau nama filenya mengandung satu huruf, misalnya "letter_A.jpg" atau
"B_percobaan.png", huruf itu dianggap jawaban yang benar dan program akan
menghitung berapa persen tebakan modelnya tepat.
"""

import glob
import os
import sys
import warnings

import cv2
import joblib
import mediapipe as mp
import numpy as np

from fitur import ambil_fitur

# Dua peringatan ini cuma berisik di layar, tidak bikin program rusak.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message=".*GetPrototype.*")

PATH_MODEL = "rf_bisindo_v2.pkl"
FOLDER_CONTOH = "contoh"


def kumpulkan_gambar(argumen: list[str]) -> list[str]:
    """Ubah apa pun yang diketik pengguna jadi daftar file gambar."""
    if not argumen:
        return sorted(glob.glob(os.path.join(FOLDER_CONTOH, "*.jpg")))

    daftar: list[str] = []
    for item in argumen:
        if os.path.isdir(item):
            for pola in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                daftar += glob.glob(os.path.join(item, pola))
        else:
            daftar.append(item)
    return sorted(daftar)


def tebak_jawaban_benar(path: str) -> str | None:
    """Cari huruf tunggal di nama file, itu dianggap jawaban yang benar."""
    nama = os.path.splitext(os.path.basename(path))[0]
    for potongan in nama.replace("-", "_").split("_"):
        if len(potongan) == 1 and potongan.isalpha():
            return potongan.upper()
    return None


def vektor_titik_tangan(hasil) -> np.ndarray | None:
    """Sama seperti di gesture_voice.py: ubah titik tangan jadi 126 angka."""
    if not hasil.multi_hand_landmarks:
        return None

    angka: list[float] = []
    for tangan in hasil.multi_hand_landmarks[:2]:
        for titik in tangan.landmark:
            angka.extend((titik.x, titik.y, titik.z))

    if len(hasil.multi_hand_landmarks) == 1:
        angka.extend([0.0] * 63)  # tangan kedua dikosongkan
    return np.asarray(angka, dtype=np.float32).reshape(1, -1)


def utama() -> int:
    daftar_gambar = kumpulkan_gambar(sys.argv[1:])
    if not daftar_gambar:
        print("Tidak ada gambar yang bisa diuji.")
        return 1

    model = joblib.load(PATH_MODEL)
    # static_image_mode=True karena ini foto diam, bukan video.
    pendeteksi = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    )

    benar = 0
    ada_jawaban = 0

    for path in daftar_gambar:
        gambar = cv2.imread(path)
        nama_file = os.path.basename(path)

        if gambar is None:
            print(f"{nama_file:24s} -> file tidak bisa dibaca")
            continue

        angka = vektor_titik_tangan(
            pendeteksi.process(cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB))
        )
        if angka is None:
            print(f"{nama_file:24s} -> tangan tidak terdeteksi")
            continue

        # Diolah dulu lewat fitur.py, sama seperti waktu model dilatih.
        peluang = model.predict_proba(ambil_fitur(angka))[0]
        indeks = int(np.argmax(peluang))
        tebakan = model.classes_[indeks]
        keyakinan = float(peluang[indeks])

        jawaban = tebak_jawaban_benar(path)
        catatan = ""
        if jawaban:
            ada_jawaban += 1
            if tebakan == jawaban:
                benar += 1
                catatan = "  BENAR"
            else:
                catatan = f"  SALAH (harusnya {jawaban})"

        print(f"{nama_file:24s} -> {tebakan}  keyakinan {keyakinan:.2f}{catatan}")

    pendeteksi.close()

    if ada_jawaban:
        print(f"\nKetepatan: {benar}/{ada_jawaban} = {benar / ada_jawaban:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(utama())
