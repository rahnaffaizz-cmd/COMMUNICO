"""Mencari ambang deteksi tangan terbaik dari video contoh berlabel per huruf.

Nama video menjadi label: A.mp4, A_02.mp4, B.mp4, dan seterusnya.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from fitur import ambil_fitur, sesuaikan_untuk
from gesture_voice import (PATH_MODEL, muat_ambang_huruf, muat_pengaturan_deteksi,
                           vektor_titik_tangan)

FOLDER = Path(__file__).resolve().parent
EKSTENSI_VIDEO = {".avi", ".mkv", ".mov", ".mp4", ".wmv"}
HURUF = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CALON_BAWAAN = (0.3, 0.5, 0.7)


def cari_video(folder: Path) -> list[tuple[Path, str]]:
    """Ambil video berlabel dari awalan nama berkas satu huruf."""
    if not folder.is_dir():
        raise ValueError(f"Folder video tidak ditemukan: {folder}")

    hasil = []
    for berkas in sorted(folder.iterdir()):
        if not berkas.is_file() or berkas.suffix.lower() not in EKSTENSI_VIDEO:
            continue
        bagian = berkas.stem.split("_", maxsplit=1)[0].split("-", maxsplit=1)[0]
        label = bagian.upper()
        if len(label) == 1 and label in HURUF:
            hasil.append((berkas, label))
    if not hasil:
        raise ValueError(
            f"Tidak ada video berlabel di {folder}. Gunakan nama seperti A.mp4 atau A_01.mp4.")
    return hasil


def cek_calon(nilai: list[float], nama: str) -> list[float]:
    if not nilai:
        raise ValueError(f"Daftar calon {nama} tidak boleh kosong.")
    if any(not 0.0 <= angka <= 1.0 for angka in nilai):
        raise ValueError(f"Semua nilai {nama} harus dari 0 sampai 1.")
    return sorted(set(nilai))


def evaluasi(video: list[tuple[Path, str]], model, deteksi: float, tracking: float,
             setiap_frame: int) -> dict:
    """Ukur cakupan tangan dan prediksi benar untuk satu kombinasi ambang."""
    mp_tangan = mp.solutions.hands
    per_huruf = defaultdict(lambda: {"sampel": 0, "terdeteksi": 0, "benar": 0})
    prediksi = []

    for jalur, label in video:
        kamera = cv2.VideoCapture(str(jalur))
        if not kamera.isOpened():
            kamera.release()
            raise ValueError(f"Video tidak bisa dibuka: {jalur}")

        nomor_frame = 0
        try:
            with mp_tangan.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=deteksi,
                    min_tracking_confidence=tracking) as pendeteksi:
                while True:
                    berhasil, frame = kamera.read()
                    if not berhasil:
                        break
                    nomor_frame += 1
                    frame = cv2.flip(frame, 1)
                    hasil = pendeteksi.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if (nomor_frame - 1) % setiap_frame:
                        continue

                    nilai = per_huruf[label]
                    nilai["sampel"] += 1
                    angka, _ = vektor_titik_tangan(hasil)
                    if angka is None:
                        continue

                    nilai["terdeteksi"] += 1
                    fitur = sesuaikan_untuk(
                        ambil_fitur(angka, getattr(model, "urut_tangan", False)),
                        model.n_features_in_)
                    peluang = model.predict_proba(fitur)[0]
                    indeks = int(peluang.argmax())
                    tebakan = str(model.classes_[indeks])
                    keyakinan = float(peluang[indeks])
                    prediksi.append({
                        "label": label, "tebakan": tebakan, "keyakinan": keyakinan,
                    })
                    if tebakan == label:
                        nilai["benar"] += 1
        finally:
            kamera.release()

        if nomor_frame == 0:
            raise ValueError(f"Video tidak berisi frame: {jalur}")

    total = sum(data["sampel"] for data in per_huruf.values())
    terdeteksi = sum(data["terdeteksi"] for data in per_huruf.values())
    benar = sum(data["benar"] for data in per_huruf.values())
    if total == 0:
        raise ValueError("Tidak ada frame sampel yang dapat dievaluasi.")

    return {
        "min_detection_confidence": deteksi,
        "min_tracking_confidence": tracking,
        "sampel": total,
        "terdeteksi": terdeteksi,
        "benar": benar,
        "cakupan": terdeteksi / total,
        "akurasi_efektif": benar / total,
        "per_huruf": dict(per_huruf),
        "prediksi": prediksi,
    }


def skor_urut(hasil: dict, acuan: dict[str, float]) -> tuple[float, float, float]:
    """Utamakan tebakan benar; jika seri, pilih cakupan lalu nilai dekat bawaan."""
    jarak_bawaan = (
        abs(hasil["min_detection_confidence"] - acuan["min_detection_confidence"])
        + abs(hasil["min_tracking_confidence"] - acuan["min_tracking_confidence"])
    )
    return hasil["akurasi_efektif"], hasil["cakupan"], -jarak_bawaan


def optimalkan_ambang_huruf(prediksi: list[dict], kelas, target: float,
                            huruf_dikalibrasi: set[str],
                            ambang_lama: dict) -> tuple[dict, dict]:
    """Cari ambang keyakinan tiap label berdasarkan video yang diberi label."""
    from diagnosa import AMBANG_MENYERAH, cari_ambang

    ambang_huruf = dict(ambang_lama)
    ringkasan = {}
    for huruf in sorted(huruf_dikalibrasi & set(map(str, kelas))):
        milik_huruf = [baris for baris in prediksi if baris["tebakan"] == huruf]
        if not milik_huruf:
            ambang_huruf[huruf] = AMBANG_MENYERAH
            ringkasan[huruf] = {"ambang": AMBANG_MENYERAH, "presisi": 0.0, "tersisa": 0.0}
            continue
        benar = [baris["label"] == huruf for baris in milik_huruf]
        yakin = [baris["keyakinan"] for baris in milik_huruf]
        batas, presisi, sisa = cari_ambang(
            np.asarray(benar), np.asarray(yakin), target)
        ambang_huruf[huruf] = batas
        ringkasan[huruf] = {
            "ambang": batas,
            "presisi": round(presisi, 4),
            "tersisa": round(sisa, 4),
        }
    return ambang_huruf, ringkasan


def utama(argv=None) -> int:
    acuan_deteksi = muat_pengaturan_deteksi()
    p = argparse.ArgumentParser(
        description="Kalibrasi ambang deteksi tangan memakai video BISINDO berlabel")
    p.add_argument("--video-dir", type=Path, required=True,
                   help="folder video; nama harus A.mp4, A_01.mp4, B.mp4, dan seterusnya")
    p.add_argument("--model", default=PATH_MODEL, help="model pengenal huruf")
    p.add_argument("--setiap-frame", type=int, default=5,
                   help="nilai setiap N frame (semua frame tetap diproses untuk tracking)")
    p.add_argument("--target-presisi", type=float, default=80.0,
                   help="presisi minimal yang dicari per huruf (persen, bawaan 80)")
    p.add_argument("--calon-deteksi", type=float, nargs="+", default=list(CALON_BAWAAN),
                   help="nilai min_detection_confidence yang diuji")
    p.add_argument("--calon-tracking", type=float, nargs="+", default=list(CALON_BAWAAN),
                   help="nilai min_tracking_confidence yang diuji")
    p.add_argument("--keluaran", type=Path, default=FOLDER / "deteksi.json",
                   help="berkas pengaturan yang dipakai aplikasi")
    arg = p.parse_args(argv)

    if arg.setiap_frame < 1:
        p.error("--setiap-frame harus minimal 1.")
    try:
        calon_deteksi = cek_calon(arg.calon_deteksi, "deteksi")
        calon_tracking = cek_calon(arg.calon_tracking, "tracking")
        video = cari_video(arg.video_dir)
    except ValueError as e:
        print(f"GAGAL: {e}")
        return 1

    try:
        model = joblib.load(FOLDER / arg.model)
    except Exception as e:  # noqa: BLE001 - joblib dapat gagal karena format atau dependensi model
        print(f"GAGAL: model '{arg.model}' tidak bisa dimuat ({e}).")
        return 1
    label_di_luar_model = sorted({label for _, label in video} - set(map(str, model.classes_)))
    if label_di_luar_model:
        print(f"GAGAL: model tidak memiliki kelas: {' '.join(label_di_luar_model)}.")
        return 1
    if not 0.0 < arg.target_presisi <= 100.0:
        p.error("--target-presisi harus lebih dari 0 dan paling tinggi 100.")

    jumlah_per_huruf = defaultdict(int)
    for _, label in video:
        jumlah_per_huruf[label] += 1
    print(f"Video: {len(video)} klip, {len(jumlah_per_huruf)} huruf; "
          f"menguji {len(calon_deteksi) * len(calon_tracking)} kombinasi.")
    if any(jumlah < 2 for jumlah in jumlah_per_huruf.values()):
        print("CATATAN: ada huruf dengan satu klip saja; hasil kalibrasi hanya "
              "mengukur klip yang diberikan.")

    semua_hasil = []
    jumlah_uji = len(calon_deteksi) * len(calon_tracking)
    for nomor, (deteksi, tracking) in enumerate(
            ((d, t) for d in calon_deteksi for t in calon_tracking), 1):
        mulai = time.perf_counter()
        try:
            nilai = evaluasi(video, model, deteksi, tracking, arg.setiap_frame)
        except (ValueError, cv2.error) as e:
            print(f"GAGAL: evaluasi video tidak selesai ({e}).")
            return 1
        semua_hasil.append(nilai)
        print(f"[{nomor}/{jumlah_uji}] deteksi={deteksi:.2f} tracking={tracking:.2f} "
              f"benar={nilai['akurasi_efektif'] * 100:.1f}% "
              f"tangan={nilai['cakupan'] * 100:.1f}% ({time.perf_counter() - mulai:.1f} dtk)")

    terbaik = max(semua_hasil, key=lambda hasil: skor_urut(hasil, acuan_deteksi))
    ambang_lama = muat_ambang_huruf(nama_model=arg.model)
    ambang_huruf, ringkasan_ambang = optimalkan_ambang_huruf(
        terbaik["prediksi"], model.classes_, arg.target_presisi / 100,
        set(jumlah_per_huruf), ambang_lama)
    pengaturan_baru = {
        "min_detection_confidence": terbaik["min_detection_confidence"],
        "min_tracking_confidence": terbaik["min_tracking_confidence"],
        "model": arg.model,
        "video": len(video),
        "sampel": terbaik["sampel"],
        "akurasi_efektif": round(terbaik["akurasi_efektif"], 4),
        "cakupan_tangan": round(terbaik["cakupan"], 4),
    }
    berkas_ambang = FOLDER / f"ambang_kalibrasi_{Path(arg.model).stem}.json"
    if arg.keluaran.resolve() == berkas_ambang.resolve():
        print("GAGAL: lokasi --keluaran tidak boleh sama dengan berkas ambang huruf.")
        return 1
    profil_ambang = {
        "target_presisi": arg.target_presisi,
        "ambang": ambang_huruf,
        "keterangan": (f"Kalibrasi video berlabel untuk {Path(arg.model).name}; "
                       "dihasilkan oleh optimasi_deteksi.py."),
    }
    try:
        arg.keluaran.parent.mkdir(parents=True, exist_ok=True)
        arg.keluaran.write_text(json.dumps(pengaturan_baru, indent=2), encoding="utf-8")
        berkas_ambang.write_text(json.dumps(profil_ambang, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"GAGAL: tidak bisa menyimpan hasil kalibrasi ({e}).")
        return 1

    print("\nPENGATURAN TERBAIK")
    print(f"  deteksi : {terbaik['min_detection_confidence']:.2f}")
    print(f"  tracking: {terbaik['min_tracking_confidence']:.2f}")
    print(f"  tebakan benar dari seluruh frame sampel: "
          f"{terbaik['akurasi_efektif'] * 100:.1f}%")
    print(f"  tangan terdeteksi: {terbaik['cakupan'] * 100:.1f}%")
    print(f"  tersimpan ke {arg.keluaran}")
    print(f"  ambang per huruf tersimpan ke {berkas_ambang.name}")
    print("\nHASIL PER HURUF")
    for huruf, data in sorted(terbaik["per_huruf"].items()):
        akurasi = data["benar"] / data["sampel"]
        cakupan = data["terdeteksi"] / data["sampel"]
        print(f"  {huruf}: benar {akurasi * 100:5.1f}%, "
              f"tangan {cakupan * 100:5.1f}% "
              f"ambang {ringkasan_ambang[huruf]['ambang']:.2f}, "
              f"presisi {ringkasan_ambang[huruf]['presisi'] * 100:.1f}% "
              f"({data['terdeteksi']}/{data['sampel']} frame terdeteksi)")
    return 0


if __name__ == "__main__":
    sys.exit(utama())
