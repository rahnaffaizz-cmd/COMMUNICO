r"""Mengoptimalkan pembacaan huruf secara otomatis: ukur, impor, latih, bandingkan.

Anda cukup memperagakan hurufnya di depan kamera. Sisanya dikerjakan sendiri:

    1. ukur       gesture_voice.py --ukur <huruf>, beberapa sesi per huruf
    2. impor      impor_ukur.py memasukkan sesi itu jadi data latih
    3. cadangkan  model dan ambang yang lama disalin ke log/cadangan_<waktu>/
    4. latih      latih_cabang.py melatih ulang model v7 beserta ambang hurufnya
    5. bandingkan (pilihan --verifikasi) mengukur lagi, lalu menampilkan
                  sebelum dan sesudah

CARA PAKAI
----------
    .\run.ps1 -Optimasi K                 ukur K dua sesi, lalu latih ulang
    .\run.ps1 -Optimasi K H P             beberapa huruf, dilatih sekali di akhir
    .\run.ps1 -Optimasi K --sesi 3        tiga sesi per huruf
    .\run.ps1 -Optimasi K --verifikasi    ukur lagi sesudah dilatih, lalu bandingkan
    .\run.ps1 -Optimasi --lewati-ukur     pakai sesi ukur yang sudah ada, langsung latih

TIPS AGAR HASILNYA BAGUS
------------------------
- Ubah posisi tiap sesi: lebih dekat dan lebih jauh, naik dan turun, sedikit
  miring. Sesi yang hampir tidak bergerak membuat model hanya menghafal jarak
  tangan, bukan bentuknya (itulah sebabnya P dan D sempat tertukar).
- Huruf dua tangan harus diperagakan dengan kedua tangan terlihat.
- 200 sampai 300 frame per sesi sudah cukup.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
FOLDER_LOG = FOLDER / "log"
BERKAS_DICADANGKAN = ["rf_bisindo_v7.pkl", "ambang_rf_bisindo_v7.json"]
FRAME_MINIMAL = 150


@dataclass
class RingkasanSesi:
    huruf: str
    berkas: Path
    frame: int
    benar: float            # persen frame yang tebakan teratasnya huruf ini
    yakin: float            # rata-rata keyakinan model untuk huruf ini
    satu_tangan: float      # persen frame yang cuma terlihat satu tangan
    pesaing: dict           # huruf lain yang paling sering jadi tebakan teratas


def ringkas_sesi(berkas: Path, huruf: str) -> RingkasanSesi:
    with open(berkas, encoding="utf-8", newline="") as f:
        baris = list(csv.DictReader(f))
    n = len(baris)
    if n == 0:
        return RingkasanSesi(huruf, berkas, 0, 0.0, 0.0, 0.0, {})
    benar = sum(1 for b in baris if b["tebakan"] == huruf)
    pesaing: dict[str, int] = {}
    for b in baris:
        if b["tebakan"] != huruf:
            pesaing[b["tebakan"]] = pesaing.get(b["tebakan"], 0) + 1
    urut = dict(sorted(pesaing.items(), key=lambda kv: -kv[1])[:3])
    return RingkasanSesi(
        huruf, berkas, n, benar / n * 100,
        sum(float(b["yakin_target"]) for b in baris) / n,
        sum(1 for b in baris if b["tangan"] == "1") / n * 100,
        {h: c / n * 100 for h, c in urut.items()})


def daftar_ukur(huruf: str) -> set[Path]:
    return set(FOLDER_LOG.glob(f"ukur_{huruf}_*.csv"))


def sesi_baru(huruf: str, sebelum: set[Path]) -> Path | None:
    """Berkas ukur `huruf` yang muncul sesudah daftar `sebelum` dibuat.

    Dibandingkan lewat daftar berkas, bukan lewat jam, supaya tidak terpengaruh
    jam komputer atau berkas yang disalin dengan waktu lama.
    """
    baru = daftar_ukur(huruf) - sebelum
    return max(baru, key=lambda p: p.name) if baru else None


def jalankan_ukur(huruf: str, model: str | None) -> None:
    perintah = [sys.executable, str(FOLDER / "gesture_voice.py"), "--ukur", huruf]
    if model:
        perintah += ["--model", model]
    subprocess.run(perintah, cwd=FOLDER)


def tunggu_siap(pesan: str) -> None:
    """Tampilkan petunjuk dan tunggu ENTER. Tanpa keyboard (EOF), lanjut saja."""
    print(f"\n{pesan}")
    if sys.stdin.isatty():
        try:
            input("  Tekan ENTER kalau sudah siap ... ")
        except EOFError:
            pass


def ukur_satu_sesi(huruf: str, model: str | None, urutan: str) -> RingkasanSesi | None:
    from cabang import HURUF_SATU_TANGAN

    tangan = "satu tangan" if huruf in HURUF_SATU_TANGAN else "KEDUA tangan terlihat"
    tunggu_siap(f"[{urutan}] Peragakan huruf {huruf} ({tangan}) terus-menerus.\n"
                f"  Jendela kamera akan terbuka. Tekan Q sesudah 200 sampai 300 frame.")
    sebelum = daftar_ukur(huruf)
    jalankan_ukur(huruf, model)
    berkas = sesi_baru(huruf, sebelum)
    if berkas is None:
        print(f"  Tidak ada data baru untuk {huruf} (tidak ada tangan terbaca?).")
        return None
    r = ringkas_sesi(berkas, huruf)
    cetak_sesi(r)
    return r


def cetak_sesi(r: RingkasanSesi) -> None:
    print(f"  Hasil {r.huruf}: {r.frame} frame | terbaca benar {r.benar:.0f}% | "
          f"keyakinan {r.yakin:.2f} | satu tangan saja {r.satu_tangan:.0f}%")
    if r.pesaing:
        print("  Sering terbaca sebagai: "
              + ", ".join(f"{h} {p:.0f}%" for h, p in r.pesaing.items()))
    if r.frame < FRAME_MINIMAL:
        print(f"  PERHATIAN: baru {r.frame} frame, kurang dari {FRAME_MINIMAL}. "
              f"Sesi ini kurang berguna; ulangi dengan lebih lama.")
    from cabang import HURUF_SATU_TANGAN
    if r.huruf not in HURUF_SATU_TANGAN and r.satu_tangan > 20:
        print(f"  PERHATIAN: {r.satu_tangan:.0f}% frame cuma satu tangan. {r.huruf} huruf dua "
              f"tangan; pastikan kedua tangan masuk kamera.")


def cadangkan() -> Path:
    tujuan = FOLDER_LOG / f"cadangan_{datetime.now():%Y%m%d_%H%M%S}"
    tujuan.mkdir(parents=True, exist_ok=True)
    for nama in BERKAS_DICADANGKAN:
        if (FOLDER / nama).exists():
            shutil.copy2(FOLDER / nama, tujuan / nama)
    return tujuan


def impor(huruf: list[str]) -> None:
    perintah = [sys.executable, str(FOLDER / "impor_ukur.py")]
    if huruf:
        perintah += ["--huruf", *huruf]
    subprocess.run(perintah, cwd=FOLDER, check=True)


def latih(huruf: list[str]) -> int:
    """Latih ulang, dan tampilkan bagian yang menjawab 'apakah ini membaik?'."""
    print("\nMelatih ulang model (sekitar 1 sampai 2 menit) ...")
    hasil = subprocess.run([sys.executable, str(FOLDER / "latih_cabang.py")], cwd=FOLDER,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    if hasil.returncode != 0:
        print("PELATIHAN GAGAL:\n" + "\n".join(hasil.stderr.strip().splitlines()[-12:]))
        return hasil.returncode

    penting = ("Data pengguna", "semua sampel", "Ambang ditulis", "Disimpan")
    for baris in hasil.stdout.splitlines():
        kunci = baris.strip()
        huruf_ini = any(kunci.startswith(f"{h} (") for h in huruf)
        if huruf_ini or any(k in baris for k in penting):
            print("  " + kunci)
    return 0


def tabel_perbandingan(sebelum: dict[str, list[RingkasanSesi]],
                       sesudah: dict[str, RingkasanSesi]) -> None:
    print("\nSEBELUM DAN SESUDAH DILATIH (sesi verifikasi belum pernah dilihat model)")
    print("  huruf | sebelum | sesudah | keyakinan")
    for h, lama in sebelum.items():
        if h not in sesudah or not lama:
            continue
        rata_lama = sum(s.benar for s in lama) / len(lama)
        yakin_lama = sum(s.yakin for s in lama) / len(lama)
        baru = sesudah[h]
        print(f"    {h}   | {rata_lama:5.0f}%  | {baru.benar:5.0f}%  | "
              f"{yakin_lama:.2f} -> {baru.yakin:.2f}")


def utama() -> int:
    p = argparse.ArgumentParser(
        description="Optimasi pembacaan huruf: ukur, impor, latih, bandingkan")
    p.add_argument("huruf", nargs="*", help="huruf yang dioptimalkan, misalnya K H P")
    p.add_argument("--sesi", type=int, default=2, help="jumlah sesi ukur per huruf (bawaan 2)")
    p.add_argument("--verifikasi", action="store_true",
                   help="ukur lagi sesudah dilatih dan bandingkan dengan sebelumnya")
    p.add_argument("--lewati-ukur", action="store_true",
                   help="jangan mengukur; pakai sesi ukur yang sudah ada")
    p.add_argument("--model", default=None, help="model untuk pengukuran (bawaan seperti run.ps1)")
    arg = p.parse_args()

    huruf = [h.upper() for h in arg.huruf]
    salah = [h for h in huruf if len(h) != 1 or not h.isalpha()]
    if salah:
        print(f"Huruf harus satu karakter A sampai Z, bukan: {', '.join(salah)}")
        return 1
    if not huruf and not arg.lewati_ukur:
        print("Sebutkan hurufnya, misalnya:  .\\run.ps1 -Optimasi K\n"
              "Atau pakai --lewati-ukur untuk melatih dari sesi yang sudah ada.")
        return 1

    sebelum: dict[str, list[RingkasanSesi]] = {h: [] for h in huruf}
    if not arg.lewati_ukur:
        for h in huruf:
            for nomor in range(1, arg.sesi + 1):
                r = ukur_satu_sesi(h, arg.model, f"{h}, sesi {nomor} dari {arg.sesi}"
                                   + ("" if nomor == 1 else "; UBAH posisi dan jarak tangan"))
                if r:
                    sebelum[h].append(r)

    print("\nMemasukkan sesi ukur jadi data latih ...")
    impor(huruf)

    cadangan = cadangkan()
    print(f"Model dan ambang yang lama disalin ke {cadangan.relative_to(FOLDER)}/")
    kode = latih(huruf)
    if kode:
        return kode

    if arg.verifikasi and huruf and not arg.lewati_ukur:
        print("\nVERIFIKASI: ukur sekali lagi dengan model yang baru.")
        sesudah = {}
        for h in huruf:
            r = ukur_satu_sesi(h, arg.model, f"{h}, verifikasi; ubah posisi dari sesi tadi")
            if r:
                sesudah[h] = r
        tabel_perbandingan(sebelum, sesudah)

    print(f"\nSelesai. Coba langsung dengan  .\\run.ps1\n"
          f"Kalau hasilnya justru lebih buruk, kembalikan dari {cadangan.relative_to(FOLDER)}/.\n"
          f"Kalau bagus, commit: rf_bisindo_v7.pkl dan ambang_rf_bisindo_v7.json.")
    return 0


if __name__ == "__main__":
    sys.exit(utama())
