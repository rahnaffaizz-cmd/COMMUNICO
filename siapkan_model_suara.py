"""Menyiapkan model pengenal suara bahasa Indonesia untuk percakapan.py.

KENAPA BUKAN WHISPER BIASA
--------------------------
Whisper bawaan dilatih untuk banyak bahasa, dan bahasa Indonesia kebagian
sedikit. Dua model di bawah ini adalah Whisper yang disetel khusus untuk bahasa
Indonesia (maleo-ai, lisensi Apache-2.0). Diukur pada 64 ucapan manusia nyata
dari FLEURS bahasa Indonesia (panjang sampai 15 detik), tingkat kesalahan kata
(WER):

    Whisper base biasa     45,6%   (yang dipakai sebelumnya)
    Whisper small biasa    19,2%
    id-base                13,9%   sama cepat dengan base biasa (sekitar 1 detik)
    id-small               10,8%   paling teliti, tapi sekitar 3 detik per ucapan

Karena itu `id-base` jadi bawaan: lebih tepat daripada small biasa dengan
sepertiga waktunya.

MODELNYA DISIAPKAN SEKALI
-------------------------
    python siapkan_model_suara.py              siapkan id-base
    python siapkan_model_suara.py id-small     siapkan yang lebih teliti
    python siapkan_model_suara.py --semua

- id-small sudah tersedia dalam format yang siap pakai, jadi cukup diunduh
  (sekitar 250 MB).
- id-base harus DIKONVERSI dulu ke format CTranslate2 (hasilnya sekitar 75 MB).
  Konversi butuh torch dan transformers, yang besar dan tidak dipakai program
  sehari-hari. Karena itu keduanya dipasang di lingkungan sementara, dipakai,
  lalu dibuang, jadi .venv Anda tidak ikut membesar. Butuh `uv` dan internet,
  dan memakan beberapa menit.

Hasilnya disimpan di folder model_suara/ (tidak ikut ke GitHub). Percakapan.py
memanggil ini otomatis kalau modelnya belum ada.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
FOLDER_MODEL = FOLDER / "model_suara"

# Revisi dikunci supaya yang dipakai persis yang sudah diukur.
MODEL = {
    "id-base": {
        "repo": "maleo-ai/whisper-base-id",
        "revisi": "fb3ebb61d48afc5cd10330eb509adb62a231e65d",
        "cara": "konversi",
    },
    "id-small": {
        "repo": "maleo-ai/whisper-small-id",
        "revisi": "37a5b9a0fa5739e8e859984dabca44337a45fa55",
        "cara": "unduh",
        "subfolder": "ct2-int8",
    },
}


def _jalankan(perintah: list[str]) -> None:
    hasil = subprocess.run(perintah, capture_output=True, text=True)
    if hasil.returncode != 0:
        ujung = (hasil.stderr or hasil.stdout).strip().splitlines()[-8:]
        raise RuntimeError(f"perintah gagal: {' '.join(perintah[:3])} ...\n" + "\n".join(ujung))


def folder_model(alias: str) -> Path:
    return FOLDER_MODEL / alias


def sudah_ada(alias: str) -> bool:
    return (folder_model(alias) / "model.bin").exists()


def siapkan(alias: str, lapor=print) -> Path:
    """Pastikan model `alias` ada di model_suara/. Balik folder modelnya."""
    if alias not in MODEL:
        raise ValueError(f"model '{alias}' tidak dikenal. Pilihan: {', '.join(MODEL)}")
    tujuan = folder_model(alias)
    if sudah_ada(alias):
        return tujuan

    info = MODEL[alias]
    FOLDER_MODEL.mkdir(exist_ok=True)
    # Dikerjakan di folder sementara lalu diganti nama, supaya model yang
    # setengah jadi tidak dianggap sudah siap kalau prosesnya terhenti.
    sementara = FOLDER_MODEL / f"{alias}.sedang-dibuat"
    shutil.rmtree(sementara, ignore_errors=True)
    try:
        if info["cara"] == "unduh":
            _unduh(info, sementara, lapor)
        else:
            _konversi(info, sementara, lapor)
        if not (sementara / "model.bin").exists():
            raise RuntimeError("hasilnya tidak lengkap (model.bin tidak ada)")
        shutil.rmtree(tujuan, ignore_errors=True)
        sementara.rename(tujuan)
    finally:
        shutil.rmtree(sementara, ignore_errors=True)
    lapor(f"Model {alias} siap di {tujuan}")
    return tujuan


def _unduh(info: dict, tujuan: Path, lapor) -> None:
    from huggingface_hub import snapshot_download

    lapor(f"Mengunduh {info['repo']} (sekitar 250 MB) ...")
    with tempfile.TemporaryDirectory(dir=FOLDER_MODEL) as unduhan:
        snapshot_download(info["repo"], revision=info["revisi"],
                          allow_patterns=[f"{info['subfolder']}/*"], local_dir=unduhan)
        shutil.copytree(Path(unduhan) / info["subfolder"], tujuan)


def _cari_uv() -> str | None:
    """Cari uv di PATH, lalu di folder Scripts milik Python yang sedang jalan.

    run.ps1 memanggil .venv\\Scripts\\python.exe tanpa mengaktifkan .venv, jadi
    uv yang dipasang lewat `pip install uv` tidak ada di PATH.
    """
    uv = shutil.which("uv")
    if uv:
        return uv
    lokal = Path(sys.executable).parent / ("uv.exe" if sys.platform == "win32" else "uv")
    return str(lokal) if lokal.exists() else None


def _konversi(info: dict, tujuan: Path, lapor) -> None:
    uv = _cari_uv()
    if not uv:
        raise RuntimeError(
            "Konversi model butuh `uv` (https://docs.astral.sh/uv/) dan tidak ditemukan.\n"
            "Pasang dengan: .\\.venv\\Scripts\\python.exe -m pip install uv\n"
            "atau pakai model yang tidak perlu konversi: --model-ucapan id-small")

    with tempfile.TemporaryDirectory(dir=FOLDER_MODEL) as sementara:
        venv = Path(sementara) / "venv"
        py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        lapor("Menyiapkan lingkungan sementara untuk konversi (torch, transformers) ...")
        _jalankan([uv, "venv", "--python", "3.12", str(venv)])
        _jalankan([uv, "pip", "install", "--python", str(py), "ctranslate2",
                   "transformers<5", "torch", "sentencepiece"])

        lapor(f"Mengonversi {info['repo']} ke format CTranslate2 (int8) ...")
        # Lewat `python -m`, bukan ct2-transformers-converter.exe: pembuka .exe
        # yang baru diunduh bisa diblokir Smart App Control di Windows.
        _jalankan([str(py), "-m", "ctranslate2.converters.transformers",
                   "--model", info["repo"], "--revision", info["revisi"],
                   "--output_dir", str(tujuan), "--quantization", "int8",
                   "--copy_files", "tokenizer.json", "preprocessor_config.json"])


def utama() -> int:
    p = argparse.ArgumentParser(description="Menyiapkan model pengenal suara bahasa Indonesia")
    p.add_argument("model", nargs="*", help=f"salah satu dari: {', '.join(MODEL)} "
                                            f"(bawaan id-base)")
    p.add_argument("--semua", action="store_true", help="siapkan semuanya")
    arg = p.parse_args()

    daftar = list(MODEL) if arg.semua else (arg.model or ["id-base"])
    for alias in daftar:
        try:
            siapkan(alias)
        except Exception as e:  # noqa: BLE001
            print(f"GAGAL menyiapkan {alias}: {e}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(utama())
