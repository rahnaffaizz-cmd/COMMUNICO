"""Pencatat jalannya program, supaya kelihatan apa yang terjadi di tiap tahap.

Tanpa ini, kalau hasilnya salah kita cuma tahu "hurufnya keliru" tanpa tahu di
mana macetnya. Dengan pencatatan, setiap tahap kelihatan:

    kamera -> tangan terdeteksi?        berapa tangan, seberapa yakin
           -> model menebak apa?        tebakan teratas dan pesaingnya
           -> voting sudah cukup?       berapa kali menang dari berapa frame
           -> huruf diterima            masuk ke teks
           -> suara dibunyikan          diunduh atau ambil dari simpanan

Catatan ditulis ke dua tempat sekaligus: ke layar terminal supaya bisa dipantau
langsung, dan ke berkas di folder log/ supaya bisa dibaca ulang setelah selesai.

Ada dua tingkat kerincian:
    biasa  - hanya kejadian penting: huruf diterima, suara dibunyikan
    detail - ditambah tebakan tiap kali berubah dan kemajuan voting
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

FOLDER_LOG = Path(__file__).resolve().parent / "log"

# Supaya tidak membanjiri layar, tebakan yang sama berturut-turut tidak dicatat
# ulang. Hanya perubahan yang menarik untuk dilihat.
_tebakan_terakhir = ""


class _WarnaRingkas(logging.Formatter):
    """Format satu baris pendek untuk layar: jam, tahap, pesan."""

    def format(self, record: logging.LogRecord) -> str:
        tahap = getattr(record, "tahap", "-")
        waktu = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        return f"{waktu} [{tahap:<8}] {record.getMessage()}"


class Pencatat:
    """Pembungkus logging supaya pemanggilannya pendek dan seragam."""

    def __init__(self, aktif: bool = True, detail: bool = False,
                 ke_berkas: bool = True):
        self.aktif = aktif
        self.detail = detail
        self.path_berkas: Path | None = None

        self._log = logging.getLogger("gesture_voice")
        self._log.handlers.clear()
        self._log.setLevel(logging.DEBUG if detail else logging.INFO)
        self._log.propagate = False

        if not aktif:
            return

        ke_layar = logging.StreamHandler(sys.stdout)
        ke_layar.setFormatter(_WarnaRingkas())
        self._log.addHandler(ke_layar)

        if ke_berkas:
            FOLDER_LOG.mkdir(exist_ok=True)
            nama = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
            self.path_berkas = FOLDER_LOG / nama
            ke_file = logging.FileHandler(self.path_berkas, encoding="utf-8")
            ke_file.setFormatter(logging.Formatter(
                "%(asctime)s [%(tahap)-8s] %(message)s", datefmt="%H:%M:%S"))
            self._log.addHandler(ke_file)

    # --- pemanggilan umum ---------------------------------------------------
    def kabar(self, tahap: str, pesan: str) -> None:
        """Kejadian penting, selalu dicatat."""
        if self.aktif:
            self._log.info(pesan, extra={"tahap": tahap})

    def rinci(self, tahap: str, pesan: str) -> None:
        """Kejadian kecil, hanya dicatat kalau mode detail menyala."""
        if self.aktif and self.detail:
            self._log.debug(pesan, extra={"tahap": tahap})

    # --- pemanggilan khusus tiap tahap --------------------------------------
    def mulai(self, model: str, jumlah_fitur: int, huruf: str,
              bahasa: str, kamera: int) -> None:
        self.kabar("MULAI", f"model {model}, butuh {jumlah_fitur} angka masukan")
        self.kabar("MULAI", f"huruf yang dikenali: {huruf}")
        self.kabar("MULAI", f"kamera {kamera}, suara bahasa '{bahasa}'")
        if self.path_berkas:
            self.kabar("MULAI", f"catatan disimpan di log/{self.path_berkas.name}")

    def tangan(self, jumlah: int) -> None:
        self.rinci("TANGAN", f"{jumlah} tangan terdeteksi")

    def tebakan(self, huruf: str, keyakinan: float, pesaing: list) -> None:
        """Dicatat hanya kalau tebakan teratasnya berubah, biar tidak banjir."""
        global _tebakan_terakhir
        if huruf == _tebakan_terakhir:
            return
        _tebakan_terakhir = huruf
        daftar = ", ".join(f"{h} {k:.2f}" for h, k in pesaing)
        self.rinci("TEBAK", f"{huruf} ({keyakinan:.2f})   pesaing: {daftar}")

    def voting(self, huruf: str, menang: int, butuh: int, rata_yakin: float) -> None:
        self.rinci("VOTING",
                   f"{huruf}: menang {menang}/{butuh} frame, "
                   f"rata-rata keyakinan {rata_yakin:.2f}")

    def voting_gagal(self, huruf: str, alasan: str) -> None:
        self.rinci("VOTING", f"{huruf} belum diterima, {alasan}")

    def huruf_diterima(self, huruf: str, teks: str) -> None:
        self.kabar("HURUF", f"'{huruf}' diterima  ->  teks sekarang: \"{teks}\"")

    def teks_berubah(self, apa: str, teks: str) -> None:
        self.kabar("TEKS", f"{apa}  ->  \"{teks}\"")

    def suara_minta(self, teks: str) -> None:
        self.kabar("SUARA", f"minta dibunyikan: \"{teks}\"")

    def suara_simpanan(self, teks: str) -> None:
        self.rinci("SUARA", f"\"{teks}\" diambil dari simpanan, tidak perlu internet")

    def suara_unduh(self, teks: str, detik: float) -> None:
        self.rinci("SUARA", f"\"{teks}\" diunduh dari Google TTS ({detik:.2f} detik)")

    def suara_selesai(self, teks: str, detik: float) -> None:
        self.kabar("SUARA", f"selesai membunyikan \"{teks}\" ({detik:.2f} detik)")

    def suara_gagal(self, teks: str, sebab: str) -> None:
        self.kabar("SUARA", f"GAGAL membunyikan \"{teks}\": {sebab}")

    def selesai(self, teks: str, jumlah_frame: int, detik: float) -> None:
        laju = jumlah_frame / detik if detik > 0 else 0
        self.kabar("SELESAI", f"{jumlah_frame} frame dalam {detik:.1f} detik "
                              f"({laju:.1f} frame per detik)")
        self.kabar("SELESAI", f"teks terakhir: \"{teks}\"")
        if self.path_berkas:
            self.kabar("SELESAI", f"catatan lengkap ada di log/{self.path_berkas.name}")
