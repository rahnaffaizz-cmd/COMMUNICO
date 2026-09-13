"""Mengukur seberapa baik satu huruf terbaca oleh kamera Anda sendiri.

KENAPA PERLU
------------
Ambang di ambang_huruf.json disetel berdasarkan dataset Kaggle 312 foto, bukan
berdasarkan tangan Anda, kamera Anda, dan ruangan Anda. Bisa jadi sebuah huruf
yang di dataset selalu gagal ternyata terbaca bagus di tempat Anda, atau
sebaliknya.

Alat ini merekam setiap frame saat Anda memperagakan satu huruf, lalu di akhir
melaporkan angka sebenarnya: seberapa sering huruf itu jadi tebakan teratas,
berapa keyakinannya, huruf apa saja yang jadi pesaing, dan ambang berapa yang
sebaiknya dipakai.

CARA PAKAI
----------
    python gesture_voice.py --ukur I

Peragakan huruf I terus-menerus sampai terkumpul cukup banyak frame, lalu tekan
Q. Laporannya muncul di layar dan tersimpan di folder log/.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

FOLDER_LOG = Path(__file__).resolve().parent / "log"


class PengukurHuruf:
    """Mencatat tiap pembacaan saat satu huruf sedang diperagakan."""

    def __init__(self, huruf_target: str, ambang_berlaku: float):
        self.target = huruf_target.upper()
        self.ambang_berlaku = ambang_berlaku
        self.mulai = datetime.now()

        # Tiap frame yang ada tangannya dicatat satu baris.
        self._baris: list[dict] = []

    def catat_frame(self, tebakan_teratas: str, keyakinan_teratas: float,
                    peluang_target: float, tiga_besar: list,
                    jumlah_tangan: int = 0) -> None:
        self._baris.append({
            "detik": (datetime.now() - self.mulai).total_seconds(),
            "tangan": jumlah_tangan,
            "tebakan": tebakan_teratas,
            "yakin_tebakan": keyakinan_teratas,
            "yakin_target": peluang_target,
            "peringkat_target": self._peringkat(tiga_besar),
            "pesaing": " ".join(f"{h}:{n:.2f}" for h, n in tiga_besar),
        })

    def _peringkat(self, tiga_besar: list) -> int:
        """Urutan ke berapa huruf target muncul. 0 kalau di luar tiga besar."""
        for nomor, (huruf, _) in enumerate(tiga_besar, start=1):
            if huruf == self.target:
                return nomor
        return 0

    @property
    def jumlah_frame(self) -> int:
        return len(self._baris)

    def ringkas_singkat(self) -> str:
        """Satu baris pendek untuk ditampilkan di layar kamera."""
        if not self._baris:
            return f"ukur {self.target}: belum ada data"
        benar = sum(1 for b in self._baris if b["tebakan"] == self.target)
        rata = np.mean([b["yakin_target"] for b in self._baris])
        return (f"ukur {self.target}: {benar}/{len(self._baris)} frame "
                f"({benar / len(self._baris) * 100:.0f}%) yakin {rata:.2f}")

    # ------------------------------------------------------------ laporan akhir
    def laporan(self) -> str:
        if not self._baris:
            return (f"\nTidak ada frame yang terekam untuk huruf {self.target}.\n"
                    f"Pastikan tangan terlihat kamera saat mengukur.\n")

        n = len(self._baris)
        yakin_target = np.array([b["yakin_target"] for b in self._baris])
        teratas = [b["tebakan"] for b in self._baris]

        benar = sum(1 for t in teratas if t == self.target)
        dalam_tiga = sum(1 for b in self._baris if b["peringkat_target"] > 0)

        baris = []
        t = baris.append
        t("")
        t("=" * 66)
        t(f"HASIL PENGUKURAN HURUF {self.target}")
        t("=" * 66)
        t(f"  frame terekam            : {n}")
        t(f"  jadi tebakan teratas     : {benar} ({benar / n * 100:.1f}%)")
        t(f"  masuk tiga besar         : {dalam_tiga} ({dalam_tiga / n * 100:.1f}%)")
        t("")
        t(f"  keyakinan untuk {self.target}:")
        t(f"    rata-rata  {yakin_target.mean():.3f}")
        t(f"    tengah     {np.median(yakin_target):.3f}")
        t(f"    terendah   {yakin_target.min():.3f}")
        t(f"    tertinggi  {yakin_target.max():.3f}")

        # Huruf apa saja yang jadi tebakan teratas, selain target.
        lain = {}
        for huruf in teratas:
            if huruf != self.target:
                lain[huruf] = lain.get(huruf, 0) + 1
        t("")
        if lain:
            t("  Saat bukan tebakan teratas, yang muncul:")
            for huruf, jumlah in sorted(lain.items(), key=lambda x: -x[1])[:5]:
                t(f"    {huruf}: {jumlah} frame ({jumlah / n * 100:.1f}%)")
        else:
            t(f"  {self.target} selalu jadi tebakan teratas, tidak ada pesaing.")

        # Jumlah tangan sangat menentukan. Beberapa huruf di dataset dipelajari
        # sebagai huruf satu tangan, yang lain dua tangan. Kalau tangan kedua
        # ikut terbaca padahal tidak dimaksudkan, tebakannya bisa melenceng
        # ke huruf dua tangan yang sama sekali berbeda.
        tangan = [b["tangan"] for b in self._baris]
        hitung_tangan = {j: tangan.count(j) for j in sorted(set(tangan))}
        if len(hitung_tangan) > 0:
            t("")
            t("  Jumlah tangan yang terbaca kamera:")
            for jumlah, berapa in hitung_tangan.items():
                t(f"    {jumlah} tangan: {berapa} frame ({berapa / n * 100:.1f}%)")

            # Kalau jumlah tangannya berubah-ubah, bandingkan hasilnya.
            if len(hitung_tangan) > 1:
                t("")
                t("  Ketepatan dipisah menurut jumlah tangan:")
                for jumlah in hitung_tangan:
                    pilih = [b for b in self._baris if b["tangan"] == jumlah]
                    tepat = sum(1 for b in pilih if b["tebakan"] == self.target)
                    rata = np.mean([b["yakin_target"] for b in pilih])
                    t(f"    {jumlah} tangan: {tepat}/{len(pilih)} benar "
                      f"({tepat / len(pilih) * 100:.1f}%), keyakinan {rata:.2f}")
                t("")
                t("  Kalau salah satunya jauh lebih baik, usahakan peragaannya")
                t("  konsisten memakai jumlah tangan itu saja.")

        # Bagaimana nasibnya dengan ambang yang berlaku sekarang.
        t("")
        t("-" * 66)
        lolos = (yakin_target >= self.ambang_berlaku).sum()
        if self.ambang_berlaku >= 0.99:
            t(f"  Ambang {self.target} saat ini: 0.99, artinya DITOLAK sepenuhnya.")
            t(f"  Padahal keyakinan rata-ratanya di sini {yakin_target.mean():.2f}.")
        else:
            t(f"  Ambang {self.target} saat ini: {self.ambang_berlaku:.2f}")
            t(f"  Frame yang lolos ambang ini: {lolos} dari {n} "
              f"({lolos / n * 100:.1f}%)")

        # Usulan ambang: cukup rendah supaya sebagian besar frame lolos, tapi
        # tidak lebih rendah dari yang perlu.
        if benar > 0:
            usul = float(np.percentile(yakin_target[np.array(teratas) == self.target], 20))
            usul = max(0.20, round(usul - 0.02, 2))
            t("")
            t(f"  Usulan ambang berdasarkan pengukuran ini: {usul:.2f}")
            t(f"  Dengan ambang itu, {(yakin_target >= usul).sum()} dari {n} frame lolos.")
            if self.ambang_berlaku >= 0.99:
                t("")
                t(f"  Kalau angka di atas meyakinkan, huruf {self.target} sebaiknya")
                t("  tidak lagi ditolak. Ubah nilainya di ambang_huruf.json.")
        else:
            t("")
            t(f"  {self.target} tidak pernah jadi tebakan teratas sekali pun.")
            t("  Kemungkinan peragaannya belum cocok dengan yang dipelajari model,")
            t("  atau huruf ini memang tidak bisa dibedakan dari satu frame diam.")
        t("=" * 66)
        return "\n".join(baris)

    def simpan_rincian(self) -> Path | None:
        """Tulis tiap frame ke berkas csv supaya bisa diperiksa lebih lanjut."""
        if not self._baris:
            return None
        FOLDER_LOG.mkdir(exist_ok=True)
        nama = f"ukur_{self.target}_{self.mulai.strftime('%Y%m%d_%H%M%S')}.csv"
        path = FOLDER_LOG / nama
        with open(path, "w", newline="", encoding="utf-8") as f:
            penulis = csv.DictWriter(f, fieldnames=list(self._baris[0].keys()))
            penulis.writeheader()
            penulis.writerows(self._baris)
        return path
