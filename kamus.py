"""Rekomendasi dan koreksi kata dari huruf yang terbaca.

Model huruf kadang salah satu huruf, misalnya P terbaca D, atau satu huruf
masuk dua kali. Modul ini mencocokkan kata yang sedang dieja dengan daftar
kata di kamus_kata.txt (dan kamus_pribadi.txt kalau ada), lalu:

    - menampilkan saran kata selama huruf masih dieja
    - mengganti kata yang tidak ada di kamus dengan saran terbaik begitu kata
      itu selesai (spasi masuk, atau teks diucapkan / dikirim)

Pencocokannya bukan sekadar jarak huruf biasa. Untuk tiap huruf yang
diterima, peluang semua huruf dari model ikut disimpan (rata-rata selama
voting). Jadi kalau model membaca D tapi P di urutan kedua dengan peluang
hampir sama, mengganti D dengan P dianggap murah. Mengganti D dengan huruf
yang sama sekali tidak dilirik model dianggap mahal.

Biaya tiap langkah:
    ganti huruf     0.3 (pesaing sekuat huruf terbaca) sampai 1.0 (tidak dilirik)
    huruf berlebih  1.0, atau 0.5 kalau sama dengan huruf sebelumnya (masuk dua kali)
    huruf terlewat  1.0, atau 0.4 untuk huruf yang ambangnya 0.99 (sengaja
                    ditolak, misalnya I di model v7, jadi memang tidak akan
                    pernah terbaca)

Kata baru diganti kalau biayanya cukup kecil: satu kesalahan untuk kata tiga
huruf atau lebih, dan sedikit lebih longgar untuk kata panjang.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
BERKAS_KAMUS = FOLDER / "kamus_kata.txt"
BERKAS_PRIBADI = FOLDER / "kamus_pribadi.txt"

BIAYA_GANTI_MIN = 0.3
BIAYA_LEBIH = 1.0
BIAYA_LEBIH_KEMBAR = 0.5
BIAYA_LEWAT = 1.0
BIAYA_LEWAT_DITOLAK = 0.4
# Bobot kecil untuk kata yang lebih jarang (urutan lebih bawah di kamus),
# supaya dua kandidat yang sama mahal dimenangkan kata yang lebih umum.
BOBOT_FREKUENSI = 0.08
# Selisih minimal antara saran pertama dan kedua agar kata diganti otomatis.
# Kalau keduanya terlalu mirip, saran tetap tampil tapi tidak dipaksakan.
SELISIH_YAKIN = 0.05


def baca_daftar_kata(berkas: Path) -> list[str]:
    """Satu kata per baris, baris diawali # diabaikan. Hanya huruf A-Z."""
    if not berkas.exists():
        return []
    kata = []
    for baris in berkas.read_text(encoding="utf-8").splitlines():
        baris = baris.split("#", 1)[0].strip().upper()
        if baris and re.fullmatch(r"[A-Z]+", baris):
            kata.append(baris)
    return kata


def batas_biaya(panjang: int) -> float:
    """Seberapa banyak kesalahan yang masih boleh dikoreksi untuk kata sepanjang ini."""
    if panjang <= 2:
        return 0.5
    if panjang <= 6:
        return 1.0
    return 1.5


class PengoreksiKata:
    """Menyimpan peluang tiap huruf di kata yang sedang dieja, lalu mencari kata kamus."""

    def __init__(self, daftar_kata: list[str], huruf_ditolak=(), aktif: bool = True,
                 catat=None):
        self.aktif = aktif
        self.catat = catat
        self.huruf_ditolak = set(huruf_ditolak)
        # Urutan pertama yang dipakai kalau sebuah kata tercantum dua kali.
        self.peringkat: dict[str, int] = {}
        for k in daftar_kata:
            self.peringkat.setdefault(k, len(self.peringkat))
        self._per_panjang: dict[int, list[str]] = {}
        for k in self.peringkat:
            self._per_panjang.setdefault(len(k), []).append(k)

        self._frame: list[dict[str, float]] = []     # peluang sejak huruf terakhir
        self._huruf: list[tuple[str, dict[str, float]]] = []   # kata yang sedang dieja
        self.koreksi_terakhir: tuple[str, str] | None = None   # (asli, pengganti)
        # Saran diminta tiap frame untuk layar, padahal teks jarang berubah.
        self._saran_simpan: tuple[str, list[str]] = ("", [])

    @classmethod
    def muat(cls, ambang_huruf: dict | None = None, aktif: bool = True, catat=None):
        kata = baca_daftar_kata(BERKAS_PRIBADI) + baca_daftar_kata(BERKAS_KAMUS)
        ditolak = {h for h, a in (ambang_huruf or {}).items() if a >= 0.99}
        diri = cls(kata, ditolak, aktif=aktif, catat=catat)
        if catat:
            catat.kabar("KAMUS", f"{len(diri.peringkat)} kata dimuat"
                                 + (f", huruf ditolak: {''.join(sorted(ditolak))}"
                                    if ditolak else "")
                                 + ("" if aktif else " (koreksi mati)"))
        return diri

    # ------------------------------------------------------ dari loop kamera
    def catat_frame(self, kelas, peluang) -> None:
        """Simpan peluang semua huruf untuk satu frame. Dipanggil tiap frame yang dibaca."""
        self._frame.append({str(k): float(p) for k, p in zip(kelas, peluang) if p >= 0.01})
        if len(self._frame) > 60:
            del self._frame[0]

    def lupakan_frame(self) -> None:
        """Voting direset (tangan hilang): frame yang terkumpul tidak dipakai lagi."""
        self._frame.clear()

    def huruf_masuk(self, huruf: str, teks: str) -> None:
        """Huruf diterima voting. Peluangnya dirata-rata dari frame yang terkumpul."""
        self._selaraskan(teks[:-1])
        sebaran: dict[str, float] = {}
        for f in self._frame:
            for h, p in f.items():
                sebaran[h] = sebaran.get(h, 0.0) + p / len(self._frame)
        if sebaran.get(huruf, 0.0) <= 0.0:
            sebaran = {huruf: 1.0}
        self._huruf.append((huruf, sebaran))
        self._frame.clear()
        self._saran_simpan = ("", [])

    # ----------------------------------------------------- menjaga keselarasan
    def _selaraskan(self, teks: str) -> None:
        """Samakan catatan huruf dengan kata terakhir di teks.

        Teks bisa berubah tanpa lewat huruf_masuk (backspace, kosongkan, spasi,
        koreksi). Bagian awal yang masih sama dipertahankan, sisanya dianggap
        diketik tanpa info peluang.
        """
        kata = kata_terakhir(teks)
        sama = 0
        while (sama < len(kata) and sama < len(self._huruf)
               and self._huruf[sama][0] == kata[sama]):
            sama += 1
        del self._huruf[sama:]
        for h in kata[sama:]:
            self._huruf.append((h, {h: 1.0}))

    # --------------------------------------------------------- pencocokan
    def _biaya(self, huruf: list[tuple[str, dict[str, float]]], calon: str,
               batas: float) -> float:
        """Jarak edit berbobot antara huruf terbaca dan kata calon."""
        n, m = len(huruf), len(calon)
        sebelum = [0.0] * (m + 1)
        for j in range(1, m + 1):
            sebelum[j] = sebelum[j - 1] + self._biaya_lewat(calon[j - 1])
        for i in range(1, n + 1):
            terbaca, sebaran = huruf[i - 1]
            lebih = (BIAYA_LEBIH_KEMBAR if i > 1 and huruf[i - 2][0] == terbaca
                     else BIAYA_LEBIH)
            kini = [sebelum[0] + lebih] + [0.0] * m
            for j in range(1, m + 1):
                c = calon[j - 1]
                if c == terbaca:
                    ganti = 0.0
                else:
                    p_terbaca = max(sebaran.get(terbaca, 0.0), 1e-6)
                    mirip = min(1.0, sebaran.get(c, 0.0) / p_terbaca)
                    ganti = 1.0 - (1.0 - BIAYA_GANTI_MIN) * mirip
                kini[j] = min(sebelum[j - 1] + ganti,
                              sebelum[j] + lebih,
                              kini[j - 1] + self._biaya_lewat(c))
            if min(kini) > batas:
                return math.inf                  # sudah pasti terlalu mahal
            sebelum = kini
        return sebelum[m]

    def _biaya_lewat(self, huruf: str) -> float:
        return BIAYA_LEWAT_DITOLAK if huruf in self.huruf_ditolak else BIAYA_LEWAT

    def _skor(self, kata: str, biaya: float) -> float:
        return biaya + BOBOT_FREKUENSI * self.peringkat[kata] / max(1, len(self.peringkat))

    def cari(self, teks: str, jumlah: int = 3, awalan: bool = False) -> list[tuple[str, float]]:
        """Kata kamus terdekat dengan kata terakhir di teks, urut dari yang terbaik.

        awalan=True dipakai selama huruf masih dieja: kata yang lebih panjang
        dan diawali huruf-huruf itu juga ikut disarankan (MAK -> MAKAN).
        """
        self._selaraskan(teks)
        huruf = self._huruf
        if not huruf or not self.peringkat:
            return []
        n = len(huruf)
        batas = batas_biaya(n)
        hasil: list[tuple[str, float]] = []
        for panjang, daftar in self._per_panjang.items():
            if abs(panjang - n) > 2 and not (awalan and panjang > n):
                continue
            for k in daftar:
                if awalan and len(k) > n:
                    # Bandingkan dengan awal kata saja, sisanya belum dieja.
                    biaya = self._biaya(huruf, k[:n], batas) + 0.05 * (len(k) - n)
                else:
                    biaya = self._biaya(huruf, k, batas)
                if biaya <= batas:
                    hasil.append((k, biaya))
        hasil.sort(key=lambda kb: self._skor(*kb))
        return hasil[:jumlah]

    def saran(self, teks: str, jumlah: int = 3) -> list[str]:
        """Saran untuk ditampilkan selama kata masih dieja."""
        if not self.aktif or not kata_terakhir(teks) or teks.endswith(" "):
            return []
        kunci = f"{jumlah}|{teks}"
        if self._saran_simpan[0] != kunci:
            self._saran_simpan = (kunci, [k for k, _ in self.cari(teks, jumlah, awalan=True)])
        return self._saran_simpan[1]

    def pengganti(self, teks: str) -> str | None:
        """Kata pengganti untuk kata terakhir, atau None kalau tidak perlu diganti."""
        kata = kata_terakhir(teks)
        if not self.aktif or not kata or kata in self.peringkat:
            return None
        calon = self.cari(teks, 2)
        if not calon:
            return None
        if len(calon) > 1 and (self._skor(*calon[1]) - self._skor(*calon[0])
                               < SELISIH_YAKIN):
            return None                          # dua kandidat sama kuat, jangan menebak
        return calon[0][0]

    # ----------------------------------------------------- mengubah teks
    def ganti_kata_terakhir(self, teks: str, pengganti: str, alasan: str) -> str:
        kata = kata_terakhir(teks)
        ujung = teks[len(teks.rstrip()):]
        dasar = teks.rstrip()
        baru = dasar[:len(dasar) - len(kata)] + pengganti + ujung
        self.koreksi_terakhir = (kata, pengganti)
        if self.catat:
            self.catat.teks_berubah(f"kata dikoreksi ({alasan}) {kata} -> {pengganti}", baru)
        return baru

    def akhiri_kata(self, teks: str) -> str:
        """Kata terakhir selesai dieja: koreksi kalau tidak ada di kamus."""
        pengganti = self.pengganti(teks)
        if pengganti:
            teks = self.ganti_kata_terakhir(teks, pengganti, "otomatis")
        self._huruf.clear()
        self._frame.clear()
        return teks

    def pakai_saran(self, teks: str) -> str:
        """Ganti kata yang sedang dieja dengan saran teratas (tombol TAB)."""
        saran = self.saran(teks, 1)
        if not saran or saran[0] == kata_terakhir(teks):
            return teks
        return self.ganti_kata_terakhir(teks, saran[0], "dipilih")

    def batalkan(self, teks: str) -> str:
        """Kembalikan kata asli dari koreksi terakhir, kalau kata itu masih di ujung teks."""
        if not self.koreksi_terakhir:
            return teks
        asli, pengganti = self.koreksi_terakhir
        if kata_terakhir(teks) != pengganti:
            return teks
        self.koreksi_terakhir = None
        ujung = teks[len(teks.rstrip()):]
        dasar = teks.rstrip()
        baru = dasar[:len(dasar) - len(pengganti)] + asli + ujung
        if self.catat:
            self.catat.teks_berubah(f"koreksi dibatalkan {pengganti} -> {asli}", baru)
        return baru


def kata_terakhir(teks: str) -> str:
    bagian = teks.rstrip().rsplit(" ", 1)
    return bagian[-1] if bagian else ""


if __name__ == "__main__":
    # Coba cepat tanpa kamera:  python kamus.py MAKAM BADAK SAJA
    import sys

    from gesture_voice import PATH_MODEL, muat_ambang_huruf

    p = PengoreksiKata.muat(muat_ambang_huruf(nama_model=PATH_MODEL))
    for kata in sys.argv[1:] or ["MAKAM", "SAYA", "TRIMA", "BU", "MINNUM"]:
        kata = kata.upper()
        print(f"{kata:10} -> pengganti {p.pengganti(kata) or '-':10} "
              f"saran {[k for k, _ in p.cari(kata, 5)]}")
        p.akhiri_kata(kata)
