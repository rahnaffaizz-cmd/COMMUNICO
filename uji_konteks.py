"""Uji konteks suara untuk koreksi kata, tanpa kamera dan tanpa mikrofon.

Tiap skenario berisi ucapan lawan bicara dan kata yang dieja dengan isyarat.
Satu huruf sengaja dibuat salah terbaca memakai pasangan yang paling sering
tertukar menurut AKURASI.md (misalnya M jadi N, I jadi J, H jadi Y). Lalu
kata itu dicoba dua kali di kamus.py:

    tanpa konteks   hanya peluang huruf dari model isyarat
    dengan konteks  ditambah ucapan lawan bicara (seperti hasil Whisper)

Jalankan:
    python uji_konteks.py
    python uji_konteks.py "Mau beli obat apa?" PARACETANOL   # coba sendiri

Catatan: ini simulasi kesalahan model, bukan rekaman kamera asli. Untuk angka
yang dilaporkan di makalah, ulangi dengan rekaman nyata dan percakapan.py.
"""

from __future__ import annotations

import sys

from kamus import BERKAS_KAMUS, BERKAS_PRIBADI, BERKAS_TOPIK, PengoreksiKata, baca_daftar_kata, baca_topik

# Huruf -> huruf yang paling sering muncul saat model salah (dari AKURASI.md).
TERTUKAR = {
    "M": "N", "N": "M", "I": "J", "H": "Y", "B": "E", "A": "W", "D": "A",
    "C": "D", "L": "Y", "P": "A", "U": "Y", "Y": "T", "K": "P", "S": "D",
    "X": "Y", "V": "Y", "W": "U", "Z": "E", "R": "P", "F": "Y", "Q": "O",
}

# (ucapan lawan bicara, kata yang sebenarnya dieja, posisi huruf yang salah terbaca)
SKENARIO = [
    ("Mau beli obat apa?", "PARACETAMOL", 8),
    ("Rumahnya di jalan apa?", "MERDEKA", 0),
    ("Kamu sakit apa?", "PUSING", 3),
    ("Mau makan apa?", "NASI", 3),
    ("Kamu mau minum apa?", "TEH", 2),
    ("Sekolahnya di mana?", "SEKOLAH", 6),
    ("Siapa nama gurumu?", "GURU", 2),
    ("Mau naik apa ke sana?", "ANGKOT", 0),
    ("Obatnya untuk apa?", "DEMAM", 4),
    ("Belanja di warung mana?", "WARUNG", 3),
    ("Mau ke apotek mana?", "APOTEK", 1),
    ("Kerjanya di mana?", "KANTOR", 0),
]


def baca_salah(kata: str, posisi: int) -> str:
    h = kata[posisi]
    return kata[:posisi] + TERTUKAR.get(h, h) + kata[posisi + 1:]


def eja(pengoreksi: PengoreksiKata, terbaca: str, benar: str) -> str:
    """Kirim huruf satu per satu seperti dari kamera, lalu akhiri kata."""
    teks = ""
    for h_terbaca, h_benar in zip(terbaca, benar):
        sebaran = {h_terbaca: 0.45}
        if h_benar != h_terbaca:
            sebaran[h_benar] = 0.38          # huruf benar jadi pilihan kedua model
        pengoreksi.catat_frame(list(sebaran), list(sebaran.values()))
        teks += h_terbaca
        pengoreksi.huruf_masuk(h_terbaca, teks)
    return pengoreksi.akhiri_kata(teks)


def coba(ucapan: str, terbaca: str, benar: str | None = None) -> tuple[str, str]:
    benar = benar or terbaca
    kata = baca_daftar_kata(BERKAS_PRIBADI) + baca_daftar_kata(BERKAS_KAMUS)
    topik = baca_topik(BERKAS_TOPIK)
    tanpa = PengoreksiKata(kata, topik=topik, pakai_konteks=False)
    dengan = PengoreksiKata(kata, topik=topik, pakai_konteks=True)
    dengan.tambah_konteks(ucapan)
    return eja(tanpa, terbaca, benar), eja(dengan, terbaca, benar)


def main() -> None:
    if len(sys.argv) == 3:
        ucapan, terbaca = sys.argv[1], sys.argv[2].upper()
        a, b = coba(ucapan, terbaca)
        print(f"Ucapan : {ucapan}\nTerbaca: {terbaca}\n"
              f"Tanpa konteks : {a}\nDengan konteks: {b}")
        return

    benar_tanpa = benar_dengan = 0
    print(f"{'ucapan lawan bicara':28} {'terbaca':12} {'tanpa konteks':14} {'dengan konteks':14} benar")
    print("-" * 82)
    for ucapan, benar, pos in SKENARIO:
        terbaca = baca_salah(benar, pos)
        a, b = coba(ucapan, terbaca, benar)
        benar_tanpa += a == benar
        benar_dengan += b == benar
        tanda = lambda x: ("v " if x == benar else "x ") + x
        print(f"{ucapan:28} {terbaca:12} {tanda(a):14} {tanda(b):14} {benar}")
    n = len(SKENARIO)
    print("-" * 82)
    print(f"Akurasi kata tanpa konteks : {benar_tanpa}/{n} ({100 * benar_tanpa / n:.0f}%)")
    print(f"Akurasi kata dengan konteks: {benar_dengan}/{n} ({100 * benar_dengan / n:.0f}%)")


if __name__ == "__main__":
    main()
