"""Komponen layar bersudut bulat untuk percakapan.py.

Tkinter tidak punya kotak bersudut bulat, gelembung obrolan, atau tulisan di
atas video. Semuanya di sini digambar di Canvas. Warna dan ukuran mengikuti
desain "Tema asli dirapikan" (krem, cokelat zaitun, oranye bata).

Ukuran huruf ditulis dalam piksel (angka negatif untuk Tk), sama dengan desain.
"""

from __future__ import annotations

import math
import tkinter as tk
import tkinter.font as tkfont

# ------------------------------------------------------------------- warna
LATAR = "#EFE4CF"          # jendela
KARTU = "#FAF4E8"          # kartu kiri/kanan
GARIS_KARTU = "#E1D3BA"
PUTIH = "#FFFDF8"          # kotak isian
GARIS = "#D9C6A6"
TEKS = "#4A4A2C"           # cokelat zaitun tua
REDUP = "#7A6A52"
BATA = "#A9542B"           # oranye bata: sisi bahasa isyarat
BATA_TUA = "#7E3D1E"
BATA_MUDA = "#F3E0CC"
ZAITUN = "#566030"         # hijau zaitun: sisi lawan bicara
VIDEO = "#3A3A24"
EMAS = "#E8B87B"
GARIS_FOKUS = "#AF6039"
HIJAU = "#6E8A2E"
ABU = "#B9AA8E"

SANS = "Segoe UI"
SERIF = "Georgia"


def huruf(px: int, *gaya: str, keluarga: str = SANS) -> tuple:
    """Huruf berukuran piksel, seperti di desain."""
    return (keluarga, -px, *gaya)


def campur(warna: str, ke: str, t: float) -> str:
    """Campur dua warna hex; dipakai untuk efek tombol disorot."""
    a = [int(warna[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(ke[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def kotak_bulat(kanvas: tk.Canvas, x1, y1, x2, y2, r, **opsi) -> int:
    """Gambar kotak bersudut bulat. r boleh satu angka atau (kiri-atas, kanan-atas,
    kanan-bawah, kiri-bawah)."""
    if isinstance(r, (int, float)):
        r = (r, r, r, r)
    batas = max(0.0, min((x2 - x1) / 2, (y2 - y1) / 2))
    a, b, c, d = (max(0.0, min(v, batas)) for v in r)
    # Titik digandakan supaya garis lurus tetap lurus saat smooth=True.
    titik = [x1 + a, y1, x1 + a, y1, x2 - b, y1, x2 - b, y1, x2, y1,
             x2, y1 + b, x2, y1 + b, x2, y2 - c, x2, y2 - c, x2, y2,
             x2 - c, y2, x2 - c, y2, x1 + d, y2, x1 + d, y2, x1, y2,
             x1, y2 - d, x1, y2 - d, x1, y1 + a, x1, y1 + a, x1, y1]
    return kanvas.create_polygon(titik, smooth=True, **opsi)


# -------------------------------------------------------------------- ikon
# Ikon garis sederhana, digambar langsung (tanpa berkas gambar).
def ikon_suara(k: tk.Canvas, x: float, cy: float, warna: str, *, mati: bool = False) -> int:
    k.create_polygon(x, cy - 3, x + 4, cy - 3, x + 9, cy - 7, x + 9, cy + 7, x + 4, cy + 3,
                     x, cy + 3, fill="", outline=warna, width=2, joinstyle="round")
    if mati:
        k.create_line(x + 12, cy - 4, x + 18, cy + 4, fill=warna, width=2, capstyle="round")
        k.create_line(x + 18, cy - 4, x + 12, cy + 4, fill=warna, width=2, capstyle="round")
    else:
        k.create_arc(x + 5, cy - 6, x + 15, cy + 6, start=-50, extent=100,
                     style="arc", outline=warna, width=2)
    return 18


def ikon_mikrofon(k: tk.Canvas, cx: float, cy: float, warna: str) -> None:
    kotak_bulat(k, cx - 4, cy - 10, cx + 4, cy + 3, 4, fill="", outline=warna, width=2)
    k.create_arc(cx - 8, cy - 6, cx + 8, cy + 7, start=180, extent=180,
                 style="arc", outline=warna, width=2)
    k.create_line(cx, cy + 7, cx, cy + 11, fill=warna, width=2, capstyle="round")


def ikon_berhenti(k: tk.Canvas, cx: float, cy: float, warna: str) -> None:
    kotak_bulat(k, cx - 7, cy - 7, cx + 7, cy + 7, 2, fill=warna, outline=warna)


# ------------------------------------------------------------------ panel
class PanelBulat(tk.Canvas):
    """Wadah bersudut bulat. Isinya diletakkan di frame `.isi`.

    ikut_isi=True   tinggi panel mengikuti isinya (kotak kecil, baris tombol)
    ikut_isi=False  isi mengikuti ukuran panel (kartu besar yang mengisi jendela)
    """

    def __init__(self, induk, *, latar_induk: str, isi: str, garis: str | None = None,
                 r=18, pad: int | tuple[int, int] = 20, tebal: float = 1,
                 ikut_isi: bool = True, **opsi):
        super().__init__(induk, bg=latar_induk, highlightthickness=0, bd=0, **opsi)
        self.warna_isi, self.warna_garis = isi, garis
        self.r, self.tebal, self.ikut_isi = r, tebal, ikut_isi
        self.padx, self.pady = (pad, pad) if isinstance(pad, int) else pad
        self.isi = tk.Frame(self, bg=isi)
        self._jendela = self.create_window(self.padx, self.pady, window=self.isi, anchor="nw")
        self.bind("<Configure>", self._ubah_ukuran)
        if ikut_isi:
            self.isi.bind("<Configure>", self._isi_berubah)

    def _isi_berubah(self, _e=None) -> None:
        lebar = self.isi.winfo_reqwidth() + 2 * self.padx
        tinggi = self.isi.winfo_reqheight() + 2 * self.pady
        if int(self.cget("height")) != tinggi:
            self.configure(height=tinggi)
        if int(self.cget("width")) != lebar:
            self.configure(width=lebar)

    def _ubah_ukuran(self, e=None) -> None:
        lebar, tinggi = self.winfo_width(), self.winfo_height()
        self.delete("latar")
        setengah = self.tebal / 2
        kotak_bulat(self, setengah, setengah, lebar - setengah - 1, tinggi - setengah - 1,
                    self.r, fill=self.warna_isi, outline=self.warna_garis or self.warna_isi,
                    width=self.tebal if self.warna_garis else 0, tags="latar")
        self.tag_lower("latar")
        self.itemconfigure(self._jendela, width=max(1, lebar - 2 * self.padx))
        if not self.ikut_isi:
            self.itemconfigure(self._jendela, height=max(1, tinggi - 2 * self.pady))

    def warnai(self, *, isi: str | None = None, garis: str | None = None) -> None:
        if isi:
            self.warna_isi = isi
            self.isi.configure(bg=isi)
        if garis:
            self.warna_garis = garis
        self._ubah_ukuran()


# ------------------------------------------------------------------ tombol
class TombolBulat(tk.Canvas):
    """Tombol bersudut bulat. perintah=None membuatnya jadi lencana biasa."""

    def __init__(self, induk, teks: str, perintah=None, *, latar_induk: str, isi: str,
                 warna_teks: str, garis: str | None = None, r=12, tinggi: int = 50,
                 lebar: int | None = None, font=None, hint: str | None = None,
                 warna_hint: str | None = None, ikon=None, padx: int = 14,
                 isi_sorot: str | None = None):
        super().__init__(induk, bg=latar_induk, highlightthickness=0, bd=0, height=tinggi,
                         takefocus=1 if perintah else 0,
                         cursor="hand2" if perintah else "")
        self.perintah = perintah
        self.teks, self.hint, self.ikon = teks, hint, ikon
        self.warna_isi, self.warna_teks, self.warna_garis = isi, warna_teks, garis
        self.warna_hint = warna_hint or warna_teks
        self.isi_sorot = isi_sorot
        self.r, self.padx, self.lebar_tetap = r, padx, lebar
        self.font = tkfont.Font(font=font or huruf(15, "bold"))
        self.font_hint = tkfont.Font(font=huruf(11, "bold"))
        self._sorot = self._tekan = self._fokus = False
        self._sesuaikan_lebar()
        self.bind("<Configure>", lambda _e: self._gambar())
        if perintah:
            self.bind("<Enter>", lambda _e: self._ganti("_sorot", True))
            self.bind("<Leave>", lambda _e: (self._ganti("_sorot", False),
                                             self._ganti("_tekan", False)))
            self.bind("<ButtonPress-1>", lambda _e: self._ganti("_tekan", True))
            self.bind("<ButtonRelease-1>", self._lepas)
            self.bind("<FocusIn>", lambda _e: self._ganti("_fokus", True))
            self.bind("<FocusOut>", lambda _e: self._ganti("_fokus", False))
            self.bind("<Return>", lambda _e: self.perintah())
            self.bind("<space>", lambda _e: self.perintah())

    def _lebar_isi(self) -> int:
        lebar = self.font.measure(self.teks)
        if self.hint:
            lebar += 8 + self.font_hint.measure(self.hint)
        if self.ikon:
            lebar += 26
        return lebar

    def _sesuaikan_lebar(self) -> None:
        self.configure(width=self.lebar_tetap or self._lebar_isi() + 2 * self.padx)

    def _ganti(self, nama: str, nilai: bool) -> None:
        setattr(self, nama, nilai)
        self._gambar()

    def _lepas(self, e) -> None:
        dalam = 0 <= e.x <= self.winfo_width() and 0 <= e.y <= self.winfo_height()
        self._ganti("_tekan", False)
        if dalam and self.perintah:
            self.perintah()

    def atur(self, **ubah) -> None:
        """Ganti teks atau warna, misalnya atur(teks="Putuskan", isi=ZAITUN)."""
        for kunci, nilai in ubah.items():
            setattr(self, {"isi": "warna_isi", "garis": "warna_garis",
                           "warna_teks": "warna_teks"}.get(kunci, kunci), nilai)
        if "warna_teks" in ubah and "warna_hint" not in ubah:
            self.warna_hint = ubah["warna_teks"]
        self._sesuaikan_lebar()
        self._gambar()

    def _gambar(self) -> None:
        self.delete("all")
        lebar, tinggi = self.winfo_width(), self.winfo_height()
        if lebar < 4 or tinggi < 4:
            return
        isi = self.warna_isi
        if self._tekan:
            isi = campur(isi, "#000000", 0.12)
        elif self._sorot:
            isi = self.isi_sorot or campur(isi, "#000000", 0.05)
        garis = BATA if self._fokus else self.warna_garis
        kotak_bulat(self, 1, 1, lebar - 2, tinggi - 2, self.r, fill=isi,
                    outline=garis or isi, width=2 if self._fokus else (1 if garis else 0))
        x = (lebar - self._lebar_isi()) / 2
        cy = tinggi / 2
        if self.ikon:
            self.ikon(self, x, cy, self.warna_teks)
            x += 26
        self.create_text(x, cy, text=self.teks, anchor="w", font=self.font,
                         fill=self.warna_teks)
        if self.hint:
            x += self.font.measure(self.teks) + 8
            self.create_text(x, cy + 1, text=self.hint, anchor="w", font=self.font_hint,
                             fill=self.warna_hint)


# ----------------------------------------------------------------- isian
class IsianBulat(PanelBulat):
    """Kotak ketik bersudut bulat; Entry-nya ada di `.entry`."""

    def __init__(self, induk, *, latar_induk: str, garis: str = GARIS, garis_fokus: str = BATA,
                 tinggi: int = 50, r=12, font=None, lebar: int | None = None, **opsi_entry):
        super().__init__(induk, latar_induk=latar_induk, isi=PUTIH, garis=garis, r=r,
                         pad=(14, 4), tebal=1.5, ikut_isi=False, height=tinggi,
                         **({"width": lebar} if lebar else {}))
        self._garis_biasa, self._garis_fokus = garis, garis_fokus
        self.isi.rowconfigure(0, weight=1)
        self.isi.columnconfigure(0, weight=1)
        self.entry = tk.Entry(self.isi, relief="flat", bd=0, bg=PUTIH, fg=TEKS,
                              insertbackground=BATA, selectbackground=GARIS,
                              disabledbackground=PUTIH, disabledforeground=REDUP,
                              highlightthickness=0, font=font or huruf(16), **opsi_entry)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.entry.bind("<FocusIn>", lambda _e: self.warnai(garis=self._garis_fokus))
        self.entry.bind("<FocusOut>", lambda _e: self.warnai(garis=self._garis_biasa))


# ----------------------------------------------------------------- video
class VideoBulat(tk.Canvas):
    """Kotak video gelap bersudut bulat, dengan lencana tangan dan kartu huruf."""

    def __init__(self, induk, *, latar_induk: str, r: int = 14, **opsi):
        super().__init__(induk, bg=latar_induk, highlightthickness=0, bd=0, **opsi)
        self.latar_induk, self.r = latar_induk, r
        self._foto = None
        self._teks_kosong = "membuka kamera ..."
        self._tangan = 0
        self._huruf: str | None = None
        self._yakin: float | None = None
        self.bind("<Configure>", lambda _e: self._gambar())

    @property
    def ukuran(self) -> tuple[int, int]:
        return max(1, self.winfo_width()), max(1, self.winfo_height())

    def tampilkan(self, foto: tk.PhotoImage | None, teks_kosong: str | None = None) -> None:
        self._foto = foto
        if teks_kosong is not None:
            self._teks_kosong = teks_kosong
        self._gambar()

    def atur_status(self, tangan: int, huruf_terbaca: str | None, yakin: float | None) -> None:
        self._tangan, self._huruf, self._yakin = tangan, huruf_terbaca, yakin

    def _kotak(self, lebar: int, tinggi: int) -> tuple[float, float, float, float]:
        """Kotak video mengikuti bentuk gambar kamera, di tengah ruang yang tersedia."""
        if self._foto is not None:
            w, h = self._foto.width(), self._foto.height()
            self._rasio = w / max(1, h)
        else:
            rasio = getattr(self, "_rasio", 4 / 3)   # bentuk kamera terakhir, awalnya 4:3
            w = min(lebar, tinggi * rasio)
            h = w / rasio
        x0, y0 = (lebar - w) / 2, (tinggi - h) / 2
        return x0, y0, x0 + w, y0 + h

    def _gambar(self) -> None:
        self.delete("all")
        lebar, tinggi = self.ukuran
        x0, y0, x1, y1 = self._kotak(lebar, tinggi)
        self.create_rectangle(x0, y0, x1, y1, fill=VIDEO, outline="")
        if self._foto is not None:
            self.create_image((x0 + x1) / 2, (y0 + y1) / 2, image=self._foto)
        else:
            self.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=self._teks_kosong,
                             fill=GARIS, font=huruf(15))
            self._sudut(x0, y0, x1, y1)
            return

        # Lencana tangan, kiri atas gambar.
        teks = ("TIDAK ADA TANGAN" if not self._tangan else
                "TANGAN TERDETEKSI" if self._tangan == 1 else "2 TANGAN TERDETEKSI")
        f = tkfont.Font(font=huruf(13, "bold"))
        kotak_bulat(self, x0 + 14, y0 + 14, x0 + 14 + f.measure(teks) + 24, y0 + 46, 16,
                    fill=KARTU, outline="")
        self.create_text(x0 + 26, y0 + 30, text=teks, anchor="w", font=f,
                         fill=TEKS if self._tangan else REDUP)

        # Kartu huruf terbaca, kiri bawah gambar.
        if self._tangan:
            # Kecil saja, supaya tidak menutupi tangan.
            kx, ky = x0 + 12, y1 - 12 - 44
            kotak_bulat(self, kx, ky, kx + 132, ky + 44, 10, fill=KARTU, outline="")
            kotak_bulat(self, kx + 6, ky + 6, kx + 38, ky + 38, 7, fill=BATA, outline="")
            self.create_text(kx + 22, ky + 22, text=self._huruf or "…", fill=KARTU,
                             font=huruf(19, "bold", keluarga=SERIF))
            self.create_text(kx + 46, ky + 14, text="Huruf terbaca", anchor="w", fill=REDUP,
                             font=huruf(10))
            yakin = f"{self._yakin * 100:.0f}% yakin" if self._yakin is not None else "membaca…"
            self.create_text(kx + 46, ky + 30, text=yakin, anchor="w", fill=TEKS,
                             font=huruf(13, "bold"))
        self._sudut(x0, y0, x1, y1)

    def _sudut(self, kiri: float, atas: float, kanan: float, bawah: float) -> None:
        # Tutup keempat sudut gambar dengan warna kartu supaya tampak membulat.
        r = self.r
        # Sudut busur dihitung dengan sumbu y ke atas (y layar dibalik di bawah).
        for (cx, cy, x0, y0, mulai) in ((kiri + r, atas + r, kiri, atas, 90),
                                         (kanan - r, atas + r, kanan, atas, 0),
                                         (kanan - r, bawah - r, kanan, bawah, 270),
                                         (kiri + r, bawah - r, kiri, bawah, 180)):
            titik = [x0, y0]
            for i in range(9):
                sudut = math.radians(mulai + i * 90 / 8)
                titik += [cx + r * math.cos(sudut), cy - r * math.sin(sudut)]
            self.create_polygon(titik, fill=self.latar_induk, outline="")


# ----------------------------------------------------------- teks disusun
class KotakSusun(tk.Canvas):
    """Kotak teks yang sedang disusun dari isyarat, dengan kursor dan hitung mundur."""

    def __init__(self, induk, *, latar_induk: str, tinggi: int = 60):
        super().__init__(induk, bg=latar_induk, highlightthickness=0, bd=0, height=tinggi)
        self._teks = ""
        self._sisa: float | None = None          # 0..1 menuju terkirim otomatis
        self.font = tkfont.Font(font=huruf(26, "bold"))
        self.bind("<Configure>", lambda _e: self._gambar())

    def atur(self, teks: str, kemajuan: float | None) -> None:
        if teks == self._teks and kemajuan == self._sisa:
            return
        self._teks, self._sisa = teks, kemajuan
        self._gambar()

    def _gambar(self) -> None:
        self.delete("all")
        lebar, tinggi = self.winfo_width(), self.winfo_height()
        if lebar < 10:
            return
        kotak_bulat(self, 1, 1, lebar - 2, tinggi - 2, 12, fill=PUTIH, outline=GARIS)
        if self._teks:
            teks = self._teks
            # Teks panjang: tampilkan ujungnya saja, yang sedang diketik.
            while len(teks) > 1 and self.font.measure(teks) > lebar - 50:
                teks = teks[1:]
            self.create_text(16, tinggi / 2, text=teks, anchor="w", font=self.font, fill=TEKS)
            x = 16 + self.font.measure(teks) + 3
            self.create_rectangle(x, tinggi / 2 - 14, x + 3, tinggi / 2 + 14, fill=BATA,
                                  outline="")
        else:
            self.create_text(16, tinggi / 2, text="Peragakan huruf di depan kamera…",
                             anchor="w", font=huruf(15), fill=REDUP)
        if self._sisa:
            # Garis tipis di bawah: makin penuh, makin dekat terkirim otomatis.
            self.create_line(12, tinggi - 5, 12 + (lebar - 24) * min(1.0, self._sisa),
                             tinggi - 5, fill=BATA, width=3, capstyle="round")


# ----------------------------------------------------------------- obrolan
class ObrolanBulat(tk.Canvas):
    """Riwayat percakapan berupa gelembung, bisa digulir dengan roda tetikus."""

    def __init__(self, induk, *, latar: str, nama: dict[str, str]):
        super().__init__(induk, bg=latar, highlightthickness=0, bd=0, height=200)
        self.nama = nama
        self.pesan: list = []
        self.f_nama = tkfont.Font(font=huruf(13, "bold"))
        self.f_jam = tkfont.Font(font=huruf(13))
        self.f_isyarat = tkfont.Font(font=huruf(18, "bold"))
        self.f_lawan = tkfont.Font(font=huruf(17))
        self.bind("<Configure>", lambda _e: self._tata())
        self.bind("<Enter>", lambda _e: self.bind_all("<MouseWheel>", self._gulir))
        self.bind("<Leave>", lambda _e: self.unbind_all("<MouseWheel>"))

    def _gulir(self, e) -> None:
        self.yview_scroll(int(-e.delta / 120) * 30, "units")

    def tambah(self, pesan) -> None:
        self.pesan.append(pesan)
        self._tata()
        self.yview_moveto(1.0)

    def kosongkan(self) -> None:
        self.pesan.clear()
        self._tata()

    def _tata(self) -> None:
        self.delete("all")
        lebar, tinggi = self.winfo_width(), self.winfo_height()
        if lebar < 50:
            return
        if not self.pesan:
            self.create_text(lebar / 2, tinggi / 2, text="Belum ada percakapan.",
                             fill=REDUP, font=huruf(15))
            self.configure(scrollregion=(0, 0, lebar, tinggi))
            return

        batas = int(lebar * 0.8)
        y = 6
        for p in self.pesan:
            isyarat = p.pembicara == "tuli"
            nama = self.nama[p.pembicara] + ("" if isyarat else
                                             f" · {'Suara' if p.via == 'suara' else 'Ketik'}")
            jam = f"{p.waktu:%H.%M}"
            warna_nama = BATA if isyarat else ZAITUN

            # Baris nama + jam.
            lebar_nama = self.f_nama.measure(nama) + 6 + self.f_jam.measure(jam)
            x_nama = 4 if isyarat else lebar - 4 - lebar_nama
            self.create_text(x_nama, y, text=nama, anchor="nw", font=self.f_nama,
                             fill=warna_nama)
            self.create_text(x_nama + self.f_nama.measure(nama) + 6, y, text=jam,
                             anchor="nw", font=self.f_jam, fill=REDUP)
            y += self.f_nama.metrics("linespace") + 4

            # Gelembung: tulis teksnya dulu untuk tahu ukurannya.
            teks = p.teks.upper() if isyarat else p.teks
            font = self.f_isyarat if isyarat else self.f_lawan
            item = self.create_text(0, 0, text=teks, anchor="nw", font=font,
                                    width=batas - 32,
                                    fill=TEKS if isyarat else KARTU)
            x1, y1, x2, y2 = self.bbox(item)
            lebar_g, tinggi_g = (x2 - x1) + 32, (y2 - y1) + 22
            x = 4 if isyarat else lebar - 4 - lebar_g
            self.coords(item, x + 16, y + 11)
            latar = kotak_bulat(self, x, y, x + lebar_g, y + tinggi_g,
                                (4, 16, 16, 16) if isyarat else (16, 4, 16, 16),
                                fill=BATA_MUDA if isyarat else ZAITUN, outline="")
            self.tag_lower(latar, item)
            y += tinggi_g + 14
        self.configure(scrollregion=(0, 0, lebar, max(y, tinggi)))


class TitikStatus(tk.Canvas):
    """Titik kecil berwarna di depan teks keadaan."""

    def __init__(self, induk, *, latar_induk: str, warna: str = ABU):
        super().__init__(induk, bg=latar_induk, width=8, height=8, highlightthickness=0, bd=0)
        self._titik = self.create_oval(0, 0, 8, 8, fill=warna, outline="")

    def warnai(self, warna: str) -> None:
        self.itemconfigure(self._titik, fill=warna)


class TombolMik(tk.Canvas):
    """Tombol bundar mikrofon: mulai / berhenti mendengarkan."""

    def __init__(self, induk, perintah, *, latar_induk: str):
        super().__init__(induk, bg=latar_induk, width=52, height=52, highlightthickness=0,
                         bd=0, cursor="hand2", takefocus=1)
        self.perintah = perintah
        self.dengar = False
        self.bind("<ButtonRelease-1>", lambda _e: self.perintah())
        self.bind("<Return>", lambda _e: self.perintah())
        self.bind("<space>", lambda _e: self.perintah())
        self._gambar()

    def atur(self, dengar: bool) -> None:
        if dengar != self.dengar:
            self.dengar = dengar
            self._gambar()

    def _gambar(self) -> None:
        self.delete("all")
        if self.dengar:
            self.create_oval(1, 1, 51, 51, fill=BATA_TUA, outline=EMAS, width=4)
            ikon_berhenti(self, 26, 26, KARTU)
        else:
            self.create_oval(1, 1, 51, 51, fill=BATA, outline="")
            ikon_mikrofon(self, 26, 25, KARTU)


class Tingkat(tk.Canvas):
    """Batang kecil kerasnya suara."""

    def __init__(self, induk, *, latar_induk: str, lebar: int = 120):
        super().__init__(induk, bg=latar_induk, width=lebar, height=8, highlightthickness=0,
                         bd=0)
        self._lebar = lebar
        kotak_bulat(self, 0, 0, lebar, 8, 4, fill=LATAR, outline="")
        self._isi = kotak_bulat(self, 0, 0, 0, 8, 4, fill=BATA, outline="")

    def atur(self, nilai: float) -> None:
        self.delete(self._isi)
        if nilai > 0.01:
            self._isi = kotak_bulat(self, 0, 0, max(8, self._lebar * min(1.0, nilai)), 8, 4,
                                    fill=BATA, outline="")
