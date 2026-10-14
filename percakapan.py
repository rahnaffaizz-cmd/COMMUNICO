"""Percakapan dua sisi: penyandang tuli (isyarat) dan lawan bicara (mengetik).

Jendelanya dibagi dua, berdampingan:

    KIRI   penyandang tuli. Video kamera, dan di bawahnya teks hasil
           pembacaan isyarat (huruf yang sedang disusun, lalu riwayat
           percakapan).
    KANAN  lawan bicara yang tidak tuli. Mengetik pesan, dan riwayat
           percakapan ditulis di atasnya.

Tiap baris percakapan diberi tanda siapa yang bicara: warna biru dan rata kiri
untuk penyandang tuli, hijau dan rata kanan untuk lawan bicara, lengkap dengan
nama dan jam. Riwayat yang sama muncul di kedua sisi, jadi penyandang tuli ikut
membaca ketikan lawan bicaranya.

Cara pakai:
    python percakapan.py
    python percakapan.py --kamera 1 --model rf_bisindo_v7.pkl
    python percakapan.py --tanpa-suara        # pesan isyarat tidak diucapkan

Tombol (selain tombol di layar):
    ENTER di kotak ketik   kirim pesan lawan bicara
    F2   kirim teks isyarat sekarang juga
    F3   hapus huruf isyarat terakhir
    F4   kosongkan teks isyarat
    F5   tambah spasi pada teks isyarat
    F6   suara menyala / mati

Pesan isyarat terkirim sendiri kalau tidak ada huruf baru selama --ucap-setelah
detik (bawaan 2,5). Pembacaan hurufnya memakai bagian yang sama dengan
gesture_voice.py: voting, ambang per huruf, dan tangan hilang sesaat.

Riwayat percakapan juga disimpan ke log/percakapan_<waktu>.txt.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import tkinter as tk
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np

from fitur import ambil_fitur, sesuaikan_untuk

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message=".*GetPrototype.*")

FOLDER = Path(__file__).resolve().parent
FOLDER_LOG = FOLDER / "log"

LEBAR_VIDEO, TINGGI_VIDEO = 480, 360

# Warna tiap pembicara. Dipakai di kedua sisi supaya konsisten.
NAMA = {"tuli": "Bahasa Isyarat", "dengar": "Lawan bicara"}
WARNA_NAMA = {"tuli": "#0d47a1", "dengar": "#1b5e20"}
WARNA_LATAR = {"tuli": "#e3f2fd", "dengar": "#e8f5e9"}

TANPA_WAKTU = float("-inf")


# ------------------------------------------------------------------- percakapan
@dataclass
class Pesan:
    pembicara: str          # "tuli" atau "dengar"
    teks: str
    waktu: datetime


class Riwayat:
    """Daftar pesan, dan catatan teksnya di log/ (satu baris satu pesan)."""

    def __init__(self, simpan: bool):
        self.pesan: list[Pesan] = []
        self._berkas: Path | None = None
        if simpan:
            FOLDER_LOG.mkdir(exist_ok=True)
            self._berkas = FOLDER_LOG / f"percakapan_{datetime.now():%Y%m%d_%H%M%S}.txt"

    def tambah(self, pembicara: str, teks: str) -> Pesan:
        baru = Pesan(pembicara, teks, datetime.now())
        self.pesan.append(baru)
        if self._berkas:
            with open(self._berkas, "a", encoding="utf-8") as f:
                f.write(f"[{baru.waktu:%H:%M:%S}] {pembicara.upper():7s}: {teks}\n")
        return baru


# --------------------------------------------------- dari isyarat jadi teks
class PenyusunTeks:
    """Mengubah aliran tebakan model jadi huruf, lalu jadi pesan.

    Aturannya sama dengan loop di gesture_voice.py: voting menyaring huruf,
    tangan hilang lama menambah spasi, dan teks terkirim sendiri kalau tidak ada
    huruf baru selama `ucap_setelah` detik. Dipisah dari kamera dan layar supaya
    bisa diuji sendiri.
    """

    def __init__(self, model, voting, catat, *, ucap_setelah: float,
                 spasi_setelah: float, tahan_tangan_hilang: int,
                 kirim_otomatis: bool = True):
        self.model = model
        self.voting = voting
        self.catat = catat
        self.ucap_setelah = ucap_setelah
        self.spasi_setelah = spasi_setelah
        self.tahan_tangan_hilang = tahan_tangan_hilang
        self.kirim_otomatis = kirim_otomatis

        self.teks = ""
        self._waktu_huruf = TANPA_WAKTU
        self._sejak_hilang: float | None = None
        # Mulai dari angka besar supaya satu tangan di awal (huruf satu tangan)
        # tidak dianggap tangan yang hilang.
        self._frame_sejak_dua_tangan = 10 ** 9

    def langkah(self, angka, jumlah_tangan: int, sekarang: float):
        """Proses satu frame. Balik (status untuk layar, daftar kejadian)."""
        kejadian: list[tuple[str, str]] = []
        mode, tebakan, keyakinan = "diam", "", 0.0

        if jumlah_tangan == 2:
            self._frame_sejak_dua_tangan = 0
        elif jumlah_tangan == 1:
            self._frame_sejak_dua_tangan += 1
        else:
            self._frame_sejak_dua_tangan = 10 ** 9
        tangan_goyah = (jumlah_tangan == 1
                        and self._frame_sejak_dua_tangan <= self.tahan_tangan_hilang)

        if angka is not None and tangan_goyah:
            # Satu tangan sesaat setelah dua tangan: hampir pasti tangan yang
            # hilang dari deteksi, bukan huruf satu tangan. Dilewati.
            self._sejak_hilang = None
            mode = "baca"
        elif angka is not None:
            self._sejak_hilang = None
            mode = "baca"
            fitur = sesuaikan_untuk(
                ambil_fitur(angka, getattr(self.model, "urut_tangan", False)),
                self.model.n_features_in_)
            peluang = self.model.predict_proba(fitur)[0]
            tertinggi = int(np.argmax(peluang))
            tebakan = str(self.model.classes_[tertinggi])
            keyakinan = float(peluang[tertinggi])

            diterima = self.voting.masukkan(tebakan, keyakinan, sekarang)
            if diterima:
                self.teks += diterima
                self._waktu_huruf = sekarang
                self.catat.huruf_diterima(diterima, self.teks)
                kejadian.append(("huruf", diterima))
        else:
            # Tangan hilang dari layar. Kalau lama, satu kata dianggap selesai.
            self.voting.kosongkan()
            if self._sejak_hilang is None:
                self._sejak_hilang = sekarang
            elif (sekarang - self._sejak_hilang >= self.spasi_setelah
                  and self.teks and not self.teks.endswith(" ")):
                self.teks += " "
                self.catat.teks_berubah("spasi ditambah otomatis", self.teks)

        sisa = max(0.0, self.ucap_setelah - (sekarang - self._waktu_huruf))
        if (self.kirim_otomatis and sisa == 0.0 and self.teks.strip()
                and self._waktu_huruf != TANPA_WAKTU):
            kejadian.append(("pesan", self.kirim()))
            sisa = self.ucap_setelah

        huruf_unggul, kemajuan = self.voting.kemajuan
        status = {
            "mode": mode, "tebakan": tebakan, "keyakinan": keyakinan,
            "tangan": jumlah_tangan, "huruf_unggul": huruf_unggul,
            "kemajuan": kemajuan, "teks": self.teks,
            # None kalau tidak ada yang menunggu terkirim.
            "sisa": sisa if self.teks.strip() and self._waktu_huruf != TANPA_WAKTU
            else None,
            "ucap_setelah": self.ucap_setelah,
        }
        return status, kejadian

    def kirim(self) -> str:
        """Ambil teks sekarang sebagai satu pesan, lalu kosongkan."""
        hasil = self.teks.strip()
        self.teks = ""
        self._waktu_huruf = TANPA_WAKTU
        return hasil

    def perintah(self, nama: str) -> str | None:
        """Perintah dari tombol. Balik teks pesan kalau perintahnya `kirim`."""
        if nama == "kirim":
            return self.kirim() or None
        if nama == "hapus":
            self.teks = self.teks[:-1]
            self.catat.teks_berubah("huruf terakhir dihapus", self.teks)
        elif nama == "kosong":
            self.teks = ""
            self.catat.teks_berubah("teks dikosongkan", self.teks)
        elif nama == "spasi" and self.teks and not self.teks.endswith(" "):
            self.teks += " "
            self.catat.teks_berubah("spasi ditambah manual", self.teks)
        return None


# ------------------------------------------------------------ kamera di thread
class PekerjaKamera(threading.Thread):
    """Membaca kamera dan menjalankan pengenalan di thread sendiri.

    Layar (tkinter) hanya boleh disentuh dari thread utamanya. Jadi pekerjaan
    berat, yaitu kamera, MediaPipe, dan model, dikerjakan di sini, lalu hasilnya
    dititipkan lewat antrian. Perintah dari tombol datang lewat antrian lain.
    """

    def __init__(self, sumber, pendeteksi, penyusun: PenyusunTeks, *, gambar_titik=None):
        super().__init__(daemon=True)
        self.sumber = sumber                  # punya .read() seperti cv2.VideoCapture
        self.pendeteksi = pendeteksi          # punya .process(rgb) seperti MediaPipe Hands
        self.penyusun = penyusun
        self.gambar_titik = gambar_titik      # fungsi (frame, hasil) -> None, boleh kosong
        self.keluar: queue.Queue = queue.Queue(maxsize=3)
        self.perintah: queue.Queue = queue.Queue()
        self.berhenti = threading.Event()
        self.galat: str | None = None

    def _titip(self, isi) -> None:
        # Layar lambat? Buang kiriman terlama, jangan menumpuk.
        while True:
            try:
                self.keluar.put_nowait(isi)
                return
            except queue.Full:
                try:
                    self.keluar.get_nowait()
                except queue.Empty:
                    pass

    def run(self) -> None:
        from gesture_voice import vektor_titik_tangan

        try:
            while not self.berhenti.is_set():
                kejadian_perintah: list[tuple[str, str]] = []
                while True:
                    try:
                        nama = self.perintah.get_nowait()
                    except queue.Empty:
                        break
                    teks = self.penyusun.perintah(nama)
                    if teks:
                        kejadian_perintah.append(("pesan", teks))

                berhasil, frame = self.sumber.read()
                if not berhasil:
                    self.galat = "kamera berhenti mengirim gambar"
                    return
                frame = cv2.flip(frame, 1)

                hasil = self.pendeteksi.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                angka, jumlah_tangan = vektor_titik_tangan(hasil)
                status, kejadian = self.penyusun.langkah(angka, jumlah_tangan,
                                                         time.monotonic())
                if self.gambar_titik is not None:
                    self.gambar_titik(frame, hasil)

                self._titip({"gambar": _ke_ppm(frame), "status": status,
                             "kejadian": kejadian_perintah + kejadian})
        except Exception as e:  # noqa: BLE001 - dilaporkan ke layar, bukan ditelan
            self.galat = f"{type(e).__name__}: {e}"
        finally:
            self._titip({"gambar": None, "status": None, "kejadian": [],
                         "selesai": True})


def _ke_ppm(frame_bgr: np.ndarray) -> bytes:
    """Kecilkan frame sesuai kotak video, lalu ubah ke format PPM yang dibaca Tk."""
    tinggi, lebar = frame_bgr.shape[:2]
    skala = min(LEBAR_VIDEO / lebar, TINGGI_VIDEO / tinggi)
    kecil = cv2.resize(frame_bgr, (max(1, int(lebar * skala)), max(1, int(tinggi * skala))))
    rgb = cv2.cvtColor(kecil, cv2.COLOR_BGR2RGB)
    return b"P6 %d %d 255\n" % (rgb.shape[1], rgb.shape[0]) + rgb.tobytes()


# ------------------------------------------------------------------------ layar
class PanelObrolan(tk.Text):
    """Kotak teks hanya-baca untuk riwayat. Tiap pesan: nama + jam, lalu isinya."""

    def __init__(self, master, *, ukuran_font: int, **opsi):
        super().__init__(master, wrap="word", state="disabled", relief="flat",
                         padx=10, pady=8, cursor="arrow", bg="#fafafa",
                         font=("Segoe UI", ukuran_font), spacing1=0, spacing3=0, **opsi)
        for siapa in ("tuli", "dengar"):
            rata = "left" if siapa == "tuli" else "right"
            tepi_dekat, tepi_jauh = 6, 70
            kiri, kanan = ((tepi_dekat, tepi_jauh) if siapa == "tuli"
                           else (tepi_jauh, tepi_dekat))
            self.tag_configure(f"{siapa}_nama", justify=rata, lmargin1=kiri,
                               lmargin2=kiri, rmargin=kanan, spacing1=10,
                               foreground=WARNA_NAMA[siapa],
                               font=("Segoe UI", max(8, ukuran_font - 3), "bold"))
            self.tag_configure(f"{siapa}_isi", justify=rata, lmargin1=kiri,
                               lmargin2=kiri, rmargin=kanan, spacing1=1, spacing3=2,
                               background=WARNA_LATAR[siapa])

    def tambah(self, pesan: Pesan) -> None:
        siapa = pesan.pembicara
        self.configure(state="normal")
        self.insert("end", f"{NAMA[siapa]}  {pesan.waktu:%H:%M:%S}\n", f"{siapa}_nama")
        self.insert("end", f" {pesan.teks} \n", f"{siapa}_isi")
        self.configure(state="disabled")
        self.see("end")


class Aplikasi:
    def __init__(self, akar: tk.Tk, pekerja: PekerjaKamera, pengucap, riwayat: Riwayat,
                 catat, *, suara: bool):
        self.akar = akar
        self.pekerja = pekerja
        self.pengucap = pengucap
        self.riwayat = riwayat
        self.catat = catat
        self.suara = tk.BooleanVar(value=suara)
        self._antri_suara: list[str] = []
        self._gambar_sekarang = None          # simpan rujukan, kalau tidak Tk membuangnya

        akar.title("BISINDO - Percakapan dua sisi")
        # Ukuran mengikuti layar, supaya riwayat di bawah video tetap kelihatan
        # walaupun layarnya kecil atau pembesaran Windows-nya tinggi.
        lebar = min(1400, akar.winfo_screenwidth() - 80)
        tinggi = min(900, akar.winfo_screenheight() - 120)
        akar.geometry(f"{lebar}x{tinggi}+20+20")
        akar.minsize(900, 600)
        akar.configure(bg="#eceff1")
        akar.columnconfigure(0, weight=1, uniform="sisi")
        akar.columnconfigure(1, weight=1, uniform="sisi")
        akar.rowconfigure(0, weight=1)

        self._bangun_kiri()
        self._bangun_kanan()
        self._pasang_tombol()
        akar.protocol("WM_DELETE_WINDOW", self.tutup)

    # ---- sisi kiri: penyandang tuli
    def _bangun_kiri(self) -> None:
        kiri = tk.Frame(self.akar, bg="#eceff1")
        kiri.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        kiri.columnconfigure(0, weight=1)
        kiri.rowconfigure(7, weight=1)

        tk.Label(kiri, text="BAHASA ISYARAT  -  kamera", anchor="w",
                 font=("Segoe UI", 12, "bold"), fg=WARNA_NAMA["tuli"],
                 bg="#eceff1").grid(row=0, column=0, sticky="ew")

        kotak = tk.Frame(kiri, bg="black", width=LEBAR_VIDEO, height=TINGGI_VIDEO)
        kotak.grid(row=1, column=0, pady=(4, 6))
        kotak.grid_propagate(False)
        kotak.pack_propagate(False)
        self.video = tk.Label(kotak, bg="black", fg="#bbbbbb", text="membuka kamera ...",
                              font=("Segoe UI", 12))
        self.video.pack(expand=True)

        self.status = tk.Label(kiri, text="", anchor="w", bg="#eceff1", fg="#37474f",
                               font=("Segoe UI", 10))
        self.status.grid(row=2, column=0, sticky="ew")

        tk.Label(kiri, text="Sedang disusun dari isyarat:", anchor="w", bg="#eceff1",
                 fg="#546e7a", font=("Segoe UI", 9)).grid(row=3, column=0, sticky="ew",
                                                          pady=(6, 0))
        self.sedang = tk.Label(kiri, text="", anchor="w", justify="left", wraplength=500,
                               bg="white", fg="#0d47a1", relief="solid", bd=1,
                               padx=8, pady=4, font=("Segoe UI", 18, "bold"), height=1)
        self.sedang.grid(row=4, column=0, sticky="ew")
        kiri.bind("<Configure>", lambda e: self.sedang.configure(
            wraplength=max(200, e.width - 30)))

        # Dua baris: batang waktu kirim otomatis + suara, lalu tombol-tombol.
        waktu = tk.Frame(kiri, bg="#eceff1")
        waktu.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        tk.Label(waktu, text="Terkirim otomatis:", bg="#eceff1", fg="#546e7a",
                 font=("Segoe UI", 9)).pack(side="left")
        self.hitung = ttk.Progressbar(waktu, maximum=1.0, length=110)
        self.hitung.pack(side="left", padx=6)
        ttk.Checkbutton(waktu, text="Suara (F6)", variable=self.suara
                        ).pack(side="right")

        tombol = tk.Frame(kiri, bg="#eceff1")
        tombol.grid(row=6, column=0, sticky="ew", pady=(2, 6))
        for nomor, (teks, perintah) in enumerate((("Kirim F2", "kirim"),
                                                  ("Hapus F3", "hapus"),
                                                  ("Kosong F4", "kosong"),
                                                  ("Spasi F5", "spasi"))):
            tombol.columnconfigure(nomor, weight=1, uniform="tombol")
            ttk.Button(tombol, text=teks,
                       command=lambda p=perintah: self.pekerja.perintah.put(p)
                       ).grid(row=0, column=nomor, sticky="ew", padx=2)

        self.obrolan_kiri = PanelObrolan(kiri, ukuran_font=12, height=6)
        self.obrolan_kiri.grid(row=7, column=0, sticky="nsew")

    # ---- sisi kanan: lawan bicara
    def _bangun_kanan(self) -> None:
        kanan = tk.Frame(self.akar, bg="#eceff1")
        kanan.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        kanan.columnconfigure(0, weight=1)
        kanan.rowconfigure(1, weight=1)

        tk.Label(kanan, text="LAWAN BICARA  -  mengetik", anchor="w",
                 font=("Segoe UI", 12, "bold"), fg=WARNA_NAMA["dengar"],
                 bg="#eceff1").grid(row=0, column=0, sticky="ew")

        self.obrolan_kanan = PanelObrolan(kanan, ukuran_font=16)
        self.obrolan_kanan.grid(row=1, column=0, sticky="nsew", pady=(4, 8))

        bawah = tk.Frame(kanan, bg="#eceff1")
        bawah.grid(row=2, column=0, sticky="ew")
        bawah.columnconfigure(0, weight=1)
        self.ketik = tk.Entry(bawah, font=("Segoe UI", 16), relief="solid", bd=1)
        self.ketik.grid(row=0, column=0, sticky="ew", ipady=7)
        self.ketik.bind("<Return>", lambda _e: self.kirim_ketikan())
        ttk.Button(bawah, text="Kirim", command=self.kirim_ketikan
                   ).grid(row=0, column=1, padx=(8, 0), ipady=4)
        tk.Label(kanan, text="Ketik pesan lalu tekan Enter. Pesan muncul juga di sisi kiri.",
                 anchor="w", bg="#eceff1", fg="#78909c", font=("Segoe UI", 9)
                 ).grid(row=3, column=0, sticky="ew", pady=(4, 0))
        self.ketik.focus_set()

    def _pasang_tombol(self) -> None:
        for tombol, perintah in (("<F2>", "kirim"), ("<F3>", "hapus"),
                                 ("<F4>", "kosong"), ("<F5>", "spasi")):
            self.akar.bind(tombol, lambda _e, p=perintah: self.pekerja.perintah.put(p))
        self.akar.bind("<F6>", lambda _e: self.suara.set(not self.suara.get()))

    # ---- pesan
    def tambah_pesan(self, pembicara: str, teks: str) -> None:
        pesan = self.riwayat.tambah(pembicara, teks)
        self.obrolan_kiri.tambah(pesan)
        self.obrolan_kanan.tambah(pesan)
        self.catat.kabar("PESAN", f"{pembicara}: {teks}")
        if pembicara == "tuli" and self.suara.get():
            self._antri_suara.append(teks.lower())

    def kirim_ketikan(self) -> None:
        teks = self.ketik.get().strip()
        if teks:
            self.tambah_pesan("dengar", teks)
            self.ketik.delete(0, "end")

    def _ucapkan_antrian(self) -> None:
        # Pengucap membuang permintaan kalau masih bicara, jadi pesan yang
        # datang berurutan ditahan di sini sampai gilirannya.
        if self._antri_suara and not self.pengucap.sibuk:
            self.pengucap.ucapkan(self._antri_suara.pop(0))

    # ---- putaran layar
    def putar(self) -> None:
        terakhir = None
        try:
            while True:
                kiriman = self.pekerja.keluar.get_nowait()
                if kiriman.get("selesai"):
                    if self.pekerja.galat:
                        messagebox.showerror("Kamera berhenti", self.pekerja.galat)
                    self.tutup()
                    return
                for jenis, teks in kiriman["kejadian"]:
                    if jenis == "pesan":
                        self.tambah_pesan("tuli", teks.capitalize())
                terakhir = kiriman
        except queue.Empty:
            pass

        if terakhir is not None:
            self._gambar(terakhir["gambar"], terakhir["status"])
        self._ucapkan_antrian()
        self.akar.after(15, self.putar)

    def _gambar(self, ppm: bytes | None, status: dict | None) -> None:
        if ppm is not None:
            self._gambar_sekarang = tk.PhotoImage(data=ppm, format="PPM")
            self.video.configure(image=self._gambar_sekarang, text="")
        if status is None:
            return

        if status["tangan"]:
            baca = (f"Terbaca: {status['tebakan']} ({status['keyakinan'] * 100:.0f}%)"
                    if status["mode"] == "baca" and status["tebakan"] else "Tangan terlihat")
        else:
            baca = "Tidak ada tangan"
        self.status.configure(text=f"Tangan: {status['tangan']}    {baca}")

        # Garis kursor kecil supaya jelas bahwa teks ini masih disusun.
        self.sedang.configure(text=(status["teks"] + "▌") if status["teks"] else "")
        if status["sisa"] is None:
            self.hitung.configure(value=0)
        else:
            self.hitung.configure(value=1.0 - status["sisa"] / status["ucap_setelah"])

    def tutup(self) -> None:
        self.pekerja.berhenti.set()
        self.akar.destroy()


# --------------------------------------------------------------------- program
def baca_argumen(argv=None) -> argparse.Namespace:
    import gesture_voice as gv

    p = argparse.ArgumentParser(
        description="Percakapan dua sisi: isyarat BISINDO dan ketikan lawan bicara")
    p.add_argument("--kamera", type=int, default=0)
    p.add_argument("--model", default=gv.PATH_MODEL,
                   help="berkas model yang dipakai (bawaan sama dengan gesture_voice.py)")
    p.add_argument("--bahasa", default="id", help="bahasa suara")
    p.add_argument("--pelan", action="store_true", help="suara dibaca lebih pelan")
    p.add_argument("--tanpa-suara", action="store_true",
                   help="pesan isyarat tidak diucapkan (bisa dinyalakan lagi dengan F6)")
    p.add_argument("--ucap-setelah", type=float, default=2.5,
                   help="detik tanpa huruf baru sebelum pesan terkirim sendiri")
    p.add_argument("--spasi-setelah", type=float, default=2.0,
                   help="detik tangan hilang sebelum spasi ditambah sendiri")
    p.add_argument("--jendela-voting", type=int, default=12)
    p.add_argument("--frame-konfirmasi", type=int, default=8)
    p.add_argument("--min-keyakinan", type=float, default=0.45)
    p.add_argument("--jeda", type=float, default=1.2)
    p.add_argument("--tahan-tangan-hilang", type=int, default=15, metavar="FRAME",
                   help="abaikan frame satu tangan sesaat setelah dua tangan (0 = mati)")
    p.add_argument("--keyakinan-deteksi", type=float, default=0.6)
    p.add_argument("--keyakinan-tracking", type=float, default=0.5)
    p.add_argument("--ambang-seragam", action="store_true",
                   help="abaikan ambang per huruf, pakai --min-keyakinan untuk semua")
    p.add_argument("--tanpa-catatan", action="store_true",
                   help="jangan simpan log sesi dan riwayat percakapan")
    return p.parse_args(argv)


def utama() -> int:
    import joblib

    import gesture_voice as gv
    from catatan import Pencatat

    arg = baca_argumen()
    catat = Pencatat(aktif=not arg.tanpa_catatan, detail=False)

    jalur_model = FOLDER / arg.model
    try:
        model = joblib.load(jalur_model)
    except Exception as e:  # noqa: BLE001
        print(f"GAGAL: model '{arg.model}' tidak bisa dimuat ({e}).")
        return 1
    catat.mulai(arg.model, model.n_features_in_, "".join(model.classes_),
                arg.bahasa, arg.kamera)

    ambang = {} if arg.ambang_seragam else gv.muat_ambang_huruf(catat, arg.model)
    voting = gv.VotingHuruf(arg.jendela_voting, arg.frame_konfirmasi, arg.min_keyakinan,
                            arg.jeda, catat=catat, ambang_huruf=ambang)
    penyusun = PenyusunTeks(model, voting, catat, ucap_setelah=arg.ucap_setelah,
                            spasi_setelah=arg.spasi_setelah,
                            tahan_tangan_hilang=arg.tahan_tangan_hilang)

    kamera = (cv2.VideoCapture(arg.kamera, cv2.CAP_DSHOW) if gv.WINDOWS
              else cv2.VideoCapture(arg.kamera))
    if not kamera.isOpened():
        print(f"GAGAL: kamera {arg.kamera} tidak bisa dibuka. "
              f"Coba --kamera 1, atau tutup aplikasi lain yang sedang memakai kamera.")
        return 1

    # Dimuat di sini karena lambat; lihat catatan yang sama di gesture_voice.py.
    import mediapipe as mp

    mp_tangan, mp_gambar = mp.solutions.hands, mp.solutions.drawing_utils
    pendeteksi = mp_tangan.Hands(static_image_mode=False, max_num_hands=2,
                                 min_detection_confidence=arg.keyakinan_deteksi,
                                 min_tracking_confidence=arg.keyakinan_tracking)

    def gambar_titik(frame, hasil) -> None:
        for tangan in hasil.multi_hand_landmarks or []:
            mp_gambar.draw_landmarks(frame, tangan, mp_tangan.HAND_CONNECTIONS)

    pengucap = gv.Pengucap(bahasa=arg.bahasa, pelan=arg.pelan, catat=catat)
    riwayat = Riwayat(simpan=not arg.tanpa_catatan)
    pekerja = PekerjaKamera(kamera, pendeteksi, penyusun, gambar_titik=gambar_titik)

    akar = tk.Tk()
    aplikasi = Aplikasi(akar, pekerja, pengucap, riwayat, catat, suara=not arg.tanpa_suara)
    pekerja.start()
    akar.after(15, aplikasi.putar)
    waktu_mulai = time.monotonic()
    try:
        akar.mainloop()
    finally:
        pekerja.berhenti.set()
        pekerja.join(timeout=2.0)
        catat.selesai(penyusun.teks, 0, time.monotonic() - waktu_mulai)
        pengucap.tutup()
        pendeteksi.close()
        kamera.release()
    return 0


if __name__ == "__main__":
    sys.exit(utama())
