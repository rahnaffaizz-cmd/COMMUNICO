"""Percakapan dua sisi: penyandang tuli (isyarat) dan lawan bicara (mengetik).

Jendelanya dibagi dua, berdampingan:

    KIRI   penyandang tuli. Video kamera, dan di bawahnya teks hasil
           pembacaan isyarat (huruf yang sedang disusun, lalu riwayat
           percakapan).
    KANAN  lawan bicara yang tidak tuli. Riwayat percakapan, dan di bawahnya
           pilihan memasukkan pesan: KETIK, atau SUARA lewat mikrofon yang
           diubah jadi teks (suara_ke_teks.py, jalan di CPU saja).

Tiap baris percakapan diberi tanda siapa yang bicara: warna biru dan rata kiri
untuk penyandang tuli, hijau dan rata kanan untuk lawan bicara, lengkap dengan
nama dan jam. Riwayat yang sama muncul di kedua sisi, jadi penyandang tuli ikut
membaca ketikan lawan bicaranya.

Cara pakai:
    python percakapan.py
    python percakapan.py --kamera 1 --model rf_bisindo_v7.pkl
    python percakapan.py --tanpa-suara        # pesan isyarat tidak diucapkan
    python percakapan.py --mulai-dengan-suara # lawan bicara langsung berbicara
    python percakapan.py --model-ucapan id-small # pengenal suara lebih teliti, lebih lambat
    python percakapan.py --esp32 192.168.1.50    # kamera, mikrofon, dan OLED dari ESP32-CAM

ESP32: alamatnya juga bisa diisi di kolom ESP32 di kanan atas jendela. Video
dan suara datang dari ESP32, dibaca di laptop, lalu tiap pesan dikirim balik ke
layar OLED ESP32. Lihat esp32.py dan firmware/comunico_esp32/.

Tombol (selain tombol di layar):
    ENTER di kotak ketik   kirim pesan lawan bicara
    F2   kirim teks isyarat sekarang juga
    F3   hapus huruf isyarat terakhir
    F4   kosongkan teks isyarat
    F5   tambah spasi pada teks isyarat
    F6   suara menyala / mati
    F7   lawan bicara berganti antara ketik dan suara
    F8   bersihkan percakapan di layar (catatan di log/ tetap lengkap)
    F9   pakai saran kata teratas (tampil di kanan kotak teks isyarat)
    CTRL+Z  batalkan koreksi kata terakhir

Kata yang dieja dicocokkan dengan kamus_kata.txt (lihat kamus.py). Kalau satu
huruf salah terbaca, kata itu diganti dengan kata kamus terdekat begitu kata
selesai. Matikan dengan --tanpa-koreksi.

Pesan isyarat terkirim sendiri kalau tidak ada huruf baru selama --ucap-setelah
detik (bawaan 4). Antar huruf diberi jeda --jeda detik (bawaan 2,7) supaya
ada waktu berganti bentuk tangan. Pembacaan hurufnya memakai bagian yang sama dengan
gesture_voice.py: voting, ambang per huruf, dan tangan hilang sesaat.

Riwayat percakapan juga disimpan ke log/percakapan_<waktu>.txt.

Mode SUARA: ucapan lawan bicara dikenali lalu muncul sebagai satu pesan, sama
seperti ketikan, dengan tanda "(suara)" di namanya. Model dimuat dan diunduh
saat mode suara pertama kali dinyalakan. Selama suara hasil pembacaan isyarat
sedang berbunyi, mikrofon diabaikan supaya tidak terbaca ulang.
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
from tkinter import messagebox

import cv2
import numpy as np

from fitur import ambil_fitur, sesuaikan_untuk
from tampilan import (ABU, BATA, BATA_MUDA, BATA_TUA, GARIS, GARIS_FOKUS, GARIS_KARTU,
                      HIJAU, KARTU, LATAR, PUTIH, REDUP, SERIF, TEKS, ZAITUN, IsianBulat,
                      KotakSusun, ObrolanBulat, PanelBulat, TitikStatus, Tingkat,
                      TombolBulat, TombolMik, VideoBulat, huruf, ikon_suara)

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message=".*GetPrototype.*")

FOLDER = Path(__file__).resolve().parent
FOLDER_LOG = FOLDER / "log"
FOLDER_ASET = FOLDER / "aset"

LEBAR_VIDEO, TINGGI_VIDEO = 480, 360     # ukuran awal; kotak video yang menentukan

# Nama tiap pembicara di gelembung percakapan.
NAMA = {"tuli": "Bahasa Isyarat", "dengar": "Lawan Bicara"}

TANPA_WAKTU = float("-inf")


# ------------------------------------------------------------------- percakapan
@dataclass
class Pesan:
    pembicara: str          # "tuli" atau "dengar"
    teks: str
    waktu: datetime
    via: str = ""           # "suara" kalau berasal dari mikrofon, kosong kalau diketik


class Riwayat:
    """Daftar pesan, dan catatan teksnya di log/ (satu baris satu pesan)."""

    def __init__(self, simpan: bool):
        self.pesan: list[Pesan] = []
        self._berkas: Path | None = None
        if simpan:
            FOLDER_LOG.mkdir(exist_ok=True)
            self._berkas = FOLDER_LOG / f"percakapan_{datetime.now():%Y%m%d_%H%M%S}.txt"

    def tambah(self, pembicara: str, teks: str, via: str = "") -> Pesan:
        baru = Pesan(pembicara, teks, datetime.now(), via)
        self.pesan.append(baru)
        if self._berkas:
            tanda = f"({via}) " if via else ""
            with open(self._berkas, "a", encoding="utf-8") as f:
                f.write(f"[{baru.waktu:%H:%M:%S}] {pembicara.upper():7s}: {tanda}{teks}\n")
        return baru

    def kosongkan_layar(self) -> None:
        """Layar dibersihkan. Catatan di log/ tidak dihapus, hanya diberi penanda."""
        self.pesan.clear()
        if self._berkas:
            with open(self._berkas, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now():%H:%M:%S}] ------- layar dibersihkan -------\n")


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
                 kirim_otomatis: bool = True, kamus=None):
        self.model = model
        self.kamus = kamus                    # PengoreksiKata, boleh kosong
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
            if self.kamus:
                self.kamus.catat_frame(self.model.classes_, peluang)

            diterima = self.voting.masukkan(tebakan, keyakinan, sekarang)
            if diterima:
                self.teks += diterima
                if self.kamus:
                    self.kamus.huruf_masuk(diterima, self.teks)
                self._waktu_huruf = sekarang
                self.catat.huruf_diterima(diterima, self.teks)
                kejadian.append(("huruf", diterima))
        else:
            # Tangan hilang dari layar. Kalau lama, satu kata dianggap selesai.
            self.voting.kosongkan()
            if self.kamus:
                self.kamus.lupakan_frame()
            if self._sejak_hilang is None:
                self._sejak_hilang = sekarang
            elif (sekarang - self._sejak_hilang >= self.spasi_setelah
                  and self.teks and not self.teks.endswith(" ")):
                self.teks = self._akhiri_kata() + " "
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
            "saran": self.kamus.saran(self.teks) if self.kamus else [],
            # None kalau tidak ada yang menunggu terkirim.
            "sisa": sisa if self.teks.strip() and self._waktu_huruf != TANPA_WAKTU
            else None,
            "ucap_setelah": self.ucap_setelah,
        }
        return status, kejadian

    def _akhiri_kata(self) -> str:
        return self.kamus.akhiri_kata(self.teks) if self.kamus else self.teks

    def kirim(self) -> str:
        """Ambil teks sekarang sebagai satu pesan, lalu kosongkan."""
        hasil = self._akhiri_kata().strip()
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
            self.teks = self._akhiri_kata() + " "
            self.catat.teks_berubah("spasi ditambah manual", self.teks)
        elif nama == "saran" and self.kamus:
            self.teks = self.kamus.pakai_saran(self.teks)
        elif nama == "batal" and self.kamus:
            self.teks = self.kamus.batalkan(self.teks)
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
        # Diatur layar mengikuti ukuran kotak video; gambar dikecilkan di thread ini.
        self.ukuran_gambar = (LEBAR_VIDEO, TINGGI_VIDEO)
        self._sumber_baru = None
        self._kunci = threading.Lock()

    def ganti_sumber(self, sumber) -> None:
        """Pindah ke kamera lain (misalnya ESP32) tanpa menghentikan thread.

        Sumber lama dilepas di thread ini juga, di antara dua frame, supaya
        tidak dilepas saat sedang dibaca.
        """
        with self._kunci:
            self._sumber_baru = [sumber]      # dibungkus supaya None juga bisa dipilih

    def _ambil_sumber_baru(self) -> None:
        with self._kunci:
            bungkus, self._sumber_baru = self._sumber_baru, None
        if bungkus is not None and bungkus[0] is not self.sumber:
            lama, self.sumber = self.sumber, bungkus[0]
            if lama is not None:
                lama.release()

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

                self._ambil_sumber_baru()
                if self.sumber is None:             # belum ada kamera sama sekali
                    if kejadian_perintah:
                        self._titip({"gambar": None, "status": None,
                                     "kejadian": kejadian_perintah})
                    self.berhenti.wait(0.1)
                    continue

                berhasil, frame = self.sumber.read()
                if not berhasil:
                    self.galat = "kamera berhenti mengirim gambar"
                    return
                if frame is None:
                    # Kamera jaringan (ESP32) belum mengirim gambar baru.
                    if kejadian_perintah:
                        self._titip({"gambar": None, "status": None,
                                     "kejadian": kejadian_perintah})
                    continue
                frame = cv2.flip(frame, 1)

                hasil = self.pendeteksi.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                angka, jumlah_tangan = vektor_titik_tangan(hasil)
                status, kejadian = self.penyusun.langkah(angka, jumlah_tangan,
                                                         time.monotonic())
                if self.gambar_titik is not None:
                    self.gambar_titik(frame, hasil)

                self._titip({"gambar": _ke_ppm(frame, *self.ukuran_gambar), "status": status,
                             "kejadian": kejadian_perintah + kejadian})
        except Exception as e:  # noqa: BLE001 - dilaporkan ke layar, bukan ditelan
            self.galat = f"{type(e).__name__}: {e}"
        finally:
            self._titip({"gambar": None, "status": None, "kejadian": [],
                         "selesai": True})


def _ke_ppm(frame_bgr: np.ndarray, lebar_kotak: int = LEBAR_VIDEO,
            tinggi_kotak: int = TINGGI_VIDEO) -> bytes:
    """Kecilkan frame sesuai kotak video, lalu ubah ke format PPM yang dibaca Tk."""
    tinggi, lebar = frame_bgr.shape[:2]
    skala = min(lebar_kotak / lebar, tinggi_kotak / tinggi)
    kecil = cv2.resize(frame_bgr, (max(1, int(lebar * skala)), max(1, int(tinggi * skala))))
    rgb = cv2.cvtColor(kecil, cv2.COLOR_BGR2RGB)
    return b"P6 %d %d 255\n" % (rgb.shape[1], rgb.shape[0]) + rgb.tobytes()


# ------------------------------------------------------------------------ layar
class Aplikasi:
    """Jendela percakapan. Susunannya mengikuti desain "Tema asli dirapikan":
    kepala (logo, ESP32, BISINDO), kartu BAHASA ISYARAT di kiri, kartu
    PERCAKAPAN di kanan."""

    def __init__(self, akar: tk.Tk, pekerja: PekerjaKamera, pengucap, riwayat: Riwayat,
                 catat, *, suara: bool, pendengar=None, buka_kamera_laptop=None,
                 buka_mikrofon_laptop=None):
        self.akar = akar
        self.pekerja = pekerja
        self.pendengar = pendengar            # PendengarSuara, kosong kalau tidak dipakai
        # Dipakai saat ESP32 diputus, untuk kembali ke kamera dan mikrofon laptop.
        self.buka_kamera_laptop = buka_kamera_laptop
        self.buka_mikrofon_laptop = buka_mikrofon_laptop
        self.esp = None                       # esp32.SambunganESP32 yang sedang dipakai
        self.alamat_esp = tk.StringVar()
        self.mode = tk.StringVar(value="ketik")
        self.pengucap = pengucap
        self.riwayat = riwayat
        self.catat = catat
        self.suara = tk.BooleanVar(value=suara)
        self._antri_suara: list[str] = []
        self._gambar_sekarang = None          # simpan rujukan, kalau tidak Tk membuangnya
        self._kabar_suara = "Ucapan lawan bicara diubah jadi teks."

        akar.title("COMMUNICO - Percakapan dua sisi")
        # Ukuran mengikuti layar, supaya semuanya tetap kelihatan walaupun
        # layarnya kecil atau pembesaran Windows-nya tinggi.
        lebar = min(1440, akar.winfo_screenwidth() - 80)
        tinggi = min(900, akar.winfo_screenheight() - 120)
        akar.geometry(f"{lebar}x{tinggi}+20+20")
        akar.minsize(980, 640)
        akar.configure(bg=LATAR)
        akar.columnconfigure(0, weight=1)
        akar.rowconfigure(2, weight=1)

        self._pasang_ikon()
        self._bangun_kepala()
        self._bangun_isyarat()
        self._bangun_percakapan()
        self._pasang_tombol()
        self.suara.trace_add("write", lambda *_: self._gambar_tombol_suara())
        self._ganti_mode()
        akar.protocol("WM_DELETE_WINDOW", self.tutup)

    def _pasang_ikon(self) -> None:
        """Logo COMUNICO di sudut jendela dan di taskbar."""
        try:
            if sys.platform == "win32":
                self.akar.iconbitmap(default=str(FOLDER_ASET / "ikon.ico"))
            self._ikon = tk.PhotoImage(file=str(FOLDER_ASET / "ikon.png"))
            self.akar.iconphoto(True, self._ikon)
        except (tk.TclError, OSError):
            pass  # berkas ikon hilang: pakai ikon bawaan

    # ---- kepala
    def _bangun_kepala(self) -> None:
        kepala = tk.Frame(self.akar, bg=LATAR)
        kepala.grid(row=0, column=0, sticky="ew", padx=32, pady=(18, 16))
        kepala.columnconfigure(2, weight=1)

        try:
            self._logo = tk.PhotoImage(file=str(FOLDER_ASET / "logo_kepala.png"))
            tk.Label(kepala, image=self._logo, bg=LATAR).grid(row=0, column=0, padx=(0, 16))
        except (tk.TclError, OSError):
            pass  # berkas logo hilang: cukup tulisan saja

        identitas = tk.Frame(kepala, bg=LATAR)
        identitas.grid(row=0, column=1, sticky="w")
        tk.Label(identitas, text="COMMUNICO", bg=LATAR, fg=TEKS,
                 font=huruf(32, "bold", keluarga=SERIF)).pack(anchor="w")
        tk.Label(identitas, text="ISYARAT   SUARA   PESAN", bg=LATAR, fg=BATA,
                 font=huruf(13)).pack(anchor="w", pady=(2, 0))

        # Ruang kosong yang mendorong kartu WEARABLE MODE dan BISINDO ke kanan.
        tk.Frame(kepala, bg=LATAR).grid(row=0, column=2, sticky="ew")

        self._bangun_esp32(kepala).grid(row=0, column=3, padx=(0, 14))
        TombolBulat(kepala, "BISINDO", latar_induk=LATAR, isi=TEKS, warna_teks=KARTU,
                    r=8, tinggi=40, font=huruf(14, "bold"), padx=16
                    ).grid(row=0, column=4)

        garis = tk.Frame(self.akar, bg=BATA, height=2)
        garis.grid(row=1, column=0, sticky="ew", padx=32)

    def _bangun_esp32(self, induk) -> PanelBulat:
        """Kartu ESP32: video dan suara dari ESP32, hasilnya dikirim balik ke OLED."""
        kartu = PanelBulat(induk, latar_induk=LATAR, isi=KARTU, garis=GARIS_KARTU, r=14,
                           pad=(12, 10))
        baris = tk.Frame(kartu.isi, bg=KARTU)
        baris.pack(anchor="w")
        tk.Label(baris, text="WEARABLE MODE", bg=KARTU, fg=TEKS,
                 font=huruf(13, "bold")).pack(side="left", padx=(0, 8))
        self.kolom_esp = IsianBulat(baris, latar_induk=KARTU, tinggi=38, r=10, lebar=170,
                                    font=huruf(14), textvariable=self.alamat_esp)
        self.kolom_esp.pack(side="left")
        self.kolom_esp.entry.bind("<Return>", lambda _e: self.hubungkan_esp32())
        self.tombol_esp = TombolBulat(baris, "Hubungkan", self._tombol_esp32,
                                      latar_induk=KARTU, isi=KARTU, warna_teks=ZAITUN,
                                      garis=ZAITUN, r=10, tinggi=38, font=huruf(14, "bold"))
        self.tombol_esp.pack(side="left", padx=(8, 0))

        keadaan = tk.Frame(kartu.isi, bg=KARTU)
        keadaan.pack(anchor="w", pady=(6, 0))
        self.titik_esp = TitikStatus(keadaan, latar_induk=KARTU)
        self.titik_esp.pack(side="left", padx=(0, 6))
        self.status_esp = tk.Label(keadaan, text="Belum terhubung · lihat alamat IP di layar OLED",
                                   bg=KARTU, fg=REDUP, font=huruf(12))
        self.status_esp.pack(side="left")
        return kartu

    # ---- kartu kiri: bahasa isyarat
    def _bangun_isyarat(self) -> None:
        utama = tk.Frame(self.akar, bg=LATAR)
        utama.grid(row=2, column=0, sticky="nsew", padx=32, pady=(20, 28))
        utama.columnconfigure(0, weight=1, uniform="kartu")
        utama.columnconfigure(1, weight=1, uniform="kartu")
        utama.rowconfigure(0, weight=1)
        self._utama = utama

        kartu = PanelBulat(utama, latar_induk=LATAR, isi=KARTU, garis=GARIS_KARTU, r=18,
                           pad=20, ikut_isi=False, width=520, height=560)
        kartu.grid(row=0, column=0, sticky="nsew", padx=(0, 11))
        isi = kartu.isi
        isi.columnconfigure(0, weight=1)
        isi.rowconfigure(1, weight=1)

        atas = tk.Frame(isi, bg=KARTU)
        atas.grid(row=0, column=0, sticky="ew")
        tk.Label(atas, text="BAHASA ISYARAT", bg=KARTU, fg=BATA,
                 font=huruf(21, "bold", keluarga=SERIF)).pack(side="left")
        tk.Label(atas, text="kamera", bg=KARTU, fg=REDUP,
                 font=huruf(16, "italic", keluarga=SERIF)).pack(side="left", padx=(8, 0),
                                                               pady=(4, 0))
        self.tombol_suara = TombolBulat(atas, "", lambda: self.suara.set(not self.suara.get()),
                                        latar_induk=KARTU, isi=BATA_MUDA, warna_teks=BATA_TUA,
                                        garis=BATA, r=20, tinggi=40, font=huruf(13, "bold"))
        self.tombol_suara.pack(side="right")
        self._gambar_tombol_suara()

        self.video = VideoBulat(isi, latar_induk=KARTU, height=300)
        self.video.grid(row=1, column=0, sticky="nsew", pady=16)
        self.video.bind("<Configure>", self._video_berubah, add="+")

        label = tk.Frame(isi, bg=KARTU)
        label.grid(row=2, column=0, sticky="ew")
        tk.Label(label, text="Sedang disusun dari isyarat", bg=KARTU, fg=REDUP,
                 font=huruf(13, "bold")).pack(side="left")
        self.terkirim = tk.Label(label, text="-", bg=KARTU, fg=TEKS, font=huruf(12, "bold"))
        self.terkirim.pack(side="right")
        tk.Label(label, text="Terkirim otomatis:", bg=KARTU, fg=REDUP,
                 font=huruf(12)).pack(side="right", padx=(0, 4))

        self.sedang = KotakSusun(isi, latar_induk=KARTU)
        self.sedang.grid(row=3, column=0, sticky="ew", pady=(8, 16))

        tombol = tk.Frame(isi, bg=KARTU)
        tombol.grid(row=4, column=0, sticky="ew")
        for nomor, (teks, kunci, perintah) in enumerate((("Kirim", "F2", "kirim"),
                                                         ("Hapus", "F3", "hapus"),
                                                         ("Kosong", "F4", "kosong"),
                                                         ("Spasi", "F5", "spasi"))):
            tombol.columnconfigure(nomor, weight=1, uniform="tombol")
            utama_tombol = nomor == 0
            TombolBulat(tombol, teks, lambda p=perintah: self.pekerja.perintah.put(p),
                        latar_induk=KARTU, isi=BATA if utama_tombol else PUTIH,
                        warna_teks=KARTU if utama_tombol else TEKS,
                        garis=None if utama_tombol else GARIS, hint=kunci,
                        warna_hint=KARTU if utama_tombol else REDUP, lebar=60,
                        font=huruf(15, "bold" if utama_tombol else "normal"),
                        ).grid(row=0, column=nomor, sticky="ew",
                               padx=(0 if nomor == 0 else 5, 0 if nomor == 3 else 5))

    def _gambar_tombol_suara(self) -> None:
        if self.suara.get():
            self.tombol_suara.atur(teks="Suara aktif · F6", isi=BATA_MUDA, warna_teks=BATA_TUA,
                                   garis=BATA, ikon=ikon_suara)
        else:
            self.tombol_suara.atur(teks="Suara mati · F6", isi=KARTU, warna_teks=REDUP,
                                   garis=GARIS,
                                   ikon=lambda k, x, cy, w: ikon_suara(k, x, cy, w, mati=True))

    def _video_berubah(self, _e=None) -> None:
        self.pekerja.ukuran_gambar = self.video.ukuran

    # ---- kartu kanan: percakapan
    def _bangun_percakapan(self) -> None:
        kartu = PanelBulat(self._utama, latar_induk=LATAR, isi=KARTU, garis=GARIS_KARTU,
                           r=18, pad=20, ikut_isi=False, width=520, height=560)
        kartu.grid(row=0, column=1, sticky="nsew", padx=(11, 0))
        isi = kartu.isi
        isi.columnconfigure(0, weight=1)
        isi.rowconfigure(1, weight=1)

        atas = tk.Frame(isi, bg=KARTU)
        atas.grid(row=0, column=0, sticky="ew")
        tk.Label(atas, text="PERCAKAPAN", bg=KARTU, fg=ZAITUN,
                 font=huruf(21, "bold", keluarga=SERIF)).pack(side="left")
        TombolBulat(atas, "Bersihkan percakapan · F8", self.bersihkan_percakapan,
                    latar_induk=KARTU, isi=PUTIH, warna_teks=REDUP, garis=GARIS, r=10,
                    tinggi=40, font=huruf(13, "bold")).pack(side="right")

        self.obrolan = ObrolanBulat(isi, latar=KARTU, nama=NAMA)
        self.obrolan.grid(row=1, column=0, sticky="nsew", pady=(14, 14))

        tk.Frame(isi, bg=GARIS_KARTU, height=1).grid(row=2, column=0, sticky="ew")

        pilih = tk.Frame(isi, bg=KARTU)
        pilih.grid(row=3, column=0, sticky="ew", pady=(14, 10))
        tk.Label(pilih, text="Masukan lawan bicara", bg=KARTU, fg=REDUP,
                 font=huruf(13, "bold")).pack(side="left")
        segmen = PanelBulat(pilih, latar_induk=KARTU, isi=LATAR, r=10, pad=3)
        segmen.pack(side="right")
        gaya = dict(latar_induk=LATAR, isi=LATAR, warna_teks=TEKS, r=8, tinggi=36,
                    font=huruf(13, "bold"))
        self.tombol_ketik = TombolBulat(segmen.isi, "Ketik", lambda: self._pilih_mode("ketik"),
                                        **gaya)
        self.tombol_ketik.pack(side="left")
        self.tombol_dengar = TombolBulat(segmen.isi, "Suara",
                                         lambda: self._pilih_mode("suara"), **gaya)
        self.tombol_dengar.pack(side="left", padx=(3, 0))
        tk.Label(segmen.isi, text="F7", bg=LATAR, fg=REDUP,
                 font=huruf(11, "bold")).pack(side="left", padx=8)

        self.ruang_masukan = tk.Frame(isi, bg=KARTU)
        self.ruang_masukan.grid(row=4, column=0, sticky="ew")
        self.ruang_masukan.columnconfigure(0, weight=1)

        # Mode KETIK: kotak ketik + tombol Kirim.
        self.bawah_ketik = tk.Frame(self.ruang_masukan, bg=KARTU)
        self.bawah_ketik.grid(row=0, column=0, sticky="ew")
        self.bawah_ketik.columnconfigure(0, weight=1)
        self.isian_ketik = IsianBulat(self.bawah_ketik, latar_induk=KARTU, garis=GARIS_FOKUS,
                                      garis_fokus=BATA)
        self.isian_ketik.grid(row=0, column=0, sticky="ew")
        self.ketik = self.isian_ketik.entry
        self.ketik.bind("<Return>", lambda _e: self.kirim_ketikan())
        TombolBulat(self.bawah_ketik, "Kirim", self.kirim_ketikan, latar_induk=KARTU,
                    isi=ZAITUN, warna_teks=KARTU, padx=22).grid(row=0, column=1, padx=(10, 0))

        # Mode SUARA: tombol mikrofon bundar + keadaan + kerasnya suara.
        self.bawah_suara = PanelBulat(self.ruang_masukan, latar_induk=KARTU, isi=PUTIH,
                                      garis=GARIS_FOKUS, r=12, pad=(14, 10), tebal=1.5)
        self.bawah_suara.grid(row=0, column=0, sticky="ew")
        suara = self.bawah_suara.isi
        suara.columnconfigure(1, weight=1)
        self.tombol_mik = TombolMik(suara, self.ganti_dengar, latar_induk=PUTIH)
        self.tombol_mik.grid(row=0, column=0, rowspan=2, padx=(0, 14))
        self.judul_suara = tk.Label(suara, text="", anchor="w", bg=PUTIH, fg=TEKS,
                                    font=huruf(15, "bold"))
        self.judul_suara.grid(row=0, column=1, sticky="sew")
        self.keterangan_suara = tk.Label(suara, text="", anchor="w", bg=PUTIH, fg=REDUP,
                                         font=huruf(13))
        self.keterangan_suara.grid(row=1, column=1, sticky="new")
        self.tingkat_suara = Tingkat(suara, latar_induk=PUTIH)
        self.tingkat_suara.grid(row=0, column=2, rowspan=2, padx=(10, 0))
        self.bawah_suara.grid_remove()

    # ---- ESP32
    def _tombol_esp32(self) -> None:
        if self.esp is None:
            self.hubungkan_esp32()
        else:
            self.putuskan_esp32()

    def hubungkan_esp32(self) -> None:
        from esp32 import SambunganESP32

        alamat = self.alamat_esp.get().strip()
        if not alamat:
            self.status_esp.configure(text="Isi dulu alamat IP ESP32", fg=BATA)
            self.titik_esp.warnai(BATA)
            return
        if self.esp is not None:
            self.putuskan_esp32()
        self.esp = SambunganESP32(alamat)
        self.pekerja.ganti_sumber(self.esp.kamera)
        self._mikrofon_berganti()
        self.kolom_esp.entry.configure(state="disabled")
        self.tombol_esp.atur(teks="Putuskan", isi=ZAITUN, warna_teks=KARTU)
        self.video.tampilkan(None, "menunggu video ESP32 ...")
        self.catat.kabar("ESP32", f"menghubungkan ke {self.esp.host}")

    def putuskan_esp32(self) -> None:
        if self.esp is None:
            return
        esp, self.esp = self.esp, None
        esp.tutup()
        kamera = self.buka_kamera_laptop() if self.buka_kamera_laptop else None
        if kamera is None:
            self.status_esp.configure(text="ESP32 diputus · kamera laptop tidak tersedia",
                                      fg=BATA)
            self.titik_esp.warnai(BATA)
            self.video.tampilkan(None, "tidak ada kamera")
        else:
            self.status_esp.configure(text="Belum terhubung · kembali ke kamera laptop",
                                      fg=REDUP)
            self.titik_esp.warnai(ABU)
        # Kamera ESP32 dilepas oleh thread kamera sendiri, di antara dua frame.
        self.pekerja.ganti_sumber(kamera)
        self._mikrofon_berganti()
        self.kolom_esp.entry.configure(state="normal")
        self.tombol_esp.atur(teks="Hubungkan", isi=KARTU, warna_teks=ZAITUN)
        self.catat.kabar("ESP32", f"diputus dari {esp.host}")

    def _perbarui_status_esp(self) -> None:
        esp = self.esp
        if esp is None:
            return
        keadaan = esp.keadaan
        if esp.kamera.keadaan.endswith("fps"):
            teks, warna, titik = f"Terhubung ke {esp.host} · {keadaan}", ZAITUN, HIJAU
        else:
            teks, warna, titik = f"{esp.host} · {keadaan}", REDUP, ABU
        if self.status_esp.cget("text") != teks:
            self.status_esp.configure(text=teks, fg=warna)
            self.titik_esp.warnai(titik)

    def _mikrofon_berganti(self) -> None:
        # Kalau mode suara sedang menyala, mikrofonnya dibuka ulang dari sumber baru.
        if self.pendengar is not None and self.pendengar.aktif.is_set():
            self.pendengar.buka_ulang.set()

    def buka_mikrofon(self):
        """Dipanggil pendengar suara: mikrofon ESP32 kalau tersambung, kalau tidak laptop."""
        esp = self.esp
        if esp is not None:
            return esp.buka_mikrofon()
        return self.buka_mikrofon_laptop()

    # ---- ketik / suara
    def _pilih_mode(self, mode: str) -> None:
        self.mode.set(mode)
        self._ganti_mode()

    def _ganti_mode(self) -> None:
        aktif, pasif = dict(isi=ZAITUN, warna_teks=KARTU), dict(isi=LATAR, warna_teks=TEKS)
        if self.mode.get() == "suara" and self.pendengar is None:
            self.mode.set("ketik")
            self.catat.kabar("SUARA", "pengenal suara tidak tersedia")
        if self.mode.get() == "suara":
            self.tombol_ketik.atur(**pasif)
            self.tombol_dengar.atur(**aktif)
            self.bawah_ketik.grid_remove()
            self.bawah_suara.grid()
            self._kabar_suara = "Menyiapkan mikrofon..."
            self.pendengar.aktif.set()        # langsung mendengarkan, seperti sebelumnya
            self.catat.kabar("SUARA", "masukan lawan bicara: suara")
            self._gambar_suara()
            return
        self.tombol_ketik.atur(**aktif)
        self.tombol_dengar.atur(**pasif)
        if self.pendengar is not None:
            self.pendengar.aktif.clear()
        self.bawah_suara.grid_remove()
        self.bawah_ketik.grid()
        self.ketik.focus_set()

    def ganti_dengar(self) -> None:
        """Tombol mikrofon: berhenti atau mulai lagi mendengarkan."""
        if self.pendengar is None:
            return
        if self.pendengar.aktif.is_set():
            self.pendengar.aktif.clear()
            self._kabar_suara = "Ucapan lawan bicara diubah jadi teks."
        else:
            self._kabar_suara = "Menyiapkan mikrofon..."
            self.pendengar.aktif.set()
        self._gambar_suara()

    def _gambar_suara(self) -> None:
        dengar = self.pendengar is not None and self.pendengar.aktif.is_set()
        self.tombol_mik.atur(dengar)
        if dengar:
            self.judul_suara.configure(text="Mendengarkan…", fg=BATA)
            kabar = self._kabar_suara
            if kabar.startswith("Mendengarkan"):
                kabar = "Tekan lagi kalau sudah selesai bicara."
            self.keterangan_suara.configure(text=kabar)
        else:
            self.judul_suara.configure(text="Tekan untuk bicara", fg=TEKS)
            self.keterangan_suara.configure(text="Ucapan lawan bicara diubah jadi teks.")
            self.tingkat_suara.atur(0.0)

    def _pasang_tombol(self) -> None:
        for tombol, perintah in (("<F2>", "kirim"), ("<F3>", "hapus"),
                                 ("<F4>", "kosong"), ("<F5>", "spasi")):
            self.akar.bind(tombol, lambda _e, p=perintah: self.pekerja.perintah.put(p))
        self.akar.bind("<F6>", lambda _e: self.suara.set(not self.suara.get()))
        self.akar.bind("<F7>", lambda _e: self._pilih_mode(
            "ketik" if self.mode.get() == "suara" else "suara"))
        self.akar.bind("<F8>", lambda _e: self.bersihkan_percakapan())
        self.akar.bind("<F9>", lambda _e: self.pekerja.perintah.put("saran"))
        self.akar.bind("<Control-z>", lambda _e: self.pekerja.perintah.put("batal"))

    # ---- pesan
    def tambah_pesan(self, pembicara: str, teks: str, via: str = "") -> None:
        pesan = self.riwayat.tambah(pembicara, teks, via)
        self.obrolan.tambah(pesan)
        self.catat.kabar("PESAN", f"{pembicara}: {teks}")
        if pembicara == "tuli":
            self.terkirim.configure(text=teks.upper())
        if self.esp is not None:
            self.esp.pengirim.kirim("isyarat" if pembicara == "tuli" else "lawan", teks)
        if pembicara == "tuli" and self.suara.get():
            self._antri_suara.append(teks.lower())

    def bersihkan_percakapan(self) -> None:
        """Kosongkan riwayat di layar. Catatan di log/ tetap ada."""
        self.obrolan.kosongkan()
        self.riwayat.kosongkan_layar()
        self.catat.kabar("PESAN", "percakapan di layar dibersihkan")
        if self.mode.get() == "ketik":
            self.ketik.focus_set()

    def kirim_ketikan(self) -> None:
        teks = self.ketik.get().strip()
        if teks:
            self.tambah_pesan("dengar", teks)
            self.ketik.delete(0, "end")

    def sedang_menyuarakan(self) -> bool:
        """Benar kalau ada suara yang sedang atau akan diputar lewat pengeras suara.

        Dipakai pendengar untuk mengabaikan mikrofon, supaya suara hasil
        pembacaan isyarat tidak terbaca ulang sebagai ucapan lawan bicara.
        """
        return self.pengucap.sibuk or bool(self._antri_suara)

    def _urus_suara_masuk(self) -> None:
        """Ambil kabar dari pendengar suara: status, kerasnya suara, dan teks."""
        if self.pendengar is None:
            return
        try:
            while True:
                jenis, isi = self.pendengar.keluar.get_nowait()
                if jenis == "teks":
                    self.tambah_pesan("dengar", isi, via="suara")
                elif jenis == "tingkat":
                    self.tingkat_suara.atur(isi)
                elif jenis == "status" and self.mode.get() == "suara":
                    if self.pendengar.aktif.is_set():
                        self._kabar_suara = isi
                        self._gambar_suara()
                elif jenis == "galat":
                    self.catat.kabar("SUARA", f"gagal: {isi}")
                    self._pilih_mode("ketik")
                    messagebox.showerror("Suara ke teks gagal", isi)
        except queue.Empty:
            pass

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
        self._urus_suara_masuk()
        self._perbarui_status_esp()
        self.akar.after(15, self.putar)

    def _gambar(self, ppm: bytes | None, status: dict | None) -> None:
        if status is not None:
            tebakan = status["tebakan"] if status["mode"] == "baca" else None
            self.video.atur_status(status["tangan"], tebakan,
                                   status["keyakinan"] if tebakan else None)
            kemajuan = (None if status["sisa"] is None
                        else 1.0 - status["sisa"] / status["ucap_setelah"])
            self.sedang.atur(status["teks"], kemajuan, status.get("saran", []))
        if ppm is not None:
            self._gambar_sekarang = tk.PhotoImage(data=ppm, format="PPM")
            self.video.tampilkan(self._gambar_sekarang)

    def tutup(self) -> None:
        self.pekerja.berhenti.set()
        if self.pendengar is not None:
            self.pendengar.aktif.clear()
            self.pendengar.berhenti.set()
        if self.esp is not None:
            self.esp.tutup()
        self.akar.destroy()


# --------------------------------------------------------------------- program
def baca_argumen(argv=None) -> argparse.Namespace:
    import gesture_voice as gv

    pengaturan_deteksi = gv.muat_pengaturan_deteksi()
    p = argparse.ArgumentParser(
        description="Percakapan dua sisi: isyarat BISINDO dan ketikan lawan bicara")
    p.add_argument("--kamera", type=int, default=0)
    p.add_argument("--model", default=gv.PATH_MODEL,
                   help="berkas model yang dipakai (bawaan sama dengan gesture_voice.py)")
    p.add_argument("--bahasa", default="id", help="bahasa suara")
    p.add_argument("--pelan", action="store_true", help="suara dibaca lebih pelan")
    p.add_argument("--tanpa-suara", action="store_true",
                   help="pesan isyarat tidak diucapkan (bisa dinyalakan lagi dengan F6)")
    # Harus lebih lama dari --jeda, kalau tidak pesan terkirim sebelum huruf
    # berikutnya sempat terbaca.
    p.add_argument("--ucap-setelah", type=float, default=4.0,
                   help="detik tanpa huruf baru sebelum pesan terkirim sendiri")
    p.add_argument("--spasi-setelah", type=float, default=2.0,
                   help="detik tangan hilang sebelum spasi ditambah sendiri")
    p.add_argument("--jendela-voting", type=int, default=12)
    p.add_argument("--frame-konfirmasi", type=int, default=8)
    p.add_argument("--min-keyakinan", type=float, default=0.45)
    p.add_argument("--jeda", type=float, default=2.7,
                   help="detik minimal antar huruf, waktu untuk berganti bentuk tangan")
    p.add_argument("--tahan-tangan-hilang", type=int, default=15, metavar="FRAME",
                   help="abaikan frame satu tangan sesaat setelah dua tangan (0 = mati)")
    p.add_argument("--keyakinan-deteksi", type=float,
                   default=pengaturan_deteksi["min_detection_confidence"])
    p.add_argument("--keyakinan-tracking", type=float,
                   default=pengaturan_deteksi["min_tracking_confidence"])
    p.add_argument("--ambang-seragam", action="store_true",
                   help="abaikan ambang per huruf, pakai --min-keyakinan untuk semua")
    p.add_argument("--tanpa-koreksi", action="store_true",
                   help="jangan koreksi kata dengan kamus_kata.txt")
    p.add_argument("--model-ucapan", default="id-base",
                   help="model pengenal suara (CPU, int8). id-base/id-small disetel "
                        "untuk bahasa Indonesia; untuk bahasa lain pakai tiny/base/small/medium")
    p.add_argument("--beam-ucapan", type=int, default=5,
                   help="lebar pencarian pengenal suara; makin besar makin teliti dan "
                        "sedikit lebih lambat (bawaan 5)")
    p.add_argument("--mikrofon", default=None,
                   help="nomor atau nama perangkat mikrofon (bawaan: yang aktif di Windows)")
    p.add_argument("--thread-suara", type=int, default=4,
                   help="jumlah thread CPU untuk pengenal suara (bawaan 4)")
    p.add_argument("--mulai-dengan-suara", action="store_true",
                   help="lawan bicara langsung dalam mode suara, bukan ketik")
    p.add_argument("--esp32", default=None, metavar="IP",
                   help="langsung tersambung ke ESP32-CAM ini (video, mikrofon, layar OLED); "
                        "kamera laptop tidak wajib ada")
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
    from kamus import PengoreksiKata
    kamus = PengoreksiKata.muat(gv.muat_ambang_huruf(nama_model=arg.model),
                                aktif=not arg.tanpa_koreksi, catat=catat)
    penyusun = PenyusunTeks(model, voting, catat, ucap_setelah=arg.ucap_setelah,
                            spasi_setelah=arg.spasi_setelah,
                            tahan_tangan_hilang=arg.tahan_tangan_hilang,
                            kamus=kamus)

    def buka_kamera_laptop():
        kamera = (cv2.VideoCapture(arg.kamera, cv2.CAP_DSHOW) if gv.WINDOWS
                  else cv2.VideoCapture(arg.kamera))
        if kamera.isOpened():
            return kamera
        kamera.release()
        return None

    # Dengan --esp32, gambar datang dari ESP32, jadi kamera laptop tidak dibuka dulu.
    kamera = None if arg.esp32 else buka_kamera_laptop()
    if kamera is None and not arg.esp32:
        print(f"GAGAL: kamera {arg.kamera} tidak bisa dibuka. "
              f"Coba --kamera 1, --esp32 <IP>, atau tutup aplikasi lain yang sedang "
              f"memakai kamera.")
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
    if not arg.tanpa_suara:
        pengucap.siapkan(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list(kamus.peringkat))
    riwayat = Riwayat(simpan=not arg.tanpa_catatan)

    # Pengenal suara tidak memuat apa-apa sampai mode suara pertama kali dinyalakan.
    from suara_ke_teks import Mikrofon, PendengarSuara, PengenalSuara

    perangkat = arg.mikrofon
    if perangkat is not None and perangkat.isdigit():
        perangkat = int(perangkat)
    model_ucapan = arg.model_ucapan
    if model_ucapan.startswith("id-") and arg.bahasa != "id":
        print(f"Catatan: {model_ucapan} khusus bahasa Indonesia, jadi untuk --bahasa "
              f"{arg.bahasa} dipakai Whisper 'base'.")
        model_ucapan = "base"
    pendengar = PendengarSuara(
        PengenalSuara(model_ucapan, arg.bahasa, arg.thread_suara, arg.beam_ucapan),
        buka_mikrofon=lambda: aplikasi.buka_mikrofon(),
        sedang_bicara=lambda: aplikasi.sedang_menyuarakan())
    pekerja = PekerjaKamera(kamera, pendeteksi, penyusun, gambar_titik=gambar_titik)

    if sys.platform == "win32":
        # Tanpa ini Windows mengelompokkan jendela di bawah ikon python.exe,
        # jadi logo COMUNICO tidak muncul di taskbar.
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "IOH-AIRAN.COMUNICO.Percakapan")
        except Exception:  # noqa: BLE001 - hanya ikon, abaikan kalau gagal
            pass
    akar = tk.Tk()
    aplikasi = Aplikasi(akar, pekerja, pengucap, riwayat, catat,
                        suara=not arg.tanpa_suara, pendengar=pendengar,
                        buka_kamera_laptop=buka_kamera_laptop,
                        buka_mikrofon_laptop=lambda: Mikrofon(perangkat))
    if arg.esp32:
        aplikasi.alamat_esp.set(arg.esp32)
        aplikasi.hubungkan_esp32()
    pekerja.start()
    pendengar.start()
    if arg.mulai_dengan_suara:
        aplikasi.mode.set("suara")
        aplikasi._ganti_mode()
    akar.after(15, aplikasi.putar)
    waktu_mulai = time.monotonic()
    try:
        akar.mainloop()
    finally:
        pekerja.berhenti.set()
        pendengar.aktif.clear()
        pendengar.berhenti.set()
        pekerja.join(timeout=2.0)
        catat.selesai(penyusun.teks, 0, time.monotonic() - waktu_mulai)
        pengucap.tutup()
        pendeteksi.close()
        if pekerja.sumber is not None:
            pekerja.sumber.release()
    return 0


if __name__ == "__main__":
    sys.exit(utama())
