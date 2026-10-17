"""Menghubungkan percakapan.py dengan ESP32-CAM lewat Wi-Fi.

ESP32 hanya mengirim gambar dan suara, lalu menampilkan hasilnya. Semua
pembacaan (isyarat dan suara ke teks) tetap dikerjakan laptop:

    ESP32-CAM  -- video MJPEG   port 81  GET /stream  -->  laptop: MediaPipe + model huruf
    INMP441    -- suara PCM     port 82  GET /audio   -->  laptop: suara ke teks
    OLED       <-- pesan teks   port 80  POST /pesan  --   laptop: hasil kedua sisi

Suara dikirim mentah: 16 kHz, mono, 16 bit little-endian, tanpa kepala berkas.
Firmware yang cocok ada di firmware/comunico_esp32/comunico_esp32.ino.

Semua kelas di sini menyambung ulang sendiri kalau Wi-Fi putus sebentar, jadi
program tidak berhenti hanya karena ESP32 dimulai ulang.
"""

from __future__ import annotations

import queue
import threading
import time
import unicodedata
import urllib.parse
import urllib.request

import cv2
import numpy as np

PORT_PESAN, PORT_VIDEO, PORT_SUARA = 80, 81, 82
LAJU = 16000                    # sama dengan suara_ke_teks.LAJU
TUNGGU_JARINGAN = 5.0           # detik tanpa data sebelum dianggap putus


def rapikan_alamat(teks: str) -> str:
    """'http://192.168.1.50/' -> '192.168.1.50'. Port diabaikan, dipakai port tetap."""
    alamat = teks.strip()
    if "://" in alamat:
        alamat = alamat.split("://", 1)[1]
    alamat = alamat.split("/", 1)[0]
    return alamat.split(":", 1)[0]


def _pendek(e: Exception) -> str:
    alasan = getattr(e, "reason", None) or e
    return f"{type(e).__name__}: {alasan}"[:80]


# --------------------------------------------------------------------- video
class KameraESP32:
    """Pengganti cv2.VideoCapture untuk aliran MJPEG dari ESP32-CAM.

    read() tidak pernah memblokir lama. Kalau belum ada gambar baru, hasilnya
    (True, None), supaya PekerjaKamera tetap bisa mengurus tombol. Gambar yang
    sama tidak pernah dikembalikan dua kali, jadi voting huruf tidak terhitung
    ganda saat Wi-Fi tersendat.
    """

    def __init__(self, host: str, *, tunggu_baca: float = 0.2):
        self.url = f"http://{host}:{PORT_VIDEO}/stream"
        self.tunggu_baca = tunggu_baca
        self.keadaan = "menghubungkan video..."
        self._frame = None
        self._nomor = 0
        self._nomor_dibaca = 0
        self._syarat = threading.Condition()
        self._berhenti = threading.Event()
        self._thread = threading.Thread(target=self._jalan, daemon=True)
        self._thread.start()

    def isOpened(self) -> bool:
        return not self._berhenti.is_set()

    def read(self):
        if self._berhenti.is_set():
            return False, None
        with self._syarat:
            self._syarat.wait_for(
                lambda: self._nomor != self._nomor_dibaca or self._berhenti.is_set(),
                timeout=self.tunggu_baca)
            if self._nomor == self._nomor_dibaca:
                return True, None
            self._nomor_dibaca = self._nomor
            return True, self._frame

    def release(self) -> None:
        self._berhenti.set()
        with self._syarat:
            self._syarat.notify_all()

    def _jalan(self) -> None:
        while not self._berhenti.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=TUNGGU_JARINGAN) as r:
                    self.keadaan = "video tersambung"
                    self._baca_aliran(r)
                if not self._berhenti.is_set():
                    self.keadaan = "video terputus, menyambung ulang..."
            except Exception as e:  # noqa: BLE001 - dicoba lagi, alasannya ditampilkan
                self.keadaan = f"video gagal ({_pendek(e)})"
            self._berhenti.wait(1.0)

    def _baca_aliran(self, r) -> None:
        # Tiap gambar JPEG diawali FF D8 dan diakhiri FF D9. Mencari dua penanda
        # ini lebih tahan banting daripada mengurai pembatas multipart.
        sisa = b""
        hitung, mulai = 0, time.monotonic()
        while not self._berhenti.is_set():
            data = r.read1(16384)
            if not data:
                return
            sisa += data
            while True:
                awal = sisa.find(b"\xff\xd8")
                if awal < 0:
                    sisa = sisa[-1:]
                    break
                akhir = sisa.find(b"\xff\xd9", awal + 2)
                if akhir < 0:
                    sisa = sisa[awal:]
                    break
                jpg, sisa = sisa[awal:akhir + 2], sisa[akhir + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                with self._syarat:
                    self._frame = frame
                    self._nomor += 1
                    self._syarat.notify_all()
                hitung += 1
                lewat = time.monotonic() - mulai
                if lewat >= 1.0:
                    self.keadaan = f"video {hitung / lewat:.0f} fps"
                    hitung, mulai = 0, time.monotonic()
            if len(sisa) > 2_000_000:       # aliran rusak, jangan menumpuk memori
                sisa = b""


# --------------------------------------------------------------------- suara
class MikrofonESP32:
    """Pengganti suara_ke_teks.Mikrofon: membaca suara PCM dari ESP32.

    Hasil ambil() sama persis dengan Mikrofon: potongan float32 16 kHz mono.
    Mikrofon INMP441 punya sedikit pergeseran DC; itu dibuang di sini supaya
    pengukur energi di PemotongUcapan tidak mengira ada suara terus-menerus.
    """

    def __init__(self, host: str, *, blok: int = 1024):
        self.url = f"http://{host}:{PORT_SUARA}/audio"
        self.blok = blok
        self.keadaan = "menghubungkan mikrofon..."
        self._antrian: queue.Queue[np.ndarray] = queue.Queue(maxsize=400)
        self._dc = 0.0
        self._berhenti = threading.Event()
        self._thread = threading.Thread(target=self._jalan, daemon=True)
        self._thread.start()

    def ambil(self, tunggu: float = 0.1) -> np.ndarray | None:
        try:
            return self._antrian.get(timeout=tunggu)
        except queue.Empty:
            return None

    def tutup(self) -> None:
        self._berhenti.set()

    def _jalan(self) -> None:
        while not self._berhenti.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=TUNGGU_JARINGAN) as r:
                    self.keadaan = "mikrofon tersambung"
                    self._baca_aliran(r)
            except Exception as e:  # noqa: BLE001
                self.keadaan = f"mikrofon gagal ({_pendek(e)})"
            self._berhenti.wait(1.0)

    def _baca_aliran(self, r) -> None:
        sisa = b""
        ukuran = self.blok * 2                      # 2 byte per sampel
        while not self._berhenti.is_set():
            data = r.read1(8192)
            if not data:
                return
            sisa += data
            while len(sisa) >= ukuran:
                mentah, sisa = sisa[:ukuran], sisa[ukuran:]
                potongan = np.frombuffer(mentah, dtype="<i2").astype(np.float32) / 32768.0
                self._dc = 0.98 * self._dc + 0.02 * float(potongan.mean())
                try:
                    self._antrian.put_nowait(potongan - self._dc)
                except queue.Full:
                    pass                # lebih baik kehilangan audio daripada macet


# -------------------------------------------------------------------- pesan
class PengirimESP32:
    """Mengirim pesan ke layar OLED ESP32 di thread sendiri, tanpa menahan layar."""

    def __init__(self, host: str):
        self.host = host
        self.keadaan = ""
        self._antrian: queue.Queue = queue.Queue(maxsize=50)
        self._berhenti = threading.Event()
        self._thread = threading.Thread(target=self._jalan, daemon=True)
        self._thread.start()

    def kirim(self, dari: str, teks: str) -> None:
        """dari: 'isyarat' (penyandang tuli) atau 'lawan' (lawan bicara)."""
        try:
            self._antrian.put_nowait((dari, teks))
        except queue.Full:
            pass

    def tutup(self) -> None:
        self._berhenti.set()
        self._antrian.put(None)

    def _jalan(self) -> None:
        while not self._berhenti.is_set():
            isi = self._antrian.get()
            if isi is None:
                return
            dari, teks = isi
            # Huruf OLED hanya ASCII: 'é' jadi 'e', huruf lain dibuang.
            polos = unicodedata.normalize("NFKD", teks).encode("ascii", "ignore")
            url = (f"http://{self.host}:{PORT_PESAN}/pesan?"
                   + urllib.parse.urlencode({"dari": dari}))
            permintaan = urllib.request.Request(
                url, data=polos, method="POST",
                headers={"Content-Type": "text/plain; charset=us-ascii"})
            try:
                with urllib.request.urlopen(permintaan, timeout=3) as r:
                    r.read()
                self.keadaan = "pesan terkirim"
            except Exception as e:  # noqa: BLE001
                self.keadaan = f"pesan gagal ({_pendek(e)})"


# ---------------------------------------------------------------- sambungan
class SambunganESP32:
    """Satu ESP32: video, pengirim pesan, dan pembuka mikrofonnya."""

    def __init__(self, alamat: str):
        self.host = rapikan_alamat(alamat)
        if not self.host:
            raise ValueError("alamat ESP32 kosong")
        self.kamera = KameraESP32(self.host)
        self.pengirim = PengirimESP32(self.host)
        self._mikrofon: MikrofonESP32 | None = None

    def buka_mikrofon(self) -> MikrofonESP32:
        self._mikrofon = MikrofonESP32(self.host)
        return self._mikrofon

    @property
    def keadaan(self) -> str:
        bagian = [self.kamera.keadaan]
        if self._mikrofon is not None and not self._mikrofon._berhenti.is_set():
            bagian.append(self._mikrofon.keadaan)
        if self.pengirim.keadaan:
            bagian.append(self.pengirim.keadaan)
        return "  |  ".join(bagian)

    def tutup(self) -> None:
        # Kamera tidak ditutup di sini: PekerjaKamera yang melepasnya saat ganti sumber.
        self.pengirim.tutup()
        if self._mikrofon is not None:
            self._mikrofon.tutup()
