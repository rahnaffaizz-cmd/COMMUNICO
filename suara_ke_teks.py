"""Suara ke teks, dijalankan di CPU saja.

Dipakai percakapan.py supaya lawan bicara bisa berbicara, bukan mengetik.

ALURNYA
-------
    mikrofon -> potongan audio kecil
             -> PemotongUcapan   mencari kapan orang mulai dan selesai bicara
             -> PengenalSuara    model Whisper (faster-whisper) mengubah satu
                                 ucapan jadi teks
             -> teks jadi satu pesan di percakapan

Modelnya berjalan di CPU dengan kuantisasi int8, tanpa kartu grafis. Model
`id-base` disetel untuk bahasa Indonesia; model dan dependensi konversinya
disiapkan sekali saat pertama dipakai.

DUA HAL YANG SENGAJA DIJAGA
---------------------------
1. Suara dari pengeras suara (suara hasil pembacaan isyarat) ikut tertangkap
   mikrofon. Kalau dibiarkan, suara itu terbaca ulang sebagai ucapan lawan
   bicara. Selama program sedang berbicara, dan sebentar sesudahnya, audio dari
   mikrofon dibuang.
2. Whisper suka "mengarang" teks dari keheningan atau derau. Ucapan hanya
   diproses kalau energinya benar-benar di atas derau ruangan, dan hasil yang
   model sendiri ragukan dibuang.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque

import numpy as np

LAJU = 16000                    # Whisper mengharapkan audio 16 kHz mono


# ------------------------------------------------------------ memotong ucapan
class PemotongUcapan:
    """Mencari awal dan akhir ucapan dari aliran audio, berdasarkan energinya.

    Cara kerjanya sederhana: energi (RMS) tiap potongan dibandingkan dengan
    batas yang mengikuti derau ruangan. Di atas batas berarti orang bicara.
    Ucapan dianggap selesai setelah `jeda_diam` detik tenang. Sedikit audio
    sebelum ucapan (`pra_rekam`) ikut disimpan supaya huruf pertama tidak
    terpotong.
    """

    def __init__(self, *, laju: int = LAJU, ambang_dasar: float = 0.004,
                 kali_derau: float = 4.0, jeda_diam: float = 0.8,
                 panjang_min: float = 0.4, panjang_maks: float = 15.0,
                 pra_rekam: float = 0.3):
        self.laju = laju
        self.ambang_dasar = ambang_dasar
        self.kali_derau = kali_derau
        self.jeda_diam = jeda_diam
        self.panjang_min = panjang_min
        self.panjang_maks = panjang_maks
        self._pra_rekam_detik = pra_rekam

        self.tingkat = 0.0                  # energi potongan terakhir, untuk penunjuk
        self.derau = 0.004
        self.reset()

    def reset(self) -> None:
        self._pra: deque[np.ndarray] = deque()
        self._pra_panjang = 0
        self._bicara: list[np.ndarray] = []
        self._sedang_bicara = False
        self._diam = 0.0                    # detik tenang berturut-turut
        self._panjang_bicara = 0.0          # detik yang benar-benar bersuara

    @property
    def batas(self) -> float:
        return max(self.ambang_dasar, self.derau * self.kali_derau)

    @property
    def sedang_bicara(self) -> bool:
        return self._sedang_bicara

    def masuk(self, potongan: np.ndarray) -> list[np.ndarray]:
        """Beri satu potongan audio (float32 mono). Balik ucapan yang baru selesai."""
        selesai: list[np.ndarray] = []
        durasi = len(potongan) / self.laju
        rms = float(np.sqrt(np.mean(np.square(potongan, dtype=np.float64)))) if len(potongan) else 0.0
        self.tingkat = rms
        bersuara = rms > self.batas

        if not self._sedang_bicara:
            if bersuara:
                self._sedang_bicara = True
                self._bicara = list(self._pra) + [potongan]
                self._pra.clear()
                self._pra_panjang = 0
                self._diam = 0.0
                self._panjang_bicara = durasi
            else:
                # Derau ruangan diikuti pelan-pelan, hanya saat tidak ada yang bicara.
                self.derau = 0.95 * self.derau + 0.05 * rms
                self._pra.append(potongan)
                self._pra_panjang += len(potongan)
                batas_pra = int(self._pra_rekam_detik * self.laju)
                while self._pra_panjang > batas_pra and len(self._pra) > 1:
                    self._pra_panjang -= len(self._pra.popleft())
            return selesai

        self._bicara.append(potongan)
        if bersuara:
            self._diam = 0.0
            self._panjang_bicara += durasi
        else:
            self._diam += durasi

        total = sum(len(p) for p in self._bicara) / self.laju
        if self._diam >= self.jeda_diam or total >= self.panjang_maks:
            if self._panjang_bicara >= self.panjang_min:
                selesai.append(np.concatenate(self._bicara).astype(np.float32))
            self.reset()
        return selesai


# ------------------------------------------------------------------ mengenali
class PengenalSuara:
    """Whisper di CPU. Modelnya baru dimuat saat pertama dibutuhkan."""

    def __init__(self, ukuran: str = "id-base", bahasa: str = "id",
                 jumlah_thread: int = 4, beam: int = 5, lapor=None):
        # Bisa memakai model Indonesia yang disiapkan lokal atau ukuran Whisper biasa.
        self.ukuran = ukuran
        self.bahasa = bahasa
        self.jumlah_thread = jumlah_thread
        self.beam = beam
        self.lapor = lapor or (lambda _teks: None)
        self._model = None

    @property
    def siap(self) -> bool:
        return self._model is not None

    def muat(self) -> None:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ModuleNotFoundError as e:
                # Program bisa jalan dengan Python yang berbeda dari tempat paket
                # dipasang (ada .venv di folder ini dan di folder induk), jadi
                # pesannya menyebut Python mana yang sedang dipakai.
                raise RuntimeError(
                    f"Paket '{e.name}' belum terpasang di Python yang sedang dipakai:\n"
                    f"{sys.executable}\n\nPasang dengan:\n"
                    f'uv pip install --python "{sys.executable}" -r requirements.txt'
                ) from e

            from siapkan_model_suara import MODEL, siapkan

            sumber = str(siapkan(self.ukuran, self.lapor)) if self.ukuran in MODEL \
                else self.ukuran
            self.lapor(f"Memuat model suara '{self.ukuran}' ...")
            # device="cpu" dan int8: tidak butuh kartu grafis, dan memori kecil.
            self._model = WhisperModel(sumber, device="cpu", compute_type="int8",
                                       cpu_threads=self.jumlah_thread)

    def kenali(self, audio: np.ndarray) -> str:
        """Ubah satu ucapan jadi teks. Balik string kosong kalau tidak yakin."""
        self.muat()
        segmen, _ = self._model.transcribe(
            audio.astype(np.float32), language=self.bahasa, beam_size=self.beam,
            condition_on_previous_text=False, temperature=0.0)

        bagian = []
        for s in segmen:
            # Buang yang model sendiri ragukan: kemungkinan besar bukan ucapan.
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                continue
            if s.avg_logprob < -1.6:
                continue
            bagian.append(s.text.strip())
        hasil = " ".join(b for b in bagian if b).strip()
        return hasil[:1].upper() + hasil[1:]


# ------------------------------------------------------------------- mikrofon
class Mikrofon:
    """Membaca mikrofon lewat sounddevice, selalu menghasilkan audio 16 kHz mono."""

    def __init__(self, perangkat=None, blok: int = 1024):
        import sounddevice as sd

        self._antrian: queue.Queue[np.ndarray] = queue.Queue(maxsize=400)
        self._laju_asli = LAJU

        def isi(data, _frames, _waktu, _status):
            try:
                self._antrian.put_nowait(data[:, 0].copy())
            except queue.Full:
                pass                        # lebih baik kehilangan audio daripada macet

        try:
            self._aliran = sd.InputStream(samplerate=LAJU, channels=1, dtype="float32",
                                          blocksize=blok, device=perangkat, callback=isi)
        except Exception:                   # noqa: BLE001 - perangkat tak mendukung 16 kHz
            info = sd.query_devices(perangkat, "input")
            self._laju_asli = int(info["default_samplerate"])
            self._aliran = sd.InputStream(samplerate=self._laju_asli, channels=1,
                                          dtype="float32", device=perangkat,
                                          blocksize=int(blok * self._laju_asli / LAJU),
                                          callback=isi)
        self._aliran.start()

    def ambil(self, tunggu: float = 0.1) -> np.ndarray | None:
        try:
            potongan = self._antrian.get(timeout=tunggu)
        except queue.Empty:
            return None
        if self._laju_asli != LAJU:
            n = max(1, int(len(potongan) * LAJU / self._laju_asli))
            potongan = np.interp(np.linspace(0, len(potongan) - 1, n),
                                 np.arange(len(potongan)), potongan).astype(np.float32)
        return potongan

    def tutup(self) -> None:
        try:
            self._aliran.stop()
            self._aliran.close()
        except Exception:                   # noqa: BLE001
            pass


# ------------------------------------------------------------------ pendengar
class PendengarSuara(threading.Thread):
    """Thread yang mendengarkan mikrofon selama `aktif` menyala.

    Hasilnya dititipkan ke antrian `keluar` sebagai pasangan (jenis, isi):
        ("status", teks)   keadaan sekarang, untuk ditampilkan
        ("tingkat", 0..1)  kerasnya suara, untuk penunjuk
        ("teks", teks)     satu ucapan yang sudah jadi teks
        ("galat", teks)    sesuatu gagal, mikrofon dimatikan
    """

    def __init__(self, pengenal: PengenalSuara, *, buka_mikrofon, sedang_bicara=None,
                 tenang_setelah: float = 0.8, pemotong: PemotongUcapan | None = None):
        super().__init__(daemon=True)
        self.pengenal = pengenal
        self.buka_mikrofon = buka_mikrofon          # fungsi tanpa argumen -> objek Mikrofon
        self.sedang_bicara = sedang_bicara or (lambda: False)
        self.tenang_setelah = tenang_setelah
        self.pemotong = pemotong or PemotongUcapan()
        self.aktif = threading.Event()
        self.berhenti = threading.Event()
        self.keluar: queue.Queue = queue.Queue()

    def _titip(self, jenis: str, isi) -> None:
        self.keluar.put((jenis, isi))

    def run(self) -> None:
        while not self.berhenti.is_set():
            if not self.aktif.wait(timeout=0.1):
                continue
            try:
                self._dengarkan()
            except Exception as e:  # noqa: BLE001 - ditampilkan, bukan ditelan
                self.aktif.clear()
                self._titip("galat", f"{type(e).__name__}: {e}")

    def _dengarkan(self) -> None:
        if not self.pengenal.siap:
            self._titip("status", f"Memuat model suara '{self.pengenal.ukuran}' ...")
            self.pengenal.lapor = lambda teks: self._titip("status", teks)
            self.pengenal.muat()

        mik = self.buka_mikrofon()
        self.pemotong.reset()
        self._titip("status", "Mendengarkan...")
        # Sisa waktu mengabaikan mikrofon, dihitung dalam detik AUDIO, bukan jam
        # dinding. Kalau pengenalan sedang lambat dan audio menumpuk, hitungan
        # jam dinding akan membuang ucapan yang sebenarnya sudah lewat masanya.
        abaikan_detik = 0.0
        sebelumnya_bicara = False
        try:
            while self.aktif.is_set() and not self.berhenti.is_set():
                potongan = mik.ambil(0.1)
                if potongan is None:
                    continue

                # Suara dari pengeras suara jangan sampai terbaca sebagai ucapan.
                bicara = self.sedang_bicara()
                if bicara or sebelumnya_bicara:
                    abaikan_detik = self.tenang_setelah
                sebelumnya_bicara = bicara
                if abaikan_detik > 0:
                    abaikan_detik -= len(potongan) / LAJU
                    self.pemotong.reset()
                    self._titip("tingkat", 0.0)
                    continue

                ucapan = self.pemotong.masuk(potongan)
                self._titip("tingkat", min(1.0, self.pemotong.tingkat * 8))
                for audio in ucapan:
                    self._titip("status", "Mengenali...")
                    teks = self.pengenal.kenali(audio)
                    if teks:
                        self._titip("teks", teks)
                    self._titip("status", "Mendengarkan...")
        finally:
            mik.tutup()
            self._titip("tingkat", 0.0)
            self._titip("status", "Mikrofon mati")
