"""Pengenal huruf BISINDO secara langsung, lalu diucapkan dengan suara.

Alur kerjanya begini:
    kamera -> MediaPipe cari titik-titik tangan -> 126 angka koordinat
           -> model Random Forest menebak hurufnya (rf_bisindo_99.pkl)
           -> huruf disaring dulu lewat voting biar tidak asal masuk
           -> huruf ditumpuk jadi kata -> diucapkan pakai Google TTS

Google TTS butuh internet waktu pertama kali mengucapkan sebuah kata.
Setelah itu suaranya disimpan di komputer, jadi kalau kata yang sama
diucapkan lagi tidak perlu internet.

Cara pakai:
    python gesture_voice.py
    python gesture_voice.py --kamera 1 --bahasa id --pelan
    python gesture_voice.py --tanpa-eja-huruf     # cuma kata yang diucapkan

Tombol saat jendela kamera terbuka:
    Q / ESC    keluar
    BACKSPACE  hapus huruf terakhir
    C          hapus semua teks
    SPASI      tambah spasi
    ENTER      ucapkan teks sekarang juga
    P          nyalakan / matikan ucap otomatis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import tempfile
import threading
import time
import warnings
from collections import Counter, deque
from ctypes import c_buffer, windll
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from catatan import Pencatat
from fitur import ambil_fitur
from pengukur import PengukurHuruf

# Dua peringatan ini cuma berisik di layar, tidak bikin program rusak.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message=".*GetPrototype.*")

PATH_MODEL = "rf_bisindo_v2.pkl"

# Warna dipakai OpenCV dengan urutan BGR (biru, hijau, merah), bukan RGB.
ABU, HIJAU, BIRU, PUTIH, KUNING, GELAP = (
    (160, 160, 160), (80, 220, 80), (255, 160, 60),
    (255, 255, 255), (60, 200, 255), (30, 30, 30),
)
HURUF_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------------ suara / TTS
def _perintah_mci(perintah: str) -> str:
    """Kirim satu perintah ke pemutar bawaan Windows (winmm) dan ambil balasannya."""
    penampung = c_buffer(255)
    windll.winmm.mciSendStringW(perintah, penampung, 254, 0)
    return penampung.value


class Pengucap:
    """Tukang bicara. Jalan di thread sendiri supaya video tidak ikut macet.

    Setiap teks diubah jadi file mp3 oleh Google TTS, lalu diputar lewat
    pemutar bawaan Windows. File mp3-nya disimpan (di-cache) supaya kata yang
    sama tidak perlu diunduh ulang.
    """

    def __init__(self, bahasa: str = "id", pelan: bool = False, catat=None):
        self.bahasa = bahasa
        self.pelan = pelan
        self.catat = catat
        self._folder_simpan = Path(tempfile.gettempdir()) / "gesture_voice_tts"
        self._folder_simpan.mkdir(exist_ok=True)
        self._antrian: queue.Queue[str | None] = queue.Queue()
        self._sedang_bicara = threading.Event()
        self._sudah_pernah_lapor = False
        self._thread = threading.Thread(target=self._jalan, daemon=True)
        self._thread.start()

    def _path_klip(self, teks: str) -> Path:
        """Nama file dibuat dari sidik jari teksnya, jadi tiap kata punya file sendiri."""
        sidik = hashlib.md5(
            f"{self.bahasa}|{self.pelan}|{teks}".encode()
        ).hexdigest()
        return self._folder_simpan / f"{sidik}.mp3"

    def _buat_suara(self, teks: str, path: Path) -> bool:
        """Minta Google TTS bikin file mp3. Balik False kalau gagal (misal tidak ada internet)."""
        from gtts import gTTS

        mulai = time.monotonic()
        try:
            gTTS(text=teks, lang=self.bahasa, slow=self.pelan).save(str(path))
            if self.catat:
                self.catat.suara_unduh(teks, time.monotonic() - mulai)
            return True
        except Exception as e:  # noqa: BLE001 - biasanya masalah internet
            if self.catat:
                self.catat.suara_gagal(teks, str(e))
            if not self._sudah_pernah_lapor:
                print(f"Google TTS tidak bisa dihubungi ({e}). Suara dimatikan "
                      f"sampai internet kembali. Kata yang sudah tersimpan tetap bisa dibunyikan.")
                self._sudah_pernah_lapor = True
            path.unlink(missing_ok=True)
            return False

    def _putar(self, path: Path) -> None:
        """Putar file mp3 sampai selesai."""
        nama = f"gv{threading.get_ident()}"
        _perintah_mci(f'open "{path}" type mpegvideo alias {nama}')
        try:
            _perintah_mci(f"play {nama} wait")
        finally:
            _perintah_mci(f"close {nama}")

    def _jalan(self) -> None:
        """Loop di thread: ambil teks dari antrian, bunyikan, ulangi."""
        while True:
            teks = self._antrian.get()
            if teks is None:  # tanda dari tutup(): waktunya berhenti
                break
            self._sedang_bicara.set()
            mulai = time.monotonic()
            try:
                klip = self._path_klip(teks)
                tersimpan = klip.exists()
                if tersimpan and self.catat:
                    self.catat.suara_simpanan(teks)
                if tersimpan or self._buat_suara(teks, klip):
                    self._putar(klip)
                    if self.catat:
                        self.catat.suara_selesai(teks, time.monotonic() - mulai)
            except Exception as e:  # noqa: BLE001
                if self.catat:
                    self.catat.suara_gagal(teks, str(e))
                print(f"Gagal memutar suara: {e}")
            finally:
                self._sedang_bicara.clear()

    def ucapkan(self, teks: str) -> None:
        """Titip teks untuk diucapkan. Kalau masih bicara, permintaan diabaikan."""
        teks = teks.strip()
        if teks and not self._sedang_bicara.is_set():
            if self.catat:
                self.catat.suara_minta(teks)
            self._antrian.put(teks)

    @property
    def sibuk(self) -> bool:
        return self._sedang_bicara.is_set() or not self._antrian.empty()

    def tutup(self) -> None:
        self._antrian.put(None)


# --------------------------------------------------------- voting & kumpulan teks
class VotingHuruf:
    """Penyaring tebakan biar huruf tidak gampang salah masuk.

    Idenya sederhana: tebakan dari beberapa frame terakhir dikumpulkan, lalu
    huruf yang paling sering muncul dianggap menang. Huruf itu baru diterima
    kalau menangnya cukup banyak, rata-rata keyakinannya cukup tinggi, dan
    jarak waktunya dengan huruf sebelumnya sudah cukup jauh. Tanpa ini, satu
    pose tangan yang ditahan sebentar bisa jadi sepuluh huruf sekaligus.
    """

    def __init__(self, jumlah_frame: int, frame_konfirmasi: int,
                 min_keyakinan: float, jeda: float, catat=None,
                 ambang_huruf: dict | None = None):
        self.jumlah_frame = jumlah_frame
        self.frame_konfirmasi = frame_konfirmasi
        self.min_keyakinan = min_keyakinan
        self.jeda = jeda
        self.catat = catat
        # Ambang khusus tiap huruf, hasil diagnosa.py. Kalau kosong, semua huruf
        # memakai min_keyakinan yang sama.
        self.ambang_huruf = ambang_huruf or {}
        self._riwayat: deque[tuple[str, float]] = deque(maxlen=jumlah_frame)
        self._waktu_terima_terakhir = float("-inf")
        self._lapor_terakhir = ""

    def ambang_untuk(self, huruf: str) -> float:
        """Ambang keyakinan yang berlaku untuk satu huruf."""
        return self.ambang_huruf.get(huruf, self.min_keyakinan)

    def kosongkan(self) -> None:
        self._riwayat.clear()

    def _lapor(self, pesan_kunci: str, tulis) -> None:
        """Catat sekali saja selama alasannya belum berubah, biar tidak banjir."""
        if self.catat and pesan_kunci != self._lapor_terakhir:
            self._lapor_terakhir = pesan_kunci
            tulis()

    def masukkan(self, huruf: str, keyakinan: float, sekarang: float) -> str | None:
        """Masukkan satu tebakan. Balik nama huruf kalau diterima, None kalau belum."""
        self._riwayat.append((huruf, keyakinan))

        jumlah = Counter(h for h, _ in self._riwayat)
        juara, banyak = jumlah.most_common(1)[0]
        nilai = [k for h, k in self._riwayat if h == juara]
        rata = sum(nilai) / len(nilai)

        if banyak < self.frame_konfirmasi:
            # Belum cukup sering muncul.
            self._lapor(f"kurang-{juara}-{banyak}",
                        lambda: self.catat.voting(juara, banyak,
                                                  self.frame_konfirmasi, rata))
            return None

        batas = self.ambang_untuk(juara)
        if rata < batas:
            # Sering muncul, tapi modelnya sendiri ragu-ragu.
            if batas >= 0.99:
                # Huruf ini memang sengaja ditolak. Diagnosa menunjukkan model
                # justru lebih yakin saat menebaknya salah, jadi ambang berapa
                # pun tidak menolong dan lebih baik tidak dimasukkan sama sekali.
                self._lapor(f"tolak-{juara}", lambda: self.catat.voting_gagal(
                    juara, "huruf ini ditolak, model tidak bisa diandalkan untuknya"))
            else:
                self._lapor(f"ragu-{juara}", lambda: self.catat.voting_gagal(
                    juara, f"keyakinan {rata:.2f} di bawah ambang {juara} "
                           f"yang {batas:.2f}"))
            return None

        sisa = self.jeda - (sekarang - self._waktu_terima_terakhir)
        if sisa > 0:
            # Masih terlalu cepat dari huruf sebelumnya.
            self._lapor(f"jeda-{juara}", lambda: self.catat.voting_gagal(
                juara, f"masih menunggu jeda {sisa:.1f} detik lagi"))
            return None

        if self.catat:
            self.catat.voting(juara, banyak, self.frame_konfirmasi, rata)
        self._waktu_terima_terakhir = sekarang
        self._riwayat.clear()
        self._lapor_terakhir = ""
        return juara

    @property
    def kemajuan(self) -> tuple[str, float]:
        """Huruf yang sedang unggul dan seberapa dekat dia ke ambang konfirmasi."""
        if not self._riwayat:
            return "", 0.0
        juara, banyak = Counter(h for h, _ in self._riwayat).most_common(1)[0]
        return juara, banyak / self.frame_konfirmasi


# ------------------------------------------------------------------ titik tangan
def vektor_titik_tangan(hasil) -> tuple[np.ndarray | None, int]:
    """Ubah hasil MediaPipe jadi 126 angka, sesuai bentuk yang dipelajari model.

    Satu tangan = 21 titik x 3 koordinat (x, y, z) = 63 angka.
    Model dilatih dengan dua tangan, jadi kalau cuma satu tangan yang terlihat,
    sisanya diisi angka nol.
    """
    if not hasil.multi_hand_landmarks:
        return None, 0

    angka: list[float] = []
    for tangan in hasil.multi_hand_landmarks[:2]:
        for titik in tangan.landmark:
            angka.extend((titik.x, titik.y, titik.z))

    jumlah_tangan = len(hasil.multi_hand_landmarks[:2])
    if jumlah_tangan == 1:
        angka.extend([0.0] * 63)
    return np.asarray(angka, dtype=np.float32).reshape(1, -1), jumlah_tangan


# ------------------------------------------------------------------ ambang huruf
def muat_ambang_huruf(catat=None) -> dict:
    """Baca ambang keyakinan tiap huruf dari ambang_huruf.json, kalau ada.

    Berkas itu dibuat oleh diagnosa.py. Isinya ambang yang berbeda-beda karena
    tiap huruf perilakunya tidak sama: G hampir selalu benar sehingga ambang
    rendah pun aman, sedangkan E justru lebih sering yakin waktu salah sehingga
    lebih baik ditolak sama sekali.
    """
    berkas = Path(__file__).resolve().parent / "ambang_huruf.json"
    if not berkas.exists():
        if catat:
            catat.kabar("AMBANG", "ambang_huruf.json tidak ada, semua huruf "
                                  "memakai ambang yang sama")
        return {}

    isi = json.loads(berkas.read_text(encoding="utf-8"))
    ambang = isi.get("ambang", {})
    ditolak = sorted(h for h, nilai in ambang.items() if nilai >= 0.99)

    if catat:
        catat.kabar("AMBANG", f"memakai ambang per huruf dari {berkas.name} "
                              f"(target presisi {isi.get('target_presisi', '?')}%)")
        if ditolak:
            catat.kabar("AMBANG", f"huruf yang ditolak: {' '.join(ditolak)} "
                                  f"-- pakai --ambang-seragam kalau tetap ingin dicoba")
    return ambang


# ----------------------------------------------------------------------- argumen
def baca_argumen() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pengenal huruf BISINDO dengan suara (Google TTS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kamera", type=int, default=0,
                   help="nomor kamera, coba 1 atau 2 kalau yang kebuka salah")
    p.add_argument("--jendela-voting", type=int, default=12,
                   help="berapa frame terakhir yang ikut voting")
    p.add_argument("--frame-konfirmasi", type=int, default=8,
                   help="berapa frame harus sama sebelum huruf diterima")
    p.add_argument("--min-keyakinan", type=float, default=0.45,
                   help="keyakinan minimal model, naikkan kalau sering salah huruf")
    p.add_argument("--jeda", type=float, default=1.2,
                   help="jarak waktu minimal antar huruf, dalam detik")
    p.add_argument("--ucap-setelah", type=float, default=2.5,
                   help="berapa detik diam sebelum kata diucapkan")
    p.add_argument("--spasi-setelah", type=float, default=2.0,
                   help="berapa detik tangan hilang sebelum spasi ditambah")
    p.add_argument("--bahasa", default="id",
                   help="kode bahasa Google TTS (id = Indonesia, en = Inggris)")
    p.add_argument("--pelan", action="store_true", help="suara dibaca lebih pelan")
    p.add_argument("--tanpa-eja-huruf", action="store_true",
                   help="jangan bunyikan tiap huruf, cuma kata yang jadi")
    p.add_argument("--tanpa-auto-ucap", action="store_true",
                   help="jangan bunyikan otomatis, tetap bisa lewat tombol ENTER")
    p.add_argument("--keyakinan-deteksi", type=float, default=0.6,
                   help="ambang MediaPipe untuk mulai mendeteksi tangan")
    p.add_argument("--keyakinan-tracking", type=float, default=0.5,
                   help="ambang MediaPipe untuk terus mengikuti tangan")
    p.add_argument("--ukur", metavar="HURUF", default=None,
                   help="mode pengukuran: peragakan satu huruf terus-menerus, "
                        "nanti dilaporkan seberapa baik huruf itu terbaca oleh "
                        "kamera Anda sendiri. Contoh: --ukur I")
    p.add_argument("--ambang-seragam", action="store_true",
                   help="abaikan ambang_huruf.json, pakai --min-keyakinan untuk "
                        "semua huruf. Huruf yang biasanya ditolak jadi ikut masuk")
    p.add_argument("--tanpa-catatan", action="store_true",
                   help="matikan pencatatan jalannya program")
    p.add_argument("--catat-detail", action="store_true",
                   help="catat juga tiap tebakan dan kemajuan voting, bukan cuma "
                        "kejadian penting. Berguna untuk mencari tahu kenapa "
                        "sebuah huruf tidak pernah diterima")
    return p.parse_args()


# ---------------------------------------------------------------------- tampilan
def gambar_tampilan(frame, *, mode, tebakan, keyakinan, huruf_voting, kemajuan_voting,
                    teks, sedang_bicara, hitung_mundur, ucap_setelah, auto_ucap,
                    titik_tangan, mp_gambar, mp_tangan, tiga_besar=()):
    """Tulis semua info ke atas gambar kamera."""
    tinggi, lebar = frame.shape[:2]

    # Rangka tangan, biar kelihatan MediaPipe benar-benar menemukan tangannya.
    for tangan in titik_tangan:
        mp_gambar.draw_landmarks(frame, tangan, mp_tangan.HAND_CONNECTIONS)

    warna_mode = {"diam": ABU, "baca": HIJAU}[mode]
    cv2.putText(frame, f"MODE: {mode.upper()}", (20, 35), HURUF_FONT, 0.7, warna_mode, 2)

    # Huruf tebakan sementara, ditulis besar di tengah atas.
    if tebakan:
        cv2.putText(frame, tebakan, (lebar // 2 - 40, 120), HURUF_FONT, 4.0, PUTIH, 8)
        x, y, panjang = lebar // 2 - 100, 140, 200
        cv2.rectangle(frame, (x, y), (x + panjang, y + 14), PUTIH, 1)
        isi = int(panjang * min(1.0, keyakinan))
        if isi:
            cv2.rectangle(frame, (x, y), (x + isi, y + 14), HIJAU, -1)
        cv2.putText(frame, f"{keyakinan:.2f}", (x + panjang + 12, y + 13),
                    HURUF_FONT, 0.5, PUTIH, 1)

    # Batang kecil yang menunjukkan huruf ini sudah menang berapa kali.
    # Dibatasi 10 karena jendela voting lebih panjang dari syarat konfirmasi,
    # jadi kemajuannya bisa lewat dari 100 persen dan batangnya meluber.
    if huruf_voting:
        isi = min(10, int(round(kemajuan_voting * 10)))
        batang = "#" * isi
        cv2.putText(frame, f"voting {huruf_voting} [{batang:<10}]", (20, 65),
                    HURUF_FONT, 0.6, KUNING, 1)

    # Tiga tebakan teratas di pojok kanan. Kalau huruf yang Anda maksud muncul
    # di urutan kedua atau ketiga, berarti model mengenalinya tapi kalah saing.
    for baris, (h, nilai) in enumerate(tiga_besar):
        warna = PUTIH if baris == 0 else ABU
        cv2.putText(frame, f"{h} {nilai:.2f}", (lebar - 110, 35 + baris * 24),
                    HURUF_FONT, 0.6, warna, 2 if baris == 0 else 1)

    # Kotak gelap di bawah untuk teks yang sudah terkumpul.
    cv2.rectangle(frame, (0, tinggi - 96), (lebar, tinggi), GELAP, -1)
    tampil = teks if len(teks) < 40 else "..." + teks[-37:]
    cv2.putText(frame, f"TEKS: {tampil}", (20, tinggi - 60), HURUF_FONT, 0.9, PUTIH, 2)

    if sedang_bicara:
        cv2.putText(frame, "SEDANG BICARA...", (20, tinggi - 26),
                    HURUF_FONT, 0.7, HIJAU, 2)
    elif auto_ucap and teks.strip():
        # Batang hitung mundur menuju saat kata diucapkan.
        maju = 1.0 - hitung_mundur / ucap_setelah
        cv2.putText(frame, "diucapkan dalam", (20, tinggi - 26), HURUF_FONT, 0.55, PUTIH, 1)
        x, panjang = 190, 160
        cv2.rectangle(frame, (x, tinggi - 36), (x + panjang, tinggi - 24), PUTIH, 1)
        isi = int(panjang * max(0.0, min(1.0, maju)))
        if isi:
            cv2.rectangle(frame, (x, tinggi - 36), (x + isi, tinggi - 24), KUNING, -1)
    elif not auto_ucap:
        cv2.putText(frame, "ucap otomatis MATI (tekan P)", (20, tinggi - 26),
                    HURUF_FONT, 0.55, ABU, 1)

    bantuan = "Q keluar  BKSP hapus  C bersih  SPASI _  ENTER ucap  P otomatis"
    cv2.putText(frame, bantuan, (lebar - 600, tinggi - 8), HURUF_FONT, 0.45, ABU, 1)


# ------------------------------------------------------------------ program utama
def utama() -> int:
    arg = baca_argumen()

    catat = Pencatat(aktif=not arg.tanpa_catatan, detail=arg.catat_detail)

    model = joblib.load(PATH_MODEL)
    catat.mulai(PATH_MODEL, model.n_features_in_, "".join(model.classes_),
                arg.bahasa, arg.kamera)

    ambang_huruf = {} if arg.ambang_seragam else muat_ambang_huruf(catat)

    pengucap = Pengucap(bahasa=arg.bahasa, pelan=arg.pelan, catat=catat)
    voting = VotingHuruf(arg.jendela_voting, arg.frame_konfirmasi,
                         arg.min_keyakinan, arg.jeda, catat=catat,
                         ambang_huruf=ambang_huruf)

    # Mode pengukuran satu huruf.
    pengukur = None
    if arg.ukur:
        target = arg.ukur.strip().upper()
        if target not in model.classes_:
            print(f"GAGAL: '{target}' bukan huruf yang dikenali model.")
            return 1
        pengukur = PengukurHuruf(target, voting.ambang_untuk(target))
        catat.kabar("UKUR", f"mode pengukuran huruf {target}. Peragakan huruf itu "
                            f"terus-menerus, lalu tekan Q untuk melihat hasilnya.")
        catat.kabar("UKUR", f"ambang {target} yang berlaku sekarang: "
                            f"{pengukur.ambang_berlaku:.2f}")

    # CAP_DSHOW dipakai supaya kamera di Windows kebuka lebih cepat.
    kamera = cv2.VideoCapture(arg.kamera, cv2.CAP_DSHOW)
    if not kamera.isOpened():
        print(f"GAGAL: kamera {arg.kamera} tidak bisa dibuka. "
              f"Coba --kamera 1, atau tutup aplikasi lain yang sedang memakai kamera.")
        return 1

    mp_tangan = mp.solutions.hands
    mp_gambar = mp.solutions.drawing_utils
    # Objek ini dibuat sekali saja di luar loop. Kalau dibuat tiap frame, berat sekali.
    pendeteksi = mp_tangan.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=arg.keyakinan_deteksi,
        min_tracking_confidence=arg.keyakinan_tracking,
    )

    teks = ""
    waktu_huruf_terakhir = float("-inf")
    sejak_tangan_hilang: float | None = None
    auto_ucap = not arg.tanpa_auto_ucap
    jumlah_frame = 0
    waktu_mulai = time.monotonic()

    try:
        while True:
            berhasil, frame = kamera.read()
            if not berhasil:
                catat.kabar("GAGAL", "kamera berhenti mengirim gambar")
                print("GAGAL: kamera berhenti mengirim gambar.")
                return 1
            jumlah_frame += 1

            # Dibalik supaya terasa seperti bercermin, lebih enak dipakai.
            frame = cv2.flip(frame, 1)
            sekarang = time.monotonic()

            hasil = pendeteksi.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            angka, jumlah_tangan = vektor_titik_tangan(hasil)

            mode = "diam"
            tebakan, keyakinan = "", 0.0
            tiga_besar = []

            if angka is not None:
                if sejak_tangan_hilang is not None:
                    catat.tangan(jumlah_tangan)
                sejak_tangan_hilang = None
                mode = "baca"

                # Titik mentah diolah dulu jadi 162 angka lewat fitur.py,
                # persis seperti waktu modelnya dilatih.
                peluang = model.predict_proba(ambil_fitur(angka))[0]
                indeks_tertinggi = int(np.argmax(peluang))
                tebakan = model.classes_[indeks_tertinggi]
                keyakinan = float(peluang[indeks_tertinggi])

                # Tiga tebakan teratas. Ini yang memperlihatkan huruf apa yang
                # sedang bersaing, misalnya I yang selalu kalah oleh J.
                urut = np.argsort(peluang)[::-1][:3]
                tiga_besar = [(model.classes_[i], float(peluang[i])) for i in urut]
                catat.tebakan(tebakan, keyakinan, tiga_besar[1:])

                if pengukur is not None:
                    # Semua frame direkam apa adanya, termasuk yang salah, supaya
                    # laporannya menggambarkan keadaan sebenarnya.
                    i_target = list(model.classes_).index(pengukur.target)
                    pengukur.catat_frame(tebakan, keyakinan,
                                         float(peluang[i_target]), tiga_besar)

                huruf_diterima = voting.masukkan(tebakan, keyakinan, sekarang)
                if huruf_diterima:
                    teks += huruf_diterima
                    waktu_huruf_terakhir = sekarang
                    catat.huruf_diterima(huruf_diterima, teks)
                    if not arg.tanpa_eja_huruf:
                        pengucap.ucapkan(huruf_diterima)
            else:
                # Tangan hilang dari layar: voting direset, dan kalau lama
                # menghilang berarti satu kata sudah selesai, jadi diberi spasi.
                voting.kosongkan()
                if sejak_tangan_hilang is None:
                    sejak_tangan_hilang = sekarang
                    catat.rinci("TANGAN", "tangan hilang dari layar")
                elif (sekarang - sejak_tangan_hilang >= arg.spasi_setelah
                      and teks and not teks.endswith(" ")):
                    teks += " "
                    catat.teks_berubah("spasi ditambah otomatis", teks)

            hitung_mundur = max(0.0, arg.ucap_setelah - (sekarang - waktu_huruf_terakhir))
            sudah_waktunya = (
                auto_ucap
                and hitung_mundur == 0.0
                and teks.strip()
                and waktu_huruf_terakhir != float("-inf")
                and not pengucap.sibuk
            )
            if sudah_waktunya:
                pengucap.ucapkan(teks)
                teks = ""
                waktu_huruf_terakhir = float("-inf")

            huruf_unggul, kemajuan = voting.kemajuan
            gambar_tampilan(
                frame, mode=mode, tebakan=tebakan, keyakinan=keyakinan,
                huruf_voting=huruf_unggul, kemajuan_voting=kemajuan, teks=teks,
                sedang_bicara=pengucap.sibuk, hitung_mundur=hitung_mundur,
                ucap_setelah=arg.ucap_setelah, auto_ucap=auto_ucap,
                titik_tangan=hasil.multi_hand_landmarks or [],
                mp_gambar=mp_gambar, mp_tangan=mp_tangan,
                tiga_besar=tiga_besar,
            )

            if pengukur is not None:
                # Angka berjalan supaya Anda tahu kapan datanya sudah cukup.
                cv2.putText(frame, pengukur.ringkas_singkat(), (20, 95),
                            HURUF_FONT, 0.6, KUNING, 2)

            cv2.imshow("BISINDO ke Suara   (tekan Q untuk keluar)", frame)

            tombol = cv2.waitKey(1) & 0xFF
            if tombol in (ord("q"), 27):            # Q atau ESC
                break
            elif tombol in (8, 127):                # Backspace
                teks = teks[:-1]
                catat.teks_berubah("huruf terakhir dihapus", teks)
            elif tombol == ord("c"):                # C
                teks = ""
                catat.teks_berubah("teks dikosongkan", teks)
            elif tombol == ord(" "):                # Spasi
                if teks and not teks.endswith(" "):
                    teks += " "
                    catat.teks_berubah("spasi ditambah manual", teks)
            elif tombol in (13, 10):                # Enter
                pengucap.ucapkan(teks)
                teks = ""
                waktu_huruf_terakhir = float("-inf")
            elif tombol == ord("p"):                # P
                auto_ucap = not auto_ucap
                catat.kabar("ATUR", f"ucap otomatis {'menyala' if auto_ucap else 'mati'}")
    finally:
        # Bagian ini selalu dijalankan, walaupun program berhenti karena error.
        catat.selesai(teks, jumlah_frame, time.monotonic() - waktu_mulai)
        pengucap.tutup()
        pendeteksi.close()
        kamera.release()
        cv2.destroyAllWindows()

    if pengukur is not None:
        print(pengukur.laporan())
        rincian = pengukur.simpan_rincian()
        if rincian:
            print(f"\nRincian tiap frame tersimpan di log/{rincian.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(utama())
