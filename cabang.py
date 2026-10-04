"""Model dua cabang: satu cabang untuk huruf satu tangan, satu untuk dua tangan.

ATURANNYA
---------
Banyak huruf di dataset diperagakan dengan satu atau dua tangan. Model tunggal
harus menebak jumlah tangan dan hurufnya sekaligus, dan itu membuat huruf yang
mirip (A dengan S, misalnya) saling tertukar. Di sini jumlah tangan dijadikan
aturan tetap, bukan sesuatu yang ditebak:

    terlihat satu tangan   ->  cabang satu tangan, hanya bisa menjawab
                               C E I J L O R U V Z
    terlihat dua tangan    ->  cabang dua tangan, hanya bisa menjawab
                               huruf sisanya (A B D F G H K M N P Q S T W X Y)

Jumlah tangan dibaca dari fitur, tanpa masukan tambahan. Kolom 165 (jarak antar
pergelangan, dari fitur.py) hanya bernilai di atas nol kalau KEDUA tangan
terlihat, jadi cukup untuk menentukan cabang.

Kelas ini meniru bagian model scikit-learn yang dipakai program
(`classes_`, `n_features_in_`, `predict_proba`), jadi gesture_voice.py,
uji_gambar.py, dan putar_ulang.py bisa memakainya tanpa diubah.

Berkas ini harus ada di folder yang sama saat model dimuat, karena joblib
mencatat nama kelasnya.
"""

from __future__ import annotations

import numpy as np

HURUF_SATU_TANGAN = list("CEIJLORUVZ")
HURUF_DUA_TANGAN = list("ABDFGHKMNPQSTWXY")
KOLOM_JARAK_PERGELANGAN = 165


def dua_tangan_terlihat(fitur: np.ndarray) -> np.ndarray:
    """Benar untuk baris yang kedua tangannya terlihat."""
    return fitur[:, KOLOM_JARAK_PERGELANGAN] > 0


class ModelDuaCabang:
    def __init__(self, model_satu, model_dua, jumlah_fitur: int):
        self.model_satu = model_satu
        self.model_dua = model_dua
        self.n_features_in_ = jumlah_fitur
        self.classes_ = np.array(sorted(set(model_satu.classes_) | set(model_dua.classes_)))

    def _isi(self, model, fitur: np.ndarray, hasil: np.ndarray, baris: np.ndarray) -> None:
        if not baris.any():
            return
        peluang = model.predict_proba(fitur[baris])
        kolom = [int(np.searchsorted(self.classes_, k)) for k in model.classes_]
        sub = np.zeros((int(baris.sum()), len(self.classes_)))
        sub[:, kolom] = peluang
        hasil[baris] = sub

    def predict_proba(self, fitur) -> np.ndarray:
        fitur = np.asarray(fitur, dtype=np.float32)
        hasil = np.zeros((len(fitur), len(self.classes_)))
        dua = dua_tangan_terlihat(fitur)
        self._isi(self.model_satu, fitur, hasil, ~dua)
        self._isi(self.model_dua, fitur, hasil, dua)
        return hasil

    def predict(self, fitur) -> np.ndarray:
        return self.classes_[self.predict_proba(fitur).argmax(axis=1)]
