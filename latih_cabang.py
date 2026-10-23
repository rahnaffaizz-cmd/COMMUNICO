"""Melatih model dua cabang (v7) dan membandingkannya dengan model satu cabang (v6).

    python latih_cabang.py                 ukur, lalu simpan rf_bisindo_v7.pkl
    python latih_cabang.py --keluaran x.pkl

Aturan cabangnya ada di cabang.py. Data tiap cabang dipilih begini:
  cabang satu tangan : sampel huruf C E I J L O R U V Z yang satu tangan
  cabang dua tangan  : sampel huruf sisanya yang DUA tangan terlihat
Sampel dua tangan dari huruf satu tangan, atau satu tangan dari huruf dua
tangan, tidak dipakai melatih, karena aturannya memang tidak membolehkan huruf
itu keluar dari cabang tersebut.

Pengujian pakai pembagian per foto asli (lihat evaluasi.py), jadi angkanya
jujur. Rekaman sendiri dari rekam.py (data/data_pengguna.csv) ikut dipakai.

Tiap sampel juga diberi salinan tiruan tangan anak (tangan_anak.py), bawaannya
1 salinan. Salinan hanya dipakai melatih, tidak pernah ikut diuji, dan selalu
berada di lipatan yang sama dengan sampel asalnya.
    python latih_cabang.py --anak 0        tanpa salinan anak (seperti sebelumnya)
    python latih_cabang.py --anak 2        2 salinan per sampel
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from cabang import (HURUF_DUA_TANGAN, HURUF_SATU_TANGAN, ModelDuaCabang,  # noqa: E402
                    dua_tangan_terlihat)
from evaluasi import kelompokkan_per_foto, siapkan_data  # noqa: E402
from fitur import ambil_fitur  # noqa: E402
from tangan_anak import tiru_tangan_anak  # noqa: E402

FOLDER = Path(__file__).resolve().parent
BERKAS_PENGGUNA = FOLDER / "data" / "data_pengguna.csv"
KELUARAN_BAWAAN = FOLDER / "rf_bisindo_v7.pkl"


def judul(teks: str) -> None:
    print(f"\n{'=' * 62}\n{teks}\n{'=' * 62}")


def buat_model():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)


def latih_dua_cabang(X, y, bobot):
    """Latih kedua cabang dari fitur lengkap X. Mengembalikan ModelDuaCabang."""
    dua = dua_tangan_terlihat(X)
    satu_huruf = np.isin(y, HURUF_SATU_TANGAN)

    ada_satu = ~dua & satu_huruf
    ada_dua = dua & ~satu_huruf
    model_satu = buat_model().fit(X[ada_satu], y[ada_satu], sample_weight=bobot[ada_satu])
    model_dua = buat_model().fit(X[ada_dua], y[ada_dua], sample_weight=bobot[ada_dua])
    return ModelDuaCabang(model_satu, model_dua, X.shape[1])


def muat_data():
    X_mentah, y = siapkan_data()
    grup = np.asarray(kelompokkan_per_foto(X_mentah, y), dtype=object)
    bobot = np.ones(len(y))

    if BERKAS_PENGGUNA.exists():
        import pandas as pd

        p = pd.read_csv(BERKAS_PENGGUNA)
        X_p = np.nan_to_num(p[[f"k{i}" for i in range(126)]].values)
        y_p = p["label"].values
        # Satu sesi rekaman dipotong jadi 5 bagian menurut urutan waktu. Tanpa ini,
        # satu sesi = satu kelompok utuh yang tidak pernah dilihat saat latihan,
        # dan ketepatannya jadi terlalu pesimis. Dengan dipotong, kebocoran antar
        # frame berdekatan membuat angka huruf milik Anda sedikit terlalu optimis.
        urutan = p.groupby("sesi").cumcount() * 5 // p.groupby("sesi")["sesi"].transform("size")
        grup_p = np.array([f"{h}_pengguna_{s}_{u}" for h, s, u in
                           zip(p["label"], p["sesi"], urutan)], dtype=object)
        X_mentah = np.vstack([X_mentah, X_p])
        y = np.concatenate([y, y_p])
        grup = np.concatenate([grup, grup_p])
        bobot = np.concatenate([bobot, np.full(len(y_p), 3.0)])
        print(f"Data pengguna ikut dipakai: {len(y_p)} frame, bobot 3")
    return X_mentah, y, grup, bobot


def ukur(X, y, grup, bobot):
    """Uji dua cabang dan satu cabang pada pembagian per foto yang SAMA."""
    from sklearn.model_selection import GroupKFold

    tebak_cabang = np.empty(len(y), dtype=object)
    tebak_satu = np.empty(len(y), dtype=object)
    yakin_cabang = np.zeros(len(y))
    for nomor, (latih, uji) in enumerate(GroupKFold(5).split(X, y, groups=grup), 1):
        cabang = latih_dua_cabang(X[latih], y[latih], bobot[latih])
        peluang = cabang.predict_proba(X[uji])
        tebak_cabang[uji] = cabang.classes_[peluang.argmax(axis=1)]
        yakin_cabang[uji] = peluang.max(axis=1)

        satu = buat_model().fit(X[latih], y[latih], sample_weight=bobot[latih])
        tebak_satu[uji] = satu.predict(X[uji])
        print(f"    lipatan {nomor}: dua cabang {np.mean(tebak_cabang[uji] == y[uji]) * 100:6.2f}%"
              f"   satu cabang {np.mean(tebak_satu[uji] == y[uji]) * 100:6.2f}%")
    return tebak_cabang, tebak_satu, yakin_cabang


# Salinan anak dibobot setengah dari sampel asalnya, supaya data asli tetap
# lebih menentukan daripada tiruan.
BOBOT_ANAK = 0.5


def buat_salinan_anak(X_mentah, bobot, jumlah: int, benih: int):
    """`jumlah` salinan tiruan anak per sampel. Mengembalikan fitur, indeks asal, bobot."""
    rng = np.random.default_rng(benih)
    asal = np.tile(np.arange(len(X_mentah)), jumlah)
    X_anak = ambil_fitur(tiru_tangan_anak(X_mentah[asal], rng), urut_tangan=True)
    return X_anak, asal, bobot[asal] * BOBOT_ANAK


def ukur_anak(X, y, grup, bobot, X_anak, asal, bobot_anak, X_uji_anak):
    """Bandingkan dua cabang tanpa dan dengan salinan anak, pada lipatan yang SAMA.

    Diuji pada sampel asli dan pada tiruan anak dari sampel uji (benih lain dari
    salinan latih, jadi bukan salinan yang sama).
    """
    from sklearn.model_selection import GroupKFold

    n = len(y)
    tebak = {k: np.empty(n, dtype=object) for k in ("polos", "anak", "polos_tiru", "anak_tiru")}
    yakin_anak = np.zeros(n)
    for nomor, (latih, uji) in enumerate(GroupKFold(5).split(X, y, groups=grup), 1):
        polos = latih_dua_cabang(X[latih], y[latih], bobot[latih])
        ikut = np.isin(asal, latih)
        anak = latih_dua_cabang(np.vstack([X[latih], X_anak[ikut]]),
                                np.concatenate([y[latih], y[asal[ikut]]]),
                                np.concatenate([bobot[latih], bobot_anak[ikut]]))
        for nama, model in (("polos", polos), ("anak", anak)):
            for akhiran, data in (("", X[uji]), ("_tiru", X_uji_anak[uji])):
                peluang = model.predict_proba(data)
                tebak[nama + akhiran][uji] = model.classes_[peluang.argmax(axis=1)]
                if nama == "anak" and not akhiran:
                    yakin_anak[uji] = peluang.max(axis=1)
        b = {k: np.mean(v[uji] == y[uji]) * 100 for k, v in tebak.items()}
        print(f"    lipatan {nomor}: asli {b['polos']:6.2f}% -> {b['anak']:6.2f}%   "
              f"tiruan anak {b['polos_tiru']:6.2f}% -> {b['anak_tiru']:6.2f}%")
    return tebak, yakin_anak


def laporkan_anak(X, y, tebak) -> None:
    dua = dua_tangan_terlihat(X)
    sah = ~((dua & np.isin(y, HURUF_SATU_TANGAN)) | (~dua & ~np.isin(y, HURUF_SATU_TANGAN)))

    judul("HASIL SALINAN ANAK (pembagian per foto asli, sampel sesuai aturan)")
    print("                         tanpa salinan   dengan salinan")
    for label, akhiran in (("sampel asli", ""), ("tiruan tangan anak", "_tiru")):
        a = np.mean((tebak["polos" + akhiran] == y)[sah]) * 100
        b = np.mean((tebak["anak" + akhiran] == y)[sah]) * 100
        print(f"  {label:<22} {a:10.2f}%   {b:10.2f}%   {b - a:+5.1f}")

    print("\n  Tiap huruf pada tiruan anak (tanpa -> dengan), lalu sampel asli:")
    for huruf in sorted(np.unique(y)):
        p = (y == huruf) & sah
        if not p.any():
            continue
        ta, tb = (np.mean(tebak[k][p] == huruf) * 100 for k in ("polos_tiru", "anak_tiru"))
        aa, ab = (np.mean(tebak[k][p] == huruf) * 100 for k in ("polos", "anak"))
        print(f"    {huruf}: tiruan {ta:6.2f}% -> {tb:6.2f}% {tb - ta:+6.1f}   "
              f"asli {aa:6.2f}% -> {ab:6.2f}% {ab - aa:+6.1f}")

    # Uji coba anak-anak memperlihatkan J, Z, U sering muncul padahal tidak
    # diperagakan. Ini seberapa sering huruf lain terbaca sebagai huruf itu.
    print("\n  Huruf LAIN pada tiruan anak yang salah terbaca sebagai:")
    for huruf in "JZUE":
        lain = (y != huruf) & sah
        a, b = (np.mean(tebak[k][lain] == huruf) * 100 for k in ("polos_tiru", "anak_tiru"))
        print(f"    {huruf}: {a:5.2f}% -> {b:5.2f}%")


def laporkan(X, y, tebak_cabang, tebak_satu) -> None:
    dua = dua_tangan_terlihat(X)
    satu_huruf = np.isin(y, HURUF_SATU_TANGAN)
    # Sampel yang TIDAK mungkin benar oleh aturan: huruf dua tangan yang di
    # sampelnya cuma satu tangan terlihat, atau sebaliknya.
    mustahil = (dua & satu_huruf) | (~dua & ~satu_huruf)

    judul("HASIL (pembagian per foto asli)")
    for nama, tebak in (("satu cabang (v6)", tebak_satu), ("dua cabang (v7)", tebak_cabang)):
        print(f"  {nama}: semua sampel {np.mean(tebak == y) * 100:6.2f}%   "
              f"hanya sampel yang sesuai aturan {np.mean((tebak == y)[~mustahil]) * 100:6.2f}%")
    print(f"\n  Sampel yang tidak mungkin benar menurut aturan: "
          f"{mustahil.sum()} dari {len(y)} ({mustahil.mean() * 100:.1f}%)")
    print("  (huruf dua tangan yang dataset-nya merekam dengan satu tangan, "
          "atau sebaliknya)")

    print("\n  Ketepatan tiap huruf, satu cabang -> dua cabang:")
    for huruf in sorted(np.unique(y)):
        p = y == huruf
        a, b = np.mean(tebak_satu[p] == huruf), np.mean(tebak_cabang[p] == huruf)
        cabang = "1 tangan" if huruf in HURUF_SATU_TANGAN else "2 tangan"
        pm = mustahil[p].mean()
        print(f"    {huruf} ({cabang}): {a * 100:6.2f}% -> {b * 100:6.2f}%  "
              f"{(b - a) * 100:+6.1f}   tidak mungkin benar: {pm * 100:3.0f}%")

    judul("A DAN S")
    for dari, ke in (("A", "S"), ("S", "A")):
        p = y == dari
        print(f"  {dari} terbaca {ke}:  satu cabang {np.mean(tebak_satu[p] == ke) * 100:5.1f}%   "
              f"dua cabang {np.mean(tebak_cabang[p] == ke) * 100:5.1f}%")


def tulis_ambang(y, tebak, yakin, sah, berkas_model: Path, target: float = 0.80) -> None:
    """Hitung ambang tiap huruf dari tebakan yang belum pernah dilihat model.

    Cara yang sama dengan diagnosa.py: ambang terendah yang membuat tebakan
    huruf itu benar minimal `target`. Hasilnya ke ambang_<nama model>.json,
    yang dipakai gesture_voice.py hanya untuk model ini.
    """
    import json

    from diagnosa import AMBANG_MENYERAH, cari_ambang

    # Hanya sampel yang sesuai aturan satu/dua tangan yang dinilai. Sampel yang
    # tidak mungkin benar (huruf dua tangan yang direkam dengan satu tangan di
    # dataset) akan salah diterbaca sebagai huruf satu tangan dan merusak
    # presisi C, E, O, V, padahal saat dipakai Anda mengikuti aturannya.
    y, tebak, yakin = y[sah], tebak[sah], yakin[sah]

    ambang = {}
    for huruf in sorted(np.unique(y)):
        ditebak = tebak == huruf
        if not ditebak.any():
            ambang[huruf] = AMBANG_MENYERAH
            continue
        nilai, _, _ = cari_ambang(y[ditebak] == huruf, yakin[ditebak], target)
        ambang[huruf] = nilai

    berkas = berkas_model.with_name(f"ambang_{berkas_model.stem}.json")

    # Ambang yang disetel manual dari uji kamera (punya alasan di
    # catatan_penyetelan) tidak ditimpa angka dataset. Dataset sudah beberapa
    # kali terbukti keliru, misalnya I yang selalu jadi J di dataset padahal
    # terbaca baik di kamera pengguna.
    catatan = {}
    if berkas.exists():
        lama = json.loads(berkas.read_text(encoding="utf-8"))
        catatan = lama.get("catatan_penyetelan", {})
        for huruf in catatan:
            if huruf in lama.get("ambang", {}):
                print(f"  Ambang {huruf} dipertahankan {lama['ambang'][huruf]} "
                      f"(disetel manual, hitungan dataset {ambang.get(huruf)})")
                ambang[huruf] = lama["ambang"][huruf]

    isi = {
        "target_presisi": target * 100,
        "ambang": ambang,
        "keterangan": (f"Ambang untuk {berkas_model.name}, dihitung latih_cabang.py "
                       f"dari tebakan per foto asli. {AMBANG_MENYERAH} berarti ditolak."),
    }
    if catatan:
        isi["catatan_penyetelan"] = catatan
    berkas.write_text(json.dumps(isi, indent=2, ensure_ascii=False), encoding="utf-8")
    ditolak = " ".join(h for h, v in ambang.items() if v >= AMBANG_MENYERAH) or "-"
    print(f"  Ambang ditulis ke {berkas.name}. Huruf yang ditolak: {ditolak}")


def utama() -> int:
    p = argparse.ArgumentParser(description="Melatih model dua cabang (v7)")
    p.add_argument("--keluaran", type=Path, default=KELUARAN_BAWAAN)
    p.add_argument("--anak", type=int, default=1, metavar="N",
                   help="salinan tiruan tangan anak per sampel (0 = tanpa)")
    arg = p.parse_args()

    import joblib

    X_mentah, y, grup, bobot = muat_data()
    print(f"Data: {len(y)} sampel, {len(np.unique(y))} huruf")
    X = ambil_fitur(X_mentah, urut_tangan=True)

    if arg.anak > 0:
        X_anak, asal, bobot_anak = buat_salinan_anak(X_mentah, bobot, arg.anak, benih=1)
        X_uji_anak, _, _ = buat_salinan_anak(X_mentah, bobot, 1, benih=2)
        print(f"Salinan tiruan anak: {len(X_anak)} sampel, bobot {BOBOT_ANAK} x asalnya")

        judul("MENGUKUR (5 lipatan per foto asli, tanpa vs dengan salinan anak)")
        tebak, yakin_cabang = ukur_anak(X, y, grup, bobot, X_anak, asal, bobot_anak, X_uji_anak)
        laporkan_anak(X, y, tebak)
        tebak_cabang = tebak["anak"]

        judul("MELATIH MODEL AKHIR")
        model = latih_dua_cabang(np.vstack([X, X_anak]), np.concatenate([y, y[asal]]),
                                 np.concatenate([bobot, bobot_anak]))
    else:
        judul("MENGUKUR (5 lipatan per foto asli)")
        tebak_cabang, tebak_satu, yakin_cabang = ukur(X, y, grup, bobot)
        laporkan(X, y, tebak_cabang, tebak_satu)

        judul("MELATIH MODEL AKHIR")
        model = latih_dua_cabang(X, y, bobot)
    joblib.dump(model, arg.keluaran, compress=3)
    print(f"  Disimpan: {arg.keluaran.name} ({arg.keluaran.stat().st_size / 1048576:.1f} MB)")
    dua = dua_tangan_terlihat(X)
    sah = ~((dua & np.isin(y, HURUF_SATU_TANGAN)) | (~dua & ~np.isin(y, HURUF_SATU_TANGAN)))
    tulis_ambang(y, tebak_cabang, yakin_cabang, sah, arg.keluaran)
    print(f"  Coba:  .\\run.ps1 --model {arg.keluaran.name}")
    return 0


if __name__ == "__main__":
    sys.exit(utama())
