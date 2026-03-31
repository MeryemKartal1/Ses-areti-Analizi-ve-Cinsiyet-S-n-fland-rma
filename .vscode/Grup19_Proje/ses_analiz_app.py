"""
=============================================================
  Grup 19 – Ses Analizi ve Cinsiyet Sınıflandırma
  Dönemiçi Proje  |  2025-2026 Bahar
=============================================================
Kurulum:
    pip install streamlit librosa numpy scipy pandas matplotlib openpyxl

Çalıştırma:
    streamlit run ses_analiz_app.py

Klasör yapısı (kodun yanında olmalı):
    proje_1/
        Midterm_Dataset_2026/
            GRUP_01/   (veya Grup_01)
                G01_D01_C_11_Angry_C3.wav ...
            GRUP_02/ ...
            GRUP_19/ ...
    birlesik_metadata.xlsx
=============================================================
"""

import os
import sys
import re
import glob
import warnings
import tempfile
warnings.filterwarnings("ignore")

# ── Kodun kendi bulunduğu klasörü baz al ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
import streamlit as st

# ─────────────────────────────────────────────
#  SAYFA AYARLARI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ses Analizi – Grup 19",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  SABİTLER
# ─────────────────────────────────────────────
FRAME_MS      = 25
HOP_MS        = 10
ENERGY_THRESH = 0.02
ZCR_THRESH    = 0.15

F0_COCUK_SINIR = 250
F0_KADIN_SINIR = 165

CINSIYET_RENK = {"E": "#4A90D9", "K": "#E57373", "C": "#81C784"}
CINSIYET_AD   = {"E": "Erkek",   "K": "Kadin",   "C": "Cocuk"}

# ─────────────────────────────────────────────
#  CİNSİYET NORMALIZE
#  Excel'de E, K, C, M, F, k, 'C ' gibi farklı değerler var
# ─────────────────────────────────────────────
def normalize_cinsiyet(ham) -> str:
    if ham is None:
        return "?"
    s = str(ham).strip().upper()
    if s in ("E", "M", "MALE", "ERKEK"):
        return "E"
    if s in ("K", "F", "FEMALE", "KADIN"):
        return "K"
    if s in ("C", "CH", "CHILD", "COCUK", "COCUK"):
        return "C"
    return "?"


# ─────────────────────────────────────────────
#  DOSYA YOLU BULMA
#  Dosya adından grup numarası çıkarılır, alt klasörde aranır
# ─────────────────────────────────────────────
def dosya_yolu_bul(dosya_adi: str, dataset_koku: str):
    dosya_adi = dosya_adi.strip()

    # 1. Grup numarasından klasör türet: G19_ -> GRUP_19
    m = re.match(r'[Gg](\d+)_', dosya_adi)
    if m:
        grup_no = int(m.group(1))
        for variant in [
            f"GRUP_{grup_no:02d}", f"Grup_{grup_no:02d}",
            f"grup_{grup_no:02d}", f"GRUP{grup_no:02d}",
            f"Grup{grup_no:02d}",  f"G{grup_no:02d}",
        ]:
            yol = os.path.join(dataset_koku, variant, dosya_adi)
            if os.path.exists(yol):
                return yol

    # 2. Dataset kökünde doğrudan
    yol = os.path.join(dataset_koku, dosya_adi)
    if os.path.exists(yol):
        return yol

    # 3. Tüm alt klasörleri tara (yavaş ama garantili)
    for root, dirs, files in os.walk(dataset_koku):
        if dosya_adi in files:
            return os.path.join(root, dosya_adi)

    return None


# ─────────────────────────────────────────────
#  SES ANALİZ FONKSİYONLARI
# ─────────────────────────────────────────────
def yukle_ses(dosya_yolu: str, sr_hedef: int = 22050):
    y, sr = librosa.load(dosya_yolu, sr=sr_hedef, mono=True)
    return y, sr


def hesapla_enerji_zcr(y, sr):
    frame_len   = int(FRAME_MS * sr / 1000)
    hop_len     = int(HOP_MS   * sr / 1000)
    zcr         = librosa.feature.zero_crossing_rate(
                      y, frame_length=frame_len, hop_length=hop_len)[0]
    rms         = librosa.feature.rms(
                      y=y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_norm    = rms / (rms.max() + 1e-9)
    frame_times = librosa.frames_to_time(
                      np.arange(len(rms)), sr=sr, hop_length=hop_len)
    return rms_norm, zcr, frame_times


def sesli_bolge_maskesi(energy, zcr):
    return (energy > ENERGY_THRESH) & (zcr < ZCR_THRESH)


def otokorelasyon_f0(cerceve, sr, f0_min=50, f0_max=500):
    n       = len(cerceve)
    cerceve = cerceve - cerceve.mean()
    r       = np.correlate(cerceve, cerceve, mode='full')
    r       = r[n - 1:]
    r       = r / (r[0] + 1e-9)
    lag_min = int(sr / f0_max)
    lag_max = min(int(sr / f0_min), len(r) - 1)
    if lag_min >= lag_max:
        return None
    r_aralik  = r[lag_min:lag_max]
    tepe_idx, _ = find_peaks(r_aralik, height=0.3)
    if len(tepe_idx) == 0:
        return None
    en_iyi_lag = tepe_idx[np.argmax(r_aralik[tepe_idx])] + lag_min
    return float(sr / en_iyi_lag)


def analiz_et(dosya_yolu: str):
    y, sr     = yukle_ses(dosya_yolu)
    energy, zcr, frame_times = hesapla_enerji_zcr(y, sr)
    voiced    = sesli_bolge_maskesi(energy, zcr)
    frame_len = int(FRAME_MS * sr / 1000)
    hop_len   = int(HOP_MS   * sr / 1000)

    f0_listesi, f0_zaman = [], []
    for i, is_voiced in enumerate(voiced):
        if not is_voiced:
            continue
        bas = i * hop_len
        bit = bas + frame_len
        if bit > len(y):
            break
        f0 = otokorelasyon_f0(y[bas:bit], sr)
        if f0 is not None:
            f0_listesi.append(f0)
            f0_zaman.append(frame_times[i])

    f0_dizi     = np.array(f0_listesi)
    f0_ort      = float(np.mean(f0_dizi))        if len(f0_dizi) > 0 else 0.0
    f0_std      = float(np.std(f0_dizi))         if len(f0_dizi) > 0 else 0.0
    zcr_ort     = float(np.mean(zcr[voiced]))    if voiced.sum() > 0 else 0.0
    eng_ort     = float(np.mean(energy[voiced])) if voiced.sum() > 0 else 0.0
    voiced_oran = voiced.sum() / len(voiced)     if len(voiced) > 0 else 0.0

    if f0_ort >= F0_COCUK_SINIR:
        tahmin = "C"
    elif f0_ort >= F0_KADIN_SINIR:
        tahmin = "K"
    else:
        tahmin = "E"

    return {
        "f0_ort": f0_ort, "f0_std": f0_std,
        "zcr_ort": zcr_ort, "energy_ort": eng_ort,
        "voiced_oran": voiced_oran, "tahmin": tahmin,
        "y": y, "sr": sr,
        "energy": energy, "zcr": zcr,
        "frame_times": frame_times, "voiced_mask": voiced,
        "f0_dizisi": f0_dizi,
        "frame_f0_times": np.array(f0_zaman),
    }


# ─────────────────────────────────────────────
#  GRAFİK FONKSİYONLARI
# ─────────────────────────────────────────────
def ciz_analiz(sonuc: dict, dosya_adi: str, gercek=None):
    y, sr = sonuc["y"], sonuc["sr"]
    t     = np.linspace(0, len(y) / sr, num=len(y))

    fig = plt.figure(figsize=(16, 11), facecolor="#0E1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    tahmin = sonuc["tahmin"]
    renk   = CINSIYET_RENK[tahmin]
    baslik = f"{dosya_adi}  ->  Tahmin: {CINSIYET_AD[tahmin]}"
    if gercek and gercek in CINSIYET_AD:
        dogru   = "OK" if gercek == tahmin else "YANLIS"
        baslik += f"  |  Gercek: {CINSIYET_AD[gercek]}  [{dogru}]"
    fig.suptitle(baslik, color="white", fontsize=13, fontweight="bold", y=1.01)

    # 1. Dalga formu
    ax1.plot(t, y, color="#5a9fd4", lw=0.4, alpha=0.8)
    hop_px = int(HOP_MS * sr / 1000)
    for i, v in enumerate(sonuc["voiced_mask"]):
        if v:
            ts = i * hop_px / sr
            te = ts + FRAME_MS / 1000
            ax1.axvspan(ts, te, alpha=0.15, color="#81C784")
    ax1.set_title("Dalga Formu (yesil = sesli bolgeler)")
    ax1.set_xlabel("Zaman (s)"); ax1.set_ylabel("Genlik")

    # 2. STE
    ax2.plot(sonuc["frame_times"], sonuc["energy"], color="#FFD700", lw=1.5)
    ax2.axhline(ENERGY_THRESH, color="#FF6B6B", ls="--", lw=1,
                label=f"Esik={ENERGY_THRESH}")
    ax2.set_title("Kisa Sureli Enerji (STE)")
    ax2.set_xlabel("Zaman (s)"); ax2.set_ylabel("RMS (norm)")
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e")

    # 3. ZCR
    ax3.plot(sonuc["frame_times"], sonuc["zcr"], color="#FF9F43", lw=1.5)
    ax3.axhline(ZCR_THRESH, color="#FF6B6B", ls="--", lw=1,
                label=f"Esik={ZCR_THRESH}")
    ax3.set_title("Sifir Gecis Orani (ZCR)")
    ax3.set_xlabel("Zaman (s)"); ax3.set_ylabel("ZCR")
    ax3.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e")

    # 4. F0 Zaman serisi
    if len(sonuc["f0_dizisi"]) > 0:
        ax4.scatter(sonuc["frame_f0_times"], sonuc["f0_dizisi"],
                    color=renk, s=8, alpha=0.7, label="F0 (otokor.)")
        ax4.axhline(sonuc["f0_ort"], color="white", ls="--", lw=1.5,
                    label=f"Ort.={sonuc['f0_ort']:.1f} Hz")
        ax4.axhline(F0_COCUK_SINIR, color="#81C784", ls=":", lw=1,
                    label=f"Cocuk={F0_COCUK_SINIR} Hz")
        ax4.axhline(F0_KADIN_SINIR, color="#E57373", ls=":", lw=1,
                    label=f"Kadin={F0_KADIN_SINIR} Hz")
        ax4.set_title("F0 Zaman Serisi (Otokorelasyon)")
        ax4.set_xlabel("Zaman (s)"); ax4.set_ylabel("F0 (Hz)")
        ax4.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e")
    else:
        ax4.text(0.5, 0.5, "F0 tespit edilemedi", ha="center", va="center",
                 color="white", transform=ax4.transAxes)

    # 5. Otokorelasyon vs FFT
    frame_len_px = int(FRAME_MS * sr / 1000)
    hop_len_px   = int(HOP_MS   * sr / 1000)
    first_voiced = np.where(sonuc["voiced_mask"])[0]
    if len(first_voiced) > 0:
        idx     = first_voiced[0]
        bas     = idx * hop_len_px
        bit     = bas + frame_len_px
        cerceve = y[bas:bit] if bit <= len(y) else y[bas:]
        cerceve = cerceve - cerceve.mean()
        n       = len(cerceve)
        r       = np.correlate(cerceve, cerceve, mode='full')[n - 1:]
        r       = r / (r[0] + 1e-9)
        lag_arr = np.arange(len(r)) / sr * 1000
        lag_min = int(sr / 500)
        lag_max = min(int(sr / 50), len(r) - 1)
        ax5.plot(lag_arr[:lag_max], r[:lag_max], color="#5a9fd4",
                 lw=1.2, label="Otokorelasyon")
        freqs = np.fft.rfftfreq(n, d=1 / sr)
        mag   = np.abs(np.fft.rfft(cerceve))
        mask  = (freqs >= 50) & (freqs <= 500)
        if mask.any():
            fft_f0  = freqs[mask][np.argmax(mag[mask])]
            oto_lag = sr / (sonuc["f0_ort"] + 1e-9) / sr * 1000
            ax5.axvline(oto_lag, color="#FFD700", ls="--", lw=1.5,
                        label=f"F0 (otokor.)={sonuc['f0_ort']:.0f} Hz")
            ax5.axvline(1000 / fft_f0, color="#FF9F43", ls=":", lw=1.5,
                        label=f"F0 (FFT)={fft_f0:.0f} Hz")
        ax5.set_title("Otokorelasyon vs FFT (1. sesli cerceve)")
        ax5.set_xlabel("Gecikme (ms)"); ax5.set_ylabel("R(t) norm")
        ax5.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e")
    else:
        ax5.text(0.5, 0.5, "Sesli cerceve bulunamadi", ha="center",
                 va="center", color="white", transform=ax5.transAxes)
    return fig


def istatistik_tablosu(df: pd.DataFrame):
    satirlar = []
    for cin, ad in CINSIYET_AD.items():
        alt = df[df["Gercek"] == cin]
        if len(alt) == 0:
            continue
        satirlar.append({
            "Sinif":            ad,
            "Ornek Sayisi":     len(alt),
            "Ortalama F0 (Hz)": round(alt["F0_Ort"].mean(), 2),
            "Standart Sapma":   round(alt["F0_Ort"].std(),  2),
            "Basari (%)":       round(alt["Dogru_mu"].mean() * 100, 1),
        })
    return pd.DataFrame(satirlar)


def confusion_matrix_ciz(df: pd.DataFrame):
    siniflar  = ["E", "K", "C"]
    etiketler = ["Erkek", "Kadin", "Cocuk"]
    cm = np.zeros((3, 3), dtype=int)
    for i, g in enumerate(siniflar):
        for j, t_ in enumerate(siniflar):
            cm[i, j] = ((df["Gercek"] == g) & (df["Tahmin"] == t_)).sum()
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0E1117")
    ax.set_facecolor("#1a1a2e")
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(etiketler, color="white")
    ax.set_yticklabels(etiketler, color="white")
    ax.set_xlabel("Tahmin", color="white")
    ax.set_ylabel("Gercek", color="white")
    ax.set_title("Karisiklik Matrisi", color="white")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] < cm.max() / 2 else "#0E1117",
                    fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def f0_dagilim_grafigi(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0E1117")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    siniflar = ["E", "K", "C"]
    veriler  = [df[df["Gercek"] == c]["F0_Ort"].values for c in siniflar]
    bp = ax.boxplot(veriler, patch_artist=True,
                    medianprops=dict(color="white", lw=2))
    for patch, cin in zip(bp["boxes"], siniflar):
        patch.set_facecolor(CINSIYET_RENK[cin])
    ax.set_xticklabels([CINSIYET_AD[c] for c in siniflar], color="white")
    ax.axhline(F0_COCUK_SINIR, color="#81C784", ls=":", lw=1.5,
               label=f"Cocuk esigi {F0_COCUK_SINIR} Hz")
    ax.axhline(F0_KADIN_SINIR, color="#E57373", ls=":", lw=1.5,
               label=f"Kadin esigi {F0_KADIN_SINIR} Hz")
    ax.set_ylabel("F0 (Hz)"); ax.set_title("F0 Dagilimi (Sinif Bazli)")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e")
    return fig


# ─────────────────────────────────────────────
#  ANA UYGULAMA
# ─────────────────────────────────────────────
def main():

    # ── Yan Panel ─────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/microphone.png", width=80)
        st.title("Grup 19")
        st.markdown("**Ses Analizi ve Cinsiyet Siniflandirma**")
        st.markdown("---")

        dataset_koku = st.text_input(
            "Dataset Kok Klasoru",
            value="proje_1/Midterm_Dataset_2026",
            help="GRUP_01, GRUP_02 ... klasorlerinin bulundugu ana klasor"
        )
        excel_yolu = st.text_input(
            "Birlesik Excel MetaVeri",
            value="birlesik_metadata.xlsx",
        )

        st.markdown("---")
        st.caption("F0 Esikleri (Hz)")
        cocuk_esik = st.number_input("Cocuk Esigi",  value=250,
                                     min_value=100, max_value=600, step=10)
        kadin_esik = st.number_input("Kadin Esigi",  value=165,
                                     min_value=80,  max_value=400, step=5)
        global F0_COCUK_SINIR, F0_KADIN_SINIR
        F0_COCUK_SINIR = cocuk_esik
        F0_KADIN_SINIR = kadin_esik

        # Grup filtresi
        st.markdown("---")
        secili_gruplar = []
        if os.path.exists(excel_yolu):
            try:
                meta_on = pd.read_excel(excel_yolu)
                meta_on["Dosya_Adi"] = meta_on["Dosya_Adi"].astype(str).str.strip()
                grup_set = set()
                for d in meta_on["Dosya_Adi"]:
                    m2 = re.match(r'[Gg](\d+)_', d)
                    if m2:
                        grup_set.add(f"G{int(m2.group(1)):02d}")
                gruplar = sorted(grup_set)
                secili_gruplar = st.multiselect(
                    "Gruplar (bos = hepsi)",
                    options=gruplar,
                    default=gruplar,
                )
            except Exception:
                pass

    # ── Sekmeler ──────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "Canli Demo",
        "Veri Seti Analizi",
        "Hata Analizi",
    ])

    # ══════════════════════════════════════════
    #  SEKME 1 – CANLI DEMO
    # ══════════════════════════════════════════
    with tab1:
        st.header("Canli Demo - Ses Siniflandirma")
        st.info(
            "Bir WAV dosyasi yukleyin VEYA dataset'ten secin. "
            "Sistem otokorelasyon tabanli F0 analizi ile sinifi tahmin eder."
        )

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.subheader("Dosya Yukle (Drag & Drop)")
            yuklenen = st.file_uploader("WAV dosyasi secin", type=["wav"])

        with col_b:
            st.subheader("Dataset'ten Sec")
            tum_dosyalar = []
            if os.path.isdir(dataset_koku):
                tum_dosyalar = sorted([
                    os.path.basename(p)
                    for p in glob.glob(
                        os.path.join(dataset_koku, "**", "*.wav"),
                        recursive=True
                    )
                ])
            secili_dosya = st.selectbox(
                f"Dosya secin ({len(tum_dosyalar)} dosya mevcut)",
                ["--"] + tum_dosyalar
            )

        analiz_yap = st.button("ANALIZ ET", type="primary",
                               use_container_width=True)

        if analiz_yap:
            tmp_yol       = None
            gercek_etiket = None
            dosya_ismi    = ""

            if yuklenen is not None:
                with tempfile.NamedTemporaryFile(delete=False,
                                                 suffix=".wav") as tmp:
                    tmp.write(yuklenen.read())
                    tmp_yol   = tmp.name
                dosya_ismi = yuklenen.name

            elif secili_dosya != "--":
                dosya_ismi = secili_dosya
                tmp_yol    = dosya_yolu_bul(secili_dosya, dataset_koku)
                if tmp_yol is None:
                    st.error(f"Dosya bulunamadi: {secili_dosya}")
                    st.stop()
                if os.path.exists(excel_yolu):
                    try:
                        meta = pd.read_excel(excel_yolu)
                        meta["Dosya_Adi"] = meta["Dosya_Adi"].astype(str).str.strip()
                        eslesme = meta[meta["Dosya_Adi"] == secili_dosya.strip()]
                        if not eslesme.empty:
                            gercek_etiket = normalize_cinsiyet(
                                eslesme.iloc[0]["Cinsiyet"])
                    except Exception:
                        pass
            else:
                st.warning("Lutfen bir dosya yukleyin veya secin.")
                st.stop()

            with st.spinner("Analiz yapiliyor..."):
                sonuc = analiz_et(tmp_yol)

            renk = CINSIYET_RENK[sonuc["tahmin"]]
            st.markdown(f"""
            <div style='background:{renk}22; border:2px solid {renk};
                        border-radius:12px; padding:20px; text-align:center;'>
              <h2 style='color:{renk}; margin:0;'>
                {CINSIYET_AD[sonuc["tahmin"]]}
              </h2>
              <p style='color:white; margin:4px 0;'>Tahmin</p>
            </div>
            """, unsafe_allow_html=True)

            if gercek_etiket and gercek_etiket in CINSIYET_AD:
                if gercek_etiket == sonuc["tahmin"]:
                    st.success(f"Dogru! Gercek: {CINSIYET_AD[gercek_etiket]}")
                else:
                    st.error(f"Hatali. Gercek: {CINSIYET_AD[gercek_etiket]}")

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ortalama F0",  f"{sonuc['f0_ort']:.1f} Hz")
            c2.metric("F0 Std",       f"{sonuc['f0_std']:.1f} Hz")
            c3.metric("Ort. ZCR",     f"{sonuc['zcr_ort']:.4f}")
            c4.metric("Sesli Oran",   f"{sonuc['voiced_oran']*100:.1f}%")

            fig = ciz_analiz(sonuc, dosya_ismi, gercek_etiket)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.audio(tmp_yol, format="audio/wav")

    # ══════════════════════════════════════════
    #  SEKME 2 – VERİ SETİ ANALİZİ
    # ══════════════════════════════════════════
    with tab2:
        st.header("Tum Veri Seti Analizi")

        if not os.path.exists(excel_yolu):
            st.error(f"Excel dosyasi bulunamadi: {excel_yolu}")
            st.stop()
        if not os.path.isdir(dataset_koku):
            st.error(f"Dataset klasoru bulunamadi: {dataset_koku}")
            st.stop()

        if st.button("Tum Seti Analiz Et", type="primary"):
            meta_df = pd.read_excel(excel_yolu)
            meta_df["Dosya_Adi"]     = meta_df["Dosya_Adi"].astype(str).str.strip()
            meta_df["Cinsiyet_Norm"] = meta_df["Cinsiyet"].apply(normalize_cinsiyet)
            meta_df = meta_df[meta_df["Cinsiyet_Norm"].isin(["E", "K", "C"])]

            # Grup filtresi
            if secili_gruplar:
                def grup_kodu(dosya):
                    m2 = re.match(r'[Gg](\d+)_', dosya)
                    return f"G{int(m2.group(1)):02d}" if m2 else ""
                meta_df["_grup"] = meta_df["Dosya_Adi"].apply(grup_kodu)
                meta_df = meta_df[meta_df["_grup"].isin(secili_gruplar)]

            toplam   = len(meta_df)
            progress = st.progress(0, text="Analiz basliyor...")
            sonuc_listesi = []

            for i, (_, satir) in enumerate(meta_df.iterrows()):
                progress.progress(
                    (i + 1) / max(toplam, 1),
                    text=f"({i+1}/{toplam}) {satir['Dosya_Adi']}"
                )
                yol = dosya_yolu_bul(satir["Dosya_Adi"], dataset_koku)
                if yol is None:
                    continue
                try:
                    s = analiz_et(yol)
                except Exception:
                    continue

                sonuc_listesi.append({
                    "Dosya_Adi":   satir["Dosya_Adi"],
                    "Denek_ID":    satir.get("Denek_ID", ""),
                    "Gercek":      satir["Cinsiyet_Norm"],
                    "Yas":         satir.get("Yas", ""),
                    "Duygu":       satir.get("Duygu", ""),
                    "Tahmin":      s["tahmin"],
                    "F0_Ort":      round(s["f0_ort"],      2),
                    "F0_Std":      round(s["f0_std"],      2),
                    "ZCR_Ort":     round(s["zcr_ort"],     4),
                    "Energy_Ort":  round(s["energy_ort"],  4),
                    "Voiced_Oran": round(s["voiced_oran"], 3),
                    "Dogru_mu":    satir["Cinsiyet_Norm"] == s["tahmin"],
                })

            progress.empty()
            df = pd.DataFrame(sonuc_listesi)
            st.session_state["sonuc_df"] = df

            bulunamayan = toplam - len(sonuc_listesi)
            if bulunamayan > 0:
                st.warning(
                    f"{bulunamayan} dosya dataset klasorunde bulunamadi "
                    f"ve atlandi. Klasor yolunu kontrol edin."
                )

        if "sonuc_df" in st.session_state:
            df  = st.session_state["sonuc_df"]
            acc = df["Dogru_mu"].mean() * 100

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Genel Basari",  f"%{acc:.1f}")
            col_m2.metric("Analiz Edilen", f"{len(df)} dosya")
            col_m3.metric("Hatali",        f"{(~df['Dogru_mu']).sum()} dosya")
            st.markdown("---")

            st.subheader("F0 Istatistik Tablosu")
            st.dataframe(istatistik_tablosu(df),
                         use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Karisiklik Matrisi")
                fig_cm = confusion_matrix_ciz(df)
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
            with col2:
                st.subheader("F0 Dagilimi")
                fig_f0 = f0_dagilim_grafigi(df)
                st.pyplot(fig_f0, use_container_width=True)
                plt.close(fig_f0)

            st.subheader("Tum Tahmin Sonuclari")
            goruntu_df = df.copy()
            goruntu_df["Gercek"] = goruntu_df["Gercek"].map(CINSIYET_AD)
            goruntu_df["Tahmin"] = goruntu_df["Tahmin"].map(CINSIYET_AD)
            goruntu_df["Sonuc"]  = goruntu_df["Dogru_mu"].apply(
                lambda x: "DOGRU" if x else "YANLIS")
            goruntu_df = goruntu_df.drop(columns=["Dogru_mu"])
            st.dataframe(goruntu_df, use_container_width=True, hide_index=True)

            csv = goruntu_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Sonuclari CSV Indir", csv,
                               "birlesik_sonuclar.csv", "text/csv")

    # ══════════════════════════════════════════
    #  SEKME 3 – HATA ANALİZİ
    # ══════════════════════════════════════════
    with tab3:
        st.header("Hata Analizi")

        if "sonuc_df" not in st.session_state:
            st.info("Once 'Veri Seti Analizi' sekmesinden analizi calistirin.")
            st.stop()

        df      = st.session_state["sonuc_df"]
        hatalar = df[~df["Dogru_mu"]].copy()

        if hatalar.empty:
            st.success("Hic hata yok! Tum ornekler dogru siniflandirildi.")
        else:
            st.warning(f"{len(hatalar)} adet yanlis tahmin bulundu.")
            goster = hatalar[["Dosya_Adi", "Gercek", "Tahmin",
                               "F0_Ort", "Duygu", "Yas"]].copy()
            goster["Gercek"] = goster["Gercek"].map(CINSIYET_AD)
            goster["Tahmin"] = goster["Tahmin"].map(CINSIYET_AD)
            st.dataframe(goster.rename(columns={
                "Gercek": "Gercek Sinif", "Tahmin": "Tahmin Edilen"
            }), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Otomatik Hata Yorumu")
            for _, satir in hatalar.iterrows():
                with st.expander(f"YANLIS: {satir['Dosya_Adi']}"):
                    gercek = CINSIYET_AD.get(satir["Gercek"], satir["Gercek"])
                    tahmin = CINSIYET_AD.get(satir["Tahmin"], satir["Tahmin"])
                    f0     = satir["F0_Ort"]
                    duygu  = satir.get("Duygu", "-")
                    yas    = satir.get("Yas", "-")

                    st.markdown(f"**Gercek:** {gercek} | **Tahmin:** {tahmin}")
                    st.markdown(
                        f"**F0 Ort:** {f0:.1f} Hz | "
                        f"**Duygu:** {duygu} | **Yas:** {yas}"
                    )
                    yorumlar = []
                    if satir["Gercek"] == "K" and satir["Tahmin"] == "E":
                        yorumlar.append(
                            f"Dusuk F0 ({f0:.0f} Hz): Kadin sesinin F0'i "
                            f"esik degerinin ({F0_KADIN_SINIR} Hz) altinda. "
                            f"'{duygu}' duygusu veya yas ({yas}) etkili olabilir."
                        )
                    if satir["Gercek"] == "K" and satir["Tahmin"] == "C":
                        yorumlar.append(
                            f"Yuksek F0 ({f0:.0f} Hz): Kadin sesi Cocuk "
                            f"esigini ({F0_COCUK_SINIR} Hz) asmis. "
                            f"Duygu veya yas ({yas}) etkisi."
                        )
                    if satir["Gercek"] == "E" and satir["Tahmin"] == "K":
                        yorumlar.append(
                            f"Yuksek F0 ({f0:.0f} Hz): Erkek sesi "
                            f"beklenenden tiz. '{duygu}' veya genc yas ({yas})."
                        )
                    if satir["Gercek"] == "C" and satir["Tahmin"] != "C":
                        yorumlar.append(
                            f"Dusuk F0 ({f0:.0f} Hz): Cocuk sesi esik degerinin "
                            f"({F0_COCUK_SINIR} Hz) altinda. "
                            f"Yas {yas} - erken ergenlik veya kalin tini."
                        )
                    if not yorumlar:
                        yorumlar.append(
                            f"F0={f0:.0f} Hz sinir bolgede; "
                            f"duygu/gurultu/yas etkisi olabilir."
                        )
                    for y_text in yorumlar:
                        st.markdown(f"- {y_text}")

        st.markdown("---")
        st.subheader("Genel Yorum")
        st.markdown("""
        **Sistemin guclu yanlari:**
        - Otokorelasyon tabanli F0 tespiti sesli bolgelerde guvenilir tahmin uretir.
        - ZCR + STE filtresi sessiz/gurultulu kisimlar analiz disi birakilir.
        - Cok gruplu dataset otomatik taranir, dosya yollari dinamik bulunur.
        - Farkli gruplarin cinsiyet etiket formatlari (M/F/E/K/C) otomatik normalize edilir.

        **Olasi hata kaynaklari:**
        1. Duygu durumu - Ofkeli/sevinçli ses F0'i +/-30-60 Hz kaydirabilir.
        2. Yas sinir bolgeleri - Yasli kadin/genc erkek sesi sinir degerlerde kalabilir.
        3. Farkli kayit ortamlari - Gruplar farkli cihaz ve ortamlarda kayit yapti.
        4. Ortam gurultusu - Otokorelasyon tepe tespiti gurultude sapar.
        5. Sabit esik - Kural tabanli yaklasim bireysel frekans varyasyonunu goz ardi eder.
        """)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
