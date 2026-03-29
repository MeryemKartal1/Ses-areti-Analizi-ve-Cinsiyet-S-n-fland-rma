"""
=============================================================
  GRup 19 – Ses Analizi ve Cinsiyet Sınıflandırma
  Dönemiçi Proje  |  2025-2026 Bahar
=============================================================
Kurulum:
    pip install streamlit librosa numpy scipy pandas matplotlib openpyxl

Çalıştırma:
    streamlit run ses_analiz_app.py

Klasör yapısı (kodun yanında olmalı):
    proje_1/
        ses_kayitlari/
            G19_D01_C_08_No_Mutlu_C2.wav
            ...
    Grup_19_MetaVeri.xlsx
=============================================================
"""

import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

# ── Kodun kendi bulunduğu klasörü baz al ──────────────────────────
# Streamlit hangi dizinden çalıştırılırsa çalıştırılsın,
# bu satır sayesinde Excel ve ses klasörü hep doğru bulunur.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)   # çalışma dizinini kodun yanına sabitle

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
FRAME_MS      = 25          # pencere uzunluğu (ms)
HOP_MS        = 10          # atlama (ms)
ENERGY_THRESH = 0.02        # sesli bölge enerji eşiği (normalize)
ZCR_THRESH    = 0.15        # sesli/sessiz ZCR eşiği

# Kural tabanlı sınıflandırma eşikleri (Hz)
# Kadın > 165 Hz, Çocuk > 250 Hz, Erkek ≤ 165 Hz
F0_COCUK_SINIR = 250
F0_KADIN_SINIR = 165

CINSIYET_RENK = {"E": "#4A90D9", "K": "#E57373", "C": "#81C784"}
CINSIYET_AD   = {"E": "Erkek",   "K": "Kadın",   "C": "Çocuk"}

# ─────────────────────────────────────────────
#  YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────

def yukle_ses(dosya_yolu: str, sr_hedef: int = 22050):
    """WAV dosyasını yükler, mono'ya çevirir."""
    y, sr = librosa.load(dosya_yolu, sr=sr_hedef, mono=True)
    return y, sr


def hesapla_enerji_zcr(y, sr):
    """
    Kısa süreli enerji (STE) ve ZCR hesaplar.
    Döner: energy (frame,), zcr (frame,), frame_times (frame,)
    """
    frame_len = int(FRAME_MS * sr / 1000)
    hop_len   = int(HOP_MS   * sr / 1000)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len,
                                              hop_length=hop_len)[0]
    # STE  (RMS enerji)
    rms = librosa.feature.rms(y=y, frame_length=frame_len,
                               hop_length=hop_len)[0]
    rms_norm = rms / (rms.max() + 1e-9)

    frame_times = librosa.frames_to_time(np.arange(len(rms)),
                                          sr=sr, hop_length=hop_len)
    return rms_norm, zcr, frame_times


def sesli_bolge_maskesi(energy, zcr):
    """
    Sesli (voiced) çerçeveleri belirler:
      - enerji > ENERGY_THRESH  VE  zcr < ZCR_THRESH
    """
    return (energy > ENERGY_THRESH) & (zcr < ZCR_THRESH)


def otokorelasyon_f0(cerceve, sr, f0_min=50, f0_max=500):
    """
    Tek bir çerçeve için otokorelasyon tabanlı F0 tahmini.
    R(τ) = Σ x[n]·x[n−τ]
    """
    n = len(cerceve)
    # Normalize otokorelasyon
    cerceve = cerceve - cerceve.mean()
    r = np.correlate(cerceve, cerceve, mode='full')
    r = r[n - 1:]          # sadece τ ≥ 0 kısmı
    r = r / (r[0] + 1e-9)  # normalize

    # Arama aralığı (lag cinsinden)
    lag_min = int(sr / f0_max)
    lag_max = int(sr / f0_min)
    lag_max = min(lag_max, len(r) - 1)

    if lag_min >= lag_max:
        return None

    r_aralik = r[lag_min:lag_max]
    tepe_idx, _ = find_peaks(r_aralik, height=0.3)

    if len(tepe_idx) == 0:
        return None

    en_iyi_lag = tepe_idx[np.argmax(r_aralik[tepe_idx])] + lag_min
    f0 = sr / en_iyi_lag
    return float(f0)


def analiz_et(dosya_yolu: str):
    """
    Bir ses dosyasını tam analiz eder.
    Döner sözlük: {f0_ort, f0_std, zcr_ort, energy_ort,
                   voiced_oran, tahmin,
                   y, sr, energy, zcr, frame_times, voiced_mask,
                   f0_dizisi, frame_f0_times}
    """
    y, sr = yukle_ses(dosya_yolu)
    energy, zcr, frame_times = hesapla_enerji_zcr(y, sr)
    voiced = sesli_bolge_maskesi(energy, zcr)

    frame_len = int(FRAME_MS * sr / 1000)
    hop_len   = int(HOP_MS   * sr / 1000)

    f0_listesi   = []
    f0_zaman     = []

    for i, is_voiced in enumerate(voiced):
        if not is_voiced:
            continue
        baslangic = i * hop_len
        bitis     = baslangic + frame_len
        if bitis > len(y):
            break
        cerceve = y[baslangic:bitis]
        f0 = otokorelasyon_f0(cerceve, sr)
        if f0 is not None:
            f0_listesi.append(f0)
            f0_zaman.append(frame_times[i])

    f0_dizi = np.array(f0_listesi)
    f0_ort  = float(np.mean(f0_dizi))   if len(f0_dizi) > 0 else 0.0
    f0_std  = float(np.std(f0_dizi))    if len(f0_dizi) > 0 else 0.0
    zcr_ort = float(np.mean(zcr[voiced])) if voiced.sum() > 0 else 0.0
    eng_ort = float(np.mean(energy[voiced])) if voiced.sum() > 0 else 0.0
    voiced_oran = voiced.sum() / len(voiced) if len(voiced) > 0 else 0.0

    # ── Kural Tabanlı Sınıflandırma ──────────────────────────────
    if f0_ort >= F0_COCUK_SINIR:
        tahmin = "C"
    elif f0_ort >= F0_KADIN_SINIR:
        tahmin = "K"
    else:
        tahmin = "E"

    return {
        "f0_ort": f0_ort, "f0_std": f0_std,
        "zcr_ort": zcr_ort, "energy_ort": eng_ort,
        "voiced_oran": voiced_oran,
        "tahmin": tahmin,
        "y": y, "sr": sr,
        "energy": energy, "zcr": zcr,
        "frame_times": frame_times,
        "voiced_mask": voiced,
        "f0_dizisi": f0_dizi,
        "frame_f0_times": np.array(f0_zaman),
    }


def ciz_analiz(sonuc: dict, dosya_adi: str, gercek: str | None = None):
    """
    4 panelli analiz grafiği:
    1. Dalga formu + sesli bölgeler
    2. STE & ZCR
    3. F0 zaman serisi
    4. Otokorelasyon vs FFT karşılaştırması
    """
    y, sr = sonuc["y"], sonuc["sr"]
    t = np.linspace(0, len(y) / sr, num=len(y))

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
    baslik = f"{dosya_adi}  →  Tahmin: {CINSIYET_AD[tahmin]}"
    if gercek:
        dogru = "✅" if gercek == tahmin else "❌"
        baslik += f"  |  Gerçek: {CINSIYET_AD[gercek]}  {dogru}"
    fig.suptitle(baslik, color="white", fontsize=13, fontweight="bold", y=1.01)

    # 1. Dalga formu
    ax1.plot(t, y, color="#5a9fd4", lw=0.4, alpha=0.8)
    hop_len = int(HOP_MS * sr / 1000)
    for i, v in enumerate(sonuc["voiced_mask"]):
        if v:
            ts = i * hop_len / sr
            te = ts + FRAME_MS / 1000
            ax1.axvspan(ts, te, alpha=0.15, color="#81C784")
    ax1.set_title("Dalga Formu (yeşil = sesli bölgeler)")
    ax1.set_xlabel("Zaman (s)"); ax1.set_ylabel("Genlik")

    # 2. STE
    ax2.plot(sonuc["frame_times"], sonuc["energy"], color="#FFD700", lw=1.5)
    ax2.axhline(ENERGY_THRESH, color="#FF6B6B", ls="--", lw=1, label=f"Eşik={ENERGY_THRESH}")
    ax2.set_title("Kısa Süreli Enerji (STE)")
    ax2.set_xlabel("Zaman (s)"); ax2.set_ylabel("RMS (norm)")
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e")

    # 3. ZCR
    ax3.plot(sonuc["frame_times"], sonuc["zcr"], color="#FF9F43", lw=1.5)
    ax3.axhline(ZCR_THRESH, color="#FF6B6B", ls="--", lw=1, label=f"Eşik={ZCR_THRESH}")
    ax3.set_title("Sıfır Geçiş Oranı (ZCR)")
    ax3.set_xlabel("Zaman (s)"); ax3.set_ylabel("ZCR")
    ax3.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e")

    # 4. F0 Zaman serisi
    if len(sonuc["f0_dizisi"]) > 0:
        ax4.scatter(sonuc["frame_f0_times"], sonuc["f0_dizisi"],
                    color=renk, s=8, alpha=0.7, label="F0 (otokor.)")
        ax4.axhline(sonuc["f0_ort"], color="white", ls="--", lw=1.5,
                    label=f"Ort.={sonuc['f0_ort']:.1f} Hz")
        ax4.axhline(F0_COCUK_SINIR, color="#81C784", ls=":", lw=1, label=f"Çocuk eşiği={F0_COCUK_SINIR}")
        ax4.axhline(F0_KADIN_SINIR, color="#E57373", ls=":", lw=1, label=f"Kadın eşiği={F0_KADIN_SINIR}")
        ax4.set_title("F0 Zaman Serisi (Otokorelasyon)")
        ax4.set_xlabel("Zaman (s)"); ax4.set_ylabel("F0 (Hz)")
        ax4.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e")
    else:
        ax4.text(0.5, 0.5, "F0 tespit edilemedi", ha="center", va="center",
                 color="white", transform=ax4.transAxes)

    # 5. Otokorelasyon vs FFT karşılaştırması
    #    Örnek olarak ilk sesli çerçeveyi kullan
    frame_len = int(FRAME_MS * sr / 1000)
    hop_len_s = int(HOP_MS   * sr / 1000)
    first_voiced = np.where(sonuc["voiced_mask"])[0]
    if len(first_voiced) > 0:
        idx = first_voiced[0]
        bas = idx * hop_len_s
        bit = bas + frame_len
        cerceve = y[bas:bit] if bit <= len(y) else y[bas:]
        cerceve = cerceve - cerceve.mean()

        # Otokorelasyon
        n = len(cerceve)
        r = np.correlate(cerceve, cerceve, mode='full')[n - 1:]
        r = r / (r[0] + 1e-9)
        lag_arr = np.arange(len(r)) / sr * 1000  # ms
        lag_min = int(sr / 500); lag_max = min(int(sr / 50), len(r) - 1)
        ax5.plot(lag_arr[:lag_max], r[:lag_max], color="#5a9fd4", lw=1.2, label="Otokorelasyon")

        # FFT tepe noktası (Hz) – yalnızca işaret olarak çiz
        freqs = np.fft.rfftfreq(n, d=1 / sr)
        mag   = np.abs(np.fft.rfft(cerceve))
        mask  = (freqs >= 50) & (freqs <= 500)
        if mask.any():
            fft_f0 = freqs[mask][np.argmax(mag[mask])]
            oto_lag = sr / (sonuc["f0_ort"] + 1e-9) / sr * 1000
            ax5.axvline(oto_lag, color="#FFD700", ls="--", lw=1.5,
                        label=f"F0 (otokor.)={sonuc['f0_ort']:.0f} Hz")
            ax5.axvline(1000 / fft_f0, color="#FF9F43", ls=":", lw=1.5,
                        label=f"F0 (FFT)={fft_f0:.0f} Hz")
        ax5.set_title("Otokorelasyon vs FFT (1. sesli çerçeve)")
        ax5.set_xlabel("Gecikme (ms)"); ax5.set_ylabel("R(τ) norm")
        ax5.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e")
    else:
        ax5.text(0.5, 0.5, "Sesli çerçeve bulunamadı", ha="center", va="center",
                 color="white", transform=ax5.transAxes)

    return fig


def toplu_analiz(meta_df: pd.DataFrame, ses_klasoru: str):
    """Tüm veri seti üzerinde analiz yapar, sonuç DataFrame döner."""
    sonuclar = []
    for _, satir in meta_df.iterrows():
        dosya_yolu = os.path.join(ses_klasoru, satir["Dosya_Adi"])
        if not os.path.exists(dosya_yolu):
            continue
        s = analiz_et(dosya_yolu)
        sonuclar.append({
            "Dosya_Adi":   satir["Dosya_Adi"],
            "Denek_ID":    satir["Denek_ID"],
            "Gercek":      satir["Cinsiyet"],
            "Yas":         satir["Yas"],
            "Duygu":       satir["Duygu"],
            "Tahmin":      s["tahmin"],
            "F0_Ort":      round(s["f0_ort"], 2),
            "F0_Std":      round(s["f0_std"], 2),
            "ZCR_Ort":     round(s["zcr_ort"], 4),
            "Energy_Ort":  round(s["energy_ort"], 4),
            "Voiced_Oran": round(s["voiced_oran"], 3),
            "Dogru_mu":    satir["Cinsiyet"] == s["tahmin"],
        })
    return pd.DataFrame(sonuclar)


def istatistik_tablosu(df: pd.DataFrame):
    """
    Proje talimatında istenen istatistik tablosu.
    Sınıf | Örnek Sayısı | Ort F0 | Std F0 | Başarı %
    """
    satirlar = []
    for cin, ad in CINSIYET_AD.items():
        alt = df[df["Gercek"] == cin]
        if len(alt) == 0:
            continue
        satirlar.append({
            "Sınıf":         ad,
            "Örnek Sayısı":  len(alt),
            "Ortalama F0 (Hz)": round(alt["F0_Ort"].mean(), 2),
            "Standart Sapma":   round(alt["F0_Ort"].std(), 2),
            "Başarı (%)":       round(alt["Dogru_mu"].mean() * 100, 1),
        })
    return pd.DataFrame(satirlar)


def confusion_matrix_ciz(df: pd.DataFrame):
    """3×3 karışıklık matrisi görselleştirmesi."""
    siniflar  = ["E", "K", "C"]
    etiketler = ["Erkek", "Kadın", "Çocuk"]
    cm = np.zeros((3, 3), dtype=int)
    for i, gercek in enumerate(siniflar):
        for j, tahmin in enumerate(siniflar):
            cm[i, j] = ((df["Gercek"] == gercek) & (df["Tahmin"] == tahmin)).sum()

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0E1117")
    ax.set_facecolor("#1a1a2e")
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(etiketler, color="white")
    ax.set_yticklabels(etiketler, color="white")
    ax.set_xlabel("Tahmin", color="white"); ax.set_ylabel("Gerçek", color="white")
    ax.set_title("Karışıklık Matrisi", color="white")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] < cm.max() / 2 else "#0E1117",
                    fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def f0_dagilim_grafigi(df: pd.DataFrame):
    """F0 dağılımı – cinsiyet bazlı boxplot."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0E1117")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    siniflar = ["E", "K", "C"]
    veriler  = [df[df["Gercek"] == c]["F0_Ort"].values for c in siniflar]
    bp = ax.boxplot(veriler, patch_artist=True,
                    medianprops=dict(color="white", lw=2))
    for patch, cin in zip(bp["boxes"], siniflar):
        patch.set_facecolor(CINSIYET_RENK[cin])

    ax.set_xticklabels([CINSIYET_AD[c] for c in siniflar], color="white")
    ax.axhline(F0_COCUK_SINIR, color="#81C784", ls=":", lw=1.5, label=f"Çocuk eşiği {F0_COCUK_SINIR} Hz")
    ax.axhline(F0_KADIN_SINIR, color="#E57373", ls=":", lw=1.5, label=f"Kadın eşiği {F0_KADIN_SINIR} Hz")
    ax.set_ylabel("F0 (Hz)"); ax.set_title("F0 Dağılımı (Sınıf Bazlı)")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e")
    return fig


# ─────────────────────────────────────────────
#  STREAMLIT ARAYÜZÜ
# ─────────────────────────────────────────────

def main():
    # ── Yan panel ─────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/microphone.png", width=80)
        st.title("Grup 19")
        st.markdown("**Ses Analizi ve Cinsiyet Sınıflandırma**")
        st.markdown("---")
        ses_klasoru = st.text_input(
            "📂 Ses Klasörü Yolu",
            value="proje_1/ses_kayitlari",
            help="WAV dosyalarının bulunduğu klasör"
        )
        excel_yolu = st.text_input(
            "📊 Excel MetaVeri Yolu",
            value="Grup_19_MetaVeri.xlsx",
        )
        st.markdown("---")
        st.caption("F0 Eşikleri (Hz)")
        cocuk_esik = st.number_input("Çocuk Eşiği", value=250, min_value=100, max_value=600, step=10)
        kadin_esik = st.number_input("Kadın Eşiği", value=165, min_value=80,  max_value=400, step=5)
        global F0_COCUK_SINIR, F0_KADIN_SINIR
        F0_COCUK_SINIR = cocuk_esik
        F0_KADIN_SINIR = kadin_esik

    # ── Sekmeler ──────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🎙️ Canlı Demo",
        "📊 Veri Seti Analizi",
        "🔍 Hata Analizi",
    ])

    # ══════════════════════════════════════════
    #  SEKMe 1 – CANLI DEMO
    # ══════════════════════════════════════════
    with tab1:
        st.header("🎙️ Canlı Demo – Ses Sınıflandırma")
        st.info(
            "Bir WAV dosyası yükleyin **VEYA** veri setinden bir dosya seçin. "
            "Sistem otokorelasyon tabanlı F0 analizi ile sınıfı tahmin eder."
        )

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.subheader("📤 Dosya Yükle (Drag & Drop)")
            yuklenen = st.file_uploader("WAV dosyası seçin", type=["wav"])

        with col_b:
            st.subheader("📂 Veri Setinden Seç")
            mevcut_dosyalar = []
            if os.path.isdir(ses_klasoru):
                mevcut_dosyalar = sorted([
                    f for f in os.listdir(ses_klasoru) if f.endswith(".wav")
                ])
            secili_dosya = st.selectbox("Dosya seçin", ["—"] + mevcut_dosyalar)

        analiz_yap = st.button("▶  ANALİZ ET", type="primary", use_container_width=True)

        if analiz_yap:
            # Hangi kaynaktan?
            tmp_yol = None
            gercek_etiket = None
            dosya_ismi = ""

            if yuklenen is not None:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(yuklenen.read())
                    tmp_yol   = tmp.name
                dosya_ismi = yuklenen.name
            elif secili_dosya != "—":
                tmp_yol    = os.path.join(ses_klasoru, secili_dosya)
                dosya_ismi = secili_dosya
                # Gerçek etiketi meta veriden al
                if os.path.exists(excel_yolu):
                    meta = pd.read_excel(excel_yolu)
                    eslesme = meta[meta["Dosya_Adi"] == secili_dosya]
                    if not eslesme.empty:
                        gercek_etiket = eslesme.iloc[0]["Cinsiyet"]
            else:
                st.warning("Lütfen bir dosya yükleyin veya seçin.")
                st.stop()

            with st.spinner("Analiz yapılıyor..."):
                sonuc = analiz_et(tmp_yol)

            # Sonuç kutusu
            renk = CINSIYET_RENK[sonuc["tahmin"]]
            st.markdown(f"""
            <div style='background:{renk}22; border:2px solid {renk};
                        border-radius:12px; padding:20px; text-align:center;'>
              <h2 style='color:{renk}; margin:0;'>
                {CINSIYET_AD[sonuc['tahmin']]}
              </h2>
              <p style='color:white; margin:4px 0;'>Tahmin</p>
            </div>
            """, unsafe_allow_html=True)

            if gercek_etiket:
                dogru = gercek_etiket == sonuc["tahmin"]
                if dogru:
                    st.success(f"✅ Doğru! Gerçek: {CINSIYET_AD[gercek_etiket]}")
                else:
                    st.error(f"❌ Hatalı. Gerçek: {CINSIYET_AD[gercek_etiket]}")

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ortalama F0", f"{sonuc['f0_ort']:.1f} Hz")
            c2.metric("F0 Std", f"{sonuc['f0_std']:.1f} Hz")
            c3.metric("Ort. ZCR", f"{sonuc['zcr_ort']:.4f}")
            c4.metric("Sesli Oran", f"{sonuc['voiced_oran']*100:.1f}%")

            fig = ciz_analiz(sonuc, dosya_ismi, gercek_etiket)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Ses çalar
            st.audio(tmp_yol, format="audio/wav")

    # ══════════════════════════════════════════
    #  SEKMe 2 – VERİ SETİ ANALİZİ
    # ══════════════════════════════════════════
    with tab2:
        st.header("📊 Tüm Veri Seti Analizi")

        if not os.path.exists(excel_yolu):
            st.error(f"Excel dosyası bulunamadı: {excel_yolu}")
            st.stop()
        if not os.path.isdir(ses_klasoru):
            st.error(f"Ses klasörü bulunamadı: {ses_klasoru}")
            st.stop()

        if st.button("🔄 Tüm Seti Analiz Et", type="primary"):
            meta_df = pd.read_excel(excel_yolu)
            progress = st.progress(0, text="Analiz ediliyor...")
            toplam   = len(meta_df)

            sonuc_listesi = []
            for i, (_, satir) in enumerate(meta_df.iterrows()):
                dosya_yolu = os.path.join(ses_klasoru, satir["Dosya_Adi"])
                progress.progress((i + 1) / toplam,
                                   text=f"{satir['Dosya_Adi']} işleniyor...")
                if not os.path.exists(dosya_yolu):
                    continue
                s = analiz_et(dosya_yolu)
                sonuc_listesi.append({
                    "Dosya_Adi":   satir["Dosya_Adi"],
                    "Denek_ID":    satir["Denek_ID"],
                    "Gercek":      satir["Cinsiyet"],
                    "Yas":         satir["Yas"],
                    "Duygu":       satir["Duygu"],
                    "Tahmin":      s["tahmin"],
                    "F0_Ort":      round(s["f0_ort"], 2),
                    "F0_Std":      round(s["f0_std"], 2),
                    "ZCR_Ort":     round(s["zcr_ort"], 4),
                    "Energy_Ort":  round(s["energy_ort"], 4),
                    "Voiced_Oran": round(s["voiced_oran"], 3),
                    "Dogru_mu":    satir["Cinsiyet"] == s["tahmin"],
                })

            progress.empty()
            df = pd.DataFrame(sonuc_listesi)
            st.session_state["sonuc_df"] = df

        # Sonuçları göster
        if "sonuc_df" in st.session_state:
            df = st.session_state["sonuc_df"]
            acc = df["Dogru_mu"].mean() * 100

            st.metric("🎯 Genel Başarı", f"%{acc:.1f}")
            st.markdown("---")

            # İstatistik tablosu
            st.subheader("📋 Proje Tablosu (F0 İstatistikleri)")
            istat = istatistik_tablosu(df)
            st.dataframe(istat, use_container_width=True, hide_index=True)

            # Grafikler
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Karışıklık Matrisi")
                fig_cm = confusion_matrix_ciz(df)
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
            with col2:
                st.subheader("F0 Dağılımı")
                fig_f0 = f0_dagilim_grafigi(df)
                st.pyplot(fig_f0, use_container_width=True)
                plt.close(fig_f0)

            # Ham tablo
            st.subheader("📄 Tüm Tahmin Sonuçları")
            goruntu_df = df.copy()
            goruntu_df["Gercek"]  = goruntu_df["Gercek"].map(CINSIYET_AD)
            goruntu_df["Tahmin"]  = goruntu_df["Tahmin"].map(CINSIYET_AD)
            goruntu_df["Sonuç"]   = goruntu_df["Dogru_mu"].apply(lambda x: "✅" if x else "❌")
            goruntu_df = goruntu_df.drop(columns=["Dogru_mu"])
            st.dataframe(goruntu_df, use_container_width=True, hide_index=True)

            # CSV indir
            csv = goruntu_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Sonuçları CSV İndir", csv,
                               "grup19_sonuclar.csv", "text/csv")

    # ══════════════════════════════════════════
    #  SEKMe 3 – HATA ANALİZİ
    # ══════════════════════════════════════════
    with tab3:
        st.header("🔍 Hata Analizi")

        if "sonuc_df" not in st.session_state:
            st.info("Önce 'Veri Seti Analizi' sekmesinden analizi çalıştırın.")
            st.stop()

        df = st.session_state["sonuc_df"]
        hatalar = df[~df["Dogru_mu"]].copy()

        if hatalar.empty:
            st.success("🎉 Hiç hata yok! Tüm örnekler doğru sınıflandırıldı.")
        else:
            st.warning(f"⚠️ {len(hatalar)} adet yanlış tahmin bulundu.")
            st.dataframe(
                hatalar[["Dosya_Adi", "Gercek", "Tahmin", "F0_Ort", "Duygu", "Yas"]].rename(
                    columns={"Gercek": "Gerçek Sınıf", "Tahmin": "Tahmin Edilen"}
                ),
                use_container_width=True, hide_index=True,
            )

            st.markdown("---")
            st.subheader("🧠 Otomatik Hata Yorumu")
            for _, satir in hatalar.iterrows():
                with st.expander(f"❌ {satir['Dosya_Adi']}"):
                    gercek = CINSIYET_AD.get(satir["Gercek"], satir["Gercek"])
                    tahmin = CINSIYET_AD.get(satir["Tahmin"], satir["Tahmin"])
                    f0     = satir["F0_Ort"]
                    duygu  = satir["Duygu"]
                    yas    = satir["Yas"]

                    st.markdown(f"**Gerçek:** {gercek} | **Tahmin:** {tahmin}")
                    st.markdown(f"**F0 Ort:** {f0:.1f} Hz | **Duygu:** {duygu} | **Yaş:** {yas}")

                    # Otomatik yorum
                    yorumlar = []
                    if satir["Gercek"] == "K" and satir["Tahmin"] == "E":
                        yorumlar.append(f"**Düşük F0 ({f0:.0f} Hz):** Kadın sesinin F0'ı Kadın eşiğinin ({F0_KADIN_SINIR} Hz) altında kalmış. "
                                        f"'{duygu}' duygusu sesin tizliğini düşürmüş olabilir.")
                    if satir["Gercek"] == "K" and satir["Tahmin"] == "C":
                        yorumlar.append(f"**Yüksek F0 ({f0:.0f} Hz):** Kadın sesi Çocuk eşiğini ({F0_COCUK_SINIR} Hz) aşmış. "
                                        f"Yaşlı denek (Yaş={yas}) sesinin frekans yapısı normalden farklı olabilir.")
                    if satir["Gercek"] == "E" and satir["Tahmin"] == "K":
                        yorumlar.append(f"**Yüksek F0 ({f0:.0f} Hz):** Erkek sesinin F0'ı beklenenden yüksek. "
                                        f"'{duygu}' duygusu veya genç yaş ({yas}) etkisi olabilir.")
                    if satir["Gercek"] == "C" and satir["Tahmin"] != "C":
                        yorumlar.append(f"**Düşük F0 ({f0:.0f} Hz):** Çocuk sesinin F0'ı eşiğin ({F0_COCUK_SINIR} Hz) altında. "
                                        f"Yaş {yas} – erken ergenlik veya sesin kalın tınısı etken olabilir.")
                    if not yorumlar:
                        yorumlar.append(f"F0={f0:.0f} Hz değeri sınır bölgesinde, duygu/gürültü etkisi olabilir.")

                    for y_text in yorumlar:
                        st.markdown(f"- {y_text}")

        st.markdown("---")
        st.subheader("📌 Genel Yorum")
        st.markdown("""
        **Sistemin güçlü yanları:**
        - Otokorelasyon tabanlı F0 tespiti, sesli bölgelerde (Voiced frames) güvenilir tahmin üretir.
        - ZCR + STE filtresi sessiz/patırtılı kısımları analiz dışı bırakır.

        **Olası hata kaynakları:**
        1. **Duygu durumu** – Öfkeli veya sevinçli ses, F0'ı ±30-60 Hz kaydırabilir.
        2. **Yaş sınır bölgeleri** – 87 yaşındaki kadın sesi erkek sesine yakın F0 gösterebilir.
        3. **Ortam gürültüsü** – Otokorelasyon tepe tespiti gürültüde sapar.
        4. **Kısa kayıt süresi** – Az sayıda sesli çerçeve, ortalama F0'ı yanıltabilir.
        5. **Sabit eşik** – Kural tabanlı sınıflandırma, kişisel frekans varyasyonunu göz ardı eder.
        """)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
