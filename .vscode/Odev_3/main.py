import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

###############################################################
# 1. SES DOSYASI OKUMA
###############################################################

def load_audio(file):

    signal, fs = librosa.load(file, sr=None)

    signal = signal / np.max(np.abs(signal))

    return signal, fs


###############################################################
# 2. FRAME OLUŞTURMA
###############################################################

def framing(signal, fs):

    frame_size = int(0.02 * fs)   # 20 ms
    hop_size = int(frame_size / 2)  # %50 overlap

    frames = []

    for i in range(0, len(signal) - frame_size, hop_size):

        frame = signal[i:i+frame_size]

        window = np.hamming(frame_size)

        frame = frame * window

        frames.append(frame)

    return np.array(frames), hop_size, frame_size


###############################################################
# 3. SHORT TIME ENERGY
###############################################################

def short_time_energy(frames):

    energy = np.sum(frames**2, axis=1)

    return energy


###############################################################
# 4. ZCR
###############################################################

def zero_crossing_rate(frames):

    zcr = []

    for frame in frames:

        crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2

        zcr.append(crossings / len(frame))

    return np.array(zcr)


###############################################################
# 5. NOISE FLOOR
###############################################################

def noise_estimation(energy):

    noise_energy = np.mean(energy[:20])

    return noise_energy


###############################################################
# 6. VAD
###############################################################

def vad_detection(energy, noise_energy):

    threshold = noise_energy * 3

    vad = energy > threshold

    return vad.astype(int)


###############################################################
# 7. HANGOVER
###############################################################

def hangover(vad, hang=3):

    vad_new = vad.copy()

    for i in range(len(vad)):

        if vad[i] == 0:

            if np.sum(vad[max(0, i-hang):i]) > 0:

                vad_new[i] = 1

    return vad_new


###############################################################
# 8. SESSİZLİK KALDIRMA
###############################################################

def remove_silence(signal, frames, vad, hop, frame_size):

    speech = []

    first = True

    for i in range(len(vad)):

        if vad[i] == 1:

            start = i * hop
            frame = signal[start:start+frame_size]

            if first:
                speech.extend(frame)
                first = False
            else:
                speech.extend(frame[hop:])

    return np.array(speech)


###############################################################
# 9. VOICED / UNVOICED
###############################################################

def voiced_unvoiced(energy, zcr):

    voiced = []

    e_th = np.mean(energy)
    z_th = np.mean(zcr)

    for i in range(len(energy)):

        if energy[i] > e_th and zcr[i] < z_th:
            voiced.append("Voiced")
        else:
            voiced.append("Unvoiced")

    return voiced


###############################################################
# 10. GRAFİK (RENK MASKESİ EKLENDİ)
###############################################################

def plot_results(signal, energy, zcr, vad, vu, hop, fs):

    time = np.arange(len(signal)) / fs
    frame_time = np.arange(len(energy)) * hop / fs

    plt.figure(figsize=(12,10))

    # 1️⃣ Orijinal Ses
    plt.subplot(4,1,1)
    plt.title("Orijinal Ses Sinyali")
    plt.plot(time, signal)

    # 2️⃣ Enerji
    plt.subplot(4,1,2)
    plt.title("Short Time Energy")
    plt.plot(frame_time, energy)

    # 3️⃣ ZCR
    plt.subplot(4,1,3)
    plt.title("Zero Crossing Rate")
    plt.plot(frame_time, zcr)

    # 4️⃣ VAD + Voiced/Unvoiced renk
    plt.subplot(4,1,4)
    plt.title("VAD ve Voiced / Unvoiced Bölgeleri")

    plt.plot(time, signal, color="black", alpha=0.5)

    for i in range(len(vad)):

        start = i * hop / fs
        end = (i+1) * hop / fs

        if vad[i] == 0:
            plt.axvspan(start, end, color="gray", alpha=0.3)

        else:
            if vu[i] == "Voiced":
                plt.axvspan(start, end, color="green", alpha=0.4)
            else:
                plt.axvspan(start, end, color="yellow", alpha=0.4)

    plt.tight_layout()
    plt.show()


###############################################################
# ANA PROGRAM
###############################################################

def process(file):

    signal, fs = load_audio(file)

    frames, hop, frame_size = framing(signal, fs)

    energy = short_time_energy(frames)

    zcr = zero_crossing_rate(frames)

    noise = noise_estimation(energy)

    vad = vad_detection(energy, noise)

    vad = hangover(vad)

    speech = remove_silence(signal, frames, vad, hop, frame_size)

    vu = voiced_unvoiced(energy, zcr)

    write("clean_speech.wav", fs, speech.astype(np.float32))

    plot_results(signal, energy, zcr, vad, vu, hop, fs)

    return vu