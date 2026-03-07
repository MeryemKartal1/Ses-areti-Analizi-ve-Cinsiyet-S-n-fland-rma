import numpy as np
import scipy.io.wavfile as wav
from scipy.ndimage import median_filter

class AudioEngine:
    def __init__(self, file_path):
        self.fs, data = wav.read(file_path)
        # Normalizasyon
        self.data = data / 32768.0 if data.dtype == np.int16 else data
        self.frame_ms = 20 
        self.overlap = 0.5 
        
    def process(self):
        frame_len = int((self.frame_ms / 1000) * self.fs)
        hop_len = int(frame_len * (1 - self.overlap))
        energies, zcrs = [], []
        
        for i in range(0, len(self.data) - frame_len, hop_len):
            frame = self.data[i:i + frame_len]
            # Hamming penceresi spektral sızıntıyı önler
            windowed = frame * np.hamming(len(frame))
            energy = np.sum(windowed**2)
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            energies.append(energy)
            zcrs.append(zcr)
            
        return np.array(energies), np.array(zcrs), frame_len, hop_len

    def classify(self, energies, zcrs):
        # Eşiği 3'ten 6'ya çıkardık (Daha agresif temizlik için)
        noise_floor = np.mean(energies[:15])
        threshold = noise_floor * 6 
        
        # Enerji eşiğini karşılamayan her şeyi sıfırla
        vad_mask = energies > threshold
        
        # Medyan filtre boyutunu 5 yaparak kısa süreli 'tık' seslerini tamamen eledik
        vad_mask = median_filter(vad_mask.astype(float), size=5) > 0.5
        
        labels = []
        for i in range(len(vad_mask)):
            if not vad_mask[i]:
                labels.append(0)
            elif zcrs[i] < 0.15:
                labels.append(1) # Voiced (Yeşil)
            else:
                labels.append(2) # Unvoiced (Sarı)
        return labels

    def save_output(self, labels, base_path="sonuc"): # base_path buraya mutlaka eklenmeli
        frame_len = int((self.frame_ms / 1000) * self.fs)
        hop_len = int(frame_len * (1 - self.overlap))
        
        cleaned = []
        for i, lbl in enumerate(labels):
            if lbl > 0: # Sessizlik olmayan kısımları al
                start = i * hop_len
                end = start + hop_len
                cleaned.extend(self.data[start:end])
        
        output_p = f"{base_path}_temiz.wav"
        if cleaned:
            import scipy.io.wavfile as wav
            import numpy as np
            wav.write(output_p, self.fs, (np.array(cleaned) * 32767).astype(np.int16))
            
        return output_p # Arayüze dosya yolunu döndür