import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from audio_engine import AudioEngine
import cleanup

class VoiceAnalyzerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AURA - Akıcı Sinyal Analizi")
        self.root.geometry("1100x800")
        self.root.configure(bg="#121212")

        cleanup.clear_temp_files()

        self.side_bar = tk.Frame(self.root, bg="#1e1e26", width=250)
        self.side_bar.pack(fill=tk.Y, side=tk.LEFT)
        self.side_bar.pack_propagate(False)

        tk.Label(self.side_bar, text="AURA LABS", font=("Segoe UI Semibold", 20), 
                 bg="#1e1e26", fg="#00f2ff", pady=30).pack()

        self.btn_select = tk.Button(self.side_bar, text="Dosya Yükle", command=self.analyze,
                                    font=("Segoe UI", 11, "bold"), bg="#00f2ff", fg="#121212",
                                    relief=tk.FLAT, cursor="hand2", padx=20, pady=10)
        self.btn_select.pack(pady=20, padx=20, fill=tk.X)

        self.status_label = tk.Label(self.root, text="Sinyal işleme hazır.", bg="#121212", fg="#00f2ff")
        self.status_label.pack(pady=10)

        self.canvas_frame = tk.Frame(self.root, bg="#121212")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.canvas = None

    def analyze(self):
        file_path = filedialog.askopenfilename(filetypes=[("Wav Files", "*.wav")])
        if not file_path: return

        try:
            engine = AudioEngine(file_path)
            energies, zcrs, flen, hlen = engine.process()
            labels = engine.classify(energies, zcrs)
            
            output_file = engine.save_output(labels, base_path=f"sonuc_{engine.fs}")
            
            self.status_label.config(text=f"Akıcı dosya oluşturuldu: {output_file}", fg="#00f2ff")
            self.show_plots(engine, energies, zcrs, labels, flen, hlen)
            
        except Exception as e:
            self.status_label.config(text=f"Hata: {str(e)}", fg="#ff4b2b")

    def show_plots(self, engine, energies, zcrs, labels, flen, hlen):
        if self.canvas: self.canvas.get_tk_widget().destroy()
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), facecolor='#121212')
        
        time_axis = np.linspace(0, len(engine.data)/engine.fs, len(engine.data))
        ax1.plot(time_axis, engine.data, color='#3d3d4d', alpha=0.6)
        
        for i, lbl in enumerate(labels):
            start = (i * hlen) / engine.fs
            end = start + (flen / engine.fs)
            if lbl == 1: ax1.axvspan(start, end, color='#00ff41', alpha=0.2)
            elif lbl == 2: ax1.axvspan(start, end, color='#ffeb3b', alpha=0.2)
        
        ax1.set_title("Sinyal Segmentasyonu")
        ax2.plot(energies, color='#00f2ff')
        ax3.plot(zcrs, color='#ff4b2b')
        fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAnalyzerUI(root)
    root.mainloop()