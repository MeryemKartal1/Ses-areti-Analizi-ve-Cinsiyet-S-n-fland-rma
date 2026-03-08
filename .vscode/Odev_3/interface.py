import tkinter as tk
from tkinter import filedialog
import main

class VAD_APP:

    def __init__(self, root):

        self.root = root
        self.root.title("Speech Analysis System")

        self.label = tk.Label(root, text="Ses Dosyası Seç")
        self.label.pack(pady=10)

        self.button = tk.Button(root, text="Dosya Aç", command=self.load_file)
        self.button.pack()

        self.result = tk.Text(root, height=15, width=60)
        self.result.pack(pady=10)

    ###############################################################

    def load_file(self):

        file = filedialog.askopenfilename(filetypes=[("WAV","*.wav")])

        if file:

            result = main.process(file)

            voiced_count = result.count("Voiced")
            unvoiced_count = result.count("Unvoiced")

            self.result.insert(tk.END,"\nAnaliz Sonucu\n")
            self.result.insert(tk.END,"-----------------------\n")
            self.result.insert(tk.END,f"Toplam Frame: {len(result)}\n")
            self.result.insert(tk.END,f"Voiced Frame: {voiced_count}\n")
            self.result.insert(tk.END,f"Unvoiced Frame: {unvoiced_count}\n\n")

            self.result.insert(tk.END,"İlk 50 Frame:\n")

            for i,r in enumerate(result[:50]):

                self.result.insert(tk.END,f"{i} -> {r}\n")


root = tk.Tk()
app = VAD_APP(root)
root.mainloop()