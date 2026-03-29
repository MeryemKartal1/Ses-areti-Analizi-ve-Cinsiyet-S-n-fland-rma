import numpy as np

import tkinter as tk

import matplotlib.pyplot as plt

import sounddevice as sd
 
# DTMF frekans tablosu

dtmf = {

    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "A": (697, 1633),

    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477), "B": (770, 1633),

    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "C": (852, 1633),

    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477), "D": (941, 1633),

}
 
fs = 8000  # örnekleme frekansı

duration = 0.5  # saniye
 
 
def generate_tone(low, high):

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    signal = np.sin(2 * np.pi * low * t) + np.sin(2 * np.pi * high * t)

    signal = signal / np.max(np.abs(signal))

    return t, signal
 
 
def play_and_plot(key):

    low, high = dtmf[key]

    t, signal = generate_tone(low, high)
 
    # sesi çal

    sd.play(signal, fs)
 
    # grafiği çiz

    plt.figure(figsize=(6, 3))

    plt.plot(t[:400], signal[:400])

    plt.title(f"Tuş: {key} ({low}Hz + {high}Hz)")

    plt.xlabel("Zaman (s)")

    plt.ylabel("Genlik")

    plt.grid(True)

    plt.show()
 
 
# GUI

root = tk.Tk()

root.title("DTMF Telefon Tuş Takımı")
 
buttons = [

    ["1", "2", "3", "A"],

    ["4", "5", "6", "B"],

    ["7", "8", "9", "C"],

    ["*", "0", "#", "D"],

]
 
for r, row in enumerate(buttons):

    for c, key in enumerate(row):

        btn = tk.Button(

            root,

            text=key,

            width=6,

            height=3,

            command=lambda k=key: play_and_plot(k),

        )

        btn.grid(row=r, column=c, padx=5, pady=5)
 
root.mainloop()

 