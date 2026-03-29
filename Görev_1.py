import numpy as np
import matplotlib.pyplot as plt

# 1. Parametrelerin Belirlenmesi
f0 = 128  

f1 = f0
f2 = f0 / 2
f3 = 10 * f0

# Örnekleme frekansı Nyquist kriterine uygun (fs > 2*f3)
fs = 100 * f0 

# 2. Zaman Dizileri ve Sinyal Üretimi
t1 = np.arange(0, 3/f1, 1/fs)
x1 = np.sin(2 * np.pi * f1 * t1)

t2 = np.arange(0, 3/f2, 1/fs)
x2 = np.sin(2 * np.pi * f2 * t2)

t3 = np.arange(0, 3/f3, 1/fs)
x3 = np.sin(2 * np.pi * f3 * t3)

# Toplam İşaret İçin Zaman Dizisi
t_max = 3 / f2 
t_sum = np.arange(0, t_max, 1/fs)

# Toplam sinyali tek satırda veya parantez içinde düzgünce tanımlayalım
xs = (np.sin(2 * np.pi * f1 * t_sum) + 
      np.sin(2 * np.pi * f2 * t_sum) + 
      np.sin(2 * np.pi * f3 * t_sum))

# 3. Görselleştirme (Subplot)
plt.figure(figsize=(10, 12))

plt.subplot(4, 1, 1)
plt.plot(t1, x1)
plt.title(f"f1 = {f1} Hz")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t2, x2)
plt.title(f"f2 = {f2} Hz")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t3, x3)
plt.title(f"f3 = {f3} Hz")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t_sum, xs)
plt.title("Toplam İşaret (f1 + f2 + f3)")
plt.xlabel("Zaman (s)")
plt.grid(True)

plt.tight_layout()
plt.show()