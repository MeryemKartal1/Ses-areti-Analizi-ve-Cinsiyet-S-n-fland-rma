import os
import glob

def clear_temp_files():
    patterns = ["sonuc_*.wav"]
    for pattern in patterns:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except:
                pass