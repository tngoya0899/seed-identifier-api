import os
import shutil
import cv2
import numpy as np

DATASET_DIR = "clean_seed_dataset"
INSPECT_DIR = "supheli_gorseller_kontrol"

if os.path.exists(INSPECT_DIR):
    shutil.rmtree(INSPECT_DIR)
os.makedirs(INSPECT_DIR, exist_ok=True)

suspicious_count = 0
total_checked = 0

for root, _, files in os.walk(DATASET_DIR):
    for fname in files:
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        total_checked += 1
        fpath = os.path.join(root, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        # Merkezdeki nesneye odaklan (kenar arka planını hariç tut)
        h, w, _ = img.shape
        center = img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))

        # Renk doygunluğu çok düşükse (siyah-beyaz mühür, logo, cetvel vb.)
        if mean_sat < 18.0:
            split_name = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            copy_name = f"{split_name}__{fname}"
            shutil.copy2(fpath, os.path.join(INSPECT_DIR, copy_name))
            suspicious_count += 1

print(f"Toplam {total_checked} görsel tarandı (hiçbir orijinal dosyaya dokunulmadı).")
print(f"Şüpheli bulunan {suspicious_count} görsel '{INSPECT_DIR}/' klasörüne kopyalandı.")