import os
import sys
import json
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models

# =========================================================================
# 1. AYARLAR VE MODEL YÜKLEME
# =========================================================================
MODEL_PATH = "models/best_species_model.pth"
LABELS_PATH = "models/species_labels.json"
METADATA_PATH = "meta_data.xlsx"
IMG_SIZE = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Sınıf etiket haritasını yükle
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_labels = json.load(f)
num_classes = len(class_labels)

# Modeli eğitimdeki mimariyle birebir aynı şekilde yükle
model = models.mobilenet_v3_large(weights=None)
in_features = model.classifier[3].in_features
model.classifier[2] = torch.nn.Dropout(p=0.45)
model.classifier[3] = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Tensör dönüşümü (Eğitimdeki val/test ile birebir aynı)
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Meta veri tablosunu yükle (Tahmin edilen türün kategorisini ve bilimsel adını bulmak için)
df_meta = None
species_info_map = {}
if os.path.exists(METADATA_PATH):
    try:
        df_meta = pd.read_excel(METADATA_PATH)
        for _, r in df_meta.iterrows():
            tr_name = str(r.get("crop_name_tr", "")).strip()
            if tr_name and tr_name not in species_info_map:
                species_info_map[tr_name] = {
                    "category": r.get("category", "Bilinmiyor"),
                    "scientific_name": r.get("scientific_name", "Bilinmiyor")
                }
    except Exception as e:
        print(f"[UYARI] Meta veri tablosu okunamadı: {e}")


# =========================================================================
# 2. EĞİTİMLE BİREBİR UYUMLU GÖRSEL ÖN İŞLEME
# =========================================================================
def prepare_seed_for_inference(image_path, target_size=224):
    """
    Ham bir test veya telefon fotoğrafını eğitim verisiyle aynı formata getirir:
    - En-boy oranını asla bozmaz, küçük tohumu bulanıklaştırmaz,
    - Tohumu merkezleyip kendi arka plan rengiyle 224x224 kareye oturtur.
    """
    orig_rgb = np.array(Image.open(image_path).convert("RGB"))
    h, w, _ = orig_rgb.shape

    # Eğer görüntü zaten eğitim setinden kesilmiş 224x224 ise doğrudan kullan
    if h == target_size and w == target_size:
        return Image.fromarray(orig_rgb)

    # Ham veritabanı veya kamera fotoğrafı ise telif kenarlarını at ve ana tohumu bul
    trimmed = orig_rgb[int(h * 0.05):int(h * 0.94), int(w * 0.02):int(w * 0.98)]
    th, tw, _ = trimmed.shape

    corners = np.vstack([
        trimmed[0:15, 0:15].reshape(-1, 3),
        trimmed[0:15, -15:].reshape(-1, 3),
        trimmed[-15:, 0:15].reshape(-1, 3),
        trimmed[-15:, -15:].reshape(-1, 3),
    ])
    native_bg_rgb = np.median(corners, axis=0)
    native_bg = tuple(native_bg_rgb.astype(np.uint8).tolist())

    lab = cv2.cvtColor(trimmed, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(np.uint8([[native_bg_rgb]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, otsu = cv2.threshold(cv2.GaussianBlur(dist_u8, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(cv2.GaussianBlur(trimmed, (5, 5), 0), 25, 90)
    combined = cv2.bitwise_or(otsu, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(closed)
    cv2.drawContours(solid_mask, contours, -1, 255, thickness=cv2.FILLED)

    # En büyük geçerli tohum bileşenini seç
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(solid_mask)
    best_crop = None
    best_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-5)
        extent = area / (bw * bh + 1e-5)
        if area > best_area and aspect_ratio < 6.5 and extent > 0.45:
            best_area = area
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            pad_x, pad_y = int(bw * 0.04), int(bh * 0.04)
            best_crop = trimmed[max(0, y - pad_y):min(th, y + bh + pad_y),
                                max(0, x - pad_x):min(tw, x + bw + pad_x)]

    if best_crop is None:
        best_crop = trimmed

    # pad_or_downscale_only mantığı
    ch, cw, _ = best_crop.shape
    max_dim = max(ch, cw)
    if max_dim > target_size:
        scale = target_size / max_dim
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        processed = cv2.resize(best_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        processed = best_crop
        new_h, new_w, _ = processed.shape

    canvas = np.full((target_size, target_size, 3), native_bg, dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = processed
    return Image.fromarray(canvas)


# =========================================================================
# 3. TAHMİN FONKSİYONU
# =========================================================================
def predict_seed(image_path):
    if not os.path.exists(image_path):
        print(f"Hata: Görsel bulunamadı -> {image_path}")
        return

    processed_pil = prepare_seed_for_inference(image_path, target_size=IMG_SIZE)
    tensor = tensor_transform(processed_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.topk(probabilities, min(3, num_classes))

    pred_species = class_labels[str(top_idx[0].item())]
    confidence = top_prob[0].item() * 100
    meta_info = species_info_map.get(pred_species, {"category": "Bilinmiyor", "scientific_name": "Bilinmiyor"})

    print("\n" + "=" * 60)
    print(f"ANALİZ EDİLEN GÖRSEL : {os.path.basename(image_path)}")
    print("=" * 60)
    print(f"TAHMİN EDİLEN TÜR    : {pred_species} ({meta_info['scientific_name']})")
    print(f"AİT OLDUĞU KATEGORİ  : {meta_info['category']}")
    print(f"GÜVEN ORANI          : %{confidence:.2f}")
    print("-" * 60)
    print("En Olası İlk 3 Tür Tahmini:")
    for i in range(len(top_idx)):
        sp_name = class_labels[str(top_idx[i].item())]
        sp_cat = species_info_map.get(sp_name, {}).get("category", "-")
        prob = top_prob[i].item() * 100
        print(f"  {i + 1}. {sp_name:28s} [{sp_cat:18s}] : %{prob:.2f}")
    print("=" * 60)

    # Eğer test edilen görselin dosya adı meta tabloda varsa gerçek etiketle karşılaştır
    if df_meta is not None:
        raw_name = os.path.basename(image_path)
        prefix = "_".join(raw_name.split("_")[:2])
        match = df_meta[df_meta["relative_path"].astype(str).str.contains(prefix, na=False, case=False)]
        if not match.empty:
            row = match.iloc[0]
            print("\n[META TABLO GERÇEK ETİKET KONTROLÜ]")
            print(f"Gerçek Tür (TR) : {row.get('crop_name_tr', 'Bilinmiyor')} ({row.get('scientific_name', '')})")
            print(f"Gerçek Kategori : {row.get('category', 'Bilinmiyor')}")
            print("=" * 60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1].strip("\"'")
    else:
        print("\nTest etmek istediğin tohum görselinin yolunu yapıştır:")
        img_path = input("Görsel Yolu: ").strip("\"'")

    predict_seed(img_path)