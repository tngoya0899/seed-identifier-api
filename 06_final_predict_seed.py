import os
import sys
import json
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# =========================================================================
# 1. AYARLAR VE MODEL YÜKLEME
# =========================================================================
MODEL_PATH = "models/best_species_model.pth"
LABELS_PATH = "models/species_labels.json"
LIBRARY_PATH = "models/seed_embeddings_library.pt"
IMG_SIZE = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_labels = json.load(f)
num_classes = len(class_labels)

# Modeli eğitimdeki mimariyle birebir aynı şekilde yükle
clf_model = models.mobilenet_v3_large(weights=None)
in_features = clf_model.classifier[3].in_features
clf_model.classifier[2] = nn.Dropout(p=0.45)
clf_model.classifier[3] = nn.Linear(in_features, num_classes)
clf_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
clf_model = clf_model.to(device)
clf_model.eval()


class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


feature_extractor = FeatureExtractor(clf_model).to(device)
feature_extractor.eval()

if not os.path.exists(LIBRARY_PATH):
    raise FileNotFoundError("Embedding kütüphanesi bulunamadı! Önce 05_build_seed_library.py çalıştırılmalı.")

library = torch.load(LIBRARY_PATH, map_location="cpu", weights_only=False)

# Tür -> Kategori ve Bilimsel İsim hızlı erişim haritası (Kütüphaneden otomatik derlenir)
species_meta_map = {}
for item in library:
    sp = item["crop_name_tr"]
    if sp not in species_meta_map:
        species_meta_map[sp] = {
            "category": item.get("category", "Bilinmiyor"),
            "scientific_name": item.get("scientific_name", "Bilinmiyor")
        }

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =========================================================================
# 2. EĞİTİMLE BİREBİR UYUMLU GÖRSEL ÖN İŞLEME
# =========================================================================
def prepare_seed_for_inference(image_path, target_size=224):
    """
    Ham fotoğrafı en-boy oranını bozmadan merkezleyip kendi arka plan rengiyle
    224x224 kareye oturtur.
    """
    orig_rgb = np.array(Image.open(image_path).convert("RGB"))
    h, w, _ = orig_rgb.shape

    if h == target_size and w == target_size:
        return Image.fromarray(orig_rgb)

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
# 3. HİBRİT TAHMİN FONKSİYONU (SINIFLANDIRICI + VEKTÖR BENZERLİĞİ)
# =========================================================================
def predict_seed_full(image_path, top_k=3):
    if not os.path.exists(image_path):
        print(f"Hata: Görsel bulunamadı -> {image_path}")
        return

    processed_pil = prepare_seed_for_inference(image_path, target_size=IMG_SIZE)
    tensor = tensor_transform(processed_pil).unsqueeze(0).to(device)

    # 1. KADEME: Doğrudan Tür Tahmini (Top-3) ve Parmak İzi Çıkarımı
    with torch.no_grad():
        outputs = clf_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, min(top_k, num_classes))
        query_embedding = feature_extractor(tensor).cpu().squeeze(0)

    top1_species = class_labels[str(top_idxs[0].item())]
    top1_conf = top_probs[0].item() * 100
    top1_meta = species_meta_map.get(top1_species, {"category": "Bilinmiyor", "scientific_name": "Bilinmiyor"})

    # 2. KADEME: Vektör Benzerliği ile En Yakın Varyete ve Referans Örnekler
    # Öncelikle tahmin edilen ilk 3 türün adayları içinde ara; boşsa tüm kütüphanede ara
    top3_species_set = {class_labels[str(idx.item())] for idx in top_idxs}
    candidates = [item for item in library if item["crop_name_tr"] in top3_species_set]
    if not candidates:
        candidates = library

    cand_embeddings = torch.stack([item["embedding"] for item in candidates])
    similarities = torch.mv(cand_embeddings, query_embedding)
    top_scores, top_indices = torch.topk(similarities, min(top_k, len(candidates)))

    print("\n" + "=" * 72)
    print(f"ANALİZ EDİLEN TOHUM : {os.path.basename(image_path)}")
    print("=" * 72)
    print(f"[1. AŞAMA - SINIFLANDIRICI TAHMİNİ]")
    print(f"  Birincil Tür  : {top1_species} ({top1_meta['scientific_name']})")
    print(f"  Ana Kategori  : {top1_meta['category']}")
    print(f"  Güven Oranı   : %{top1_conf:.2f}")
    print("-" * 72)
    print("  En Olası İlk 3 Tür:")
    for i in range(len(top_idxs)):
        sp = class_labels[str(top_idxs[i].item())]
        cat = species_meta_map.get(sp, {}).get("category", "-")
        conf = top_probs[i].item() * 100
        print(f"    {i + 1}. {sp:28s} [{cat:18s}] : %{conf:.2f}")

    print("-" * 72)
    print("[2. AŞAMA - VEKTÖR BENZERLİĞİ (EN YAKIN REFERANS VE VARYETELER)]")
    for i in range(len(top_indices)):
        idx = top_indices[i].item()
        match_item = candidates[idx]
        sim_pct = max(0.0, top_scores[i].item()) * 100

        print(f"\n  #{i + 1} Görsel Eşleşme (Benzerlik: %{sim_pct:.2f})")
        print(f"     Türkçe Adı      : {match_item['crop_name_tr']}")
        print(f"     Bilimsel Adı    : {match_item['scientific_name']}")
        print(f"     Kategori        : {match_item['category']}")
        print(f"     Varyete / Tanım : {match_item['variety']}")
        print(f"     Referans Görsel : {match_item['image_name']}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1].strip("\"'")
    else:
        print("\nTest etmek istediğin tohum görselinin yolunu yapıştır:")
        img_path = input("Görsel Yolu: ").strip("\"'")

    predict_seed_full(img_path)