import io
import os
import re
import json
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# =========================================================================
# 1. UYGULAMA VE YOL AYARLARI
# =========================================================================
app = FastAPI(
    title="TÜBİTAK Tohum Teşhis ve Morfoloji API",
    description="İki Aşamalı Ensemble (Kategori + Tür + Vektör Benzerliği) API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CAT_LABELS_PATH = "models/category_labels.json"
SPECIES_LABELS_PATH = "models/species_labels.json"
CAT_MODEL1_PATH = "models/best_category_model.pth"
CAT_MODEL2_PATH = "models/best_category_pool_model.pth"
SPECIES_MODEL_PATH = "models/best_species_model.pth"
LIBRARY_PATH = "models/seed_embeddings_library.pt"


# =========================================================================
# 2. MODEL MİMARİLERİ VE YÜKLEME
# =========================================================================
with open(CAT_LABELS_PATH, "r", encoding="utf-8") as f:
    cat_labels_dict = json.load(f)
cat_classes = [cat_labels_dict[str(i)] for i in range(len(cat_labels_dict))]
num_cat_classes = len(cat_classes)

with open(SPECIES_LABELS_PATH, "r", encoding="utf-8") as f:
    species_labels_dict = json.load(f)
species_classes = [species_labels_dict[str(i)] for i in range(len(species_labels_dict))]
species_to_idx = {name: i for i, name in enumerate(species_classes)}
num_species_classes = len(species_classes)

# Model 1: Kategori Uzmanı 1 (MobileNetV3-Large)
cat_model_1 = models.mobilenet_v3_large(weights=None)
cat_model_1.classifier[2] = nn.Dropout(p=0.55)
cat_model_1.classifier[3] = nn.Linear(cat_model_1.classifier[3].in_features, num_cat_classes)
cat_model_1.load_state_dict(torch.load(CAT_MODEL1_PATH, map_location=device))
cat_model_1 = cat_model_1.to(device).eval()


# Model 2: Kategori Uzmanı 2 (EfficientNet-B0 + Avg/Max Pool)
class SeedPoolClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(2560),
            nn.Dropout(p=0.6),
            nn.Linear(2560, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        combined = torch.cat(
            [torch.flatten(self.avgpool(feat), 1), torch.flatten(self.maxpool(feat), 1)],
            dim=1
        )
        return self.head(combined)


cat_model_2 = SeedPoolClassifier(num_cat_classes)
cat_model_2.load_state_dict(torch.load(CAT_MODEL2_PATH, map_location=device))
cat_model_2 = cat_model_2.to(device).eval()

# Model 3: 120 Tür Sınıflandırıcı ve 960-Boyutlu Özellik Çıkarıcı
species_model = models.mobilenet_v3_large(weights=None)
species_model.classifier[2] = nn.Dropout(p=0.45)
species_model.classifier[3] = nn.Linear(species_model.classifier[3].in_features, num_species_classes)
species_model.load_state_dict(torch.load(SPECIES_MODEL_PATH, map_location=device))
species_model = species_model.to(device).eval()


class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return nn.functional.normalize(x, p=2, dim=1)


feature_extractor = FeatureExtractor(species_model).to(device).eval()

# Referans Kütüphanesini Yükle (1763 Tekil Tohum)
seed_library = torch.load(LIBRARY_PATH, map_location="cpu")
library_embeddings = torch.stack([item["embedding"] for item in seed_library]).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =========================================================================
# 3. GÖRÜNTÜ ÖN İŞLEME (OTOMATİK TOHUM MERKEZLEME)
# =========================================================================
def auto_crop_seed(pil_img: Image.Image, target_size: int = 224) -> Image.Image:
    w, h = pil_img.size
    aspect = w / float(h)

    # Eğer görsel zaten ön işlemeden geçmiş kare (224x224 vb.) ise tekrar Otsu ile kırpma!
    if 0.85 <= aspect <= 1.15 and max(w, h) <= 320:
        return pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_mean = float(np.mean(border_pixels))
    border_rgb = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]], axis=0)
    bg_color = tuple(int(c) for c in np.median(border_rgb, axis=0))

    thresh_type = cv2.THRESH_BINARY_INV if bg_mean > 127 else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > (h * w * 0.005)]

    if valid:
        largest = max(valid, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        pad_x, pad_y = int(bw * 0.15), int(bh * 0.15)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
        cropped = pil_img.crop((x1, y1, x2, y2))
    else:
        cropped = pil_img

    cw, ch = cropped.size
    max_side = max(cw, ch)
    canvas = Image.new("RGB", (max_side, max_side), bg_color)
    canvas.paste(cropped, ((max_side - cw) // 2, (max_side - ch) // 2))
    return canvas.resize((target_size, target_size), Image.Resampling.LANCZOS)


# =========================================================================
# 4. API ENDPOINTLERİ
# =========================================================================
@app.get("/")
def health_check():
    return {
        "status": "online",
        "device": str(device),
        "library_size": len(seed_library),
        "categories": num_cat_classes,
        "species_count": num_species_classes
    }


@app.post("/predict")
async def predict_seed(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir görsel yükleyin.")

    try:
        contents = await file.read()
        raw_img = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_img = auto_crop_seed(raw_img, target_size=224)

        # TTA: Orijinal + Yatay Çevrilmiş
        x1 = transform(processed_img).unsqueeze(0).to(device)
        x2 = transform(processed_img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)

        with torch.no_grad():
            # 1. AŞAMA: Kategori Ensemble + TTA (%85.47 Top-3)
            p_cat1 = (torch.softmax(cat_model_1(x1), dim=1) + torch.softmax(cat_model_1(x2), dim=1)) / 2.0
            p_cat2 = (torch.softmax(cat_model_2(x1), dim=1) + torch.softmax(cat_model_2(x2), dim=1)) / 2.0
            cat_probs = (0.65 * p_cat1 + 0.35 * p_cat2)[0]

            # Az örnekli Tibbi_ve_Aromatik_Bitkiler sınıfının yanlış pozitif sapmasını dengele
            if "Tibbi_ve_Aromatik_Bitkiler" in cat_classes:
                tibbi_idx = cat_classes.index("Tibbi_ve_Aromatik_Bitkiler")
                cat_probs[tibbi_idx] *= 0.45
                cat_probs = cat_probs / cat_probs.sum()

            top3_cat_vals, top3_cat_idxs = torch.topk(cat_probs, k=min(3, num_cat_classes))
            top3_category_names = {cat_classes[idx.item()]: val.item() for idx, val in zip(top3_cat_idxs, top3_cat_vals)}

            # 2. AŞAMA: 120 Tür Olasılığı + 960D Embedding Benzerliği
            sp_probs = ((torch.softmax(species_model(x1), dim=1) + torch.softmax(species_model(x2), dim=1)) / 2.0)[0]
            q_emb = nn.functional.normalize(feature_extractor(x1) + feature_extractor(x2), p=2, dim=1)

            cos_sims = torch.matmul(library_embeddings, q_emb.squeeze(0))

            # Sadece 1. Aşama Ensemble'ın onayladığı Top-3 kategori içindeki tohumlarda ara
            hybrid_scores = []
            for idx, item in enumerate(seed_library):
                item_cat = item["category"]
                if item_cat not in top3_category_names:
                    continue

                sim_score = float(cos_sims[idx].item())
                sp_name = item["crop_name_tr"]
                sp_prob = float(sp_probs[species_to_idx[sp_name]].item()) if sp_name in species_to_idx else 0.0
                cat_prob = top3_category_names[item_cat]

                total_score = (0.25 * sim_score) + (0.55 * sp_prob) + (0.20 * cat_prob)
                hybrid_scores.append((total_score, sim_score, sp_prob, cat_prob, idx))

            hybrid_scores.sort(key=lambda x: x[0], reverse=True)

            top_matches = []
            seen_species = set()
            for _, sim_score, sp_prob, cat_prob, lib_idx in hybrid_scores:
                item = seed_library[lib_idx]
                sp_name = item["crop_name_tr"]
                if sp_name in seen_species:
                    continue
                seen_species.add(sp_name)

                # "3'lü grup", "tekil" vb. çekim ifadelerini sadeleştirip standart "tohum örneği" yap
                # Sadece "3'lü grup", "4 tohum grubu", "tekil çekim" gibi toplu/çekim ifadelerini temizle
                raw_desc = item.get("description_tr", "Standart tür örneği")
                clean_desc = re.sub(
                    r"\b(?:\d+['’]?(?:l[ıiuü]|luk|adet)?\s+)?(?:tekil\s+|çoklu\s+|popülasyon\s+)?(?:aken|tohumu?|tohumcuk|dane)?\s*(?:grubu|dizilimi|varyasyonu|örneği)\b",
                    "",
                    raw_desc,
                    flags=re.IGNORECASE
                )
                clean_desc = re.sub(r"\b(?:tekil|çoklu)\s+çekimi?\b", "", clean_desc, flags=re.IGNORECASE)
                clean_desc = re.sub(r"\b\d+['’]?(?:l[ıiuü]|luk|adet)?(?:\s+tohum)?\b", "", clean_desc, flags=re.IGNORECASE)
                clean_desc = re.sub(r"\s*,\s*,+", ",", clean_desc)
                clean_desc = re.sub(r":\s*,", ":", clean_desc)
                clean_desc = re.sub(r"\s{2,}", " ", clean_desc).strip(" ,.-")
                if clean_desc and not clean_desc.lower().endswith("tohum örneği"):
                    clean_desc = f"{clean_desc} tohum örneği"

                top_matches.append({
                    "rank": len(top_matches) + 1,
                    "crop_name_tr": sp_name,
                    "scientific_name": item["scientific_name"],
                    "variety": item.get("variety", "Standart Form"),
                    "variety_description": clean_desc or "Standart tohum örneği",
                    "category": item["category"],
                    "similarity_score": round(max(0.0, sim_score) * 100, 2),
                    "classifier_confidence": round(sp_prob * 100, 2),
                    "reference_image": item["image_name"]
                })
                if len(top_matches) == 3:
                    break

        best_match = top_matches[0]
        final_category = best_match["category"]
        final_cat_conf = round(top3_category_names[final_category] * 100, 2)

        # Gerçekçi güven kontrolü
        is_reliable = (
            (best_match["classifier_confidence"] >= 25.0)
            or (final_cat_conf >= 25.0 and best_match["similarity_score"] >= 70.0)
        )
        warning_msg = None if is_reliable else "Düşük güven oranı: Benzer desenli türler karışabilir; lütfen tohumu farklı açıdan ve net odakta tekrar çekin."

        return {
            "success": True,
            "is_reliable": is_reliable,
            "warning": warning_msg,
            "filename": file.filename,
            "predicted_species": best_match["crop_name_tr"],
            "species_confidence": best_match["similarity_score"],
            "predicted_category": final_category,
            "category_confidence": final_cat_conf,
            "top_candidates": top_matches
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem hatası: {str(e)}")