import os
import re
import json
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms, models

# =========================================================================
# 1. AYARLAR
# =========================================================================
MANIFEST_PATH = "clean_seed_dataset/processed_manifest.csv"
METADATA_PATH = "meta_data.xlsx"
MODEL_PATH = "models/best_species_model.pth"
LABELS_PATH = "models/species_labels.json"
OUTPUT_LIB_PATH = "models/seed_embeddings_library.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Embedding çıkarıcı birim: {device}")


def parse_variety_and_turkish_desc(raw_text):
    """
    'Hallmark (English desc / Türkçe morfolojik açıklama)' formatındaki metni
    çeşit adına ve saf Türkçe morfolojik açıklamaya ayırır.
    """
    text = str(raw_text).strip()
    if not text or text.lower() == "nan":
        return "Standart Form", "Standart tür örneği"

    if "(" in text:
        variety_part, paren_part = text.split("(", 1)
        variety_name = variety_part.strip()
        paren_part = paren_part.rstrip(")").strip()

        # En sondaki ' / ' ayıracından Türkçe bölümü al
        if " / " in paren_part:
            _, desc_tr = paren_part.rsplit(" / ", 1)
        else:
            desc_tr = paren_part
    else:
        variety_name = text
        desc_tr = text

    # Tekil kesilmiş tohumlarda kafa karıştıracak "6'lı / 10'lu / ölçekli" ifadeleri sadeleştir
    desc_tr = re.sub(r"\b\d+\s*mm\s*ölçekli\b", "", desc_tr, flags=re.IGNORECASE)
    desc_tr = re.sub(r"\s{2,}", " ", desc_tr).strip(" ,.-")

    return variety_name, desc_tr


# Sınıf haritası
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_labels = json.load(f)
num_classes = len(class_labels)

# Manifest ve Meta veri tablosunu yükle
manifest_df = pd.read_csv(MANIFEST_PATH)

meta_info_map = {}
if os.path.exists(METADATA_PATH):
    df_meta = pd.read_excel(METADATA_PATH)
    for _, r in df_meta.iterrows():
        sid = str(r.get("seed_id", "")).strip()
        if sid:
            v_name, d_tr = parse_variety_and_turkish_desc(r.get("variety_and_image_description", ""))
            meta_info_map[sid] = {
                "variety": v_name,
                "description_tr": d_tr
            }
    print(f"Meta veri tablosu yüklendi ve Türkçe açıklamalar ayrıştırıldı ({len(df_meta)} satır).")


# =========================================================================
# 2. MODELİ ÖZELLİK ÇIKARICI (FEATURE EXTRACTOR) OLARAK YÜKLE
# =========================================================================
base_model = models.mobilenet_v3_large(weights=None)
in_features = base_model.classifier[3].in_features
base_model.classifier[2] = nn.Dropout(p=0.45)
base_model.classifier[3] = nn.Linear(in_features, num_classes)
base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))


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


feature_extractor = FeatureExtractor(base_model).to(device)
feature_extractor.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =========================================================================
# 3. KÜTÜPHANEYİ OLUŞTURMA
# =========================================================================
ref_df = manifest_df[manifest_df["split"].isin(["train", "val"])].reset_index(drop=True)
print(f"\nReferans tohum kütüphanesi oluşturuluyor ({len(ref_df)} tekil tohum)...")

library = []

for _, row in ref_df.iterrows():
    img_path = row["saved_crop_path"]
    if not os.path.exists(img_path):
        continue

    seed_id = str(row.get("seed_id", "")).strip()
    info = meta_info_map.get(seed_id, {"variety": "Standart Form", "description_tr": "Standart tür örneği"})

    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = feature_extractor(tensor).cpu().squeeze(0)

        library.append({
            "seed_id": seed_id,
            "image_name": os.path.basename(img_path),
            "category": str(row.get("category", "Bilinmiyor")),
            "crop_name_tr": str(row.get("crop_name_tr", "Bilinmiyor")),
            "scientific_name": str(row.get("scientific_name", "Bilinmiyor")),
            "variety": info["variety"],
            "description_tr": info["description_tr"],
            "embedding": embedding
        })
    except Exception as e:
        print(f"[HATA] ({img_path}): {e}")

print(f"\nToplam {len(library)} adet tekil tohum için referans parmak izi çıkarıldı.")
torch.save(library, OUTPUT_LIB_PATH)
print(f"Güncellenmiş kütüphane kaydedildi: {OUTPUT_LIB_PATH}")