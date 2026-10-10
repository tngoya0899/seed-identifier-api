import os
import json
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms, models

# ==========================================
# 1. AYARLAR
# ==========================================
MODEL_PATH = 'models/best_seed_model.pth'
LABELS_PATH = 'models/class_labels.json'
METADATA_PATH = 'meta_data.xlsx'
OUTPUT_LIB_PATH = 'models/seed_embeddings_library.pt'
DATA_DIR = 'processed_data'
IMAGE_SIZE = (224, 224)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Embedding çıkarıcı birim: {device}")

# Sınıf haritası
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    class_labels = json.load(f)
num_classes = len(class_labels)

# Meta veri tablosu
df_meta = None
if os.path.exists(METADATA_PATH):
    df_meta = pd.read_excel(METADATA_PATH)
    print(f"Meta veri tablosu başarıyla yüklendi ({len(df_meta)} satır).")

# ==========================================
# 2. MODELİ ÖZELLİK ÇIKARICI OLARAK YÜKLE
# ==========================================
base_model = models.mobilenet_v3_large(weights=None)
in_features = base_model.classifier[3].in_features
base_model.classifier[3] = nn.Linear(in_features, num_classes)
base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

feature_extractor = FeatureExtractor(base_model).to(device)
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. KÜTÜPHANEYİ OLUŞTURMA
# ==========================================
print("\nReferans tohum kütüphanesi taranıyor...")
library = []

for split in ['train', 'val']:
    split_path = os.path.join(DATA_DIR, split)
    if not os.path.exists(split_path):
        continue

    for cat_name in os.listdir(split_path):
        cat_folder = os.path.join(split_path, cat_name)
        if not os.path.isdir(cat_folder):
            continue

        images = [f for f in os.listdir(cat_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in images:
            img_path = os.path.join(cat_folder, img_name)

            # Orijinal görsel adını yakala (augmentation eklerini temizle)
            base_img_name = img_name
            for suffix in ['_orig', '_rot90', '_rot180', '_rot270', '_hflip', '_vflip']:
                base_img_name = base_img_name.replace(suffix, '')

            crop_name_tr = "Bilinmiyor"
            scientific_name = "Bilinmiyor"
            variety = "Bilinmiyor"

            if df_meta is not None:
                # 1. Öncelik: Tam görsel adına göre eşleştir
                match = df_meta[df_meta['image_name'].astype(str).str.strip() == base_img_name.strip()]
                
                # 2. Öncelik: Dosya önek koduna göre eşleştir (örn: SEB_060)
                if match.empty:
                    prefix = "_".join(img_name.split("_")[:2])
                    match = df_meta[df_meta['image_name'].astype(str).str.startswith(prefix)]

                if not match.empty:
                    row = match.iloc[0]
                    crop_name_tr = str(row.get('crop_name_tr', 'Bilinmiyor'))
                    scientific_name = str(row.get('scientific_name', 'Bilinmiyor'))
                    variety = str(row.get('variety_and_image_description', 'Bilinmiyor'))

            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = feature_extractor(tensor).cpu().squeeze(0)

                library.append({
                    'image_name': img_name,
                    'category': cat_name,
                    'crop_name_tr': crop_name_tr,
                    'scientific_name': scientific_name,
                    'variety': variety,
                    'embedding': embedding
                })
            except Exception as e:
                print(f"Hata ({img_name}): {e}")

print(f"\nToplam {len(library)} adet tohum için referans parmak izi çıkarıldı.")
torch.save(library, OUTPUT_LIB_PATH)
print(f"Güncellenmiş kütüphane kaydedildi: {OUTPUT_LIB_PATH}")