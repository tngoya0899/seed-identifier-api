import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# ==========================================
# 1. AYARLAR VE MODEL YÜKLEME
# ==========================================
MODEL_PATH = 'models/best_seed_model.pth'
LABELS_PATH = 'models/class_labels.json'
LIBRARY_PATH = 'models/seed_embeddings_library.pt'
IMAGE_SIZE = (224, 224)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    class_labels = json.load(f)
num_classes = len(class_labels)

clf_model = models.mobilenet_v3_large(weights=None)
in_features = clf_model.classifier[3].in_features
clf_model.classifier[3] = nn.Linear(in_features, num_classes)
clf_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
clf_model = clf_model.to(device)
clf_model.eval()

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

feature_extractor = FeatureExtractor(clf_model).to(device)
feature_extractor.eval()

if not os.path.exists(LIBRARY_PATH):
    raise FileNotFoundError("Embedding kütüphanesi bulunamadı! Önce 05_build_seed_library.py çalıştırılmalı.")

library = torch.load(LIBRARY_PATH, map_location='cpu')

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 2. TAHMİN FONKSİYONU
# ==========================================
def predict_seed_full(image_path, top_k=3):
    if not os.path.exists(image_path):
        print(f"Hata: Görsel bulunamadı -> {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    # 1. KADEME: Kategori
    with torch.no_grad():
        cat_outputs = clf_model(tensor)
        cat_probs = torch.nn.functional.softmax(cat_outputs, dim=1)[0]
        top_cat_prob, top_cat_idx = torch.topk(cat_probs, 1)
        query_embedding = feature_extractor(tensor).cpu().squeeze(0)

    predicted_cat = class_labels[str(top_cat_idx[0].item())]
    cat_confidence = top_cat_prob[0].item() * 100

    # 2. KADEME: Benzerlik Analizi
    candidates = [item for item in library if item['category'] == predicted_cat]
    if not candidates:
        candidates = library

    cand_embeddings = torch.stack([item['embedding'] for item in candidates])
    similarities = torch.mv(cand_embeddings, query_embedding)
    top_scores, top_indices = torch.topk(similarities, min(top_k, len(candidates)))

    print("\n" + "="*70)
    print(f"ANALİZ EDİLEN TOHUM : {os.path.basename(image_path)}")
    print("="*70)
    print(f"[1. AŞAMA - KATEGORİ] : {predicted_cat} (Güven: %{cat_confidence:.2f})")
    print("-" * 70)
    print(f"[2. AŞAMA - EN YAKIN TÜR VE VARYETELER]")
    
    for i in range(len(top_indices)):
        idx = top_indices[i].item()
        match_item = candidates[idx]
        sim_pct = max(0.0, top_scores[i].item()) * 100

        print(f"\n #{i+1} Eşleşme (Benzerlik: %{sim_pct:.2f})")
        print(f"    Türkçe Adı      : {match_item['crop_name_tr']}")
        print(f"    Bilimsel Adı    : {match_item['scientific_name']}")
        print(f"    Varyete / Tanım : {match_item['variety']}")
        print(f"    Referans Görsel : {match_item['image_name']}")
    print("="*70 + "\n")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1].strip('\"\'')
    else:
        print("\nTest etmek istediğin tohum görselinin yolunu yapıştır:")
        img_path = input("Görsel Yolu: ").strip('\"\'')

    predict_seed_full(img_path)