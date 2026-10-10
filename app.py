import os
import json
import io
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# 1. API YAPILANDIRMASI
# ==========================================
app = FastAPI(
    title="Tohum Tanıma & Karakterizasyon API",
    description="Mobil uygulama için Çift Kademeli Tohum Teşhis Servisi",
    version="1.0.0"
)

# Mobil uygulamaların ve tarayıcıların sorunsuz bağlanması için CORS izni
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. MODEL VE KÜTÜPHANENİN YÜKLENMESİ
# ==========================================
MODEL_PATH = 'models/best_seed_model.pth'
LABELS_PATH = 'models/class_labels.json'
LIBRARY_PATH = 'models/seed_embeddings_library.pt'
IMAGE_SIZE = (224, 224)

device = torch.device("cpu")

if not os.path.exists(LABELS_PATH) or not os.path.exists(MODEL_PATH) or not os.path.exists(LIBRARY_PATH):
    raise RuntimeError("Gerekli model veya kütüphane dosyaları eksik!")

with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    class_labels = json.load(f)
num_classes = len(class_labels)

# Kategori Sınıflandırıcı Modeli
clf_model = models.mobilenet_v3_large(weights=None)
in_features = clf_model.classifier[3].in_features
clf_model.classifier[3] = nn.Linear(in_features, num_classes)
clf_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
clf_model.to(device)
clf_model.eval()

# 960 Boyutlu Özellik Çıkarıcı
class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return nn.functional.normalize(x, p=2, dim=1)

feature_extractor = FeatureExtractor(clf_model).to(device)
feature_extractor.eval()

# Referans Tohum Kütüphanesi
library = torch.load(LIBRARY_PATH, map_location='cpu')

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. ENDPOINT'LER
# ==========================================
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Tohum Tanıma API Servisi Hazır!",
        "total_reference_seeds": len(library)
    }

@app.post("/predict")
async def predict_seed(file: UploadFile = File(...), top_k: int = 3):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir görsel yükleyin.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        # 1. Aşama: Kategori Tespiti
        with torch.no_grad():
            cat_outputs = clf_model(tensor)
            cat_probs = torch.nn.functional.softmax(cat_outputs, dim=1)[0]
            top_cat_prob, top_cat_idx = torch.topk(cat_probs, 1)
            query_embedding = feature_extractor(tensor).cpu().squeeze(0)

        predicted_cat = class_labels[str(top_cat_idx[0].item())]
        confidence_pct = round(top_cat_prob[0].item() * 100, 2)

        # 2. Aşama: HİYERARŞİK / KATEGORİ FİLTRELİ BENZERLİK ARAMASI
        # Güven oranı %50'nin üzerindeyse arama yalnızca tahmin edilen kategoride yapılır
        target_indices = [
            i for i, item in enumerate(library)
            if item.get('category') == predicted_cat
        ]

        # Eğer kategoride tohum bulunamazsa veya güven çok düşükse tüm kütüphaneyi tara
        if not target_indices or confidence_pct < 50.0:
            target_indices = list(range(len(library)))

        selected_embeddings = torch.stack([library[i]['embedding'] for i in target_indices])
        similarities = torch.mv(selected_embeddings, query_embedding)

        top_scores, top_sub_indices = torch.topk(similarities, len(target_indices))

        seen_species = set()
        top_matches = []
        rank = 1

        for i in range(len(top_sub_indices)):
            actual_idx = target_indices[top_sub_indices[i].item()]
            item = library[actual_idx]
            spec_name = item['crop_name_tr']

            if spec_name in seen_species or spec_name == "Bilinmiyor":
                continue
            seen_species.add(spec_name)

            sim_pct = round(max(0.0, top_scores[i].item()) * 100, 2)
            top_matches.append({
                "rank": rank,
                "crop_name_tr": item['crop_name_tr'],
                "scientific_name": item['scientific_name'],
                "variety_description": item['variety'],
                "category": item['category'],
                "similarity_score": sim_pct,
                "reference_image": item['image_name']
            })
            rank += 1
            if len(top_matches) == top_k:
                break

        return {
            "success": True,
            "filename": file.filename,
            "predicted_category": predicted_cat,
            "category_confidence": confidence_pct,
            "top_candidates": top_matches
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İşlem hatası: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)