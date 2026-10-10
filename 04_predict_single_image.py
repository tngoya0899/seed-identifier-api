import os
import json
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models

# ==========================================
# 1. AYARLAR VE MODEL YÜKLEME
# ==========================================
MODEL_PATH = 'models/best_seed_model.pth'
LABELS_PATH = 'models/class_labels.json'
METADATA_PATH = 'meta_data.xlsx'
IMAGE_SIZE = (224, 224)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Sınıf etiket haritasını yükle
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    class_labels = json.load(f)
num_classes = len(class_labels)

# Modeli yükle
model = models.mobilenet_v3_large(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Görsel ön işleme (Test dönüşümü)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Meta veri tablosunu yükle (isteğe bağlı tür doğrulaması için)
df_meta = None
if os.path.exists(METADATA_PATH):
    try:
        df_meta = pd.read_excel(METADATA_PATH)
    except Exception as e:
        print(f"Meta veri tablosu okunamadı: {e}")

# ==========================================
# 2. TAHMİN FONKSİYONU
# ==========================================
def predict_seed(image_path):
    if not os.path.exists(image_path):
        print(f"Hata: Görsel bulunamadı -> {image_path}")
        return

    # Görseli aç ve hazırla
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_prob, top_cat_idx = torch.topk(probabilities, 3)

    pred_cat = class_labels[str(top_cat_idx[0].item())]
    confidence = top_prob[0].item() * 100

    print("\n" + "="*50)
    print(f"ANALİZ EDİLEN GÖRSEL: {os.path.basename(image_path)}")
    print("="*50)
    print(f"TAHMİN EDİLEN KATEGORİ : {pred_cat}")
    print(f"GÜVEN ORANI (CONFIDENCE): %{confidence:.2f}")
    print("-"*50)
    print("En Olası İlk 3 Kategori:")
    for i in range(3):
        cat_name = class_labels[str(top_cat_idx[i].item())]
        prob = top_prob[i].item() * 100
        print(f" {i+1}. {cat_name:25} : %{prob:.2f}")
    print("="*50)

    # Dosya adından meta tablodaki gerçek tür bilgisini sorgula
    if df_meta is not None:
        raw_name = os.path.basename(image_path)
        # Dosya adının ilk iki parçasını al (örn: BAK_031)
        prefix = "_".join(raw_name.split("_")[:2])
        
        match = df_meta[df_meta['relative_path'].astype(str).str.contains(prefix, na=False, case=False)]
        if not match.empty:
            row = match.iloc[0]
            print("\n[META TABLO REFERANS BİLGİSİ]")
            print(f"Gerçek Tür (Common Name) : {row.get('common_name', 'Bilinmiyor')}")
            print(f"Varyete/Çeşit (Variety)  : {row.get('variety', 'Bilinmiyor')}")
            print(f"Gerçek Kategori          : {row.get('category', 'Bilinmiyor')}")
            print("="*50 + "\n")

# ==========================================
# 3. ÖRNEK TEST
# ==========================================
if __name__ == '__main__':
    import sys

    # Komut satırından argüman verildiyse onu kullan, verilmediyse kullanıcıya sor
    if len(sys.argv) > 1:
        img_path = sys.argv[1].strip('\"\'')
    else:
        print("\nTest etmek istediğin tohum görselinin yolunu yapıştır:")
        img_path = input("Görsel Yolu: ").strip('\"\'')

    predict_seed(img_path)