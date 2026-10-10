import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. AYARLAR VE MODELİ YÜKLEME
# ==========================================
DATA_DIR = 'processed_data'
MODEL_PATH = 'models/best_seed_model.pth'
LABELS_PATH = 'models/class_labels.json'
OUTPUT_DIR = 'reports'
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Değerlendirme birimi: {device}")

# Sınıf etiket haritasını oku
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    class_labels_dict = json.load(f)
class_names = [class_labels_dict[str(i)] for i in range(len(class_labels_dict))]
num_classes = len(class_names)

# Test veri dönüşümleri
test_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Modeli oluştur ve eğitilmiş en iyi ağırlıkları yükle
print("En iyi model ağırlıkları yükleniyor...")
model = models.mobilenet_v3_large(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ==========================================
# 2. TEST KÜMESİNDE TAHMİNLERİ TOPLAMA
# ==========================================
print("\nTest kümesi üzerinde tahminler alınıyor...")
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ==========================================
# 3. METRİK VE RAPOR ÇIKTISI
# ==========================================
print("\n" + "="*60)
print("SINIFLANDIRMA RAPORU (PRECISION, RECALL, F1-SCORE)")
print("="*60)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)

# Raporu metin dosyası olarak kaydet
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

# ==========================================
# 4. CONFUSION MATRIX ÇİZİMİ VE KAYDI
# ==========================================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=class_names, 
    yticklabels=class_names,
    cbar=True
)

plt.title('Tohum Sınıflandırma Karmaşıklık Matrisi (Confusion Matrix)', fontsize=14, pad=15)
plt.xlabel('Tahmin Edilen Sınıf (Predicted)', fontsize=12)
plt.ylabel('Gerçek Sınıf (True)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

cm_save_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_save_path, dpi=300)
plt.close()

print(f"\n[BAŞARILI]")
print(f"1. Rapor kaydedildi: {os.path.join(OUTPUT_DIR, 'classification_report.txt')}")
print(f"2. Matris grafiği kaydedildi: {cm_save_path}")
print("="*60)