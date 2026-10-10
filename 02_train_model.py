import os
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ==========================================
# 1. TEMEL AYARLAR (OVERFITTING ENGELLEME)
# ==========================================
torch.set_num_threads(4)

DATA_DIR = 'processed_data'
MODEL_SAVE_DIR = 'models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.0002   # Ezberi önlemek için daha hassas öğrenme hızı
IMAGE_SIZE = (224, 224)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Eğitim cihazı: {device}")

# ==========================================
# 2. VERİ DÖNÜŞÜMLERİ
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'val', 'test']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=0)
    for x in ['train', 'val', 'test']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

print(f"Kategoriler ({num_classes}): {class_names}")
print(f"Görsel Sayıları: Train={dataset_sizes['train']}, Val={dataset_sizes['val']}, Test={dataset_sizes['test']}\n")

with open(os.path.join(MODEL_SAVE_DIR, 'class_labels.json'), 'w', encoding='utf-8') as f:
    json.dump({idx: name for idx, name in enumerate(class_names)}, f, ensure_ascii=False, indent=4)

# ==========================================
# 3. MOBILENET_V3 (DONDURULMUŞ ÖZELLİK ÇIKARICI)
# ==========================================
print("MobileNetV3 Modeli yükleniyor...")
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

# Ana omurgayı dondur: Sadece son karar katmanını eğiteceğiz (Ezberlemeyi engeller)
for param in model.features.parameters():
    param.requires_grad = False

# Sınıflandırıcı başlığını güçlendir ve Dropout ekle
in_features = model.classifier[3].in_features
model.classifier[2] = nn.Dropout(p=0.4)  # Aşırı ezberlemeyi kesen güçlü dropout
model.classifier[3] = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Sadece gradyanı açık olan parametreleri optimize et + weight_decay ile regülarizasyon
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# ==========================================
# 4. EĞİTİM DÖNGÜSÜ
# ==========================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=15):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_model_path = os.path.join(MODEL_SAVE_DIR, 'best_seed_model.pth')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 25)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

            print(f"{phase.upper():5} Loss: {epoch_loss:.4f} | Acc: %{epoch_acc * 100:.2f}")

            if phase == 'val':
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), best_model_path)
                    print(f"--> [KAYDEDİLDİ] En iyi model başarımı: %{best_acc * 100:.2f}")

        print()

    total_time = time.time() - start_time
    print(f"Eğitim tamamlandı! Süre: {total_time // 60:.0f}dk {total_time % 60:.0f}sn")
    print(f"En İyi Val Doğruluğu: %{best_acc * 100:.2f}")

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    trained_model = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

    print("\n" + "="*50)
    print("TEST SETİ İLE NİHAİ DEĞERLENDİRME YAPILIYOR...")
    print("="*50)
    trained_model.eval()
    test_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)

    test_acc = (test_corrects.double() / dataset_sizes['test']).item()
    print(f"\nFinal Test Seti Doğruluğu: %{test_acc * 100:.2f}")
    print("="*50)