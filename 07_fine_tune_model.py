import os
import json
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==========================================
# 1. AYARLAR
# ==========================================
DATA_DIR = 'processed_data'
INITIAL_MODEL_PATH = 'models/best_seed_model.pth'
FINETUNED_MODEL_PATH = 'models/best_seed_model.pth'  # Üzerine yazarak güncelleyecek
LABELS_PATH = 'models/class_labels.json'
BATCH_SIZE = 32
EPOCHS = 6
IMAGE_SIZE = (224, 224)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"İnce ayar eğitim birimi: {device}")

# ==========================================
# 2. VERİ YÜKLEYİCİLERİ
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=0)
    for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(f"Eğitim görseli: {dataset_sizes['train']}, Doğrulama görseli: {dataset_sizes['val']}")

# ==========================================
# 3. MODELİ YÜKLE VE SEÇİLİ KATMANLARIN KİLİDİNİ AÇ
# ==========================================
model = models.mobilenet_v3_large(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, num_classes)

# Önceki en iyi ağırlıkları yükle
model.load_state_dict(torch.load(INITIAL_MODEL_PATH, map_location=device))
print("Önceki eğitilmiş model ağırlıkları başarıyla yüklendi.")

# Önce tüm omurgayı dondur
for param in model.features.parameters():
    param.requires_grad = False

# Sadece son derin evrişim bloklarının (12, 13, 14, 15, 16) kilidini aç
# Bu katmanlar tohumun mikroskobik doku, desen ve pürüzlerini öğrenir
trainable_blocks = [12, 13, 14, 15, 16]
for block_idx in trainable_blocks:
    if block_idx < len(model.features):
        for param in model.features[block_idx].parameters():
            param.requires_grad = True

# Sınıflandırıcı katman zaten eğitilebilir olmalı
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params_count = sum(p.numel() for p in model.parameters())
print(f"Eğitilebilir parametre sayısı: {trainable_params_count:,} / {total_params_count:,}")

# ==========================================
# 4. DİFERANSİYEL ÖĞRENME ORANI VE OPTIMIZER
# ==========================================
# Omurga katmanlarına çok hassas (1e-5), başlığa (5e-5) öğrenme oranı veriyoruz
optimizer = torch.optim.AdamW([
    {'params': [p for idx in trainable_blocks if idx < len(model.features) for p in model.features[idx].parameters()], 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 5e-5}
], weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ==========================================
# 5. İNCE AYAR EĞİTİM DÖNGÜSÜ
# ==========================================
print("\n" + "="*50)
print(f"İNCE AYAR (FINE-TUNING) BAŞLIYOR: {EPOCHS} EPOCH")
print("="*50)

best_acc = 0.0
start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 30)

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

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f"{phase.upper():5} - Kayıp (Loss): {epoch_loss:.4f} | Doğruluk (Acc): {epoch_acc*100:.2f}%")

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
            print(f" -> En iyi model güncellendi! Yeni Val Acc: {best_acc*100:.2f}%")

total_duration = time.time() - start_time
print("\n" + "="*50)
print(f"İnce ayar tamamlandı! Süre: {total_duration/60:.2f} dk")
print(f"En yüksek doğrulama başarısı: {best_acc*100:.2f}%")
print("="*50)