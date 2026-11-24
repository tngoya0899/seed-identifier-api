import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# 1. AYARLAR
# =========================================================================
MANIFEST_PATH = "clean_seed_dataset/processed_manifest.csv"
INITIAL_MODEL_PATH = "models/best_species_model.pth"
FINETUNED_MODEL_PATH = "models/best_species_model.pth"
LABELS_PATH = "models/species_labels.json"
TARGET_COL = "crop_name_tr"

BATCH_SIZE = 16
EPOCHS = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"İnce ayar (Fine-Tuning) eğitim birimi: {device}")


# =========================================================================
# 2. MANIFEST TABANLI DATASET VE YÜKLEYİCİLER
# =========================================================================
class SeedFineTuneDataset(Dataset):
    def __init__(self, df, class_to_idx, target_col, transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.target_col = target_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["saved_crop_path"]).convert("RGB")
        label = self.class_to_idx[row[self.target_col]]
        if self.transform:
            image = self.transform(image)
        return image, label


with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_labels_dict = json.load(f)

class_names = [class_labels_dict[str(i)] for i in range(len(class_labels_dict))]
class_to_idx = {name: i for i, name in enumerate(class_names)}
num_classes = len(class_names)

manifest_df = pd.read_csv(MANIFEST_PATH)
train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

datasets_dict = {
    "train": SeedFineTuneDataset(train_df, class_to_idx, TARGET_COL, data_transforms["train"]),
    "val": SeedFineTuneDataset(val_df, class_to_idx, TARGET_COL, data_transforms["val"])
}

dataloaders = {
    x: DataLoader(datasets_dict[x], batch_size=BATCH_SIZE, shuffle=(x == "train"), num_workers=0)
    for x in ["train", "val"]
}

print(f"Eğitim görseli: {len(train_df)} | Doğrulama görseli: {len(val_df)} | Sınıf sayısı: {num_classes}")


# =========================================================================
# 3. MODELİ YÜKLE VE SEÇİLİ DERİN BLOKLARIN KİLİDİNİ AÇ
# =========================================================================
model = models.mobilenet_v3_large(weights=None)
in_features = model.classifier[3].in_features
model.classifier[2] = nn.Dropout(p=0.45)
model.classifier[3] = nn.Linear(in_features, num_classes)

model.load_state_dict(torch.load(INITIAL_MODEL_PATH, map_location=device))
print("Önceki eğitilmiş model ağırlıkları başarıyla yüklendi.")

# Tüm omurgayı dondur, ardından son 6 derin evrişim bloğunu aç
for param in model.features.parameters():
    param.requires_grad = False

trainable_blocks = [11, 12, 13, 14, 15, 16]
for block_idx in trainable_blocks:
    if block_idx < len(model.features):
        for param in model.features[block_idx].parameters():
            param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params_count = sum(p.numel() for p in model.parameters())
print(f"Eğitilebilir parametre sayısı: {trainable_params_count:,} / {total_params_count:,}")


# =========================================================================
# 4. DİFERANSİYEL ÖĞRENME ORANI VE SINIF AĞIRLIKLARI
# =========================================================================
optimizer = torch.optim.AdamW([
    {"params": [p for idx in trainable_blocks if idx < len(model.features) for p in model.features[idx].parameters()], "lr": 1e-5},
    {"params": model.classifier.parameters(), "lr": 5e-5}
], weight_decay=1e-3)

class_counts = train_df[TARGET_COL].value_counts()
weights = [1.0 / np.sqrt(class_counts.get(cls_name, 1)) for cls_name in class_names]
class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
class_weights_tensor = class_weights_tensor / class_weights_tensor.mean()

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)


# =========================================================================
# 5. BAŞLANGIÇ DOĞRULUK ÖLÇÜMÜ VE İNCE AYAR DÖNGÜSÜ
# =========================================================================
def evaluate_val_accuracy(m):
    m.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = m(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
    return (corrects.double() / len(datasets_dict["val"])).item()


best_acc = evaluate_val_accuracy(model)
print(f"\nBaşlangıç Val Doğruluğu (Korunan Baraj): %{best_acc * 100:.2f}")

print("\n" + "=" * 55)
print(f"İNCE AYAR (FINE-TUNING) BAŞLIYOR: {EPOCHS} EPOCH")
print("=" * 55)

start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 30)

    for phase in ["train", "val"]:
        model.train() if phase == "train" else model.eval()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == "train":
            scheduler.step()

        epoch_loss = running_loss / len(datasets_dict[phase])
        epoch_acc = (running_corrects.double() / len(datasets_dict[phase])).item()

        print(f"{phase.upper():5} - Kayıp (Loss): {epoch_loss:.4f} | Doğruluk (Acc): %{epoch_acc * 100:.2f}%")

        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
            print(f" -> En iyi model güncellendi! Yeni Val Acc: %{best_acc * 100:.2f}%")

total_duration = time.time() - start_time
print("\n" + "=" * 55)
print(f"İnce ayar tamamlandı! Süre: {total_duration / 60:.2f} dk")
print(f"En yüksek doğrulama başarısı: %{best_acc * 100:.2f}%")
print("=" * 55)