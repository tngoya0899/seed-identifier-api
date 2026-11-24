import os
import json
import time
import copy
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# =========================================================================
# 1. TEMEL AYARLAR
# =========================================================================
torch.set_num_threads(4)

MANIFEST_PATH = "clean_seed_dataset/processed_manifest.csv"
MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# "species" -> 120 bitki türünü eğitir (crop_name_tr)
# "category" -> 8 ana kategoriyi eğitir (category)
TRAIN_MODE = "category"
TARGET_COL = "crop_name_tr" if TRAIN_MODE == "species" else "category"

BATCH_SIZE = 16
NUM_EPOCHS = 20          # Düşük öğrenme hızı (1.5e-4) ile dengeli yakınsama için 20 epoch
LEARNING_RATE = 1.5e-4   # Ezberlemeyi önlemek için düşürülmüş öğrenme hızı
IMAGE_SIZE = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Eğitim cihazı: {device} | Mod: {TRAIN_MODE} ({TARGET_COL}) | Mimari: Avg+Max Pooling")


# =========================================================================
# 2. MANIFEST TABANLI ÖZEL DATASET
# =========================================================================
class SeedManifestDataset(Dataset):
    def __init__(self, df, class_to_idx, target_col, transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.target_col = target_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["saved_crop_path"]
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row[self.target_col]]

        if self.transform:
            image = self.transform(image)
        return image, label


# =========================================================================
# 3. VERİ YÜKLEME VE EZBERLEME ÖNLEYİCİ AUGMENTATION (RANDOM ERASING DAHİL)
# =========================================================================
manifest_df = pd.read_csv(MANIFEST_PATH)
if "category" not in manifest_df.columns:
    meta_cat = pd.read_excel("meta_data.xlsx")[["seed_id", "category"]].drop_duplicates()
    manifest_df = manifest_df.merge(meta_cat, on="seed_id", how="left")
    manifest_df.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

all_classes = sorted(manifest_df[TARGET_COL].unique().tolist())
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(all_classes)}
num_classes = len(all_classes)

train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
val_df = manifest_df[manifest_df["split"] == "val"].reset_index(drop=True)
test_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)

print(f"Toplam Sınıf Sayısı: {num_classes}")
print(f"Tekil Tohum Sayıları -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}\n")

label_map_path = os.path.join(MODEL_SAVE_DIR, f"{TRAIN_MODE}_labels.json")
with open(label_map_path, "w", encoding="utf-8") as f:
    json.dump({str(idx): name for idx, name in enumerate(all_classes)}, f, ensure_ascii=False, indent=4)

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # Arka plan rengine ve tek bir lokal lekeye tutunmayı kesen rastgele silme
        transforms.RandomErasing(p=0.45, scale=(0.02, 0.16), ratio=(0.3, 3.3), value="random"),
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

datasets_dict = {
    "train": SeedManifestDataset(train_df, class_to_idx, TARGET_COL, data_transforms["train"]),
    "val": SeedManifestDataset(val_df, class_to_idx, TARGET_COL, data_transforms["val"]),
    "test": SeedManifestDataset(test_df, class_to_idx, TARGET_COL, data_transforms["test"]),
}

dataloaders = {
    x: DataLoader(datasets_dict[x], batch_size=BATCH_SIZE, shuffle=(x == "train"), num_workers=0)
    for x in ["train", "val", "test"]
}


# =========================================================================
# 4. EFFICIENTNET-B0 + AVG/MAX POOLING HİBRİT MİMARİ
# =========================================================================
class SeedPoolClassifier(nn.Module):
    def __init__(self, num_classes, dropout_p=0.6):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features

        # Tüm omurgayı dondur, yalnızca en derin son 2 bloğu mikro doku için aç
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.features[-2:].parameters():
            param.requires_grad = True

        # EfficientNet-B0 son kanal sayısı: 1280 -> Avg (1280) + Max (1280) = 2560
        in_channels = 1280
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.head = nn.Sequential(
            nn.BatchNorm1d(in_channels * 2),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_channels * 2, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        avg_f = torch.flatten(self.avgpool(feat), 1)
        max_f = torch.flatten(self.maxpool(feat), 1)
        combined = torch.cat([avg_f, max_f], dim=1)
        return self.head(combined)


model = SeedPoolClassifier(num_classes=num_classes, dropout_p=0.6).to(device)

# Yumuşatılmış sınıf ağırlıkları
class_counts = train_df[TARGET_COL].value_counts()
weights = [1.0 / np.sqrt(class_counts.get(cls_name, 1)) for cls_name in all_classes]
class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
class_weights_tensor = class_weights_tensor / class_weights_tensor.mean()

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

# Yüksek weight_decay (2e-2) ile ağırlıkların ezberlemesini frenle
optimizer = optim.AdamW([
    {"params": model.features[-2:].parameters(), "lr": LEARNING_RATE * 0.25},
    {"params": model.head.parameters(), "lr": LEARNING_RATE},
], weight_decay=2e-2)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)


# =========================================================================
# 5. EĞİTİM VE DEĞERLENDİRME DÖNGÜSÜ
# =========================================================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_model_path = os.path.join(MODEL_SAVE_DIR, f"best_{TRAIN_MODE}_pool_model.pth")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
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

            epoch_loss = running_loss / len(datasets_dict[phase])
            epoch_acc = (running_corrects.double() / len(datasets_dict[phase])).item()
            print(f"{phase.upper():5} Loss: {epoch_loss:.4f} | Acc: %{epoch_acc * 100:.2f}")

            if phase == "val":
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), best_model_path)
                    print(f"--> [KAYDEDİLDİ] Yeni en iyi {TRAIN_MODE} (Pool) başarımı: %{best_acc * 100:.2f}")
        print()

    total_time = time.time() - start_time
    print(f"Eğitim tamamlandı! Süre: {total_time // 60:.0f}dk {total_time % 60:.0f}sn")
    print(f"En İyi Val Doğruluğu: %{best_acc * 100:.2f}")

    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    trained_model = train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

    print("\n" + "=" * 55)
    print("TEST SETİ İLE NİHAİ DEĞERLENDİRME (TOP-1 VE TOP-3)")
    print("=" * 55)
    trained_model.eval()
    top1_corrects = 0
    top3_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)

            _, top1_preds = torch.max(outputs, 1)
            top1_corrects += torch.sum(top1_preds == labels.data)

            _, top3_preds = outputs.topk(min(3, num_classes), dim=1)
            top3_corrects += sum(labels[i] in top3_preds[i] for i in range(labels.size(0)))

    total_test = len(datasets_dict["test"])
    print(f"Top-1 Test Doğruluğu: %{(top1_corrects.double() / total_test).item() * 100:.2f}")
    print(f"Top-3 Test Doğruluğu: %{(top3_corrects / total_test) * 100:.2f}")
    print("=" * 55)