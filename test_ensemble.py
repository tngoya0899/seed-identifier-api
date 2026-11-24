import json
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

MANIFEST_PATH = "clean_seed_dataset/processed_manifest.csv"
LABELS_PATH = "models/category_labels.json"
MODEL1_PATH = "models/best_category_model.pth"       # MobileNetV3-Large (%62.21)
MODEL2_PATH = "models/best_category_pool_model.pth"  # EfficientNet-B0 Pool (%51.74)

device = torch.device("cpu")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_dict = json.load(f)
class_names = [labels_dict[str(i)] for i in range(len(labels_dict))]
class_to_idx = {name: i for i, name in enumerate(class_names)}
num_classes = len(class_names)

# Model 1: MobileNetV3-Large
m1 = models.mobilenet_v3_large(weights=None)
m1.classifier[2] = nn.Dropout(p=0.55)
m1.classifier[3] = nn.Linear(m1.classifier[3].in_features, num_classes)
m1.load_state_dict(torch.load(MODEL1_PATH, map_location=device))
m1.eval()

# Model 2: EfficientNet-B0 + Avg/Max Pool
class SeedPoolClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(2560),
            nn.Dropout(p=0.6),
            nn.Linear(2560, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        combined = torch.cat([torch.flatten(self.avgpool(feat), 1), torch.flatten(self.maxpool(feat), 1)], dim=1)
        return self.head(combined)

m2 = SeedPoolClassifier(num_classes)
m2.load_state_dict(torch.load(MODEL2_PATH, map_location=device))
m2.eval()

# Test kümesini TTA (Orijinal + Yatay Çevrilmiş) ile değerlendir
manifest_df = pd.read_csv(MANIFEST_PATH)
test_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)

tf_orig = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

top1_hits, top3_hits = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for _, row in test_df.iterrows():
        img = Image.open(row["saved_crop_path"]).convert("RGB")
        x1 = tf_orig(img).unsqueeze(0)
        x2 = tf_orig(img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0)

        # TTA + Ensemble (%65 MobileNetV3 + %35 EfficientNet-Pool)
        p1 = (torch.softmax(m1(x1), dim=1) + torch.softmax(m1(x2), dim=1)) / 2.0
        p2 = (torch.softmax(m2(x1), dim=1) + torch.softmax(m2(x2), dim=1)) / 2.0
        probs = (0.65 * p1 + 0.35 * p2)[0]

        true_idx = class_to_idx[row["category"]]
        top3 = torch.topk(probs, 3).indices.tolist()

        if top3[0] == true_idx:
            top1_hits += 1
        if true_idx in top3:
            top3_hits += 1

        all_preds.append(top3[0])
        all_labels.append(true_idx)

total = len(test_df)
print(f"\nENSEMBLE + TTA TEST SONUÇLARI ({total} Tohum):")
print(f"Top-1 Doğruluk: %{(top1_hits / total) * 100:.2f}")
print(f"Top-3 Doğruluk: %{(top3_hits / total) * 100:.2f}\n")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=3, zero_division=0))