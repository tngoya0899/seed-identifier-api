import json
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms, models

MANIFEST_PATH = "clean_seed_dataset/processed_manifest.csv"
CAT_LABELS_PATH = "models/category_labels.json"
SPECIES_LABELS_PATH = "models/species_labels.json"
CAT_MODEL1_PATH = "models/best_category_model.pth"
CAT_MODEL2_PATH = "models/best_category_pool_model.pth"
SPECIES_MODEL_PATH = "models/best_species_model.pth"
LIBRARY_PATH = "models/seed_embeddings_library.pt"

device = torch.device("cpu")

with open(CAT_LABELS_PATH, "r", encoding="utf-8") as f:
    cat_dict = json.load(f)
cat_classes = [cat_dict[str(i)] for i in range(len(cat_dict))]

with open(SPECIES_LABELS_PATH, "r", encoding="utf-8") as f:
    sp_dict = json.load(f)
sp_classes = [sp_dict[str(i)] for i in range(len(sp_dict))]
sp_to_idx = {name: i for i, name in enumerate(sp_classes)}

m1 = models.mobilenet_v3_large(weights=None)
m1.classifier[2] = nn.Dropout(p=0.55)
m1.classifier[3] = nn.Linear(m1.classifier[3].in_features, len(cat_classes))
m1.load_state_dict(torch.load(CAT_MODEL1_PATH, map_location=device))
m1.eval()

class SeedPoolClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(2560), nn.Dropout(p=0.6), nn.Linear(2560, 256),
            nn.SiLU(), nn.BatchNorm1d(256), nn.Dropout(p=0.6), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        feat = self.features(x)
        return self.head(torch.cat([torch.flatten(self.avgpool(feat), 1), torch.flatten(self.maxpool(feat), 1)], dim=1))

m2 = SeedPoolClassifier(len(cat_classes))
m2.load_state_dict(torch.load(CAT_MODEL2_PATH, map_location=device))
m2.eval()

sp_model = models.mobilenet_v3_large(weights=None)
sp_model.classifier[2] = nn.Dropout(p=0.45)
sp_model.classifier[3] = nn.Linear(sp_model.classifier[3].in_features, len(sp_classes))
sp_model.load_state_dict(torch.load(SPECIES_MODEL_PATH, map_location=device))
sp_model.eval()

class FeatureExtractor(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.features, self.avgpool = orig.features, orig.avgpool
    def forward(self, x):
        return nn.functional.normalize(torch.flatten(self.avgpool(self.features(x)), 1), p=2, dim=1)

fe = FeatureExtractor(sp_model).eval()
seed_library = torch.load(LIBRARY_PATH, map_location="cpu")
lib_embs = torch.stack([item["embedding"] for item in seed_library])

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_df = pd.read_csv(MANIFEST_PATH)
test_df = test_df[test_df["split"] == "test"].reset_index(drop=True)

top1_sp, top3_sp, top1_cat = 0, 0, 0
sample_hits = []

with torch.no_grad():
    for _, row in test_df.iterrows():
        img = Image.open(row["saved_crop_path"]).convert("RGB")
        x1 = tf(img).unsqueeze(0)
        x2 = tf(img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0)

        p_cat1 = (torch.softmax(m1(x1), dim=1) + torch.softmax(m1(x2), dim=1)) / 2.0
        p_cat2 = (torch.softmax(m2(x1), dim=1) + torch.softmax(m2(x2), dim=1)) / 2.0
        cat_probs = (0.65 * p_cat1 + 0.35 * p_cat2)[0]
        top3_vals, top3_idxs = torch.topk(cat_probs, 3)
        top3_cats = {cat_classes[i.item()]: v.item() for i, v in zip(top3_idxs, top3_vals)}

        sp_probs = ((torch.softmax(sp_model(x1), dim=1) + torch.softmax(sp_model(x2), dim=1)) / 2.0)[0]
        q_emb = nn.functional.normalize(fe(x1) + fe(x2), p=2, dim=1)
        sims = torch.matmul(lib_embs, q_emb.squeeze(0))

        scores = []
        for idx, item in enumerate(seed_library):
            if item["category"] not in top3_cats:
                continue
            sp_name = item["crop_name_tr"]
            sp_p = float(sp_probs[sp_to_idx[sp_name]].item()) if sp_name in sp_to_idx else 0.0
            tot = 0.40 * float(sims[idx].item()) + 0.35 * sp_p + 0.25 * top3_cats[item["category"]]
            scores.append((tot, item))
        scores.sort(key=lambda x: x[0], reverse=True)

        seen, top3_names = set(), []
        for _, it in scores:
            if it["crop_name_tr"] not in seen:
                seen.add(it["crop_name_tr"])
                top3_names.append(it["crop_name_tr"])
                if len(top3_names) == 3:
                    break

        true_sp, true_cat = row["crop_name_tr"], row["category"]
        if top3_names[0] == true_sp:
            top1_sp += 1
            if len(sample_hits) < 10:
                sample_hits.append(row["saved_crop_path"])
        if true_sp in top3_names:
            top3_sp += 1
        if scores[0][1]["category"] == true_cat:
            top1_cat += 1

n = len(test_df)
print(f"\nHİBRİT API NİHAİ TEST PERFORMANSI ({n} GÖRÜNMEMİŞ TOHUM):")
print(f"Kategori Top-1 Doğruluğu : %{(top1_cat / n) * 100:.2f}")
print(f"120 Tür Top-1 Doğruluğu  : %{(top1_sp / n) * 100:.2f}")
print(f"120 Tür Top-3 Doğruluğu  : %{(top3_sp / n) * 100:.2f}")
print("\nTam İsabet Eden Örnek Test Dosyaları:")
for p in sample_hits:
    print(" -", p)