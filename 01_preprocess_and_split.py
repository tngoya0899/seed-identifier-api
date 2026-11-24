import os
import re
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold

# =========================================================================
# AYARLAR
# =========================================================================
EXCEL_PATH = "meta_data.xlsx"
SHEET_NAME = "Sayfa1"
OUTPUT_DIR = "clean_seed_dataset"
TARGET_COL = "crop_name_tr"      # 120 tür için 'crop_name_tr' (veya 'scientific_name')
IMG_SIZE = 224                   # MobileNet / EfficientNet standart giriş boyutu


# =========================================================================
# 1. TEKİL TOHUM AYIKLAMA VE CETVEL/TELİF TEMİZLEME
# =========================================================================
def sanitize_folder_name(name):
    """Türkçe karakterleri ve geçersiz dosya yolu işaretlerini güvenli hale getirir."""
    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ /\\", "cgiosuCGIOSU___")
    cleaned = str(name).translate(tr_map)
    return re.sub(r"[^a-zA-Z0-9_-]", "", cleaned).strip("_")


def pad_or_downscale_only(rgb_crop, target_size=224, bg_color=(128, 128, 128)):
    """
    224'ten büyük kesitleri orantılı küçültür, küçük kesitleri ise hiç büyütmeden
    (bulanıklaştırmadan) orijinal arka plan rengiyle ortalar.
    """
    h, w, _ = rgb_crop.shape
    max_dim = max(h, w)

    if max_dim > target_size:
        scale = target_size / max_dim
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        processed = cv2.resize(rgb_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        processed = rgb_crop
        new_h, new_w, _ = processed.shape

    canvas = np.full((target_size, target_size, 3), bg_color, dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = processed
    return canvas


def extract_individual_seeds(img_path, source_institution, target_size=224):
    """
    Telif şeritlerini ve kurum bazlı kenar cetvellerini keser, LAB renk farkı ve kenar
    tespiti ile tohumları bulur, cetvel/rakam parçalarını eler ve 224x224 kareye alır.
    """
    try:
        with Image.open(img_path) as image:
            orig_rgb = np.array(image.convert("RGB"))
    except (UnidentifiedImageError, Exception) as e:
        print(f"[UYARI] Bozuk dosya atlandı ({img_path}): {e}")
        return []

    h, w, _ = orig_rgb.shape
    inst_upper = str(source_institution).upper()

    # 1. Kuruma göre telif şeridi ve kenar cetvelini kes
    if "CFIA" in inst_upper:
        trimmed = orig_rgb[int(h * 0.05):int(h * 0.94), int(w * 0.02):int(w * 0.98)]
    elif "OSU" in inst_upper or "OHIO" in inst_upper:
        # OSU görsellerinde sol kenarda dikey milimetrik cetvel şeridi bulunur
        trimmed = orig_rgb[int(h * 0.04):int(h * 0.96), int(w * 0.24):int(w * 0.98)]
    elif "WTU" in inst_upper or "BURKE" in inst_upper:
        # WTU görsellerinde alt/kenar cetveli bulunur
        trimmed = orig_rgb[int(h * 0.04):int(h * 0.84), int(w * 0.04):int(w * 0.96)]
    elif "COLORADO" in inst_upper or "IDTOOLS" in inst_upper or "USDA" in inst_upper:
        trimmed = orig_rgb[int(h * 0.04):int(h * 0.92), int(w * 0.03):int(w * 0.97)]
    else:
        trimmed = orig_rgb[int(h * 0.05):int(h * 0.95), int(w * 0.03):int(w * 0.97)]

    th, tw, _ = trimmed.shape

    # 2. Köşelerden fotoğrafın gerçek arka plan rengini ölç
    corners = np.vstack([
        trimmed[0:15, 0:15].reshape(-1, 3),
        trimmed[0:15, -15:].reshape(-1, 3),
        trimmed[-15:, 0:15].reshape(-1, 3),
        trimmed[-15:, -15:].reshape(-1, 3),
    ])
    native_bg_rgb = np.median(corners, axis=0)
    native_bg = tuple(native_bg_rgb.astype(np.uint8).tolist())

    # 3. LAB renk farkı + Canny kenarları ile dış sınırları yakala
    lab = cv2.cvtColor(trimmed, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(np.uint8([[native_bg_rgb]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, otsu = cv2.threshold(cv2.GaussianBlur(dist_u8, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(cv2.GaussianBlur(trimmed, (5, 5), 0), 25, 90)
    combined = cv2.bitwise_or(otsu, edges)

    # Dış konturu kapat ve içini tamamen doldur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(closed)
    cv2.drawContours(solid_mask, contours, -1, 255, thickness=cv2.FILLED)

    # 4. Cetvel, kenara yapışık şerit ve rakamları ele, tekil tohumları çıkar
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(solid_mask)
    min_area = (th * tw) * 0.008

    extracted_seeds = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # Kural A: Görüntünün dış çerçevesine yapışan cetvel/etiket şeritlerini ele
        touches_border = (x <= 2) or (y <= 2) or (x + bw >= tw - 2) or (y + bh >= th - 2)
        if touches_border and (bw > tw * 0.6 or bh > th * 0.6):
            continue

        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-5)
        extent = area / (bw * bh + 1e-5)

        if area > min_area and max(bw, bh) >= 130 and aspect_ratio < 5.5 and 0.45 < extent < 0.88:
            # Kural B: Cetvel arka planı / rakam filtresi (saf beyaz zemin + siyah çizgi/rakam)
            comp_mask = (labels[y:y + bh, x:x + bw] == i)
            comp_pixels = trimmed[y:y + bh, x:x + bw][comp_mask]
            white_ratio = np.mean(np.all(comp_pixels > 220, axis=1))
            if white_ratio > 0.35:
                continue

            pad_x = int(bw * 0.04)
            pad_y = int(bh * 0.04)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(tw, x + bw + pad_x)
            y2 = min(th, y + bh + pad_y)

            raw_seed_crop = trimmed[y1:y2, x1:x2]
            square_seed = pad_or_downscale_only(raw_seed_crop, target_size=target_size, bg_color=native_bg)
            extracted_seeds.append(square_seed)

    # Çok küçük tohum grupları elendiyse tüm tohum bölgesini net haliyle koru
    if not extracted_seeds:
        coords = np.argwhere(solid_mask > 0)
        if coords.size > 0:
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            extracted_seeds.append(pad_or_downscale_only(trimmed[y1:y2, x1:x2], target_size=target_size, bg_color=native_bg))
        else:
            extracted_seeds.append(pad_or_downscale_only(trimmed, target_size=target_size, bg_color=native_bg))

    return extracted_seeds
# =========================================================================
# 2. DATA LEAKAGE ÖNLEYEN GRUP BAZLI BÖLME (TEK GÖRSELLİ TÜRLER KORUMALI)
# =========================================================================
def build_sample_group(row):
    acc = re.sub(r"-\d+$", "", str(row["accession_number"]).strip())
    return f"{row['scientific_name']}_{acc}"


def split_dataset_safely(df, target_col):
    df = df.copy()
    df["sample_group"] = df.apply(build_sample_group, axis=1)

    group_counts = df.groupby(target_col)["sample_group"].nunique()
    single_group_classes = group_counts[group_counts < 2].index

    forced_train_df = df[df[target_col].isin(single_group_classes)].copy()
    splittable_df = df[~df[target_col].isin(single_group_classes)].reset_index(drop=True)

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    folds = [test_idx for _, test_idx in sgkf.split(splittable_df, splittable_df[target_col], splittable_df["sample_group"])]

    train_idx = np.concatenate(folds[:7])
    val_idx = np.concatenate(folds[7:9])
    test_idx = folds[9]

    train_df = pd.concat([splittable_df.iloc[train_idx], forced_train_df], ignore_index=True)
    val_df = splittable_df.iloc[val_idx].reset_index(drop=True)
    test_df = splittable_df.iloc[test_idx].reset_index(drop=True)

    return {"train": train_df, "val": val_df, "test": test_df}


# =========================================================================
# 3. ANA İŞLEM AKIŞI VE RAPORLAMA
# =========================================================================
def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"[HATA] {EXCEL_PATH} bulunamadı!")
        return

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    df["full_path"] = df["relative_path"].apply(lambda p: str(p).replace("\\", "/"))

    missing_files = df[~df["full_path"].apply(os.path.exists)]
    if len(missing_files) > 0:
        print(f"[UYARI] Diskte bulunamayan {len(missing_files)} dosya atlandı.")
    df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

    splits = split_dataset_safely(df, TARGET_COL)
    manifest_records = []

    for split_name, split_df in splits.items():
        print(f"--> '{split_name}' kümesi işleniyor ({len(split_df)} kaynak görsel)...")
        total_crops = 0

        for _, row in split_df.iterrows():
            class_folder = sanitize_folder_name(row[TARGET_COL])
            target_dir = os.path.join(OUTPUT_DIR, split_name, class_folder)
            os.makedirs(target_dir, exist_ok=True)

            seed_crops = extract_individual_seeds(
                row["full_path"],
                row["source_institution"],
                target_size=IMG_SIZE
            )

            base_name = os.path.splitext(os.path.basename(row["full_path"]))[0]
            for idx, crop_img in enumerate(seed_crops, start=1):
                out_filename = f"{base_name}_seed{idx:02d}.png"
                out_path = os.path.join(target_dir, out_filename)
                Image.fromarray(crop_img).save(out_path, format="PNG")
                total_crops += 1

                manifest_records.append({
                    "split": split_name,
                    "seed_id": row["seed_id"],
                    "category": row["category"],           # <-- BUNU EKLEDİK (8 Ana Kategori)
                    "crop_name_tr": row["crop_name_tr"],   # <-- 120 Alt Tür
                    "scientific_name": row["scientific_name"],
                    "class_folder": class_folder,
                    "source_image": row["full_path"],
                    "saved_crop_path": out_path
                })

        print(f"    Oluşturulan tekil tohum görseli sayısı: {total_crops}")

    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(os.path.join(OUTPUT_DIR, "processed_manifest.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"İŞLEM TAMAMLANDI! Toplam {len(manifest_df)} adet tekil tohum üretildi.")
    print(f"Detaylı eşleştirme tablosu: {OUTPUT_DIR}/processed_manifest.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()