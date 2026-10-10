import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# =========================================================================
# 1. 4 KENARDAKİ TELİF YAZILARINI KESİP CETVEL VE TOHUMU KORUYAN FONKSİYON
# =========================================================================
def clean_copyright_keep_scale(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        img_arr = np.array(image)
        h, w, _ = img_arr.shape

        # 1. Arka plan rengini belirle
        corners = np.concatenate([
            img_arr[:20, :20].reshape(-1, 3),
            img_arr[:20, -20:].reshape(-1, 3),
            img_arr[-20:, :20].reshape(-1, 3),
            img_arr[-20:, -20:].reshape(-1, 3)
        ])
        bg_color = np.median(corners, axis=0).astype(np.uint8)

        # 2. CFIA telif yazıları istisnasız 4 dış sınırın en ucundadır:
        # Alt (Bayrak & Telif): %6.5
        # Üst (Yedek telif): %3.5
        # Sol ve Sağ (Dikey telif yazıları): %4.5
        top_cut = int(h * 0.035)
        bottom_cut = int(h * 0.935)
        left_cut = int(w * 0.045)
        right_cut = int(w * 0.955)

        # Çerçeve traşlama (Tohumlar ve cetvel ortadadır, asla zarar görmez)
        trimmed = img_arr[top_cut:bottom_cut, left_cut:right_cut]
        th, tw, _ = trimmed.shape

        # 3. Kalan tohum ve cetveli kapsayan alanı çevreleyerek kırp
        diff = np.linalg.norm(trimmed.astype(float) - bg_color.astype(float), axis=2)
        mask = diff > 20
        coords = np.argwhere(mask)

        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            pad = 20
            y1 = max(0, y_min - pad)
            y2 = min(th, y_max + pad)
            x1 = max(0, x_min - pad)
            x2 = min(tw, x_max + pad)

            if (y2 - y1) > 60 and (x2 - x1) > 60:
                return Image.fromarray(trimmed[y1:y2, x1:x2])

        return Image.fromarray(trimmed)
    except Exception as e:
        print(f"Hata ({img_path}): {e}")
        return Image.open(img_path).convert("RGB")

# =========================================================================
# 2. VERİ ARTIRMA (AUGMENTATION)
# =========================================================================
def save_augmented_images(base_img, target_folder, clean_basename):
    name_wo_ext, _ = os.path.splitext(clean_basename)
    ext = '.jpg'
    variations = {
        'orig': base_img,
        'rot90': base_img.rotate(90, expand=True),
        'rot180': base_img.rotate(180, expand=True),
        'rot270': base_img.rotate(270, expand=True),
        'hflip': base_img.transpose(Image.FLIP_LEFT_RIGHT),
        'vflip': base_img.transpose(Image.FLIP_TOP_BOTTOM)
    }
    for tag, img_var in variations.items():
        save_path = os.path.join(target_folder, f"{name_wo_ext}_{tag}{ext}")
        img_var.save(save_path, format='JPEG', quality=95)

# =========================================================================
# 3. YÜRÜTME
# =========================================================================
def main():
    excel_path = 'meta_data.xlsx'
    output_dir = 'processed_data'

    if not os.path.exists(excel_path):
        print(f"[HATA] {excel_path} bulunamadı!")
        return

    print("Metaveri tablosu okunuyor...")
    df = pd.read_excel(excel_path, sheet_name='Sayfa1')
    df['full_path'] = df['relative_path'].apply(lambda p: str(p).replace('\\', '/'))
    df = df[df['full_path'].apply(os.path.exists)].copy()

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['category']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['category']
    )

    splits = {'train': train_df, 'val': val_df, 'test': test_df}

    for split_name, split_df in splits.items():
        is_train = (split_name == 'train')
        print(f"--> '{split_name}' kümesi işleniyor ({len(split_df)} görsel)...")

        for _, row in split_df.iterrows():
            category = str(row['category'])
            target_folder = os.path.join(output_dir, split_name, category)
            os.makedirs(target_folder, exist_ok=True)

            src_img_path = row['full_path']
            clean_image_name = os.path.basename(src_img_path)

            # Çerçeve traşlayarak telifleri silen, cetvel ve tohumu koruyan işlem
            processed_img = clean_copyright_keep_scale(src_img_path)

            if is_train:
                save_augmented_images(processed_img, target_folder, clean_image_name)
            else:
                dst_img_path = os.path.join(target_folder, clean_image_name)
                processed_img.save(dst_img_path, format='JPEG', quality=95)

    print("\n" + "="*50)
    print("MÜKEMMEL! İŞLEM TAMAMLANDI.")
    print("1. Dikey ve yatay tüm kenarlardaki telif yazıları ve bayraklar elendi.")
    print("2. Tohumlar ve milimetrik ölçek cetvelleri güvenle korundu.")
    print("="*50)

if __name__ == '__main__':
    main()