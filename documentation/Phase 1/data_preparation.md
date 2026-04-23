# Phase 1: Detailed Data Preparation Guide

> [!TIP]
> **Recommended Method:** Use the provided [Phase_1_Data_Prep.ipynb](../../Phase_1_Data_Prep.ipynb) notebook in Google Colab. It automates everything (filtering, splitting, and saving) directly to your Google Drive.

## 1. Data Acquisition
The primary source is the [Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset).

### How to download:
- **Option A (Manual):** Download from Kaggle and unzip into `dataset/raw/`.
- **Option B (Kaggle API):**
  ```bash
  kaggle datasets download -d iamsouravbanerjee/indian-food-images-dataset
  unzip indian-food-images-dataset.zip -d dataset/raw/
  ```

## 2. Class Selection: Filtering 80 into 12
The original dataset has 80 classes. For this assignment, we only want the **Street Food** variants (approx. 12 classes).

### Manual Method:
Simply open the unzipped dataset, select the 12 folders you need (e.g., *Samosa*, *Pani Puri*), and copy them into your project's `dataset/raw/` directory.

### Automated Method (Recommended):
Use this script to "extract" only the classes we care about. This keeps your project clean and avoids manually digging through 80 folders.

```python
import os
import shutil

# Path where you unzipped the 80 classes
full_dataset_path = 'path/to/extracted/kaggle_data'
# Path where you want your 12 classes to live
target_path = 'dataset/raw'

# Our 12 Street Food classes
street_food_classes = [
    'pani puri', 'vada pav', 'samosa', 'kachori', 
    'dhokla', 'pav bhaji', 'momos', 'jalebi', 
    'pakora', 'chana masala', 'idli', 'dosa'
]

os.makedirs(target_path, exist_ok=True)

for folder in street_food_classes:
    src = os.path.join(full_dataset_path, folder)
    dst = os.path.join(target_path, folder)
    if os.path.exists(src):
        shutil.copytree(src, dst)
        print(f"✅ Extracted: {folder}")
    else:
        print(f"❌ Not found: {folder}")
```

## 3. Data Cleaning Script (`scripts/preprocess.py`)
You should create a script to perform the following:
- **Validation:** Check if files are valid JPEG/PNG images. Remove zero-byte files.
- **Resizing:** Resize all images to `224x224` (Standard for ResNet/MobileNet).
- **Normalization:** Ensure pixel values are scaled to `[0, 1]`.

## 4. Dataset Splitting: Manual vs. Automated

### Why NOT to do it manually:
- **Time:** Moving 1,200 images into train/val/test folders will take hours.
- **Errors:** You might accidentally put the same image in both training and testing (which ruins your results).
- **Randomness:** It's hard to pick images "randomly" by hand.

### Recommended: Automated Script (`split.py`)
Run this script to automatically move images from your raw folder into the correct Train/Val/Test folders.

```python
import os
import shutil
import random

# CONFIGURATION
raw_dir = 'dataset/raw'
output_dir = 'dataset/split'
classes = ['pani_puri', 'vada_pav', 'samosa', 'dhokla'] # Add all 12 here
split_ratio = (0.7, 0.15, 0.15) # Train, Val, Test

for cls in classes:
    # 1. Get all images for this class
    src_folder = os.path.join(raw_dir, cls)
    images = os.listdir(src_folder)
    random.shuffle(images)

    # 2. Calculate split points
    train_idx = int(len(images) * split_ratio[0])
    val_idx = int(len(images) * (split_ratio[0] + split_ratio[1]))

    splits = {
        'train': images[:train_idx],
        'val': images[train_idx:val_idx],
        'test': images[val_idx:]
    }

    # 3. Create folders and move files
    for split_name, split_images in splits.items():
        dst_folder = os.path.join(output_dir, split_name, cls)
        os.makedirs(dst_folder, exist_ok=True)
        for img in split_images:
            shutil.copy(os.path.join(src_folder, img), os.path.join(dst_folder, img))

print("✅ Dataset successfully split!")
```

## 5. Directory Structure After Splitting
Once you run the script, your project should look like this:
While images are being processed, prepare the `food_info.json`.
For each of the 12 classes, verify:
- **Calories:** Average per serving (e.g., 1 plate of 6 Pani Puris ≈ 150-200 kcal).
- **Allergens:** Identify if the dish contains Gluten (wheat), Dairy (yogurt/butter), or Nuts.

---
**Next Step:** Once the folders are populated, proceed to [Phase 2: Model Training].
