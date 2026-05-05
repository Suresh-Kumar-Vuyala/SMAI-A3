# 🍛 ChaatCheck — Indian Street Food Recognizer
### SMAI Assignment 3 · Task T2.4 (Street Food Tier 1)

> **Point your phone at a thali; the app names every dish, estimates calories, and flags allergens (gluten, dairy, nuts) — in real time.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![Model](https://img.shields.io/badge/Model-MobileNetV2-green.svg)](https://pytorch.org/vision/stable/models/mobilenet_v2.html)

---

## 📌 Problem Statement

**T2 — Indian Food Recognizer (T2.4 Street Food, Tier 1)**

Classify images of **12 Indian street food dishes** using transfer learning (MobileNetV2 fine-tuned on ~1,400 images). For each prediction, the app returns:
- 🍽 **Dish name** with confidence score
- 🔥 **Calorie estimate** (per serving)
- ⚠️ **Allergen badges** (Gluten / Dairy / Nuts)
- 🧾 **Key ingredients**

**Dataset Source:** [Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset) — 4,000+ images, 80 classes. We subset 12 street-food classes.

**RESULTS**: Some of the results are added in results folder
---

## 🗂️ Repository Structure

```
SMAI_A3/
│
├── app/                          # 🚀 Deployment-ready Streamlit app
│   ├── app.py                    # Main UI — upload → predict → display results
│   ├── model_inference.py        # Model loading & inference logic
│   └── food_model.pth            # Trained MobileNetV2 weights (12 classes)
│
├── training/
│   └── Training.ipynb            # Model training loop (Google Colab + GPU)
│
└── data_preparation/
    ├── Phase_1_Data_Prep.ipynb        # Step 1 — Download, filter, and split dataset
    └── Phase_1_Perform_Augmentation.ipynb  # Step 2 — Triple training data via augmentation
```

---

## 🍱 The 12 Street Food Classes

| # | Class | Approx. Calories | Allergens | Diet |
|---|-------|-----------------|-----------|------|
| 1 | Aloo Tikki | 180–220 kcal | Gluten | Veg |
| 2 | Bhatura | 300–350 kcal | Gluten | Veg |
| 3 | Chana Masala | 220–260 kcal | None | Vegan |
| 4 | Daal Puri | 260–300 kcal | Gluten | Veg |
| 5 | Jalebi | 140–180 kcal | Gluten | Veg |
| 6 | Kachori | 250–300 kcal | Gluten | Veg |
| 7 | Lassi | 150–200 kcal | Dairy | Veg |
| 8 | Litti Chokha | 300–350 kcal | Gluten | Veg |
| 9 | Poha | 150–200 kcal | Nuts | Vegan |
| 10 | Rabri | 300–400 kcal | Dairy, Nuts | Veg |
| 11 | Kuzhi Paniyaram | 200–250 kcal | None | Veg |
| 12 | Unni Appam | 220–260 kcal | Gluten | Veg |

---

## ⚙️ How It Works — End-to-End Pipeline

```
Kaggle Dataset (80 classes, 4000+ images)
        │
        ▼
[Phase_1_Data_Prep.ipynb]
  Filter → 12 street food classes
  Split  → 70% Train / 15% Val / 15% Test
  Result → 600 total images (50/class)
        │
        ▼
[Phase_1_Perform_Augmentation.ipynb]
  Per training image → generate 2 augmented versions
  (random flip, rotation ±15°, color jitter)
  Result → 1,260 train images (105/class)
        │
        ▼
[Training.ipynb]
  MobileNetV2 pretrained on ImageNet
  Replace final classifier → 12-class head
  Fine-tune on augmented training set
  Save best weights → food_model.pth
        │
        ▼
[app/app.py + model_inference.py]
  Streamlit UI: upload image
  Inference → predicted class + confidence
  Lookup metadata → calories, allergens, diet
  Display results with color-coded badges
```

---

## 📓 Notebook Details

### `data_preparation/Phase_1_Data_Prep.ipynb`
**Purpose:** Acquire and organize the raw dataset.

**What it does, step by step:**

1. **Mount Google Drive** — all outputs persist across Colab sessions
2. **Configure Kaggle API** — paste your `KAGGLE_USERNAME` and `KAGGLE_KEY` in the form fields
3. **Download the dataset** — pulls the full 355 MB zip directly into Drive
4. **Extract & filter** — unzips and selects only the 12 street food class folders from the 80-class archive
5. **Train/Val/Test split** — shuffles images randomly, then splits `70 / 15 / 15`
6. **Verification table** — prints a per-class count table to confirm correctness

**Output split (before augmentation):**

| Split | Images per class | Total |
|-------|-----------------|-------|
| Train | 35 | 420 |
| Val   | 7  | 84  |
| Test  | 8  | 96  |
| **Total** | **50** | **600** |

---

### `data_preparation/Phase_1_Perform_Augmentation.ipynb`
**Purpose:** Expand the training set 3× using offline augmentation.

**Augmentation pipeline applied to every training image:**
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
])
```

Each original image → **2 new augmented images** saved alongside it (named `*_aug1.*`, `*_aug2.*`).

**Output after augmentation:**

| Split | Original | Augmented | Total per class | Grand Total |
|-------|----------|-----------|-----------------|-------------|
| Train | 35 | 70 | 105 | 1,260 |
| Val   | 7  | 0  | 7   | 84   |
| Test  | 8  | 0  | 8   | 96   |
| **All** | — | — | — | **1,440** |

> Val and Test sets are **not augmented** — they must represent real-world conditions.

---

### `training/Training.ipynb`
**Purpose:** Fine-tune MobileNetV2 on the augmented street food dataset.

**Key design decisions:**

- **Base model:** `torchvision.models.mobilenet_v2(pretrained=True)` — leverages ImageNet features
- **Architecture change:** Replace `classifier[1]` (originally 1000-class) with `nn.Linear(last_channel, 12)`
- **Training transforms:** Resize to 224×224, ToTensor, Normalize (ImageNet mean/std)
- **Loss / Optimizer:** CrossEntropyLoss + Adam
- **Output:** Saves `food_model.pth` — best weights by validation accuracy

---

## 🖥️ App Details

### `app/model_inference.py`
Handles all ML logic, cleanly separated from the UI.

```python
class_names = [
    "aloo_tikki", "bhatura", "chana_masala", "daal_puri",
    "jalebi", "kachori", "kuzhi_paniyaram", "lassi",
    "litti_chokha", "poha", "rabri", "unni_appam"
]
```
> ⚠️ The class list order **must exactly match** the order used during training (alphabetical, as `ImageFolder` sorts them).

**`load_model(path)`** — rebuilds MobileNetV2 architecture and loads saved weights.  
**`predict(image, model)`** — applies inference transform, runs forward pass, returns `(class_name, confidence)`.

---

### `app/app.py`
Streamlit UI with two-column layout.

| Left Column | Right Column |
|-------------|-------------|
| Uploaded image preview | Predicted dish name |
| — | Confidence progress bar |
| — | Calories per serving |
| — | Diet type (VEG / VEGAN) |
| — | Allergen badges (color-coded) |
| — | Ingredient list |

**Allergen badge colors:**
- 🔴 `st.error` → Gluten
- 🟡 `st.warning` → Dairy
- 🔵 `st.info` → Nuts
- 🟢 `st.success` → No allergens

---

## 🚀 Running the App Locally

**Prerequisites:** Python 3.8+, pip

```bash
# 1. Clone the repo
git clone https://github.com/your-username/SMAI_A3.git
cd SMAI_A3

# 2. Install dependencies
pip install streamlit torch torchvision Pillow

# 3. Make sure food_model.pth is in the app/ folder

# 4. Launch
cd app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, upload any food photo, and get instant predictions.

---

## 🔁 Reproducing Training from Scratch

1. Open **`data_preparation/Phase_1_Data_Prep.ipynb`** in Google Colab
2. Set your Kaggle credentials in the form fields and run all cells
3. Open **`data_preparation/Phase_1_Perform_Augmentation.ipynb`** and run all cells
4. Open **`training/Training.ipynb`** and run all cells
5. Download the saved `food_model.pth` and place it in `app/`

> All notebooks are designed for **Google Colab with GPU (T4)**. Dataset is stored in Google Drive across sessions.

---

## 📊 Dataset Summary

| Property | Value |
|----------|-------|
| Source | Kaggle — Indian Food Images Dataset |
| Original classes | 80 |
| Selected classes | 12 (street food) |
| Original images | 600 (50/class) |
| After augmentation | 1,440 training-system images |
| Image size (training) | 224 × 224 px |
| Train / Val / Test | 70% / 15% / 15% |

---

## 🧠 Model Summary

| Property | Value |
|----------|-------|
| Architecture | MobileNetV2 |
| Pretrained on | ImageNet |
| Fine-tuned classes | 12 |
| Input size | 224 × 224 × 3 |
| Output | Softmax over 12 classes |
| Saved as | `food_model.pth` |

---

## 📚 References

- Dataset: [Indian Food Images — Sourav Banerjee (Kaggle)](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset)
- Model: [MobileNetV2 — PyTorch Docs](https://pytorch.org/vision/stable/models/mobilenet_v2.html)
- Nutrition reference: [IFCT 2017 — National Institute of Nutrition](https://www.nin.res.in)
- Assignment spec: SMAI A3 · T2.4 Street Food Tier 1

---

