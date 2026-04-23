# Project Overview: Indian Street Food Recognizer

## Introduction
This project is part of SMAI Assignment 3. It aims to bridge the gap between AI research and real-world application by creating a tool that helps users understand what they are eating.

## Dataset Details
We are using a subset of the **Indian Food Images** dataset from Kaggle.
- **Classes:** Pani Puri, Vada Pav, Chaat, Samosa, etc. (Targeting 12 classes).
- **Scale:** Tier 1 scope (200 - 2000 examples).

## Technical Implementation
1. **Backbone:** We will use a pre-trained ImageNet model (e.g., MobileNetV3 for efficiency or ResNet50 for accuracy) and fine-tune it.
2. **Metadata Mapping:** A JSON/CSV mapping dish names to:
   - Average Calories per serving.
   - Common Allergens (Dairy, Nuts, Gluten).
3. **UI:** A Streamlit application where users can upload an image and receive instant feedback.

## Step-by-Step Execution Guide

### Phase 1: Data preparation 📂
1. **Download:** Get the "Indian Food Images" dataset from Kaggle.
2. **Filter:** Isolate the 12 target street food classes (Pani Puri, Vada Pav, etc.).
3. **Split:** Divide data into 80% Training and 20% Validation.
4. **Augmentation:** Apply simple transforms (rotation, flip) to increase robustness.

### Phase 2: Model Training 🧠
1. **Backbone:** Load a pre-trained `models.mobilenet_v3_small` or `resnet18`.
2. **Freeze:** Freeze initial layers; only train the final classification head.
3. **Train:** Run for 1-5 epochs on Google Colab (T4 GPU).
4. **Evaluate:** Save the best model weights (`model.pth`) and record the accuracy/confusion matrix.

### Phase 3: Metadata & Logic 📊
1. **Create JSON:** Build a `food_info.json` file mapping dishes to:
   - `calories`: Estimated calories per serving.
   - `allergens`: List (e.g., ["Gluten", "Dairy"]).
2. **LLM Assist:** Use Gemini to verify calorie counts for specific street foods if not in datasets.

### Phase 4: App Development 💻
1. **Setup:** Create `app.py` using Streamlit.
2. **Inference:** Load the saved `model.pth` and write a function to process uploaded images.
3. **UI Design:** 
   - Add a header and description.
   - Display the uploaded image alongside its prediction.
   - Show color-coded "Allergen Badges" (e.g., 🔴 for Gluten).

### Phase 5: Deliverables 🚀
1. **GitHub:** Push code with a clean `requirements.txt`.
2. **HuggingFace:** Deploy the app to HF Spaces for a live demo.
3. **Report:** Write the 4-6 page technical report (PDF).
4. **Pitch:** Create a 1-slide visual summary of the project.
