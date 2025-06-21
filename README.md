
# 🧠 Age Classifier using TensorFlow & MobileNetV2

This project implements an age classification model using a facial image dataset. The dataset contains images of 10 individuals, each seen at different ages (from 20 to 80 in steps of 10). The goal is to classify a given image into its correct age group.

---

## 📁 Dataset Structure

```
assessment_data/
├── 20/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 30/
│   └── ...
...
├── 80/
```

- Each `imageX.jpg` across folders represents the same person at different ages.
- Total: 10 individuals × 7 age groups = 70 images (before augmentation).

---

## 🛠️ Project Structure

```
.
├── data_processing.py   # Loads, augments, and processes data, extracts features
├── train_model.py       # Trains classifier and evaluates performance
├── *.npy                # Saved feature arrays and labels
```

---

## ⚙️ Techniques Used

- **Transfer Learning** with `MobileNetV2` (ImageNet pre-trained)
- **Image Augmentation**: rotation, zoom, shift, flip (via Keras ImageDataGenerator)
- **Label Encoding**: converting age folder names to class indices
- **Feature Extraction**: 1280-dim embeddings via MobileNetV2
- **Dense Neural Network Classifier** with dropout regularization

---

## 🔍 Model Metrics

Evaluated on test data using:
- ✅ Accuracy - 79.36%
- 🎯 Precision - 81.21%
- 🔁 Recall 79.36%

---

## 📦 Requirements

```bash
pip install tensorflow scikit-learn numpy tqdm
```

---

## 🚀 How to Run

### Step 1: Preprocess the Data
```bash
python data_processing.py
```

### Step 2: Train the Model
```bash
python train_model.py
```

---

## 🙌 Appreciation

I really enjoyed this assignment. It challenged me to structure a pipeline, handle a non-trivial dataset, and apply transfer learning effectively. I also got the chance to dig deeper into evaluation metrics and augmentation techniques.

---

## 📈 Future Improvements

- Use a time-series model (e.g., RNN) to capture aging progression per individual.
- Try different base models: ResNet, EfficientNet.
- Visualize embeddings with t-SNE for interpretability.

---
