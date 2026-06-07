# CNN_Crack_Detection
# CNN-Based Surface Crack Detection

A deep learning project that uses Convolutional Neural Networks (CNNs) to detect whether a concrete surface is **cracked or non-cracked** from RGB images.

Two models are built and compared:
- **Custom CNN** — a 3-block convolutional network trained from scratch
- **MobileNetV2** — a transfer learning model pre-trained on ImageNet

---

## Project Overview

| Field | Details |
|---|---|
| **Course** | Machine Learning — Master of Data Science |
| **University** | University of Europe for Applied Sciences, Potsdam |
| **Author** | Akunuru Abhishek Karthik |
| **Framework** | TensorFlow 2.21 / Keras |
| **Task** | Binary image classification (Cracked / Non-cracked) |

---

## Dataset

| Property | Details |
|---|---|
| Source | [Surface Crack Detection — Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) |
| Classes | 2 — Positive (Cracked), Negative (Non-cracked) |
| Images used | 10,000 (5,000 per class, balanced) |
| Image size | Resized to 128 × 128 pixels |
| Format | RGB JPEG images in class-labelled folders |

**Dataset split:**

| Split | Size |
|---|---|
| Training | 64% (~6,400 images) |
| Validation | 16% (~1,600 images) |
| Test | 20% (2,000 images) |

---

## Repository Structure

```
CNN_Crack_Detection/
├── notebooks/
│   └── Abhishek_CNN_Crack_Detection.ipynb   # Main notebook
├── figures/
│   ├── sample_images.png
│   ├── cnn_accuracy_loss_curves.png
│   ├── cnn_confusion_matrix.png
│   ├── cnn_roc_curve.png
│   ├── mobilenet_accuracy_loss_curves.png
│   ├── mobilenet_confusion_matrix.png
│   ├── roc_comparison.png
│   ├── model_comparison_chart.png
│   └── gradcam_visualisation.png
├── model_comparison.csv
```

---

## Model Architectures

### Custom CNN

```
Input (128×128×3)
→ Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
→ Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
→ Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
→ Flatten → Dense(128, ReLU) → Dropout(0.5)
→ Dense(1, Sigmoid)
```

- Optimiser: Adam (lr=0.001)
- Loss: Binary cross-entropy
- Epochs: 10

### MobileNetV2 (Transfer Learning)

- Base: MobileNetV2 pre-trained on ImageNet (all layers frozen)
- Head: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- Optimiser: Adam (lr=0.001)
- Epochs: 5

---

## Results

| Model | Test Accuracy | AUC |
|---|---|---|
| Custom CNN | ~99% | ~1.00 |
| MobileNetV2 | ~99% | ~1.00 |

> Exact values are printed at runtime. Both models perform strongly on this dataset due to the clear visual difference between cracked and non-cracked surfaces.

---

## Research Questions

| # | Question |
|---|---|
| RQ1 | How accurately can a CNN classify concrete surfaces as cracked or non-cracked? |
| RQ2 | Which model provides better overall performance across accuracy, precision, recall, and F1-score? |
| RQ3 | What effect does data augmentation have on overfitting? |
| RQ4 | Do Grad-CAM visualisations show the model focusing on crack regions? |
| RQ5 | What do the training curves reveal about model convergence and stability? |

---

## How to Run

### Option 1 — Local (recommended for this notebook)

```bash
# Clone the repository
git clone https://github.com/<your-username>/CNN_Crack_Detection.git
cd CNN_Crack_Detection

# Create and activate virtual environment (Python 3.11 required)
python3.11 -m venv cnn_env
source cnn_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

Then open `notebooks/Abhishek_CNN_Crack_Detection.ipynb`.

> **Important:** Update `POSITIVE_PATH` and `NEGATIVE_PATH` in Section 2 of the notebook to point to your local dataset folder before running.

### Option 2 — Kaggle Notebook

1. Upload the notebook to Kaggle
2. Add the [Surface Crack Detection dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) via **+ Add Data**
3. Update the paths in Section 2 to `/kaggle/input/surface-crack-detection/`
4. Enable GPU under **Settings → Accelerator → GPU**
5. Run all cells

### Option 3 — Google Colab

1. Upload the notebook to Colab
2. Download the dataset from Kaggle and upload to Google Drive
3. Mount Drive and update the paths in Section 2
4. Run all cells

---

## Requirements

```
tensorflow>=2.10
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyterlab
```

Install:

```bash
pip install -r requirements.txt
```

> **Python 3.11 is required.** TensorFlow does not support Python 3.12 or 3.13.

---

## Output Files

All figures are saved automatically as 300 DPI PNG files when the notebook runs:

| File | Description |
|---|---|
| `sample_images.png` | 3×3 grid of sample training images |
| `cnn_accuracy_loss_curves.png` | Custom CNN train vs. validation accuracy and loss |
| `cnn_confusion_matrix.png` | Custom CNN confusion matrix |
| `cnn_roc_curve.png` | Custom CNN ROC curve with AUC |
| `mobilenet_accuracy_loss_curves.png` | MobileNetV2 accuracy and loss curves |
| `mobilenet_confusion_matrix.png` | MobileNetV2 confusion matrix |
| `roc_comparison.png` | ROC curve comparison: both models |
| `model_comparison_chart.png` | Bar chart of all metrics side by side |
| `model_comparison.csv` | Comparison table saved as CSV |
| `gradcam_visualisation.png` | Grad-CAM heatmaps for 4 test images |
