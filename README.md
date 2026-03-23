# Few-Shot Visual Defect Detection using Siamese Networks

## Overview
This project implements a **few-shot anomaly detection system** using a Siamese Neural Network trained with contrastive loss.

The system detects defects in industrial components using only **5 normal reference images**, making it suitable for low-data scenarios.

A simple GUI is also provided for real-time image-based defect detection.

---

## Features
- Few-shot learning (5-shot setup)
- Siamese network with ResNet18 backbone
- Contrastive loss training
- Euclidean distance-based classification
- GPU-accelerated training (PyTorch)
- Interactive GUI for testing images
- Evaluation with accuracy, precision, recall, F1-score
- Confusion matrix visualization


---

## Methodology

### Model
- Backbone: ResNet18 (pretrained on ImageNet)
- Frozen layers: All except final block (layer4)
- Projection head: 512 → 256 → 128

### Training
- Loss: Contrastive Loss
- Optimizer: Adam (lr = 1e-4)
- Epochs: 10
- Batch size: 16

### Few-Shot Setup
- Support set: 5 normal images
- Mean embedding used as reference

### Inference
- Metric: Euclidean distance
- Threshold: 0.6 (selected empirically)

---

## Dataset

This project uses the **MVTec AD dataset (Screw category)**.

Download from:
https://www.mvtec.com/company/research/datasets/mvtec-ad

Place it as:

data/mvtec_ad/screw/


---

## Results

| Metric     | Value   |
|------------|--------|
| Accuracy   | 98.75% |
| Precision  | 100%   |
| Recall     | 98.3%  |
| F1 Score   | 99.15% |

Confusion matrix is available in `results/`.

---

## Running the Project

```bash
python train.py
python evaluate.py
python plot_confusion.py
python app.py
