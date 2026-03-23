# Few-Shot Visual Defect Detection using Siamese Network

## Overview
This project implements a GPU-accelerated few-shot anomaly detection framework for industrial defect detection.

## Methodology
- Backbone: Pretrained ResNet18
- Training Strategy: Siamese Network with Contrastive Loss
- Embedding Dimension: 128
- Support Set: 5 normal samples (5-shot)
- Distance Metric: Euclidean Distance
- Threshold: 0.6 (selected via empirical distribution analysis)

## Dataset
MVTec AD - Screw category

## Training
- 10 epochs
- Frozen backbone except layer4
- Optimizer: Adam (lr=1e-4)

## Results
Accuracy: 98.75%
Precision: 100%
Recall: 98.3%
F1 Score: 99.15%

Confusion matrix available in results folder.
