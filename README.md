
# Mahalanobis-based Anomaly Detection with EfficientNet and TensorFlow Lite

## Overview

This project implements Mahalanobis distance-based anomaly detection using EfficientNet models. The system supports both full TensorFlow models and TensorFlow Lite models with optional quantization. Features are extracted from multiple layers of the EfficientNet models, and anomaly detection is performed using the Mahalanobis distance.
It is expected to operate with edge AI

## Features

- **EfficientNet-based feature extraction**: Supports EfficientNet-B0 and EfficientNet-B4.
- **Mahalanobis distance for anomaly detection**.
- **Support for TensorFlow Lite**: Models can be converted to TensorFlow Lite, including quantized models.
- **Multiple layers feature extraction**: Extracts features from various layers of the EfficientNet model.
- **Evaluation with ROC curves and AUC**: Generates ROC curves and calculates AUC scores for anomaly detection.
- **Save extracted features**: Optionally saves extracted features to a pickle file.

## Requirements

- Python 3.6+
- TensorFlow
- TensorFlow Lite
- NumPy
- OpenCV
- scikit-learn
- SciPy
- tqdm
- Matplotlib

To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Arguments

- `--save_path`: Path to save results, models, and extracted features. Default is `./result`.
- `--save_feat`: Whether to save extracted features as a pickle file. Default is `True`.
- `--quantize`: Whether to enable quantization for the TensorFlow Lite model. Default is `True`.

- future - `--model_name`: Name of the EfficientNet model to use (`efficientnet-b0` or `efficientnet-b4`). Default is `efficientnet-b0`. 

### Example

To run the anomaly detection with EfficientNet-B0, TensorFlow Lite, and feature extraction:

```bash
python mahalanobis_anomaly_detection.py --model_name efficientnet-b0 --save_path ./result --save_feat --quantize
```

### Outputs

- **ROC Curves**: ROC curves are saved in the specified `save_path`.
- **AUC Scores**: The AUC score for the ROC curve is printed in the console.
- **Feature Files**: If `--save_feat` is enabled, extracted features are saved as pickle files in the `./result/feats` directory.

## Directory Structure

```
project_root/
├── datasets/
│   └── mvtec.py               # Dataset loader for MVTec anomaly detection dataset
├── result/
│   ├── feats/                 # Extracted features saved as pickle files
│   └── temp/                  # Temporary files during the execution
├── mahalanobis_anomaly_detection.py  # Main script for anomaly detection
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```
## References

- gaussian-ad-mvtec [source](https://github.com/ORippler/gaussian-ad-mvtec).
- MahalanobisAD-pytorch [source](https://github.com/byungjae89/MahalanobisAD-pytorch).
- 深層異常検知（１） Gaussian-AD; 学習時間ほぼなしの高性能異常検知アルゴリズム [page](https://qiita.com/makotoito/items/39bc64d30ce49a9edad8).
- The MVTec anomaly detection dataset (MVTec AD)  [page](https://www.mvtec.com/company/research/datasets/mvtec-ad)

