# 📱 Mobile Ad Fraud Detection — Click-to-Download Prediction

A full machine learning pipeline for detecting fraudulent mobile ad clicks using the lighter, feature-enriched version of the [TalkingData AdTracking Fraud Detection](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection) dataset. This version was chosen for its richer pre-computed features in parquet format compared to the raw competition data. The task is to predict whether a user will download an app after clicking a mobile advertisement (`is_attributed`).

---

## Overview

Mobile ad fraud costs the industry billions of dollars annually. Fraudsters generate massive volumes of fake clicks without any real user intent to download. This project builds and compares six classical machine learning classifiers to distinguish genuine download-intent clicks from fraudulent ones.

**Target variable:** `is_attributed` — 1 if the user downloaded the app, 0 otherwise.

The dataset was split 80/20 for training and testing — no separate test CSV was needed since this lighter version includes all necessary features in one place.

---

## Dataset

**Source:** [TalkingData AdTracking — Lighter Version with Parquet Features](https://www.kaggle.com/datasets/matleonard/feature-engineering-data) by Matt Leonard on Kaggle.

This is the lighter, feature-enriched alternative to the original TalkingData competition dataset. It includes `train_sample.csv` (100k click records) alongside pre-computed feature engineering files in parquet format that provide richer signals than the raw data alone.

| File | Description |
|------|-------------|
| `train_sample.csv` | 100k raw click records with basic features |
| `baseline_data.pqt` | Temporal features extracted from click timestamps |
| `catboost_encodings.pqt` | Target encodings for categorical columns |
| `count_encodings.pqt` | Frequency encodings per category |
| `downloads.pqt` | Aggregated download counts per IP/app/device |
| `time_deltas.pqt` | Time between consecutive clicks from same user |
| `interactions.pqt` | Interaction features between columns |
| `past_6hr_events.pqt` | Click count in past 6 hours per IP |
| `svd_encodings.pqt` | SVD-based embeddings for categorical features |

**Class imbalance:** Only ~0.2% of clicks result in a download — heavily imbalanced.

---

## Pipeline

```
train_sample.csv + Parquet Feature Files
              ↓
      Feature Engineering
  • Temporal features (hour, minute, second, day)
  • CatBoost target encoding for app, device, OS, channel
  • Download counts & time-delta aggregations
              ↓
    Train / Test Split (80% / 20%, stratified)
              ↓
    Class Balancing (RandomUnderSampler → 50/50)
              ↓
      Feature Scaling (StandardScaler)
              ↓
    Model Training & Evaluation
    • 6 classifiers compared
    • Train & test accuracy reported for each
    • Confusion Matrix, ROC/AUC Curve, Decision Boundary
```

---

## Models & Results

All models were trained on the balanced training set and evaluated on the held-out 20% test set.

| Model | Train Acc | Test Acc | AUC |
|-------|-----------|----------|-----|
| **Random Forest** | 100.00% | 94.85% | **0.93** |
| Decision Tree | 92.32% | 94.46% | 0.92 |
| Logistic Regression | 90.92% | 95.14% | 0.91 |
| KNN | 92.02% | 93.74%* | 0.91 |
| SVM | 92.00% | 94.19% | 0.90 |
| Naive Bayes | 90.02% | 93.63% | 0.90 |

> *KNN evaluated on a 10,000-sample subset due to computational constraints — KNN prediction time scales O(n) with dataset size, making full 460k evaluation impractical.

> Random Forest's 100% train accuracy indicates overfitting to the training set, but strong test performance confirms it still generalizes well.

**Winner: Random Forest** with the highest AUC of 0.93.

---

## Key Visualizations

### All Models — ROC Curve Comparison
All six models significantly outperform the random baseline, with AUC scores ranging from 0.90 to 0.93.

### Feature Importances
Across all tree-based models, **`app`** and **`channel`** consistently rank as the most predictive features — indicating that which app is being advertised and through which channel is the strongest signal for genuine download intent.

### Decision Boundaries
Each classifier partitions the feature space differently (projected onto `channel` × `app`):
- **Logistic Regression & SVM** — clean linear boundaries
- **Decision Tree** — rectangular axis-aligned splits
- **Random Forest** — complex irregular regions
- **Naive Bayes** — smooth elliptical boundary
- **KNN** — highly local, wiggly boundary

---

## How to Run

### On Kaggle (recommended)
1. Create a new Kaggle notebook
2. Add the dataset: [feature-engineering-data](https://www.kaggle.com/datasets/matleonard/feature-engineering-data)
3. Set `DATA_ROOT` in the notebook:
```python
DATA_ROOT = '/kaggle/input/feature-engineering-data/'
```
4. Run all cells
---

## Dependencies

```
pandas
numpy
scikit-learn
imbalanced-learn
mlxtend
seaborn
matplotlib
pyarrow      # for reading .pqt parquet files
```

---

## Notes

- SVM is trained on a 300-sample balanced subset for speed; it would likely improve with more data
- KNN uses an elbow plot over k = 1, 3, 5, 7, 9 to select the optimal number of neighbors automatically
- All plots are automatically saved as PNG files when the notebook is run
