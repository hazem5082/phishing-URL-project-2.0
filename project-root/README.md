# Phishing URL Detection System
**Cybersecurity ML Course — University Project**

A machine learning pipeline that ingests raw URLs from Hugging Face, engineers structural features, trains three classifier models, and exposes an interactive CLI demo for live phishing prediction.

---

## Pipeline Overview

```
data_loader.py  →  url_features.py  →  train_classifier.py  →  demo.py
  (download)        (feature eng.)       (train + evaluate)    (predict)
```

---

## Project Structure

```
project-root/
├── data/
│   ├── raw/                  # phishing_raw.csv  (output of data_loader.py)
│   └── processed/            # phishing_processed.csv  (output of url_features.py)
├── models/
│   └── phishing_rf_model.joblib   # Saved Random Forest model
├── results/
│   ├── cm_*.png              # Confusion matrix heatmaps (one per model)
│   ├── feature_importance_rf.png
│   └── roc_auc_curve.png
├── src/
│   ├── utils/
│   │   └── data_loader.py    # Downloads dataset from Hugging Face → data/raw/
│   ├── feature_engineering/
│   │   └── url_features.py   # Extracts 4 URL features → data/processed/
│   ├── models/
│   │   └── train_classifier.py  # Trains 3 models, saves charts + .joblib
│   └── demo.py               # CLI: loads .joblib and predicts any URL
├── tests/
│   └── test_basic.py
├── notebooks/
│   ├── EDA.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run (in order)

```bash
# 1. Download the dataset
python project-root/src/utils/data_loader.py

# 2. Extract URL features
python project-root/src/feature_engineering/url_features.py

# 3. Train models and generate visual artifacts
python project-root/src/models/train_classifier.py

# 4. Run the working demo
python project-root/src/demo.py
# Or pass a URL directly:
python project-root/src/demo.py "https://www.google.com"
```

---

## Script Descriptions

### `src/utils/data_loader.py`
Downloads the `ealvaradob/phishing-dataset` dataset from Hugging Face using the `datasets` library. Loads the `urls.json` split directly (bypassing the deprecated dataset script) and saves it as `data/raw/phishing_raw.csv`.

### `src/feature_engineering/url_features.py`
Reads the raw CSV and engineers 4 structural URL features:
| Feature | Description |
|---|---|
| `url_length` | Character count after stripping the protocol |
| `dot_count` | Number of dots (subdomain depth indicator) |
| `has_at_symbol` | 1 if `@` present (host-hiding trick) |
| `has_https` | 1 if original URL used HTTPS |

Outputs `data/processed/phishing_processed.csv`. Uses vectorized pandas operations for performance. Automatically samples 20,000 rows if the dataset exceeds that size.

### `src/models/train_classifier.py`
Trains and compares three models using an 80/20 train-test split:
- **Logistic Regression** — interpretable baseline
- **Random Forest** — ensemble with `class_weight='balanced'`
- **Gradient Boosting** — sequential boosting approach

Outputs to `results/`:
- Confusion matrix heatmaps per model
- Feature importance bar chart (Random Forest)
- Combined ROC-AUC comparison curve

Saves the final Random Forest to `models/phishing_rf_model.joblib`.

### `src/demo.py`
Loads the `.joblib` model and provides a CLI interface:
- Accepts a URL via command-line argument or interactive prompt
- Normalizes the URL before feature extraction (matching training data format)
- Automatically runs 10 hardcoded sample predictions (5 legitimate, 5 phishing)

---

## Features Extracted

The model operates on 4 structural signals derived from the raw URL string. No DNS lookups or external APIs are required, making predictions fully offline.

---

## Results Summary

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression (Baseline) | ~66% | 0.64 |
| Random Forest | ~72% | 0.72 |
| Gradient Boosting | ~73% | 0.72 |

> Note: Performance is bounded by the 4-feature design. Adding lexical entropy, TLD risk scoring, and domain age would push accuracy well above 90%.

---

## Known Limitations

The model cannot identify a URL by name (e.g., "google"). It classifies solely on structural patterns. Short HTTPS URLs (like `https://www.google.com`) are occasionally misclassified because the training dataset's legitimate URLs were stored without protocol prefixes — a dataset formatting artefact documented in the Technical Report.

---

## Tests

```bash
pytest --cov=project_root project-root/tests
```
