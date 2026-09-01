# Click-Through Rate Prediction with LightGBM

A practical data science project that predicts whether a user will click an online offer. The project focuses on **feature engineering, CTR modeling, imbalanced classification, and model deployment with Flask**.

## Why this project?

CTR prediction is a common machine-learning problem in digital advertising. The interesting part is not only training a classifier, but turning raw ad-impression logs into useful signals such as:

- user/offer frequency features
- hour-of-day and day-of-week features
- user-offer interaction signals
- demographic/context information such as country
- probability calibration and threshold selection

## Dataset

This project uses **KASANDR**, a publicly available advertising/recommendation dataset from the UCI Machine Learning Repository. It contains user/offer interactions from Kelkoo and includes a binary click/implicit-feedback target, user IDs, offer IDs, country, category, merchant and timestamp information. KASANDR is licensed under **CC BY 4.0**.

Dataset page: https://archive.ics.uci.edu/dataset/385/kasandr

The complete archive is large, so it is intentionally **not committed to this repository**. Download the data from UCI and place `train_de.csv` inside `data/raw/`.

## Repository structure

```text
click-through-rate-prediction/
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
├── data/
│   ├── raw/
│   │   ├── train_de.csv        # download separately
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── models/
│   ├── ctr_pipeline.joblib     # created by notebook 03
│   └── .gitkeep
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Workflow

```text
Raw interaction logs
        ↓
Data quality checks + EDA
        ↓
Feature engineering
        ↓
Time-aware train/validation split
        ↓
LightGBM classifier
        ↓
ROC-AUC / PR-AUC / Log Loss
        ↓
Saved preprocessing + model pipeline
        ↓
Flask web app
```

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/InfinitePraveen/Click-Through-Rate-Prediction.git
cd Click-Through-Rate-Prediction
```

### 2. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download KASANDR from UCI and put the German training file at:

```text
data/raw/train_de.csv
```

### 5. Run the notebooks in order

1. `01_data_exploration.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_model_training.ipynb`

For a laptop, start with a manageable sample such as 200,000–500,000 rows. The full dataset is much larger than a normal interview laptop can comfortably process.

### 6. Start the Flask app

After notebook 03 creates `models/ctr_pipeline.joblib`:

```bash
python app/app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Interview talking points

### Why LightGBM?

LightGBM is a strong choice for tabular advertising data because it handles nonlinear relationships, trains efficiently, and works well with many categorical-derived and numeric features.

### Why not accuracy?

CTR data is normally imbalanced. A model can obtain high accuracy by mostly predicting "no click". ROC-AUC, PR-AUC and log loss are more useful for evaluating ranking/probability quality.

### Why a time-aware split?

Ad behavior changes over time. Randomly mixing future impressions into training can produce optimistic results. The project therefore demonstrates a chronological validation strategy.

### Important leakage warning

Historical CTR/frequency features must be calculated using only information available before the impression being predicted. The notebook keeps this principle explicit rather than using a simple random aggregation across the entire dataset.

## Web app

The Flask app is intentionally small: it loads the saved pipeline, accepts an impression profile, returns a click probability, and explains the result in plain language.

The app footer links to my profiles:

- GitHub: https://github.com/InfinitePraveen
- LinkedIn: https://www.linkedin.com/in/infinitepraveen/

## Dataset attribution

Sidana, S., Laclau, C., & Amini, M. (2017). KASANDR. UCI Machine Learning Repository. https://doi.org/10.24432/C5PK7M

## License

The project code is MIT licensed. Dataset rights remain with the dataset provider and are governed by the KASANDR CC BY 4.0 license.
