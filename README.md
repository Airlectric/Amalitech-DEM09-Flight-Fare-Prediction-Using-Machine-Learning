# Flight Fare Prediction Using Machine Learning

A complete machine learning pipeline for predicting flight ticket prices on Bangladesh routes. The project covers data loading, cleaning, exploratory analysis, feature engineering, model training, hyperparameter tuning, and model interpretation.

**Deployed App:** [flight-price-analysis-prediction-airlectric.streamlit.app](https://flight-price-analysis-prediction-airlectric.streamlit.app/)
**Deployment Repo:** [Amalitech-DEM09-Machine-Learning-Deployment-of-Flight-Price-Analysis-](https://github.com/Airlectric/Amalitech-DEM09-Machine-Learning-Deployment-of-Flight-Price-Analysis-.git)

## Project Structure

```
Amalitech-DEM09-Flight-Fare-Prediction-Using-Machine-Learning/
├── data/
│   └── raw/
│       └── Flight_Price_Dataset_of_Bangladesh.csv   # 57,000 records
├── models/
│   ├── best_model.pkl          # Tuned Gradient Boosting model
│   ├── scaler.pkl              # Fitted RobustScaler
│   └── feature_names.txt       # 248 feature names
├── notebooks/
│   └── flight-fare-prediction.ipynb   # End-to-end pipeline notebook
├── reports/
│   └── model_comparison.csv    # All model metrics
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load and inspect CSV data
│   ├── data_cleaning.py        # Clean and preprocess raw data
│   ├── eda.py                  # Exploratory Data Analysis and visualizations
│   ├── feature_engineering.py  # Feature creation, encoding, scaling, splitting
│   ├── models.py               # Train and evaluate 7 regression models
│   ├── optimization.py         # Hyperparameter tuning (RandomizedSearchCV)
│   ├── interpretation.py       # Feature importance and business insights
│   └── logger.py               # Centralized logging
├── logs/
│   └── pipeline.log
├── requirements.txt
└── setup.py
```

## Dataset

**Source:** `data/raw/Flight_Price_Dataset_of_Bangladesh.csv`
- **57,000** flight records with **17 columns**
- No missing values or duplicates

| Column | Type | Description |
|--------|------|-------------|
| Airline | Categorical | 24 airlines (Biman Bangladesh, Emirates, Turkish Airlines, etc.) |
| Source | Categorical | 8 departure airport codes (DAC, CGP, CXB, JSR, ZYL, RJH, SPD, BZL) |
| Source Name | Categorical | Full airport names |
| Destination | Categorical | 20 arrival airport codes (domestic + international) |
| Destination Name | Categorical | Full airport names |
| Departure Date & Time | Datetime | Departure timestamp |
| Arrival Date & Time | Datetime | Arrival timestamp |
| Duration (hrs) | Float | Flight duration (0.5 - 15.8 hrs) |
| Stopovers | Categorical | Direct, 1 Stop, 2 Stops |
| Aircraft Type | Categorical | Airbus A320/A350, Boeing 737/777/787 |
| Class | Categorical | Economy, Business, First Class |
| Booking Source | Categorical | Direct Booking, Online Website, Travel Agency |
| Base Fare (BDT) | Float | Base price (dropped as data leakage) |
| Tax & Surcharge (BDT) | Float | Tax amount (dropped as data leakage) |
| **Total Fare (BDT)** | **Float** | **Target variable** (1,801 - 558,987 BDT, mean: 71,030) |
| Seasonality | Categorical | Regular, Eid, Hajj, Winter Holidays |
| Days Before Departure | Integer | Booking lead time (1 - 90 days) |

## Pipeline Overview

```
Raw CSV (57,000 x 17)
  │
  ├── [1] Data Loading ─────────── Load and inspect dataset
  ├── [2] Data Cleaning ────────── Fix invalid entries, normalize names, convert types
  ├── [3] EDA ──────────────────── Distribution plots, KPIs, seasonal analysis
  ├── [4] Feature Engineering ──── Remove leakage, extract date features, one-hot encode,
  │                                 scale with RobustScaler, 80/20 split → 248 features
  ├── [5] Model Training ───────── Train 7 models (Linear, Ridge, Lasso, Decision Tree,
  │                                 Random Forest, Gradient Boosting, XGBoost)
  ├── [6] Hyperparameter Tuning ── RandomizedSearchCV on top models
  └── [7] Interpretation ───────── Feature importance, business insights
                                    │
                                    └── Artifacts: best_model.pkl, scaler.pkl, feature_names.txt
```

## Feature Engineering

The pipeline transforms the 17 raw columns into **248 model features**:

1. **Data leakage removal** — `Base Fare` and `Tax & Surcharge` are dropped (they directly sum to the target)
2. **Date feature extraction** — `Month`, `Day`, `Weekday`, `Hour`, `Season` derived from departure datetime
3. **Route derivation** — `Route_Combined` created from Source + Destination codes (152 unique routes)
4. **One-hot encoding** — All categorical columns encoded with `pd.get_dummies(drop_first=True)`
5. **Feature scaling** — `RobustScaler` applied to 6 numerical features (resistant to fare outliers)
6. **Train/test split** — 80/20 (45,600 training / 11,400 test samples)

## Model Results

| Model | Test R² | Test RMSE (BDT) | Test MAE (BDT) | CV R² |
|-------|---------|-----------------|-----------------|-------|
| **Gradient Boosting (Tuned)** | **0.6788** | **46,273** | **28,116** | **0.6832** |
| Random Forest (Tuned) | 0.6785 | 46,293 | 27,944 | 0.6832 |
| Random Forest | 0.6776 | 46,357 | 27,848 | 0.6799 |
| XGBoost | 0.6771 | 46,395 | 27,945 | 0.6807 |
| Gradient Boosting | 0.6770 | 46,401 | 27,927 | 0.6805 |
| Decision Tree | 0.6455 | 48,610 | 28,726 | 0.6454 |
| Lasso Regression | 0.5686 | 53,625 | 40,765 | 0.5711 |
| Ridge Regression | 0.5686 | 53,629 | 40,772 | 0.5710 |
| Linear Regression | 0.5686 | 53,629 | 40,775 | 0.5710 |

**Best model:** Gradient Boosting (Tuned) — R² = 0.6788, explaining ~68% of fare variance.

## Key Findings

**Top fare drivers (feature importance):**
1. Flight Duration — 42.0%
2. First Class ticket — 38.6%
3. Economy Class — 6.3%
4. Destination (Kolkata) — 5.4%

**Seasonal pricing:**
| Season | Average Fare (BDT) |
|--------|-------------------|
| Hajj | 97,144 |
| Eid | 91,560 |
| Winter Holidays | 79,677 |
| Regular | 68,077 |

## Getting Started

### Prerequisites

- Python >= 3.9

### Installation

```bash
# Clone the repository
git clone https://github.com/Airlectric/Amalitech-DEM09-Flight-Fare-Prediction-Using-Machine-Learning.git
cd Amalitech-DEM09-Flight-Fare-Prediction-Using-Machine-Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/flight-fare-prediction.ipynb
```

The notebook executes the full pipeline from data loading through model training and generates all artifacts in the `models/` directory.

### Using the Source Modules

```python
from src.data_loader import load_dataset, inspect_dataset
from src.data_cleaning import clean_dataset
from src.feature_engineering import run_feature_pipeline
from src.models import compare_all_models

# Load and clean
df = load_dataset()
df = clean_dataset(df)

# Feature engineering
X_train, X_test, y_train, y_test, scaler, feature_names = run_feature_pipeline(df)

# Train all models
results = compare_all_models(X_train, X_test, y_train, y_test, feature_names)
```

## Saved Artifacts

| File | Size | Description |
|------|------|-------------|
| `models/best_model.pkl` | 214 KB | Tuned GradientBoostingRegressor |
| `models/scaler.pkl` | 649 B | RobustScaler fitted on 6 numerical features |
| `models/feature_names.txt` | 6.3 KB | 248 feature names in training order |
| `reports/model_comparison.csv` | — | Metrics for all 9 model variants |

## Tech Stack

- **ML:** scikit-learn, XGBoost
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Environment:** Jupyter, Python 3.11+
- **Deployment:** Streamlit

---

**Built with Streamlit & Scikit-learn** | [Live App](https://flight-price-analysis-prediction-airlectric.streamlit.app/) | [Deployment Repo](https://github.com/Airlectric/Amalitech-DEM09-Machine-Learning-Deployment-of-Flight-Price-Analysis-.git)
