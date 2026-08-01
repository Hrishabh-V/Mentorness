# 🎓 Mentorness — Data Science Internship Tasks

**Author:** Hrishabh V
**Program:** Mentorness Data Science / Machine Learning Internship
**Repository:** [Hrishabh-V/Mentorness](https://github.com/Hrishabh-V/Mentorness)

A collection of three internship tasks covering an educational content reel on gradient-boosting internals, a binary classification model for cardiac risk, and a regression model for restaurant rating prediction. Each task folder is self-contained with its dataset, Jupyter notebook, presentation deck, and a walkthrough video.




---

## 📖 Overview

| # | Task | Type | Core Technique | Deliverables |
|---|------|------|-----------------|---------------|
| 1 | XGBoost — Missing Values & Categorical Features | Educational reel | Conceptual explanation | `.mp4` |
| 2 | Heart Attack Risk Prediction | Binary classification | `XGBClassifier` + `GridSearchCV` | `.ipynb`, `.pptx`, `.mp4`, dataset |
| 3 | Restaurant Rating Prediction | Regression | `RandomForestRegressor` | `.ipynb`, `.pptx`, `.mp4`, dataset |


## 🎬 Task 1 — XGBoost: Missing Values & Categorical Features (Reel)

**Objective:** Produce a short-form educational video explaining how XGBoost natively handles two of the messiest parts of real-world tabular data — missing values and categorical features — without requiring manual imputation for the former.

**Concepts covered in the reel:**
- **Sparsity-aware split finding** — XGBoost learns a *default direction* for each tree node during training, so rows with missing values are automatically routed left or right based on which direction minimizes loss, instead of being dropped or manually imputed.
- **Native missing-value support** — `missing=nan` handling built into `DMatrix`/`XGBClassifier`, contrasted with traditional models (e.g. scikit-learn's `LogisticRegression`) that require explicit imputation.
- **Categorical feature handling** — the difference between manual one-hot/label encoding (used in this repo's own Task 2 pipeline) and XGBoost's newer built-in categorical support (`enable_categorical=True`) that splits directly on category subsets.

**Deliverable:** `Task1-REEL/Handling Missing Values and Categorical Features in XGBoostppt1.mp4`

---

## ❤️ Task 2 — Heart Attack Risk Prediction

**Objective:** Build a machine learning model that predicts the presence of heart attack risk (`target`: 0 = low risk, 1 = high risk) from a patient's clinical and demographic attributes.

### Dataset
`data (1).xlsx` — the classic **UCI Heart Disease** dataset schema, 302 records × 14 columns.

| Column | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia type |
| `target` | **Label** — heart attack risk (0/1) |

### Pipeline (`Heart attack prediction.ipynb`)

1. **EDA** — `.head()`, `.info()`, `.describe()`, histograms of all 14 features.
2. **Data cleaning**
   - Missing-value check (none found).
   - Outlier removal on `chol` using the IQR method (5 outliers dropped → 297 rows retained).
3. **Encoding** — one-hot encoding (`pd.get_dummies`, `drop_first=True`) applied to the nominal categorical columns `cp`, `restecg`, `slope`, `thal`, expanding the feature space to **20 columns**.
4. **Train/test split** — 80/20 split → 237 training rows, 60 test rows (`random_state=42`).
5. **Baseline model** — `xgboost.XGBClassifier(random_state=42)` trained on default hyperparameters.
6. **Hyperparameter tuning** — `GridSearchCV` (3-fold CV) over:
   ```python
   param_grid = {
       'n_estimators':     [100, 200, 300],
       'max_depth':        [3, 4, 5],
       'learning_rate':    [0.01, 0.1, 0.2],
       'subsample':        [0.8, 0.9, 1.0],
       'colsample_bytree': [0.8, 0.9, 1.0]
   }
   ```
   243 candidate combinations × 3 folds = **729 model fits**.
7. **Feature importance** — top-10 feature plot via `xgb.plot_importance`.
8. **Model export** — best estimator persisted to `best_xgboost_model.h5` (saved in UBJSON format internally by XGBoost).

### Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Baseline `XGBClassifier` | 78.33% | 86.21% | 73.53% | 79.37% |
| **Tuned (GridSearchCV)** | 78.33% | **88.89%** | 70.59% | 78.69% |

**Best hyperparameters found:**
```python
{'colsample_bytree': 1.0, 'learning_rate': 0.2, 'max_depth': 5,
 'n_estimators': 100, 'subsample': 1.0}
```

Tuning traded a small amount of recall for a meaningful precision gain — relevant in a medical-risk context where reducing false positives (unnecessary alarm) was weighted against catching true positives.

**Deliverables:** notebook, presentation deck, walkthrough video, source spreadsheet.

---

## 🍽️ Task 3 — Restaurant Rating Prediction

**Objective:** Predict a restaurant's `Aggregate rating` (0–5 scale) from operational and descriptive features such as cuisine, cost, and service options.

### Dataset
`Dataset.csv` — Zomato-style restaurant listing dataset, **9,551 rows × 21 columns**, including `Restaurant Name`, `City`, `Cuisines`, `Average Cost for two`, `Has Table booking`, `Has Online delivery`, `Price range`, `Votes`, `Rating color`, `Rating text`, and the target `Aggregate rating`.

### Pipeline (`Restaurant rating prediction.ipynb`)

1. **EDA** — structure, missing-value audit, numerical + categorical summary statistics.
2. **Missing-value handling** — `Cuisines` (9 missing) imputed with the placeholder `'Unknown'`.
3. **Encoding**
   - Binary Yes/No columns (`Has Table booking`, `Has Online delivery`, `Is delivering now`, `Switch to order menu`) mapped to `1`/`0`.
   - Multi-category columns (`Cuisines`, `Currency`, `Rating color`, `Rating text`) one-hot encoded.
   - `Country Code` cast to a categorical dtype.
4. **Feature engineering** — derived a `Cost per Vote` feature: `Average Cost for two / (Votes + 1)`.
5. **Column pruning** — dropped identifier/free-text/geo columns not useful for the model: `Restaurant ID`, `Restaurant Name`, `City`, `Address`, `Locality`, `Locality Verbose`, `Longitude`, `Latitude`.
6. **Train/test split** — 80/20 split, `random_state=42`.
7. **Scaling** — `StandardScaler` fit on training data and applied to `Average Cost for two`, `Votes`, and `Cost per Vote`.
8. **Model** — `RandomForestRegressor(random_state=42)`.
9. **Evaluation** — MAE, MSE, R², plus:
   - Actual-vs-predicted scatter plot with a diagonal reference line.
   - Prediction error distribution plot.
10. **Inference utility** — a `preprocess_sample_input()` + `predict_rating()` helper pair that takes a raw dict of restaurant attributes, reindexes it to match the training feature space, and returns a predicted rating — useful as a lightweight scoring function for new records.

### Results

| Metric | Value |
|---|---|
| **MAE** | 0.1169 |
| **MSE** | 0.0313 |
| **R² Score** | **0.9862** |

The Random Forest regressor explains ~98.6% of the variance in aggregate ratings, with an average prediction error of ~0.12 points on a 5-point scale — indicating `Votes` and encoded rating-text/color fields (which strongly correlate with the target) carry substantial predictive signal.

**Deliverables:** notebook, presentation deck, walkthrough video, source CSV.

---

## 📊 Results Summary

| Task | Problem Type | Model | Key Metric |
|---|---|---|---|
| 2 — Heart Attack Prediction | Binary classification | Tuned XGBoost | 78.3% accuracy · 88.9% precision |
| 3 — Restaurant Rating Prediction | Regression | Random Forest | R² = 0.986 · MAE = 0.12 |

---

## 🧰 Tech Stack

| Category | Libraries |
|---|---|
| Language | Python 3 |
| Data handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Modeling | `scikit-learn`, `xgboost` |
| File I/O | `openpyxl` (Excel), built-in `csv` |
| Environment | Jupyter Notebook |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Hrishabh-V/Mentorness.git
cd Mentorness
```



```
Mentorness/
├── README.md
├── task1_xgboost_reel/
│   └── media/xgboost_missing_categorical.mp4
├── task2_heart_attack_prediction/
│   ├── data/data.xlsx
│   ├── notebooks/heart_attack_prediction.ipynb
│   └── reports/ (pptx, mp4)
└── task3_restaurant_rating_prediction/
    ├── data/dataset.csv
    ├── notebooks/restaurant_rating_prediction.ipynb
    └── reports/ (pptx, mp4)
```


---

## 🙏 Acknowledgements

Tasks completed as part of the **Mentorness** Machine Learning internship program.
