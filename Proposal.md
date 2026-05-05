# Supervised Learning Assignment — Project Proposal

**Project Title**: NYC Taxi Fare Prediction: A Supervised Regression Learning Approach

---

## 1. Dataset Information

| Property | Details |
|---|---|
| **Dataset Name** | NYC Yellow Taxi Trip Record Data (2025 Updated) |
| **Source Link** | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page |
| **Direct Download** | https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet |
| **Number of Rows** | > 3,000,000 (subset of ~500,000 used) |
| **Number of Columns** | 19 (18 input features + 1 target) |
| **Format** | Parquet / CSV |
| **Updated** | 2025 ✅ |

---

## 2. Target Variable

**`fare_amount`** — the base fare in USD charged for a taxi trip in New York City.

---

## 3. Type of Task

**Regression** — The goal is to predict a continuous numeric value (fare amount in dollars).

---

## 4. Problem Statement

The NYC taxi industry processes millions of trips per month, yet fare pricing remains opaque to passengers. Accurate fare prediction enables passengers to budget their travel costs, allows drivers to optimize route planning, and helps regulators detect billing anomalies. This project applies supervised machine learning regression models to the NYC Taxi Trip Record Data (2025) to predict taxi fare amounts based on trip characteristics such as distance, duration, time of day, pickup/dropoff locations, and surcharge features. We systematically compare multiple models, evaluate the impact of preprocessing, identify the most influential features, and assess robustness under varying experimental conditions.

---

## 5. Research Questions

### RQ1 — Baseline Performance
How effectively can baseline supervised learning models predict NYC taxi fares on the 2025 trip dataset?

### RQ2 — Model Comparison
Which supervised learning model achieves the best predictive performance for taxi fare prediction, and how do the models compare across regression metrics?

### RQ3 — Effect of Preprocessing
How do different data preprocessing strategies (raw data, imputation, scaling, full pipeline) affect model performance?

### RQ4 — Feature Importance and Interpretability
Which input features contribute most to fare prediction, and what domain insights can be drawn from the most influential variables?

### RQ5 — Sensitivity to Evaluation Metrics
How does the relative ranking of candidate models change when RMSE, MAE, and R² are considered separately?

### RQ6 — Robustness and Generalization
How robust is the best-performing model under different train-test splits, k-fold cross-validation, and data perturbation (noise and missing values)?

### RQ7 — Practical Usefulness and Final Recommendation
To what extent is the developed supervised learning solution practically useful, interpretable, and reliable for real-world taxi fare estimation?

---

## 6. Proposed Methodology

1. **Data Collection**: Load January 2025 Yellow Taxi Parquet data directly from the NYC TLC S3 endpoint.
2. **Preprocessing Pipeline**:
   - Remove invalid records (negative fares, zero distance, extreme outliers)
   - Handle missing values via median imputation
   - Extract datetime features: pickup hour, day of week, trip duration
   - Standardize numerical features using StandardScaler
   - Encode categorical features
3. **Exploratory Data Analysis**: Fare distributions, correlation heatmaps, fare vs. distance scatter plots, hourly patterns.
4. **Model Training**: 80/20 train-test split. Five regression models trained:
   - Linear Regression
   - Decision Tree Regressor
   - k-Nearest Neighbours Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
5. **Evaluation**: RMSE, MAE, R² across all models.
6. **Advanced Analysis**: Preprocessing ablation, feature importance, cross-validation, robustness under noise and missingness, final decision matrix.

---

## 7. Machine Learning Models

| Model | Type |
|---|---|
| Linear Regression | Baseline linear |
| Decision Tree Regressor | Non-linear, interpretable |
| k-NN Regressor | Instance-based |
| Random Forest Regressor | Ensemble (bagging) |
| Gradient Boosting Regressor | Ensemble (boosting) |

---

## 8. Evaluation Metrics

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error — penalises large prediction errors |
| **MAE** | Mean Absolute Error — average dollar error per prediction |
| **R²** | Coefficient of Determination — proportion of variance explained |

---

## 9. Expected Figures and Tables

| # | Figure / Table | RQ |
|---|---|---|
| Table I | Baseline model performance (RMSE, MAE, R²) | RQ1 |
| Figure 1 | Grouped bar chart of baseline model metrics | RQ1 |
| Table II | Comparative performance of all candidate models | RQ2 |
| Figure 2 | Horizontal bar or radar chart of model rankings | RQ2 |
| Table III | Impact of preprocessing strategies on best model | RQ3 |
| Figure 3 | Ablation bar chart of preprocessing strategies | RQ3 |
| Table IV | Top-10 feature importances with direction & interpretation | RQ4 |
| Figure 4 | Feature importance bar plot (Random Forest / SHAP) | RQ4 |
| Table V | Model rankings under RMSE, MAE, and R² | RQ5 |
| Figure 5 | Slope / bump chart of metric-based rankings | RQ5 |
| Table VI | Robustness under standard split, CV, noise, missingness | RQ6 |
| Figure 6 | Boxplot or line chart of cross-validation performance | RQ6 |
| Table VII | Final decision matrix (performance, interpretability, cost) | RQ7 |
| Figure 7 | Radar chart of final model trade-off analysis | RQ7 |
