# Supervised Machine Learning Assignment
# Submission 1 – Research Proposal

**Programme**: MSc Data Science  
**Student**: Karnati Mysanthosh  
**Student ID**: 36733714  
**University**: University of Europe for Applied Sciences  
**Supervisor**: Prof. Raja Hashim Ali  
**Date**: 4th May 2026

---

## 1. Project Title
**Predicting NYC Taxi Fares Using Supervised Machine Learning: A Comparative Analysis of Regression Models**

---

## 2. Dataset Information

| Property | Value |
|---|---|
| **Dataset Title** | NYC Yellow Taxi Trip Record Data (2025) |
| **Domain** | Urban Transportation / Smart Cities |
| **Source** | [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| **Rows & Columns** | ~3.4 Million rows (sampled to 100,000) × 19 columns |
| **Data Period** | January 2025 |
| **Task Type** | Regression (Fare Prediction) |
| **Target Variable** | `fare_amount` (Continuous) |
| **Target Variable Range**| $2.50 – $300.00 (Cleaned) |

---

## 3. Problem Statement
Accurate fare estimation is essential for both passengers and ride-hailing services to ensure transparency and operational efficiency. In New York City, taxi fares are determined by a complex combination of distance, time, location-based surcharges, and traffic conditions. While traditional meters use fixed rates, supervised machine learning can provide more dynamic and accurate predictions by learning from millions of historical trips. 

In this project, we analyze the January 2025 NYC Yellow Taxi dataset to predict the `fare_amount`. We evaluate five different regression models: Linear Regression, Decision Tree, k-Nearest Neighbors (k-NN), Random Forest, and Gradient Boosting. The goal is to identify the most robust model for real-time fare estimation.

---

## 4. Research Questions
*   **RQ1.** How effectively can baseline regression models (Linear Regression, Decision Tree, k-NN) predict taxi fares on the NYC dataset?
*   **RQ2.** Which supervised learning model achieves the best predictive performance for fare regression, and how do models compare across RMSE, MAE, and R²?
*   **RQ3.** How do different data preprocessing strategies (raw data, imputation, scaling, and outlier removal) affect the accuracy of fare predictions?
*   **RQ4.** Which input features (e.g., trip distance, pickup hour, location) contribute most to the fare amount, and what insights can be derived from them?
*   **RQ5.** How does the relative ranking of candidate models change when evaluated across different metrics (RMSE vs. MAE vs. R²)?
*   **RQ6.** How robust is the best-performing regression model under 5-fold cross-validation and varying data conditions (noise/missingness)?
*   **RQ7.** To what extent is the developed supervised learning solution practically useful and reliable for a real-world ride-hailing deployment?

---

## 5. Proposed Methodology

### 5.1 Data Preprocessing
*   **Datetime Engineering**: Extracting `pickup_hour`, `day_of_week`, and `trip_duration_minutes`.
*   **Filtering**: Removing outliers (fares < $2.50, distances > 100 miles, durations > 3 hours).
*   **Feature Selection**: Dynamic selection of surcharges (`congestion_surcharge`, `airport_fee`) and location IDs.
*   **Scaling**: Standardizing numerical features for distance-based models.

### 5.2 Machine Learning Models
*   **Linear Regression**: Baseline linear relationship model.
*   **Decision Tree Regressor**: Non-linear baseline with depth optimization.
*   **k-Nearest Neighbors (k-NN)**: Instance-based regression (k=10).
*   **Random Forest Regressor**: Ensemble bagging for high-dimensional stability.
*   **Gradient Boosting Regressor**: Boosting method for minimizing residual errors.

### 5.3 Evaluation Metrics
*   **Root Mean Squared Error (RMSE)**: Primary metric for penalizing large errors.
*   **Mean Absolute Error (MAE)**: Metric for average prediction error in dollars.
*   **R-squared (R²)**: Measure of the proportion of variance explained by the model.

---

## 6. Expected Figures and Tables

| # | Figure / Table | Purpose |
|---|---|---|
| **Table I** | Baseline Regression Performance | Comparison of LR, DT, and k-NN |
| **Figure 1** | Baseline Metric Comparison | Grouped bar chart of RMSE and MAE |
| **Table II** | Full Model Comparison | Results for all 5 candidate models |
| **Figure 2** | Model Ranking (R² and RMSE) | Sorted horizontal bar charts |
| **Table III** | Preprocessing Ablation Study | Performance gains from raw to full pipeline |
| **Figure 3** | Preprocessing Impact Plot | Visualizing error reduction across strategies |
| **Table IV** | Feature Importance Ranking | Top-10 features (Random Forest) |
| **Figure 4** | Feature Importance Bar Plot | Ranking of predictors by importance score |
| **Table V** | Metric Sensitivity Table | Model ranks across RMSE, MAE, and R² |
| **Figure 5** | Model Ranking Bump Chart | Visualizing rank stability |
| **Table VI** | 5-Fold CV & Noise Analysis | Robustness testing of the best model |
| **Figure 6** | Error Distribution Plot | Histogram of residuals for the recommended model |
| **Table VII** | Final Decision Matrix | Multi-criteria selection (Accuracy vs. Latency) |
| **Figure 7** | Multi-criteria Radar Chart | Visualizing model trade-offs |
| **Figure 8** | Predicted vs. Actual Plot | Scatter plot showing model alignment |
