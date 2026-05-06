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
**Lap-Level Pit-Stop Prediction in Formula 1 Racing Using Supervised Machine Learning**

---

## 2. Dataset Information

| Property | Value |
|---|---|
| **Dataset Title** | F1 Strategy Dataset (Lap-Level Race Data) |
| **Domain** | Sports Analytics / Data Science (Motorsports) |
| **Source** | [Kaggle: F1 Strategy Dataset](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction) |
| **Rows & Columns** | 101,371 rows × 17 columns |
| **Data Period** | 2019 – 2025 Formula 1 seasons |
| **Task Type** | Binary Classification (Pit-Stop Prediction) |
| **Target Variable** | `pit_stop` (0 = No pit stop, 1 = Pit stop) |
| **Class Balance** | ~82.3% No Pit, ~17.7% Pit Stop (Imbalanced) |

---

## 3. Problem Statement
The timing of Formula 1 pit stops is one of the most critical tactical moves in motorsports. A well-timed pit stop can win a race, while a poor decision can result in a significant loss of positions. Pit stop timing is influenced by complex, multi-dimensional factors including tyre degradation, safety car interventions, weather changes, and fuel loads. Currently, many of these decisions are made manually by race engineers under high-pressure environments. This project proposes a supervised learning approach to predict the likelihood of a pit stop occurring in any given lap. By analyzing historical race data from the 2019–2025 seasons, we aim to build a predictive model that assists in optimizing race strategy through data-driven insights.

---

## 4. Research Questions
*   **RQ1.** How effectively can baseline supervised learning models (Logistic Regression, Decision Tree, k-NN) predict pit stops on the F1 Strategy Dataset?
*   **RQ2.** Which supervised learning model achieves the best predictive performance for pit-stop classification, and how do models compare across all metrics?
*   **RQ3.** How do different data preprocessing strategies (raw data, imputation, scaling/encoding, full pipeline) affect model performance?
*   **RQ4.** Which input features contribute most to pit-stop prediction, and what domain insights can be derived from the top-ranked features?
*   **RQ5.** How does the relative ranking of candidate models change when evaluated across different metrics (Accuracy, Precision, Recall, F1-score, AUC)?
*   **RQ6.** How robust is the best-performing model under 5-fold cross-validation and varying data conditions?
*   **RQ7.** To what extent is the developed supervised learning solution practically useful and reliable for real-world F1 race-strategy decision-making?

---

## 5. Proposed Methodology

### 5.1 Data Preprocessing
*   **Cleaning**: Handling missing values through median/mode imputation.
*   **Balancing**: Addressing class imbalance using stratified splitting and evaluating through Precision-Recall curves.
*   **Feature Engineering**: Encoding categorical variables (e.g., `tyre_compound`) and scaling numerical features (e.g., `tyre_life`, `fuel_load`).
*   **Splitting**: 80% training / 20% testing set split.

### 5.2 Machine Learning Models
*   **Logistic Regression**: Linear baseline with L2 regularization.
*   **Decision Tree**: Interpretable non-linear baseline.
*   **k-Nearest Neighbors (k-NN)**: Instance-based classification.
*   **Random Forest**: Ensemble bagging method for high variance reduction.
*   **XGBoost**: Advanced gradient boosting for state-of-the-art accuracy.

### 5.3 Evaluation Metrics
*   **Primary**: F1-Score and AUC-ROC (due to class imbalance).
*   **Secondary**: Accuracy, Precision, Recall.
*   **Analysis**: Confusion Matrix and Gain-based Feature Importance.

---

## 6. Expected Figures and Tables

| # | Figure / Table | Purpose |
|---|---|---|
| **Table I** | Baseline Model Performance | Comparison of LR, DT, and k-NN |
| **Figure 1** | Baseline Performance Chart | Grouped bar chart of Accuracy, Precision, Recall, F1 |
| **Table II** | Full Model Comparison | Results for all 5 candidate models |
| **Figure 2** | Model Ranking Chart | Sorted bar chart by F1-score and AUC-ROC |
| **Table III** | Preprocessing Impact | Ablation study (Raw vs. Pipeline) |
| **Figure 3** | Preprocessing Gains | Line plot showing performance increase |
| **Table IV** | Top-10 Feature Importances | Ranking of predictors (Gain-based) |
| **Figure 4** | Feature Importance Plot | Bar plot of the top 10 influencers |
| **Table V** | Metric Sensitivity Ranking | Comparison of model ranks across metrics |
| **Figure 5** | Bump Chart | Visualization of model rank shifts |
| **Table VI** | 5-Fold CV Robustness | Stability analysis of the best model |
| **Figure 6** | CV Performance Boxplot | Variance visualization across folds |
| **Table VII** | Final Decision Matrix | Criteria-based model selection |
| **Figure 7** | Multi-criteria Radar Chart | Trade-off analysis (Performance vs. Speed vs. Interpretability) |
| **Figure 8** | Confusion Matrix | Error analysis of the recommended model |
