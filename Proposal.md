# Supervised Learning Assignment Proposal

**Project Title**: NYC Taxi Fare Prediction: A Supervised Learning Approach

## Dataset Overview
- **Dataset Name**: NYC Taxi Trip Record Data (2025 Updated)
- **Source Link**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Number of Rows**: > 10,000 (Millions available, subset used for computation)
- **Number of Columns**: 18-20 features
- **Target Variable**: `fare_amount`
- **Type of Task**: Regression

## Problem Statement
The objective of this project is to build a machine learning model to accurately predict the fare amount of a taxi ride in New York City based on trip characteristics such as trip distance, pickup/dropoff locations, time of day, and passenger count. Accurate fare prediction is vital for both drivers and passengers to ensure transparency and efficiency in the transportation ecosystem.

## Research Questions
1. How does the trip distance linearly and non-linearly affect the total fare amount?
2. Are there specific times of the day or days of the week that result in higher average fares due to traffic or surcharges?
3. How do pickup and drop-off locations (zones) impact the final fare?
4. What is the impact of passenger count on the fare amount, if any?
5. Which machine learning algorithm provides the best balance between predictive accuracy and computational efficiency for this dataset?
6. Can feature engineering (such as extracting hour, day, and month from pickup datetime) significantly improve model performance?

## Proposed Methodology
1. **Data Collection**: Download the structured Parquet/CSV format data from the NYC TLC website.
2. **Data Preprocessing**:
   - Handle missing values (imputation or removal).
   - Filter out outliers (e.g., negative fares, zero distances, or unrealistic geographic coordinates).
   - Convert datetime columns into usable numerical features (hour, day of week).
3. **Exploratory Data Analysis (EDA)**: Use visualizations to understand the distribution of fares and correlations between features.
4. **Feature Engineering**: Standardize numerical features and encode categorical ones.
5. **Model Training**: Split data into 80% training and 20% testing sets. Train multiple regression models.
6. **Model Evaluation**: Compare models using standard regression metrics.

## Machine Learning Models to be Used
- **Linear Regression**: As a baseline model to establish simple linear relationships.
- **Random Forest Regressor**: To capture non-linear relationships and interactions among features.
- **Gradient Boosting Regressor (e.g., XGBoost / LightGBM)**: For state-of-the-art predictive performance on tabular data.

## Evaluation Metrics
The performance of the regression models will be evaluated using:
- **Root Mean Squared Error (RMSE)**: To penalize large errors in fare prediction.
- **Mean Absolute Error (MAE)**: To understand the average absolute dollar error of predictions.
- **R-squared ($R^2$) Score**: To determine the proportion of variance in the fare amount explained by the model.

## Expected Figures and Tables
- **Figures**:
  - Histogram of the target variable (`fare_amount`).
  - Scatter plot of `trip_distance` vs. `fare_amount`.
  - Feature Importance bar chart (from Random Forest/Gradient Boosting).
  - Actual vs. Predicted fare amount scatter plot to visualize model performance.
- **Tables**:
  - Summary statistics table of the cleaned dataset.
  - Model comparison table detailing RMSE, MAE, and $R^2$ scores across the chosen algorithms.
