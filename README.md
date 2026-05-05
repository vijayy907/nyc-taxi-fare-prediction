# 🚕 NYC Taxi Fare Prediction: A Supervised Learning Approach

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Dataset](https://img.shields.io/badge/Dataset-NYC%20TLC%202025-orange)

## 📌 Project Overview

This project applies supervised machine learning regression techniques to predict taxi fare amounts in New York City using the official NYC Taxi & Limousine Commission (TLC) Trip Record Data (2025 Updated).

**Task Type:** Regression  
**Target Variable:** `fare_amount`  
**Best Model:** Gradient Boosting Regressor  

---

## 📂 Repository Structure

```
.
├── nyc_taxi_fare_prediction.ipynb   # Main Jupyter Notebook (complete + executed)
├── Proposal.md                      # Assignment proposal document
├── requirements.txt                 # Required Python libraries
└── README.md                        # This file
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Name** | NYC Yellow Taxi Trip Records (January 2025) |
| **Source** | [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| **Direct Download** | [yellow_tripdata_2025-01.parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet) |
| **Rows** | > 3 million |
| **Columns** | 18–20 |
| **Format** | Parquet (loaded directly in the notebook) |
| **Updated** | 2025 ✅ |

The dataset is loaded **directly from the URL** inside the notebook — no manual download is needed.

---

## 🤖 Machine Learning Models

| Model | Description |
|---|---|
| Linear Regression | Baseline linear model |
| Random Forest Regressor | Ensemble tree-based model |
| Gradient Boosting Regressor | Boosted tree-based model (best performer) |

---

## 📈 Evaluation Metrics

- **RMSE** – Root Mean Squared Error
- **MAE** – Mean Absolute Error
- **R²** – Coefficient of Determination

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended — no installation needed)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook** and upload `nyc_taxi_fare_prediction.ipynb`
3. Click **Runtime → Run all**
4. All dependencies are installed automatically in the first cell

### Option 2: Local Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/nyc-taxi-fare-prediction.git
   cd nyc-taxi-fare-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook nyc_taxi_fare_prediction.ipynb
   ```

4. **Run all cells** (Kernel → Restart & Run All)

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for the full list of dependencies.

---

## 📋 Research Questions Addressed

1. How does trip distance affect fare amount?
2. Are there time-of-day patterns that influence fares?
3. How do pickup/dropoff locations impact fare?
4. Does passenger count affect the fare amount?
5. Which ML model gives the best performance for fare prediction?
6. Does feature engineering (hour, day, duration) improve model accuracy?

---

## 👤 Author

**Student Name** — Supervised Learning Assignment, 2026  
**Professor**: Prof. Raja Hashim Ali
