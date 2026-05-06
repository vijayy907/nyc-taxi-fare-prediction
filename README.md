# F1 Pit-Stop Prediction — Supervised Learning Assignment

**Programme**: MSc Data Science  
**Student**: Karnati Mysanthosh (36733714)  
**University**: University of Europe for Applied Sciences  
**Supervisor**: Prof. Raja Hashim Ali

## Project Overview
This project applies supervised machine learning to predict lap-level pit stops in Formula 1 racing. Using data from the 2019-2025 seasons, we evaluate multiple classifiers to optimize race strategy decisions.

## Research Questions
- **RQ1**: Baseline performance (Logistic Regression, Decision Tree, k-NN).
- **RQ2**: Comprehensive model comparison (including Random Forest and XGBoost).
- **RQ3**: Impact of preprocessing strategies on F1-score.
- **RQ4**: Feature importance and domain interpretation (Tyre life, Safety Car, etc.).
- **RQ5**: Model sensitivity to evaluation metrics (Accuracy, Precision, Recall, F1, AUC).
- **RQ6**: Robustness through 5-fold cross-validation.
- **RQ7**: Practical usefulness and final model recommendation.

## Dataset
The dataset is sourced from Kaggle: [F1 Strategy Dataset (Lap-Level Race Data)](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction).

## How to Run
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the Jupyter Notebook: `f1_pitstop_prediction.ipynb`
3.  The script will generate all result tables and figures in the `/outputs/` directory.

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
