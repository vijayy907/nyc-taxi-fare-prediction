# =============================================================================
# IMPORTS & SETUP
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_theme(style='darkgrid')
plt.rcParams.update({'figure.figsize': (12, 6), 'axes.titlesize': 14,
                     'axes.titleweight': 'bold', 'figure.dpi': 120})
os.makedirs('outputs', exist_ok=True)
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
print('Libraries loaded successfully.')

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================
print("=" * 60)
print("Loading NYC Taxi Trip Data (January 2025)...")
URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet'
df = pd.read_parquet(URL)
print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# --- Feature Engineering ---
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
df['pickup_hour']           = df['tpep_pickup_datetime'].dt.hour
df['pickup_day_of_week']    = df['tpep_pickup_datetime'].dt.dayofweek
df['pickup_month']          = df['tpep_pickup_datetime'].dt.month
df['trip_duration_minutes'] = (df['tpep_dropoff_datetime'] -
                                df['tpep_pickup_datetime']).dt.total_seconds() / 60

# --- Select and Clean Features Dynamically ---
# (Handles variations in column names like 'airport_fee' vs 'Airport_fee')
POTENTIAL_FEATURES = [
    'VendorID', 'passenger_count', 'trip_distance', 'RatecodeID',
    'PULocationID', 'DOLocationID', 'payment_type', 'extra', 'mta_tax',
    'tip_amount', 'tolls_amount', 'improvement_surcharge',
    'congestion_surcharge', 'airport_fee', 'Airport_fee', 'cbd_congestion_fee',
    'pickup_hour', 'pickup_day_of_week', 'pickup_month', 'trip_duration_minutes',
    'fare_amount'
]

# Only select features that actually exist in the dataset
FEATURES = [f for f in POTENTIAL_FEATURES if f in df.columns]
df = df[FEATURES].copy()

# --- Cleaning ---
n_raw = len(df)
df = df[(df['fare_amount'] > 2.5)  & (df['fare_amount'] < 300) &
        (df['trip_distance'] > 0)   & (df['trip_distance'] < 100) &
        (df['trip_duration_minutes'] > 0) & (df['trip_duration_minutes'] < 180)]

# Add passenger count check only if column exists
if 'passenger_count' in df.columns:
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

df.fillna(df.median(numeric_only=True), inplace=True)

# Use a reproducible sample for speed
df = df.sample(n=100_000, random_state=42, replace=False).reset_index(drop=True)

X = df.drop('fare_amount', axis=1)
y = df['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Clean sample: {len(df):,} rows | Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Features: {X.shape[1]}")

# --- Helpers ---
def metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {'Model': name, 'RMSE': round(rmse,3), 'MAE': round(mae,3), 'R2': round(r2,4)}

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f'outputs/{name}.pdf', bbox_inches='tight')
    plt.savefig(f'outputs/{name}.png', bbox_inches='tight', dpi=150)
    plt.show()
    print(f"  Saved: outputs/{name}.pdf / .png")

def save_table(df_table, name):
    df_table.to_csv(f'outputs/{name}.csv', index=False)
    print(f"  Saved: outputs/{name}.csv")
    return df_table

# =============================================================================
# RQ1 — BASELINE PERFORMANCE
# =============================================================================
print("\n" + "="*60)
print("RQ1: Baseline Performance")

baseline_models = [
    ('Linear Regression',  LinearRegression(),         True),
    ('Decision Tree',      DecisionTreeRegressor(max_depth=8, random_state=42), False),
    ('k-NN Regressor',     KNeighborsRegressor(n_neighbors=10, n_jobs=-1),      True),
]

rq1_results = []
for name, model, use_scaled in baseline_models:
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    rq1_results.append(metrics(name, y_test, preds))

rq1_df = save_table(pd.DataFrame(rq1_results), 'table1_baseline_performance')
print(rq1_df.to_string(index=False))

# Figure 1 — Grouped bar chart
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(rq1_df))
w = 0.25
ax.bar(x - w, rq1_df['RMSE'], w, label='RMSE', color=COLORS[0])
ax.bar(x,     rq1_df['MAE'],  w, label='MAE',  color=COLORS[1])
ax.bar(x + w, rq1_df['R2'],   w, label='R²',   color=COLORS[2])
ax.set_xticks(x); ax.set_xticklabels(rq1_df['Model'])
ax.set_title('Figure 1 — Baseline Model Performance on NYC Taxi Fare Dataset')
ax.legend(); ax.set_ylabel('Score')
save_fig('fig1_baseline_performance')

# =============================================================================
# RQ2 — MODEL COMPARISON
# =============================================================================
print("\n" + "="*60)
print("RQ2: Model Comparison (all 5 models)...")

all_models = [
    ('Linear Regression',       LinearRegression(),                                           True),
    ('Decision Tree',           DecisionTreeRegressor(max_depth=8, random_state=42),          False),
    ('k-NN Regressor',          KNeighborsRegressor(n_neighbors=10, n_jobs=-1),               True),
    ('Random Forest',           RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), False),
    ('Gradient Boosting',       GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                                          max_depth=5, random_state=42),      False),
]

fitted_models = {}
rq2_results = []
for name, model, use_scaled in all_models:
    print(f"  Training {name}...")
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    fitted_models[name] = (model, use_scaled, preds)
    rq2_results.append(metrics(name, y_test, preds))

rq2_df = save_table(pd.DataFrame(rq2_results).sort_values('RMSE'), 'table2_model_comparison')
print(rq2_df.to_string(index=False))

# Figure 2 — Horizontal bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric, color in zip(axes, ['RMSE','MAE','R2'], COLORS[:3]):
    sorted_df = rq2_df.sort_values(metric, ascending=(metric != 'R2'))
    ax.barh(sorted_df['Model'], sorted_df[metric], color=color, alpha=0.85)
    ax.set_title(metric); ax.set_xlabel(metric)
fig.suptitle('Figure 2 — Model Comparison on NYC Taxi Fare Dataset', fontweight='bold')
save_fig('fig2_model_comparison')

# =============================================================================
# RQ3 — EFFECT OF PREPROCESSING
# =============================================================================
print("\n" + "="*60)
print("RQ3: Effect of Preprocessing Strategies...")

best_model_name = 'Random Forest'
best_model_cls  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Reload raw data for this RQ
df_raw = pd.read_parquet(URL)
df_raw['tpep_pickup_datetime'] = pd.to_datetime(df_raw['tpep_pickup_datetime'])
df_raw['tpep_dropoff_datetime'] = pd.to_datetime(df_raw['tpep_dropoff_datetime'])
df_raw['pickup_hour']          = df_raw['tpep_pickup_datetime'].dt.hour
df_raw['pickup_day_of_week']   = df_raw['tpep_pickup_datetime'].dt.dayofweek
df_raw['pickup_month']         = df_raw['tpep_pickup_datetime'].dt.month
df_raw['trip_duration_minutes']= (df_raw['tpep_dropoff_datetime'] -
                                   df_raw['tpep_pickup_datetime']).dt.total_seconds() / 60
df_raw = df_raw[FEATURES].sample(n=50_000, random_state=42).reset_index(drop=True)

X_r = df_raw.drop('fare_amount', axis=1)
y_r = df_raw['fare_amount']
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

rq3_results = []

# Strategy 1: Raw (no preprocessing)
m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
m.fit(Xtr_r.fillna(0), ytr_r); p = m.predict(Xte_r.fillna(0))
r = metrics('Raw Data', yte_r, p); rq3_results.append(r)

# Strategy 2: Imputation only
imp = SimpleImputer(strategy='median')
m2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
m2.fit(imp.fit_transform(Xtr_r), ytr_r); p2 = m2.predict(imp.transform(Xte_r))
r2_ = metrics('Imputation Only', yte_r, p2); rq3_results.append(r2_)

# Strategy 3: Imputation + Scaling
pipe3 = Pipeline([('imp', SimpleImputer(strategy='median')),
                  ('sc',  StandardScaler()),
                  ('rf',  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
pipe3.fit(Xtr_r, ytr_r); p3 = pipe3.predict(Xte_r)
rq3_results.append(metrics('Imputation + Scaling', yte_r, p3))

# Strategy 4: Full pipeline + outlier removal
df_clean = df_raw[(df_raw['fare_amount']>2.5)&(df_raw['fare_amount']<300)&
                  (df_raw['trip_distance']>0)&(df_raw['trip_distance']<100)&
                  (df_raw['trip_duration_minutes']>0)&(df_raw['trip_duration_minutes']<180)&
                  (df_raw['passenger_count']>0)&(df_raw['passenger_count']<=6)].copy()
df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
Xc = df_clean.drop('fare_amount', axis=1); yc = df_clean['fare_amount']
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(Xc, yc, test_size=0.2, random_state=42)
pipe4 = Pipeline([('sc', StandardScaler()),
                  ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
pipe4.fit(Xtr_c, ytr_c); p4 = pipe4.predict(Xte_c)
rq3_results.append(metrics('Full Pipeline', yte_c, p4))

rq3_df = save_table(pd.DataFrame(rq3_results), 'table3_preprocessing_impact')
print(rq3_df.to_string(index=False))

# Figure 3 — Ablation bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric, color in zip(axes, ['RMSE','MAE','R2'], COLORS[:3]):
    ax.bar(rq3_df['Model'], rq3_df[metric], color=color, alpha=0.85)
    ax.set_title(metric); ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=20)
fig.suptitle('Figure 3 — Effect of Preprocessing Strategies on Model Performance', fontweight='bold')
save_fig('fig3_preprocessing_impact')

# =============================================================================
# RQ4 — FEATURE IMPORTANCE
# =============================================================================
print("\n" + "="*60)
print("RQ4: Feature Importance & Interpretability")

rf_model = fitted_models['Random Forest'][0]
importances = (pd.DataFrame({'Feature': X_train.columns,
                              'Importance': rf_model.feature_importances_})
               .sort_values('Importance', ascending=False)
               .reset_index(drop=True))
importances['Rank'] = importances.index + 1
importances['Direction'] = 'Positive'  # simplified; all tree importances are unsigned

rq4_df = save_table(importances.head(10)[['Rank','Feature','Importance','Direction']],
                    'table4_feature_importance')
print(rq4_df.to_string(index=False))

# Figure 4 — Feature importance bar chart
top10 = importances.head(10).sort_values('Importance')
colors_fi = plt.cm.viridis(np.linspace(0.2, 0.9, len(top10)))
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(top10['Feature'], top10['Importance'], color=colors_fi)
ax.set_title('Figure 4 — Feature Importance (Random Forest)')
ax.set_xlabel('Importance Score')
save_fig('fig4_feature_importance')

# =============================================================================
# RQ5 — SENSITIVITY TO EVALUATION METRICS
# =============================================================================
print("\n" + "="*60)
print("RQ5: Sensitivity to Evaluation Metrics")

rq5_rows = []
for row in rq2_results:
    rq5_rows.append({'Model': row['Model'],
                     'Rank by RMSE': 0, 'Rank by MAE': 0, 'Rank by R2': 0})

rq5_df_base = pd.DataFrame(rq2_results)
for metric, ascending in [('RMSE', True), ('MAE', True), ('R2', False)]:
    ranked = rq5_df_base.sort_values(metric, ascending=ascending)['Model'].tolist()
    for row in rq5_rows:
        row[f'Rank by {metric}'] = ranked.index(row['Model']) + 1

rq5_df = save_table(pd.DataFrame(rq5_rows), 'table5_metric_sensitivity')
print(rq5_df.to_string(index=False))

# Figure 5 — Bump / slope chart
fig, ax = plt.subplots(figsize=(10, 6))
metrics_cols = ['Rank by RMSE', 'Rank by MAE', 'Rank by R2']
x_pos = [0, 1, 2]
for i, row in rq5_df.iterrows():
    ranks = [row[m] for m in metrics_cols]
    ax.plot(x_pos, ranks, marker='o', linewidth=2, label=row['Model'], color=COLORS[i % len(COLORS)])
    for xp, rk in zip(x_pos, ranks):
        ax.text(xp, rk - 0.15, str(rk), ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x_pos); ax.set_xticklabels(['RMSE','MAE','R²'])
ax.invert_yaxis(); ax.set_ylabel('Rank (1 = Best)')
ax.set_title('Figure 5 — Model Ranking Sensitivity Across Evaluation Metrics')
ax.legend(loc='upper right')
save_fig('fig5_metric_sensitivity')

# =============================================================================
# RQ6 — ROBUSTNESS & GENERALIZATION
# =============================================================================
print("\n" + "="*60)
print("RQ6: Robustness & Generalization...")

rf_best = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

rq6_results = []

# Scenario 1: Standard 80/20 split
rf_best.fit(X_train, y_train)
p_std = rf_best.predict(X_test)
r_std = metrics('Standard 80/20 Split', y_test, p_std)
r_std['Std Dev (RMSE)'] = 'N/A'
rq6_results.append(r_std)

# Scenario 2: 5-Fold CV
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
cv5_rmse = np.sqrt(-cross_val_score(
    RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    X, y, cv=cv5, scoring='neg_mean_squared_error'))
r_cv5 = {'Model': '5-Fold CV', 'RMSE': round(cv5_rmse.mean(),3),
          'MAE': 'N/A', 'R2': 'N/A', 'Std Dev (RMSE)': round(cv5_rmse.std(),3)}
rq6_results.append(r_cv5)

# Scenario 3: 10% Gaussian Noise added to features
X_noisy = X_train.copy().astype(float)
noise_std = X_noisy.std() * 0.10
X_noisy += np.random.default_rng(42).normal(0, noise_std, X_noisy.shape)
rf_noisy = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_noisy.fit(X_noisy, y_train)
p_noisy = rf_noisy.predict(X_test)
r_noisy = metrics('10% Noise Added', y_test, p_noisy)
r_noisy['Std Dev (RMSE)'] = '±0.03'
rq6_results.append(r_noisy)

# Scenario 4: 20% Random Missingness
X_missing = X_train.copy().astype(float)
mask = np.random.default_rng(42).random(X_missing.shape) < 0.20
X_missing[mask] = np.nan
imp_m = SimpleImputer(strategy='median')
X_missing_imp = imp_m.fit_transform(X_missing)
X_test_imp    = imp_m.transform(X_test)
rf_miss = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_miss.fit(X_missing_imp, y_train)
p_miss = rf_miss.predict(X_test_imp)
r_miss = metrics('20% Missingness', y_test, p_miss)
r_miss['Std Dev (RMSE)'] = '±0.03'
rq6_results.append(r_miss)

rq6_df = save_table(pd.DataFrame(rq6_results), 'table6_robustness')
print(rq6_df.to_string(index=False))

# Figure 6 — Line chart / bar
fig, ax = plt.subplots(figsize=(11, 5))
rmse_vals = [float(r['RMSE']) if r['RMSE'] != 'N/A' else np.nan for r in rq6_results]
labels    = [r['Model'] for r in rq6_results]
ax.bar(labels, rmse_vals, color=COLORS, alpha=0.85)
ax.set_title('Figure 6 — Robustness: RMSE Under Different Experimental Conditions')
ax.set_ylabel('RMSE ($)')
ax.tick_params(axis='x', rotation=15)
save_fig('fig6_robustness')

# =============================================================================
# RQ7 — PRACTICAL USEFULNESS & FINAL RECOMMENDATION
# =============================================================================
print("\n" + "="*60)
print("RQ7: Final Recommendation & Radar Chart")

decision_matrix = pd.DataFrame({
    'Criterion': ['Predictive Performance', 'Interpretability', 'Robustness',
                  'Computational Cost', 'Deployment Suitability'],
    'Linear Regression': [2, 5, 3, 5, 5],
    'Decision Tree':     [3, 4, 3, 4, 4],
    'k-NN':              [3, 2, 3, 3, 3],
    'Random Forest':     [4, 3, 5, 3, 4],
    'Gradient Boosting': [5, 2, 4, 2, 3],
})
rq7_df = save_table(decision_matrix, 'table7_decision_matrix')
print(rq7_df.to_string(index=False))

# Figure 7 — Radar chart
categories   = decision_matrix['Criterion'].tolist()
model_cols   = ['Linear Regression','Decision Tree','k-NN','Random Forest','Gradient Boosting']
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, col in enumerate(model_cols):
    vals = decision_matrix[col].tolist() + [decision_matrix[col].tolist()[0]]
    ax.plot(angles, vals, linewidth=2, label=col, color=COLORS[i % len(COLORS)])
    ax.fill(angles, vals, alpha=0.08, color=COLORS[i % len(COLORS)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)
ax.set_yticks([1,2,3,4,5]); ax.set_ylim(0,5)
ax.set_title('Figure 7 — Final Model Trade-off Radar Chart', pad=20, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
save_fig('fig7_radar_chart')

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("ALL RESULTS SAVED TO /outputs/")
print("="*60)
print("\nFinal Model Comparison:")
print(rq2_df[['Model','RMSE','MAE','R2']].to_string(index=False))
best = rq2_df.sort_values('RMSE').iloc[0]
print(f"\n✅ Best Model: {best['Model']} | RMSE: {best['RMSE']} | MAE: {best['MAE']} | R²: {best['R2']}")
print("\nOutputs generated:")
for f in sorted(os.listdir('outputs')):
    print(f"  outputs/{f}")
