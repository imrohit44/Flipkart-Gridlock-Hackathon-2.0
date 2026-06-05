"""
Traffic Demand Prediction - Optimized Model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TRAFFIC DEMAND PREDICTION MODEL - OPTIMIZED")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================
print("\n1. Loading and preprocessing data...")
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# Combine for preprocessing
df_combined = pd.concat([train_data.drop('demand', axis=1), test_data], ignore_index=True)
n_train = len(train_data)

# Parse time features
df_combined['hour'] = df_combined['timestamp'].str.split(':').str[0].astype(int)
df_combined['minute'] = df_combined['timestamp'].str.split(':').str[1].astype(int)
df_combined['time_of_day'] = df_combined['hour'] + df_combined['minute'] / 60
df_combined['is_peak_hour'] = ((df_combined['hour'] >= 7) & (df_combined['hour'] <= 9)) | \
                               ((df_combined['hour'] >= 17) & (df_combined['hour'] <= 19))

# Fill missing values
df_combined['Temperature'].fillna(df_combined['Temperature'].median(), inplace=True)
df_combined['RoadType'].fillna(df_combined['RoadType'].mode()[0], inplace=True)
df_combined['Weather'].fillna(df_combined['Weather'].mode()[0], inplace=True)

# Encode categorical variables
label_encoders = {}
for col in ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']:
    le = LabelEncoder()
    df_combined[col + '_enc'] = le.fit_transform(df_combined[col].astype(str))
    label_encoders[col] = le

# Feature engineering
df_combined['temp_sq'] = df_combined['Temperature'] ** 2
df_combined['day_sin'] = np.sin(2 * np.pi * df_combined['day'] / 7)
df_combined['day_cos'] = np.cos(2 * np.pi * df_combined['day'] / 7)
df_combined['hour_sin'] = np.sin(2 * np.pi * df_combined['hour'] / 24)
df_combined['hour_cos'] = np.cos(2 * np.pi * df_combined['hour'] / 24)
df_combined['interaction'] = df_combined['NumberofLanes'] * df_combined['RoadType_enc']

# Split back
train_processed = df_combined[:n_train].copy()
test_processed = df_combined[n_train:].copy()

# Feature selection
feature_cols = ['geohash_enc', 'day', 'hour', 'minute', 'NumberofLanes', 'RoadType_enc',
                'LargeVehicles_enc', 'Landmarks_enc', 'Temperature', 'Weather_enc',
                'is_peak_hour', 'temp_sq', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos',
                'interaction', 'time_of_day']

X_train = train_processed[feature_cols].copy()
X_test = test_processed[feature_cols].copy()
y_train = train_data['demand'].values

# Fill any NaN values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Features: {len(feature_cols)}")
print(f"Train shape: {X_train_scaled.shape}")
print(f"Test shape: {X_test_scaled.shape}")

# ============================================================================
# 2. TRAIN/VAL SPLIT
# ============================================================================
print("\n2. Splitting data for validation...")
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.15, random_state=42
)

# ============================================================================
# 3. MODEL TRAINING
# ============================================================================
print("\n3. Training models...")

results = {}

# XGBoost
print("   - XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.9,
    colsample_bytree=0.9, random_state=42, n_jobs=-1, tree_method='hist'
)
xgb_model.fit(X_train_split, y_train_split, verbose=False)
xgb_val = xgb_model.predict(X_val_split)
xgb_r2 = r2_score(y_val_split, xgb_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val_split, xgb_val))
print(f"      R²: {xgb_r2:.6f}, RMSE: {xgb_rmse:.6f}")
results['XGBoost'] = (xgb_r2, xgb_rmse)
xgb_test = xgb_model.predict(X_test_scaled)

# LightGBM
print("   - LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=100, learning_rate=0.1, num_leaves=20, subsample=0.9,
    colsample_bytree=0.9, random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train_split, y_train_split)
lgb_val = lgb_model.predict(X_val_split)
lgb_r2 = r2_score(y_val_split, lgb_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val_split, lgb_val))
print(f"      R²: {lgb_r2:.6f}, RMSE: {lgb_rmse:.6f}")
results['LightGBM'] = (lgb_r2, lgb_rmse)
lgb_test = lgb_model.predict(X_test_scaled)

# ============================================================================
# 4. ENSEMBLE & SUBMISSION
# ============================================================================
print("\n4. Creating ensemble predictions...")
final_pred = xgb_test * 0.5 + lgb_test * 0.5

submission = pd.DataFrame({
    'Index': test_data['Index'],
    'demand': final_pred
})
submission.to_csv('submission.csv', index=False)

# ============================================================================
# 5. RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("ACCURACY RESULTS")
print("=" * 80)
print("\nValidation Set Performance:")
for model_name, (r2, rmse) in results.items():
    print(f"  {model_name:15} | R² Score: {r2:.6f} | RMSE: {rmse:.6f}")

best_r2 = max([v[0] for v in results.values()])
print(f"\n  Best R² Score: {best_r2:.6f}")
print(f"  Model explains {best_r2*100:.2f}% of variance in demand")

print(f"\nEnsemble Configuration:")
print(f"  XGBoost: 50%")
print(f"  LightGBM: 50%")
print(f"\nSubmission saved to 'submission.csv'")
print(f"Predictions shape: {submission.shape}")
print(f"Sample predictions:\n{submission.head(10).to_string()}")

print("\n" + "=" * 80)
