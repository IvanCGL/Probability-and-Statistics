import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 1. Data Loading and Basic Cleaning
# ==========================================
LOAD_PATH = "./dataset/IMDB_Feature_Films_Cleaned.csv" 
df = pd.read_csv(LOAD_PATH) 

# Step A: Force conversion to numeric type
df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')

# Step B: Drop rows where the target variable is null
df = df.dropna(subset=['IMDB_Rating'])

# ==========================================
# 2. Advanced Feature Engineering (The "Secret Sauce")
# ==========================================
print("Building advanced features...")

# --- A. Temporal Features ---
# Extract month from release_date (capture seasonality)
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month

# --- B. Business Features ---
# Has homepage (1=Yes, 0=No)
df['has_homepage'] = df['homepage'].notna().astype(int)
df['roi'] = df.apply(lambda x: (x['revenue'] - x['budget']) / x['budget'] if x['budget'] > 1000 else 0, axis=1)

# --- C. "Reputation" Features (Target Encoding) ---

def calculate_reputation(df, col_name, target_col='vote_average'):
    # Calculate average rating for each person
    reputation = df.groupby(col_name)[target_col].mean()
    # Map back to the original table; fill with global average if it's a new director (not in the database)
    global_mean = df[target_col].mean()
    return df[col_name].map(reputation).fillna(global_mean)

# Encode key personnel
if 'Director' in df.columns:
    df['Director_Score'] = calculate_reputation(df, 'Director')
if 'Star1' in df.columns:
    df['Star1_Score'] = calculate_reputation(df, 'Star1')
if 'Writer' in df.columns:
    df['Writer_Score'] = calculate_reputation(df, 'Writer')

# --- D. Genre Features (One-Hot) ---
# Process Genres again
if isinstance(df['genres_list'].iloc[0], str):
    df['genres_list'] = df['genres_list'].apply(ast.literal_eval)

mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres_list'])
genres_df = pd.DataFrame(genres_encoded, columns=[f"Genre_{g}" for g in mlb.classes_], index=df.index)

# ==========================================
# 3. Prepare Training Data
# ==========================================
feature_cols = [
    'runtime', 'budget', 'revenue', 'release_year', 'release_month', # Basic
    'roi', 'has_homepage', 'overview_sentiment',                     # Business & Sentiment
    'Director_Score', 'Star1_Score', 'Writer_Score'                  # Reputation (Key!)
]

# Ensure columns exist
selected_features = [c for c in feature_cols if c in df.columns]
X = pd.concat([df[selected_features], genres_df], axis=1)
y = df['IMDB_Rating'] 
df = df[(df['IMDB_Rating'] > 0) & (df['IMDB_Rating'] <= 10)]

# Fill missing values
X = X.fillna(X.median())

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. Model Upgrade: XGBoost Regressor
# ==========================================
print(f"Training XGBoost (Number of features: {X.shape[1]})...")

# XGBoost parameter configuration (can be tuned)
model = xgb.XGBRegressor(
    n_estimators=500,     # Number of trees
    learning_rate=0.05,   # Learning rate
    max_depth=6,          # Tree depth (prevent overfitting)
    subsample=0.8,        # Use 80% of data per iteration
    colsample_bytree=0.8, # Use 80% of features per iteration
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# 5. Evaluation and Visualization
# ==========================================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n🚀 Model Upgrade Results:")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Plot Feature Importance
plt.figure(figsize=(12, 10))
importances = model.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=False).head(20)

sns.barplot(x='Importance', y='Feature', data=feat_df, palette='magma')
plt.title('What REALLY drives Movie Ratings? (XGBoost Feature Importance)', fontsize=16)
plt.tight_layout()
plt.show()

# Prediction Comparison Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='#8e44ad', s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('XGBoost Prediction Accuracy', fontsize=16)
plt.legend()
plt.show()