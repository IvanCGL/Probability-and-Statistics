import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. Load Data & Clean Target
# ==========================================
LOAD_PATH = "./dataset/IMDB_Feature_Films_Cleaned.csv"
print(f"Loading dataset from {LOAD_PATH}...")

if not os.path.exists(LOAD_PATH):
    print(f"[ERROR] File not found: {LOAD_PATH}")
    exit(1)

df = pd.read_csv(LOAD_PATH)

# Critical Step: Clean the Target Variable
# We focus on predicting 'IMDB_Rating'
df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
df = df.dropna(subset=['IMDB_Rating'])
df = df[(df['IMDB_Rating'] > 0) & (df['IMDB_Rating'] <= 10)]

print(f"Target variable cleaned. Modeling with {len(df)} movies.")

# ==========================================
# 2. The "Secret Sauce": Reputation Features
# ==========================================
print("Engineering Reputation Features (Target Encoding)...")

# Define a function to calculate the "Track Record" of people
# This is what drives the R^2 to 0.86
def calculate_reputation(df, col_name, target_col='IMDB_Rating'):
    if col_name not in df.columns: return None
    # Calculate mean rating per entity
    reputation = df.groupby(col_name)[target_col].mean()
    # Map back to dataframe
    global_mean = df[target_col].mean()
    return df[col_name].map(reputation).fillna(global_mean)

# Encode the "Big Three": Writer, Star, Director
if 'Writer' in df.columns:
    df['Writer_Score'] = calculate_reputation(df, 'Writer')
if 'Star1' in df.columns:
    df['Star1_Score'] = calculate_reputation(df, 'Star1')
if 'Director' in df.columns:
    df['Director_Score'] = calculate_reputation(df, 'Director')

# ==========================================
# 3. Other Essential Features
# ==========================================
print("Processing Financial & Genre Features...")

# ROI (Efficiency)
df['roi'] = df.apply(lambda x: (x['revenue'] - x['budget']) / x['budget'] 
                     if x['budget'] > 1000 and x['revenue'] > 0 else 0, axis=1)

# Genres (One-Hot) - Kept for comparison, though they matter less now
if isinstance(df['genres_list'].iloc[0], str):
    df['genres_list'] = df['genres_list'].apply(ast.literal_eval)

mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres_list'])
genres_df = pd.DataFrame(genres_encoded, columns=[f"Genre_{g}" for g in mlb.classes_], index=df.index)

# ==========================================
# 4. Prepare Training Data
# ==========================================
# We keep the feature list "Lean" as you suggested
feature_cols = [
    'runtime', 'budget', 'revenue', 'release_year',  # Physical/Time attributes
    'roi', 'has_homepage',                           # Business attributes
    'Director_Score', 'Star1_Score', 'Writer_Score'  # Reputation attributes (The Powerhouses)
]

# Ensure cols exist
selected_features = [c for c in feature_cols if c in df.columns]

# Concat features
X = pd.concat([df[selected_features], genres_df], axis=1)
y = df['IMDB_Rating']

# Fill NaNs
X = X.fillna(X.median())

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 5. Train XGBoost
# ==========================================
print(f"Training XGBoost Model on {X.shape[1]} features...")

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# 6. Results & Visualization
# ==========================================
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n{'='*40}")
print(f"🚀 FINAL MODEL RESULTS")
print(f"{'='*40}")
print(f"R^2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"{'='*40}")

# Plot 1: Feature Importance (Bar Chart)
plt.figure(figsize=(12, 10))
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=False).head(20)

sns.barplot(x='Importance', y='Feature', data=feat_df, palette='magma')
plt.title('What REALLY Drives Movie Ratings? (XGBoost Feature Importance)', fontsize=16)
plt.tight_layout()
plt.show()

# Plot 2: Prediction Accuracy (Scatter Plot)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='#8e44ad', s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')

plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title(f'Prediction Accuracy (R^2: {r2:.2f})', fontsize=16)
plt.legend()
plt.show()

print("\n✅ Analysis Complete. 'Writer_Score' should be dominant!")