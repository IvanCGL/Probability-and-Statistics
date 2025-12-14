import pandas as pd
import ast
import os

# ==========================================
# 1. Configuration Parameters
# ==========================================
# Input file path (adjust if your folder structure is different)
LOAD_PATH = "./dataset/IMDB TMDB Movie Metadata Big Dataset (1M).csv"
# Output file path
SAVE_PATH = "./dataset/IMDB_Feature_Films_Cleaned.csv"

# Filtering Criteria
START_YEAR = 1910
END_YEAR = 2025
MIN_RUNTIME = 60  # Define feature film: > 60 minutes

print(f"Loading raw data: {LOAD_PATH} ...")

# Check if file exists to prevent crash
if not os.path.exists(LOAD_PATH):
    print(f"Error: File not found at {LOAD_PATH}")
    print("Please check your dataset folder.")
    exit(1)

try:
    df = pd.read_csv(LOAD_PATH)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

print(f"Original row count: {len(df)}")

# ==========================================
# 2. Basic Cleaning
# ==========================================
# Drop rows with missing core fields
# Note: We keep columns like revenue and budget even if they have nulls for now.
# We will handle them separately during correlation analysis.
# Here we only drop rows missing core metadata.
cols_to_check = ['id', 'title', 'release_year', 'runtime', 'genres_list']
df = df.dropna(subset=cols_to_check)

# Type conversion
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')

# ==========================================
# 3. Apply Three Main Filters
# ==========================================

# A. Time Filter
df = df[(df['release_year'] >= START_YEAR) & (df['release_year'] <= END_YEAR)]
print(f"After time filter ({START_YEAR}-{END_YEAR}): {len(df)} rows")

# B. Runtime Filter (Remove Shorts)
# Only keep feature films (> 60 min)
df = df[df['runtime'] > MIN_RUNTIME]
print(f"After runtime filter (>{MIN_RUNTIME} min): {len(df)} rows")

# C. Genre Filter (Remove 'Unknown')
# This step is slightly complex because genres_list is a string; we need to parse it first.
print("Parsing and filtering 'Unknown' genres...")

# 1. Format fix: Convert string "['Action']" to list
# Only apply literal_eval if the data is actually a string
if not df.empty and isinstance(df['genres_list'].iloc[0], str):
    df['genres_list'] = df['genres_list'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

# 2. Filtering logic:
# Drop row if list contains 'Unknown' or is empty
def is_valid_genre(g_list):
    if not isinstance(g_list, list): return False
    if len(g_list) == 0: return False
    if 'Unknown' in g_list: return False
    return True

df = df[df['genres_list'].apply(is_valid_genre)]
print(f"After filtering 'Unknown' genres: {len(df)} rows")

# ==========================================
# 4. Save Results
# ==========================================
# Ensure output directory exists
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

print(f"Saving cleaned feature film dataset to: {SAVE_PATH} ...")

df.to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')

print("\n✅ Data cleaning complete!")
print(f"Final dataset contains {len(df)} feature films.")
print("You can use this file for Phase 2 (Visualizations), Phase 3 (Modeling) and Phase 4 (Deep Insights).")