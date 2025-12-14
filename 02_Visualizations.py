import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
import os

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. Load Data
# ==========================================
LOAD_PATH = "./dataset/IMDB_Feature_Films_Cleaned.csv"

print("Loading feature film dataset...")

if not os.path.exists(LOAD_PATH):
    print(f"[ERROR] File not found: {LOAD_PATH}")
    print("Please run '01_Data_Prep.py' first to generate the cleaned dataset.")
    exit(1)

df = pd.read_csv(LOAD_PATH)

# Fix 'genres_list' format (CSV saves lists as strings)
if not df.empty and isinstance(df['genres_list'].iloc[0], str):
    df['genres_list'] = df['genres_list'].apply(ast.literal_eval)

print(f"Data loaded successfully. Sample size: {len(df)}")

# ==========================================
# Chart 1: Genre Popularity over Time (Stacked Area)
# ==========================================
print("Generating Chart 1: Mainstream Genre Evolution...")

# Explode data to count each genre
df_exploded = df.explode('genres_list')

# Create 'decade' column for macro trends
df_exploded['decade'] = (df_exploded['release_year'] // 10) * 10

# Count genre frequency per decade
genre_trend = pd.crosstab(df_exploded['decade'], df_exploded['genres_list'])
# Normalize to percentage (%)
genre_pct = genre_trend.div(genre_trend.sum(axis=1), axis=0) * 100

# Select Top 10 genres
top_genres = df_exploded['genres_list'].value_counts().head(10).index

plt.figure(figsize=(14, 7))
plt.stackplot(genre_pct.index, 
              [genre_pct[g] for g in top_genres],
              labels=top_genres, alpha=0.8)

plt.title('Evolution of Audience Preferences: Genre Market Share (Feature Films Only)', fontsize=16)
plt.ylabel('Market Share (%)', fontsize=12)
plt.xlabel('Decade', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Genre')
plt.xlim(1920, 2020)
plt.tight_layout()
plt.show()

# ==========================================
# Chart 2: Niche & Historical Genres (Line Chart)
# ==========================================
print("Generating Chart 2: Historical Genre Trends...")

# Define genres of interest (Westerns, War, History)
historical_genres = ['Western', 'War', 'History', 'Music']
target_genres = [g for g in historical_genres if g in genre_pct.columns]

plt.figure(figsize=(14, 6))

for genre in target_genres:
    plt.plot(genre_pct.index, genre_pct[genre], 
             marker='o', linewidth=2.5, label=genre)

plt.title('The Rise and Fall of Specific Historical Genres', fontsize=16)
plt.ylabel('Market Share (%)', fontsize=12)
plt.xlabel('Decade', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(1920, 2020)

# Annotation for Westerns
if 'Western' in target_genres and 1950 in genre_pct.index:
    western_peak = genre_pct['Western'].loc[1950]
    plt.annotate('Golden Age of Westerns', 
                 xy=(1950, western_peak), 
                 xytext=(1960, western_peak + 2),
                 arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

# ==========================================
# Chart 3: Hybridization Trend (Complexity)
# ==========================================
print("Generating Chart 3: Genre Complexity Trend...")

df['genre_count'] = df['genres_list'].apply(len)
complexity_trend = df.groupby('release_year')['genre_count'].mean()

plt.figure(figsize=(12, 5))
plt.plot(complexity_trend.index, complexity_trend.values, color='#8e44ad', linewidth=2)
# Smoothing curve
plt.plot(complexity_trend.rolling(5).mean(), color='black', linewidth=1, linestyle='--')

plt.title('The "Blockbuster Formula": Are Movies Becoming More Complex?', fontsize=16)
plt.ylabel('Avg. Number of Genres per Movie', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# ==========================================
# Chart 4: Budget vs. ROI (Dual Axis)
# ==========================================
print("Generating Chart 4: Budget & ROI Analysis...")

# Filter unreliable financial data
df_finance = df[(df['budget'] > 10000) & (df['revenue'] > 10000)].copy()

# Calculate ROI
df_finance['roi'] = (df_finance['revenue'] - df_finance['budget']) / df_finance['budget']
finance_trend = df_finance.groupby('release_year')[['budget', 'revenue', 'roi']].median()

fig, ax1 = plt.subplots(figsize=(14, 7))

# Left Axis: Budget
color = 'tab:blue'
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Median Budget (Nominal $)', color=color, fontsize=12)
ax1.plot(finance_trend.index, finance_trend['budget'], color=color, label='Median Budget')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log') 

# Right Axis: ROI
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Median ROI (Ratio)', color=color, fontsize=12)
ax2.plot(finance_trend.index, finance_trend['roi'].rolling(5).mean(), color=color, linestyle='--', linewidth=2, label='ROI (5y Trend)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 10)

plt.title('The Cost of Business: Rising Budgets vs. Diminishing Returns', fontsize=16)
fig.tight_layout()
plt.show()

# ==========================================
# Chart 5: Rating Trends (Survivorship Bias)
# ==========================================
print("Generating Chart 5: Rating Trends & Survivorship Bias...")

# Ensure rating col exists (using IMDB_Rating or AverageRating)
rating_col = 'IMDB_Rating' if 'IMDB_Rating' in df.columns else 'vote_average'
# Clean non-numeric
df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')

rating_stats = df.groupby('release_year')[rating_col].agg(['mean', 'std', 'count'])

plt.figure(figsize=(14, 6))

# Mean line
plt.plot(rating_stats.index, rating_stats['mean'], color='#2c3e50', linewidth=3, label='Average Rating')

# Std Dev shading
plt.fill_between(rating_stats.index, 
                 rating_stats['mean'] - rating_stats['std'], 
                 rating_stats['mean'] + rating_stats['std'], 
                 color='gray', alpha=0.2, label='1 Std Dev (Variance)')

# Volume bar (Scaled)
plt.bar(rating_stats.index, rating_stats['count'] / rating_stats['count'].max() * 2 + 4, 
        color='orange', alpha=0.3, label='Movie Volume (Scaled)')

plt.title('Rating Trends: Survivorship Bias & The "Golden Age" Illusion', fontsize=16)
plt.ylabel('Rating (0-10)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend(loc='lower left')
plt.ylim(0, 10)
plt.show()

print("\n✅ All visualizations generated successfully!")