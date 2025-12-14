import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Load the cleaned feature film dataset
LOAD_PATH = "./dataset/IMDB_Feature_Films_Cleaned.csv"
df = pd.read_csv(LOAD_PATH)

# Fix list format again (Lists become strings after saving to CSV; must use eval to recover)
if isinstance(df['genres_list'].iloc[0], str):
    df['genres_list'] = df['genres_list'].apply(ast.literal_eval)

# ==========================================
# Dimension 1: Hard Metric Correlation
# Budget, Runtime, Old Movies: What drives high scores?
# ==========================================
print("Calculating numerical correlations...")

# Select numerical columns
num_cols = ['AverageRating', 'runtime', 'budget', 'revenue', 'release_year']
corr_matrix = df[num_cols].corr()

plt.figure(figsize=(10, 8))
# Plot Heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap: What Drives High Ratings?', fontsize=16)
plt.show()

# ==========================================
# Deep Dive: Non-linear relationship between Runtime and Rating (The "Sweet Spot")
# ==========================================
plt.figure(figsize=(12, 6))

# Bin by runtime and take the average to see trends clearly
df['runtime_bin'] = (df['runtime'] // 10) * 10
runtime_score = df[df['runtime'] <= 240].groupby('runtime_bin')['AverageRating'].mean() # Limit to within 4 hours

plt.plot(runtime_score.index, runtime_score.values, marker='o', color='#2c3e50', linewidth=2)
plt.title('Runtime vs. Rating: Is Longer Better?', fontsize=16)
plt.xlabel('Runtime (Minutes)', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Annotate peak value
peak_runtime = runtime_score.idxmax()
peak_score = runtime_score.max()
plt.annotate(f'Sweet Spot: ~{peak_runtime} min', 
             xy=(peak_runtime, peak_score), 
             xytext=(peak_runtime+20, peak_score-0.2),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.show()

# ==========================================
# Dimension 2: Genre Bonus (Genre Impact Analysis)
# Which genres naturally get higher scores? (Genre Coefficient)
# ==========================================
print("Analyzing genre impact...")

# 1. Explode data
df_exploded = df.explode('genres_list')

# 2. Calculate mean, variance, and count of ratings per genre
genre_stats = df_exploded.groupby('genres_list').agg({
    'AverageRating': ['mean', 'std', 'count']
})

# Flatten column names
genre_stats.columns = ['AverageRating', 'rating_std', 'movie_count']

# 3. Filter out extremely niche genres (Small samples cause skewed scores)
# Only keep genres with at least 500 movies
genre_stats = genre_stats[genre_stats['movie_count'] > 500]

# 4. Sort: By average rating
genre_stats = genre_stats.sort_values('AverageRating', ascending=True)

# 5. Plot: Horizontal bar chart with error bars
plt.figure(figsize=(12, 8))

# Plot bar chart (Average Rating)
bars = plt.barh(genre_stats.index, genre_stats['AverageRating'], xerr=genre_stats['rating_std'], 
                capsize=4, color='#7f8c8d', alpha=0.7)

# Highlight top 3 and bottom 3
for i, bar in enumerate(bars):
    if i >= len(bars) - 3:
        bar.set_color('#27ae60') # Green: High-scoring genres
    elif i < 3:
        bar.set_color('#c0392b') # Red: Low-scoring genres

plt.title('Genre Impact: Which Genres Have the Highest Intrinsic Quality?', fontsize=16)
plt.xlabel('Average Rating (with Standard Deviation)', fontsize=12)
plt.xlim(5, 8) # Focus on the 5-8 score range for clarity
plt.grid(axis='x', linestyle='--', alpha=0.5)

plt.show()