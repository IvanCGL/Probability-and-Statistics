import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
# Ensure the filename matches your actual file
LITE_PATH = "./dataset/IMDB TMDB Movie Metadata Big Dataset (1M).csv" 

print("Loading data...")
try:
    df = pd.read_csv(LITE_PATH)
except FileNotFoundError:
    print("File not found, please check the path.")
    exit()

# Basic cleaning
df = df.dropna(subset=['runtime', 'release_year', 'genre'])
df['release_year'] = df['release_year'].astype(int)
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')

# Filter date range (1910 - 2025)
START_YEAR, END_YEAR = 1910, 2025
df = df[(df['release_year'] >= START_YEAR) & (df['release_year'] <= END_YEAR)]

# Define Short Films (<= 60 minutes)
SHORT_THRESHOLD = 60
df['is_short'] = df['runtime'] <= SHORT_THRESHOLD

print(f"Data ready. Total rows (label-level): {len(df)}")

# ==========================================
# Chart 1: The Trend of Rising Short Films (Time Trend)
# ==========================================
print("Plotting Chart 1: Short Film Trends...")

# Deduplicate by ID to ensure each movie is counted only once
df_unique_movies = df.drop_duplicates(subset='id')

# Calculate the percentage of short films per year
trend_data = df_unique_movies.groupby('release_year')['is_short'].mean() * 100

plt.figure(figsize=(12, 6))
plt.fill_between(trend_data.index, 0, trend_data.values, color='#86d0cb', alpha=0.4)
plt.plot(trend_data.index, trend_data.values, color='#2c8c88', linewidth=2.5, label='% of Short Films')

plt.title(f'The Rise of Short Content ({START_YEAR}-{END_YEAR})', fontsize=16)
plt.ylabel('Market Share (%)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.xlim(START_YEAR, END_YEAR)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# ==========================================
# Chart 2: Genre DNA Comparison: Shorts vs. Features (Butterfly Chart)
# Using original df (not deduplicated) here because we are analyzing "tags"
# ==========================================
print("Plotting Chart 2: Genre Comparison...")

df = df[df['genre'] != 'Unknown']
# Calculate genre distribution for each (normalized to percentage)
short_genres = df[df['is_short']]['genre'].value_counts(normalize=True).head(10) * 100
feature_genres = df[~df['is_short']]['genre'].value_counts(normalize=True).head(10) * 100

# Merge data
comp_df = pd.DataFrame({'Shorts': short_genres, 'Features': feature_genres})
# Sort by Short Film popularity
comp_df = comp_df.sort_values('Shorts', ascending=True)

# Plotting
comp_df.plot(kind='barh', figsize=(12, 8), width=0.8, color=['#86d0cb', '#e88c7d'])

plt.title(f'Genre Composition: Short Films (<{SHORT_THRESHOLD}m) vs Features', fontsize=16)
plt.xlabel('Percentage within Category (%)', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# Chart 3: "Short Film Penetration Rate" within each Genre
# How many Animation movies are actually short films?
# ==========================================
print("Plotting Chart 3: Penetration Analysis...")

# Focus on the top 15 most popular genres
top_15 = df['genre'].value_counts().head(15).index
df_top = df[df['genre'].isin(top_15)]

# Calculate the mean of 'is_short' for each genre
penetration = df_top.groupby('genre')['is_short'].mean() * 100
penetration = penetration.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(penetration.index, penetration.values, color='#4c72b0')

# Highlight genres > 50% in red
for bar in bars:
    if bar.get_height() > 50:
        bar.set_color('#d62728') 
    else:
        bar.set_color('#1f77b4')

plt.title('Short Film Penetration: % of Titles that are Shorts by Genre', fontsize=16)
plt.ylabel('% Short Films', fontsize=12)
plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print("✅ Analysis Complete.")