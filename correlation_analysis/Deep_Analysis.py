import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
LOAD_PATH = "./dataset/IMDB_Feature_Films_Cleaned.csv"

print("Loading data...")
try:
    df = pd.read_csv(LOAD_PATH)
except FileNotFoundError:
    print(f"File {LOAD_PATH} not found, please check the path.")
    exit()

# Critical: Ensure rating is numeric (Clean non-numeric ratings)
df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
df = df.dropna(subset=['IMDB_Rating'])

# Simple filter: Remove extremely niche movies (fewer than 100 votes)
if 'vote_count' in df.columns:
    df = df[df['vote_count'] > 100]

print(f"Data ready. Analysis sample size: {len(df)}")

# ==========================================
# Analysis 1: Verify "The Auteur Effect"
# ==========================================
print("\nGenerating Chart 1: Director Effect Analysis...")

# 1. Statistics for each director's movie count and average rating
director_stats = df.groupby('Director')['IMDB_Rating'].agg(['mean', 'count'])

# 2. Define "Top Tier Directors"
# Criteria: Directed at least 5 movies (ensures experience) and Average Rating >= 7.5 (ensures quality)
top_directors_list = director_stats[
    (director_stats['count'] >= 5) & 
    (director_stats['mean'] >= 7.5)
].index

print(f"Identified {len(top_directors_list)} Top Directors.")

# 3. Labeling
df['Director_Class'] = df['Director'].apply(
    lambda x: 'Top Tier Directors' if x in top_directors_list else 'Regular Directors'
)

# 4. Visualization: Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='Director_Class', y='IMDB_Rating', data=df, 
               palette={"Top Tier Directors": "#e74c3c", "Regular Directors": "#95a5a6"},
               inner="quartile") # Show quartile lines

plt.title('The "Auteur Effect": Do Top Directors Guarantee Quality?', fontsize=16)
plt.ylabel('IMDB Rating', fontsize=12)
plt.xlabel('')
plt.ylim(0, 10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# Analysis 2: Verify "Value for Money Rule" (Budget vs. Quality via ROI)
# ==========================================
print("Generating Chart 2: Budget vs. Rating Scatter Plot...")

# 1. Calculate ROI (if not calculated previously)
# Simple ROI = (Revenue - Budget) / Budget
df_money = df[(df['budget'] > 100000) & (df['revenue'] > 10000)].copy()
df_money['roi'] = (df_money['revenue'] - df_money['budget']) / df_money['budget']

# 2. Visualization: Bubble Scatter Plot
plt.figure(figsize=(12, 7))

plot_data = df_money[df_money['budget'] < 500000000]

scatter = sns.scatterplot(
    data=plot_data, 
    x='budget', 
    y='IMDB_Rating', 
    hue='roi', 
    size='roi',
    sizes=(20, 200), 
    palette='viridis', 
    alpha=0.7,
    legend='brief'
)

plt.xscale('log') # Use log scale because budget grows exponentially
plt.title('Budget vs. Quality: High Budget ≠ High Rating', fontsize=16)
plt.xlabel('Budget (USD, Log Scale)', fontsize=12)
plt.ylabel('IMDB Rating', fontsize=12)

# Move legend to the side
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='ROI (Ratio)')
plt.tight_layout()
plt.show()

# ==========================================
# Analysis 3: Verify "Content Sentiment" (Sentiment Analysis)
# Logic: Model shows sentiment is useful; checking if tragedy scores higher than comedy.
# ==========================================
if 'overview_sentiment' in df.columns:
    print("Generating Chart 3: Impact of Sentiment on Rating...")
    
    # 1. Binning sentiment scores
    # -1 to -0.1 is Negative (Dark/Sad), -0.1 to 0.1 is Neutral, 0.1 to 1 is Positive (Happy)
    bins = [-1, -0.1, 0.1, 1]
    labels = ['Negative (Dark/Serious)', 'Neutral', 'Positive (Light/Happy)']
    df['Tone'] = pd.cut(df['overview_sentiment'], bins=bins, labels=labels)
    
    # 2. Visualization: Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Tone', y='IMDB_Rating', data=df, palette="coolwarm")
    
    plt.title('Tone & Quality: Do "Dark" Movies Get Higher Ratings?', fontsize=16)
    plt.ylabel('IMDB Rating', fontsize=12)
    plt.xlabel('Movie Tone (Based on Overview Sentiment)', fontsize=12)
    plt.tight_layout()
    plt.show()
else:
    print("Skipping Chart 3: 'overview_sentiment' column missing from dataset.")

print("\n✅ All charts generated successfully!")