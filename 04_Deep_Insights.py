import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. Load Data & Preprocessing
# ==========================================
LOAD_PATH = "./dataset/IMDB_Feature_Films_Cleaned.csv"
print("Loading dataset for Deep Insights...")

if not os.path.exists(LOAD_PATH):
    print(f"[ERROR] File not found: {LOAD_PATH}")
    exit(1)

df = pd.read_csv(LOAD_PATH)

# Critical: Clean Ratings
df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
df = df.dropna(subset=['IMDB_Rating'])
if 'vote_count' in df.columns:
    df = df[df['vote_count'] > 100]

print(f"Data ready. Insights based on {len(df)} movies.")

# ==========================================
# Helper Function: Identify Top Tier Talent
# ==========================================
def get_top_tier_list(df, col_name, min_works=5, min_rating=7.5):
    if col_name not in df.columns: return []
    stats = df.groupby(col_name)['IMDB_Rating'].agg(['mean', 'count'])
    top_tier = stats[(stats['count'] >= min_works) & (stats['mean'] >= min_rating)].index
    return top_tier

# ==========================================
# Insight 1: The "Creative Trinity" (Writer, Star, Director)
# Logic: Validate the Model's Top 3 features.
# ==========================================
print("\nGenerating Chart 1: The Creative Trinity (Writer > Star > Director)...")

# Define columns to analyze
creatives = {
    'Writer': 'Script is King (Writer Impact)',
    'Star1': 'Star Power (Lead Actor Impact)',
    'Director': 'Auteur Effect (Director Impact)'
}

# Create a figure with 3 subplots side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, (col, title) in enumerate(creatives.items()):
    if col in df.columns:
        # 1. Identify Top Talent
        top_talent = get_top_tier_list(df, col)
        print(f"  - Found {len(top_talent)} Top Tier {col}s")
        
        # 2. Label Data
        # We use a temporary column for plotting
        temp_col = f'{col}_Tier'
        df[temp_col] = df[col].apply(lambda x: 'Top Tier' if x in top_talent else 'Regular')
        
        # 3. Plot Violin
        sns.violinplot(x=temp_col, y='IMDB_Rating', data=df, ax=axes[i],
                       palette={"Top Tier": "#e74c3c", "Regular": "#95a5a6"},
                       inner="quartile", order=['Top Tier', 'Regular'])
        
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('IMDB Rating' if i == 0 else '')
        axes[i].grid(axis='y', alpha=0.3)

plt.suptitle('Validation of Model Findings: The "Creative Trinity" Drives Quality', fontsize=16)
plt.tight_layout()
plt.show()

# ==========================================
# Insight 2: The "Efficiency Rule" (Budget vs. Quality via ROI)
# ==========================================
print("Generating Chart 2: Budget vs. Quality (Bubble Chart)...")

df_money = df[(df['budget'] > 100000) & (df['revenue'] > 10000)].copy()
df_money['roi'] = (df_money['revenue'] - df_money['budget']) / df_money['budget']

plt.figure(figsize=(12, 7))
plot_data = df_money[df_money['budget'] < 300000000] # Remove extreme outliers for clarity

scatter = sns.scatterplot(
    data=plot_data, 
    x='budget', 
    y='IMDB_Rating', 
    hue='roi', 
    size='roi',
    sizes=(20, 200), 
    palette='viridis', 
    alpha=0.7
)

plt.xscale('log')
plt.title('Budget vs. Quality: High Budget ≠ High Rating (Efficiency Matters)', fontsize=16)
plt.xlabel('Budget (USD, Log Scale)', fontsize=12)
plt.ylabel('IMDB Rating', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='ROI (Ratio)')
plt.tight_layout()
plt.show()

# ==========================================
# Insight 3: Tone & Content
# ==========================================
if 'overview_sentiment' in df.columns:
    print("Generating Chart 3: Sentiment/Tone Analysis...")
    
    bins = [-1, -0.1, 0.1, 1]
    labels = ['Negative (Dark/Serious)', 'Neutral', 'Positive (Light/Happy)']
    df['Tone'] = pd.cut(df['overview_sentiment'], bins=bins, labels=labels)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Tone', y='IMDB_Rating', data=df, palette="coolwarm")
    plt.title('Tone & Quality: Do "Dark" Movies Get Higher Ratings?', fontsize=16)
    plt.ylabel('IMDB Rating')
    plt.show()
else:
    print("[WARNING] Skipping Chart 3: 'overview_sentiment' column missing.")

print("\n✅ Deep Insights Analysis Complete!")