import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LOAD_DIR = "./dataset/IMDB TMDB Movie Metadata Big Dataset (1M).csv"

try:
    df = pd.read_csv(LOAD_DIR)

except Exception as e:
    print(f"Load csv file fails.")

# 1. Data Cleaning
df_run = df.dropna(subset=['runtime', 'release_year']).copy()
df_run = df_run[df_run['runtime'] > 0] # Remove erroneous data where runtime is 0

# 2. Key Step: Create 'decade' column
df_run['decade'] = (df_run['release_year'] // 10) * 10

# 3. Filter Years (Maintain consistency, focus on 1910 - 2020)
df_run = df_run[(df_run['decade'] >= 1910) & (df_run['decade'] <= 2020)]

# 4. Plotting
plt.figure(figsize=(14, 8))

# Use Seaborn to draw a boxplot
# x=decade, y=runtime
sns.boxplot(data=df_run, x='decade', y='runtime', 
            palette="Blues",  # Use gradient blue palette
            width=0.6,        # Box width
            showfliers=False) # Whether to show outliers
                              # False = Hide extreme values (makes chart clearer, focuses on mainstream)
                              # True = Show all points (allows seeing 200min+ movies)

# 5. Add auxiliary lines and decorations
plt.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90 min (Standard)')
plt.axhline(y=120, color='g', linestyle='--', alpha=0.5, label='120 min (Epic)')

plt.title('Distribution of Movie Runtimes per Decade', fontsize=16)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Runtime (Minutes)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, axis='y', alpha=0.3)

# Limit Y-axis range (Useful if outliers are not hidden)
# Most movies are under 300 mins; this limits the view to prevent rare 10-hour art films from compressing the visualization
plt.ylim(0, 200) 

plt.show()