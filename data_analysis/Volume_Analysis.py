import pandas as pd


LOAD_DIR = "../dataset/IMDB TMDB Movie Metadata Big Dataset (1M).csv"

try:
    df = pd.read_csv(LOAD_DIR)

except Exception as e:

    print(f"Load csv file fails.")

assert 'release_date' in df.columns, f"Lossing important information about release_date"

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

df['year'] = df['release_date'].dt.year

# Cleaning dataset
df = df.dropna(subset=['year'])

df['year'] = df['year'].astype(int)

year_counts = df.groupby('year').size()

# print plot
import matplotlib.pyplot as plt

year_counts.plot(kind='line', title='Number of Movies per Year')

plt.xlabel('Year')
plt.ylabel('Count')
plt.show()