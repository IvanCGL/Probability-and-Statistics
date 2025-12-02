import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据加载与预处理
# ==========================================
LITE_PATH = "../dataset/IMDB_Genres_Lite.csv" # 确保文件名一致

print("正在加载数据...")
try:
    df = pd.read_csv(LITE_PATH)
except FileNotFoundError:
    print("找不到文件，请检查路径。")
    exit()

# 基础清洗
df = df.dropna(subset=['runtime', 'release_year', 'genre'])
df['release_year'] = df['release_year'].astype(int)
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')

# 过滤时间范围 (1910 - 2025)
START_YEAR, END_YEAR = 1910, 2025
df = df[(df['release_year'] >= START_YEAR) & (df['release_year'] <= END_YEAR)]

# 定义短片 (<= 60分钟)
SHORT_THRESHOLD = 60
df['is_short'] = df['runtime'] <= SHORT_THRESHOLD

print(f"数据准备就绪。总行数 (标签级): {len(df)}")

# ==========================================
# 图表 1: 短片崛起的趋势 (Time Trend)
# 【关键】：必须按 ID 去重，因为这里统计的是“电影数量”，不是“标签数量”
# ==========================================
print("正在绘制图表 1: 短片趋势...")

# 按 id 去重，确保每部电影只算一次
df_unique_movies = df.drop_duplicates(subset='id')

# 计算每年的短片占比
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
# 图表 2: 短片 vs 长片 题材基因对比 (Butterfly Chart)
# 这里使用原始的 df (未去重)，因为我们要分析的是“标签”
# ==========================================
print("正在绘制图表 2: 题材对比...")

df = df[df['genre'] != 'Unknown']
# 计算各自的题材分布（归一化为百分比）
short_genres = df[df['is_short']]['genre'].value_counts(normalize=True).head(10) * 100
feature_genres = df[~df['is_short']]['genre'].value_counts(normalize=True).head(10) * 100

# 合并数据
comp_df = pd.DataFrame({'Shorts': short_genres, 'Features': feature_genres})
# 按短片流行度排序
comp_df = comp_df.sort_values('Shorts', ascending=True)

# 绘图
comp_df.plot(kind='barh', figsize=(12, 8), width=0.8, color=['#86d0cb', '#e88c7d'])

plt.title(f'Genre Composition: Short Films (<{SHORT_THRESHOLD}m) vs Features', fontsize=16)
plt.xlabel('Percentage within Category (%)', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 图表 3: 各题材内部的“短片渗透率”
# 回答：Animation 里面到底有多少是短片？
# ==========================================
print("正在绘制图表 3: 渗透率分析...")

# 只看最热门的 15 个题材
top_15 = df['genre'].value_counts().head(15).index
df_top = df[df['genre'].isin(top_15)]

# 计算每个题材中 is_short 的平均值
penetration = df_top.groupby('genre')['is_short'].mean() * 100
penetration = penetration.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(penetration.index, penetration.values, color='#4c72b0')

# 红色高亮 > 50% 的题材
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

print("✅ 分析完成。")