import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')

# 1. 加载清洗后的长篇电影数据集
LOAD_PATH = "./dataset/IMDB_Feature_Films_Cleaned.csv"
df = pd.read_csv(LOAD_PATH)

# 再次修复列表格式 (CSV保存后会变字符串)
if isinstance(df['genres_list'].iloc[0], str):
    df['genres_list'] = df['genres_list'].apply(ast.literal_eval)

# ==========================================
# 维度一：硬指标的相关性 (Numerical Correlation)
# 回答：有钱、片长、老电影，谁更容易高分？
# ==========================================
print("正在计算数值相关性...")

# 筛选数值列
# 注意：Revenue 是上映后的结果，Budget 是上映前的投入。
# 我们看它们与 AverageRating (评分) 的关系
num_cols = ['AverageRating', 'runtime', 'budget', 'revenue', 'release_year']
corr_matrix = df[num_cols].corr()

plt.figure(figsize=(10, 8))
# 绘制热力图
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap: What Drives High Ratings?', fontsize=16)
plt.show()

# ==========================================
# 深度挖掘：时长与评分的非线性关系 (The "Sweet Spot")
# ==========================================
# 相关性系数只能看线性关系，但时长往往有“甜蜜点”
plt.figure(figsize=(12, 6))

# 使用 hexbin 图或 regplot (如果数据量太大，hexbin更好)
# 这里为了看清趋势，我们按时长分桶取平均
df['runtime_bin'] = (df['runtime'] // 10) * 10
runtime_score = df[df['runtime'] <= 240].groupby('runtime_bin')['AverageRating'].mean() # 限制在4小时内

plt.plot(runtime_score.index, runtime_score.values, marker='o', color='#2c3e50', linewidth=2)
plt.title('Runtime vs. Rating: Is Longer Better?', fontsize=16)
plt.xlabel('Runtime (Minutes)', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 标注峰值
peak_runtime = runtime_score.idxmax()
peak_score = runtime_score.max()
plt.annotate(f'Sweet Spot: ~{peak_runtime} min', 
             xy=(peak_runtime, peak_score), 
             xytext=(peak_runtime+20, peak_score-0.2),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.show()

# ==========================================
# 维度二：题材红利 (Genre Impact Analysis)
# 回答：拍什么题材天生容易高分？（Genre Coefficient）
# ==========================================
print("正在分析题材影响力...")

# 1. 炸开数据
df_exploded = df.explode('genres_list')

# 2. 统计每个题材的平均分、方差、数量
genre_stats = df_exploded.groupby('genres_list').agg({
    'AverageRating': ['mean', 'std', 'count']
})

# 扁平化列名
genre_stats.columns = ['AverageRating', 'rating_std', 'movie_count']

# 3. 过滤掉极其小众的题材 (样本过少会导致分数虚高或虚低)
# 只保留至少有 500 部电影的题材
genre_stats = genre_stats[genre_stats['movie_count'] > 500]

# 4. 排序：按平均分高低
genre_stats = genre_stats.sort_values('AverageRating', ascending=True)

# 5. 绘图：带误差棒的条形图
plt.figure(figsize=(12, 8))

# 绘制条形图 (平均分)
bars = plt.barh(genre_stats.index, genre_stats['AverageRating'], xerr=genre_stats['rating_std'], 
                capsize=4, color='#7f8c8d', alpha=0.7)

# 高亮前3名和后3名
for i, bar in enumerate(bars):
    if i >= len(bars) - 3:
        bar.set_color('#27ae60') # 绿色：高分题材
    elif i < 3:
        bar.set_color('#c0392b') # 红色：低分题材

plt.title('Genre Impact: Which Genres Have the Highest Intrinsic Quality?', fontsize=16)
plt.xlabel('Average Rating (with Standard Deviation)', fontsize=12)
plt.xlim(5, 8) # 聚焦在 5-8 分区间，看得更清楚
plt.grid(axis='x', linestyle='--', alpha=0.5)

plt.show()