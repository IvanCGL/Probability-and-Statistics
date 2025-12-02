import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

LOAD_DIR = "./dataset/IMDB TMDB Movie Metadata Big Dataset (1M).csv"

try:
    df = pd.read_csv(LOAD_DIR)

except Exception as e:

    print(f"Load csv file fails.")
# ==========================================
# 1. 数据预处理：计算题材数量
# ==========================================
# 我们先定义一个辅助函数，过滤掉 'Unknown' 并计算长度
# 1. 数据清洗与预处理
df_plot = df.dropna(subset=['genres_list', 'release_year']).copy()

# 【新增这一行】：只保留 2025 年及以前的数据
df_plot = df_plot[df_plot['release_year'] <= 2025]

# 【关键检查】：如果数据读进来是字符串 "['Action', 'Drama']"，需要把它变回真正的列表
# 我们检查第一行数据，如果是字符串类型，就执行转换
if isinstance(df_plot['genres_list'].iloc[0], str):
    # 使用 ast.literal_eval 安全地将字符串转为列表
    df_plot['genres_list'] = df_plot['genres_list'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

def count_valid_genres(g_list):
    if not isinstance(g_list, list): return 0
    # 只计算非 Unknown 的有效标签
    valid_tags = [g for g in g_list if g != 'Unknown']
    return len(valid_tags)

# 应用函数，创建一个新列 'genre_count'
df_plot['genre_count'] = df_plot['genres_list'].apply(count_valid_genres)
print(df_plot.head())

# 剔除那些标签数量为 0 的异常数据（如果有的话）
df_hybrid = df_plot[df_plot['genre_count'] > 0].copy()
# 只保留 1910 年以后的数据进行绘图
df_hybrid = df_hybrid[df_hybrid['release_year'] >= 1910]
# ==========================================
# 2. 维度一：平均题材数量随时间的变化 (Average Complexity)
# ==========================================
# 计算每年的平均题材数
avg_genre_count = df_hybrid.groupby('release_year')['genre_count'].mean()
print(avg_genre_count)

plt.figure(figsize=(12, 6))
# 绘制原始数据点（灰色背景）
plt.scatter(avg_genre_count.index, avg_genre_count.values, color='lightgray', s=10, alpha=0.5, label='Yearly Avg')
# 绘制趋势线（滑动平均）
plt.plot(avg_genre_count.rolling(window=5).mean(), color='#d62728', linewidth=3, label='5-Year Trend')

plt.title('Trend of Movie Complexity: Average Number of Genres per Movie', fontsize=16)
plt.ylabel('Avg. Genre Count', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ==========================================
# 3. 维度二：“纯类型” vs “多类型” 占比演变 (Stacked Area Chart)
# ==========================================
# 我们将电影分为三类：
# 1. Single Genre (纯粹，只有1个标签)
# 2. Dual Genre (双重，有2个标签)
# 3. Multi Genre (复杂，3个及以上标签)

def classify_complexity(n):
    if n == 1: return '1. Single Genre'
    elif n == 2: return '2. Dual Genre'
    else: return '3. Multi Genre (3+)'

df_hybrid['complexity_type'] = df_hybrid['genre_count'].apply(classify_complexity)

# 生成交叉表
complexity_counts = pd.crosstab(df_hybrid['release_year'], df_hybrid['complexity_type'])
# 计算百分比
complexity_pct = complexity_counts.div(complexity_counts.sum(axis=1), axis=0) * 100

# 画堆叠面积图
plt.figure(figsize=(12, 6))
colors = ['#aec7e8', '#ffbb78', '#98df8a'] # 柔和的配色
complexity_pct.plot(kind='area', stacked=True, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8, ax=plt.gca())

plt.title('Evolution of "Hybridization": Decline of the Single-Genre Movie', fontsize=16)
plt.ylabel('Market Share (%)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend(title='Complexity Level', loc='upper left')
plt.grid(True, alpha=0.3, axis='x') # 只显示竖向网格
plt.margins(0, 0) # 让图表紧贴边缘
plt.show()