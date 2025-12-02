import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast  # 关键库：用于将字符串格式的列表 "['A','B']" 转换为真正的列表 ['A','B']

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial'] # 防止中文乱码（视系统而定，英文一般用Arial）

LOAD_DIR = "../dataset/IMDB TMDB Movie Metadata Big Dataset (1M).csv"

# ==========================================
# 0. 数据加载与全局清洗 (Global Preprocessing)
# ==========================================
try:
    df = pd.read_csv(LOAD_DIR)
    print("数据加载成功！开始清洗...")

    # 1. 强制转换数值类型（处理脏数据）
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')

    # 2. 剔除核心字段为空的数据
    # 注意：我们在这里先不剔除 'Unknown'，在具体统计题材时再剔除，以免影响时长统计
    df_clean = df.dropna(subset=['runtime', 'release_year', 'genres_list']).copy()
    
    # 3. 剔除时长异常值
    df_clean = df_clean[df_clean['runtime'] > 0]

    # 4. 【关键步骤】修复 genres_list 格式
    # 如果读取出来是字符串类型，使用 ast.literal_eval 转换回列表
    print("正在修复题材列表格式（可能需要几秒钟）...")
    if isinstance(df_clean['genres_list'].iloc[0], str):
        df_clean['genres_list'] = df_clean['genres_list'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    
    # 5. 限制总时间范围 (1910 - 2025)
    START_YEAR, END_YEAR = 1910, 2025
    df_clean = df_clean[(df_clean['release_year'] >= START_YEAR) & (df_clean['release_year'] <= END_YEAR)]

    print(f"清洗完成。有效数据量: {len(df_clean)} 行")

except Exception as e:
    print(f"数据处理出错: {e}")
    exit() # 如果出错直接退出

# ==========================================
# Part 1: 电影时长分布演变 (Boxplot)
# ==========================================
print("正在绘制：时长分布箱线图...")

# 创建年代列
df_clean['decade'] = (df_clean['release_year'] // 10) * 10

plt.figure(figsize=(14, 8))
sns.boxplot(data=df_clean, x='decade', y='runtime', 
            palette="Blues", 
            width=0.6, 
            showfliers=False) # 不显示离群点，聚焦主流趋势

# 辅助线
plt.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90 min (Standard)')
plt.axhline(y=120, color='g', linestyle='--', alpha=0.5, label='120 min (Epic)')

plt.title('Distribution of Movie Runtimes per Decade (1910-2025)', fontsize=16)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Runtime (Minutes)', fontsize=12)
plt.ylim(0, 200) 
plt.legend(loc='upper right')
plt.grid(True, axis='y', alpha=0.3)
plt.show()

# ==========================================
# Part 2: 短片占比趋势 (Area Chart)
# ==========================================
print("正在绘制：短片趋势图...")

SHORT_THRESHOLD = 60
# 创建临时列标记短片
df_clean['is_short'] = df_clean['runtime'] <= SHORT_THRESHOLD

# 按年份统计短片占比
short_trend = df_clean.groupby('release_year')['is_short'].mean() * 100

plt.figure(figsize=(12, 6))
plt.fill_between(short_trend.index, 0, short_trend.values, color='#86d0cb', alpha=0.4)
plt.plot(short_trend.index, short_trend.values, color='#2c8c88', linewidth=2.5, label='% of Short Films')

plt.title(f'Rise of Short Content: Percentage of Titles < {SHORT_THRESHOLD} Minutes', fontsize=16)
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Market Share (%)', fontsize=12)
plt.xlim(START_YEAR, END_YEAR)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# ==========================================
# Part 3: 题材统计 (Genre Analysis - Exploded)
# ==========================================
print("正在绘制：热门题材统计图...")

# 1. 炸开列表 (Explode)
# 因为我们在第0步已经用 ast.literal_eval 转成了列表，这里可以直接炸开
df_exploded = df_clean.explode('genres_list')

# 2. 统计频次
genre_counts = df_exploded['genres_list'].value_counts()

# 3. 剔除无效数据 'Unknown'
if 'Unknown' in genre_counts:
    genre_counts = genre_counts.drop('Unknown')

# 4. 可视化 Top 10
top_genres = genre_counts.head(10)

plt.figure(figsize=(12, 6))
top_genres.sort_values().plot(kind='barh', color='#4c72b0', width=0.8)

plt.title('Top 10 Most Frequent Genres (Individual Tag Count)', fontsize=16)
plt.xlabel('Number of Movies', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# 打印最终数值供查阅
print("\nTop 10 热门题材统计结果：")
print(top_genres)