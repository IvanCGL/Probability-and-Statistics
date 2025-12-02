import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

# 1. 加载我们在上一步生成的干净数据集
LOAD_PATH = "./dataset/IMDB_Feature_Films_Cleaned.csv"

print("正在加载长篇电影数据集...")
df = pd.read_csv(LOAD_PATH)

# 2. 必须再次修复 genres_list 格式
# CSV 保存后列表会变回字符串，必须用 eval 恢复
if isinstance(df['genres_list'].iloc[0], str):
    df['genres_list'] = df['genres_list'].apply(ast.literal_eval)

print(f"数据加载完毕。分析样本量: {len(df)}")

# ==========================================
# 1. 题材流行度 (Genre Popularity over Time)
# ==========================================
# 炸开数据，统计每一个题材
df_exploded = df.explode('genres_list')

# 创建年代列 (Decade) 以减少噪音，每10年看一次大趋势
df_exploded['decade'] = (df_exploded['release_year'] // 10) * 10

# 统计每年代各题材数量
genre_trend = pd.crosstab(df_exploded['decade'], df_exploded['genres_list'])
# 归一化：计算占比 (%)
genre_pct = genre_trend.div(genre_trend.sum(axis=1), axis=0) * 100

# 选取最具有代表性的题材 (Top 8) 进行可视化
top_genres = df_exploded['genres_list'].value_counts().head(10).index

plt.figure(figsize=(14, 7))
# 绘制堆叠面积图 (Stacked Area Chart)
plt.stackplot(genre_pct.index, 
              [genre_pct[g] for g in top_genres],
              labels=top_genres, alpha=0.8)

plt.title('Evolution of Audience Preferences: Genre Market Share (Feature Films Only)', fontsize=16)
plt.ylabel('Market Share (%)', fontsize=12)
plt.xlabel('Decade', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Genre')
plt.xlim(1920, 2020) # 聚焦核心年代
plt.tight_layout()
plt.show()

# 1. 找出剩余的题材 (Niche Genres)
all_genres = genre_pct.columns.tolist()
# 使用列表推导式剔除 Top 12
niche_genres = [g for g in all_genres if g not in top_genres]

# 按总占比大小对剩余题材排序（为了画图好看，大的在下面）
# 计算剩余题材的平均市场份额
niche_mean_share = genre_pct[niche_genres].mean().sort_values(ascending=False)
sorted_niche_genres = niche_mean_share.index.tolist()

print(f"剩余的小众题材包括: {sorted_niche_genres}")

# ==========================================
# 可视化 1.1: 小众市场的内部演变 (堆叠面积图)
# ==========================================
plt.figure(figsize=(14, 7))

# 使用不同的配色方案 (Set2 或 tab20) 以示区别
colors = sns.color_palette("tab20", len(sorted_niche_genres))

plt.stackplot(genre_pct.index, 
              [genre_pct[g] for g in sorted_niche_genres],
              labels=sorted_niche_genres,
              colors=colors, 
              alpha=0.8)

plt.title('Evolution of Niche Genres (The "Long Tail" Market Share)', fontsize=16)
plt.ylabel('Market Share (%)', fontsize=12)
plt.xlabel('Decade', fontsize=12)
plt.xlim(1920, 2020)

# 关键：手动设置 Y 轴范围，让图表充满画布
# 计算剩余题材每年的总和最大值，稍微加一点余量
max_share = genre_pct[sorted_niche_genres].sum(axis=1).max()
plt.ylim(0, max_share * 1.1) 

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Niche Genre')
plt.tight_layout()
plt.show()

# ==========================================
# 可视化 1.2: 重点历史题材的兴衰 (折线图)
# 西部、战争、历史片通常被大类淹没，但在 Trend Analysis 中极具价值
# ==========================================
historical_genres = ['Western', 'War', 'History', 'Music']
# 确保这些题材在你的 niche_genres 或 top_genres 里 (取交集)
target_genres = [g for g in historical_genres if g in genre_pct.columns]

plt.figure(figsize=(14, 6))

for genre in target_genres:
    plt.plot(genre_pct.index, genre_pct[genre], 
             marker='o', linewidth=2.5, label=genre)

plt.title('The Rise and Fall of Specific Historical Genres', fontsize=16)
plt.ylabel('Market Share (%)', fontsize=12)
plt.xlabel('Decade', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(1920, 2020)

# 添加标注 (Annotation) - 这是一个加分项！
# 例如标注西部片的黄金时代
if 'Western' in target_genres:
    western_peak = genre_pct['Western'].loc[1950] # 假设1950是高点
    plt.annotate('Golden Age of Westerns', 
                 xy=(1950, western_peak), 
                 xytext=(1960, western_peak + 2),
                 arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

# ==========================================
# 2. 混搭趋势 (Hybridization Trend)
# ==========================================
# 计算每部电影的标签数量
df['genre_count'] = df['genres_list'].apply(len)

# 计算每年的平均标签数
complexity_trend = df.groupby('release_year')['genre_count'].mean()

plt.figure(figsize=(12, 5))
plt.plot(complexity_trend.index, complexity_trend.values, color='#8e44ad', linewidth=2)
# 加上平滑曲线
plt.plot(complexity_trend.rolling(5).mean(), color='black', linewidth=1, linestyle='--')

plt.title('The "Blockbuster Formula": Are Movies Becoming More Complex?', fontsize=16)
plt.ylabel('Avg. Number of Genres per Movie', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# ==========================================
# 3. 预算与回报 (Budget & ROI)
# ==========================================
# 1. 清洗：剔除预算或票房过低的数据 (通常是错误数据或极小成本片)
# 假设 budget < 10000 美元通常不可信或不是商业发行
df_finance = df[(df['budget'] > 10000) & (df['revenue'] > 10000)].copy()

# 2. 计算 ROI (投资回报率) = (Revenue - Budget) / Budget
df_finance['roi'] = (df_finance['revenue'] - df_finance['budget']) / df_finance['budget']

# 3. 按年代统计中位数 (Median)
# 使用中位数是因为平均值容易被《阿凡达》这种极值拉偏
finance_trend = df_finance.groupby('release_year')[['budget', 'revenue', 'roi']].median()

# 绘图：双轴图
fig, ax1 = plt.subplots(figsize=(14, 7))

# 左轴：画预算 (柱状图)
color = 'tab:blue'
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Median Budget (Nominal $)', color=color, fontsize=12)
ax1.plot(finance_trend.index, finance_trend['budget'], color=color, label='Median Budget')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log') # 使用对数坐标，因为通胀导致数值跨度极大

# 右轴：画 ROI (折线图)
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Median ROI (Ratio)', color=color, fontsize=12)
# 使用滑动平均平滑 ROI 曲线
ax2.plot(finance_trend.index, finance_trend['roi'].rolling(5).mean(), color=color, linestyle='--', linewidth=2, label='ROI (5y Trend)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 10) # 限制 ROI 显示范围，通常 ROI > 10 很少见

plt.title('The Cost of Business: Rising Budgets vs. Diminishing Returns', fontsize=16)
fig.tight_layout()
plt.show()

# ==========================================
# 4. 评分趋势与幸存者偏差 (Rating Evolution)
# ==========================================
# 统计每年的平均分和方差
rating_stats = df.groupby('release_year')['AverageRating'].agg(['mean', 'std', 'count'])

plt.figure(figsize=(14, 6))

# 1. 绘制平均分 (主线)
plt.plot(rating_stats.index, rating_stats['mean'], color='#2c3e50', linewidth=3, label='Average Rating')

# 2. 绘制标准差范围 (阴影) - 展示当年电影质量的参差不齐程度
plt.fill_between(rating_stats.index, 
                 rating_stats['mean'] - rating_stats['std'], 
                 rating_stats['mean'] + rating_stats['std'], 
                 color='gray', alpha=0.2, label='1 Std Dev (Quality Variance)')

# 3. 添加辅助分析：每年电影产量 (Bar Chart on bottom)
# 为了展示幸存者偏差：早年电影少，分高；近年电影多，分低
plt.bar(rating_stats.index, rating_stats['count'] / rating_stats['count'].max() * 2 + 4, 
        color='orange', alpha=0.3, label='Movie Volume (Scaled)')

plt.title('Rating Trends: Survivorship Bias & The "Golden Age" Illusion', fontsize=16)
plt.ylabel('IMDb Rating (0-10)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend(loc='lower left')
plt.ylim(0, 9) # 聚焦分数段
plt.show()