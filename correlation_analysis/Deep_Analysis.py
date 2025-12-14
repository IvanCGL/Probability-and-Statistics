import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
# 解决可能的字体显示问题 (视系统而定，Windows常用 SimHei 或 Arial)
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据加载与预处理
# ==========================================
# 请确保使用你之前清洗好的长片数据集
LOAD_PATH = "./dataset/IMDB_Feature_Films_Cleaned.csv"

print("正在加载数据...")
try:
    df = pd.read_csv(LOAD_PATH)
except FileNotFoundError:
    print(f"找不到文件 {LOAD_PATH}，请检查路径。")
    exit()

# 关键：确保评分是数值型 (清洗非数值评分)
df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
df = df.dropna(subset=['IMDB_Rating'])

# 简单过滤：剔除极其冷门的电影 (少于100人评分)
if 'vote_count' in df.columns:
    df = df[df['vote_count'] > 100]

print(f"数据准备就绪，分析样本量: {len(df)}")

# ==========================================
# 分析一：验证“作者论” (The Auteur Effect)
# ==========================================
print("\n正在生成图表 1: 导演效应分析...")

# 1. 统计每位导演的执导数量和平均分
director_stats = df.groupby('Director')['IMDB_Rating'].agg(['mean', 'count'])

# 2. 定义“名导” (Top Tier Directors)
# 标准：至少执导过 5 部电影 (保证资历) 且 平均分 >= 7.5 (保证质量)
top_directors_list = director_stats[
    (director_stats['count'] >= 5) & 
    (director_stats['mean'] >= 7.5)
].index

print(f"筛选出 {len(top_directors_list)} 位顶级导演 (Top Directors)。")

# 3. 打标签
df['Director_Class'] = df['Director'].apply(
    lambda x: 'Top Tier Directors' if x in top_directors_list else 'Regular Directors'
)

# 4. 可视化：小提琴图 (Violin Plot)
# 小提琴图比箱线图更好，因为它能展示分布的形状（是集中在某处，还是两极分化）
plt.figure(figsize=(10, 6))
sns.violinplot(x='Director_Class', y='IMDB_Rating', data=df, 
               palette={"Top Tier Directors": "#e74c3c", "Regular Directors": "#95a5a6"},
               inner="quartile") # 显示四分位数线

plt.title('The "Auteur Effect": Do Top Directors Guarantee Quality?', fontsize=16)
plt.ylabel('IMDB Rating', fontsize=12)
plt.xlabel('')
plt.ylim(0, 10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 分析二：验证“性价比法则” (Budget vs. Quality via ROI)
# 逻辑：模型显示 Budget 权重低但 ROI 高，我们验证这一点。
# ==========================================
print("正在生成图表 2: 预算与评分散点图...")

# 1. 计算 ROI (如果之前没算过)
# 简单的 ROI = (票房 - 成本) / 成本
# 过滤掉成本极低的数据（可能是错误数据）防止除以零
df_money = df[(df['budget'] > 100000) & (df['revenue'] > 10000)].copy()
df_money['roi'] = (df_money['revenue'] - df_money['budget']) / df_money['budget']

# 2. 可视化：气泡散点图
plt.figure(figsize=(12, 7))

plot_data = df_money[df_money['budget'] < 500000000] # 3亿美元以下

scatter = sns.scatterplot(
    data=plot_data, 
    x='budget', 
    y='IMDB_Rating', 
    hue='roi',       # 颜色代表 ROI 高低
    size='roi',      # 大小也代表 ROI
    sizes=(20, 200), # 设置点的大小范围
    palette='viridis', 
    alpha=0.7,
    legend='brief'
)

plt.xscale('log') # 使用对数坐标，因为预算是指数级增长的
plt.title('Budget vs. Quality: High Budget ≠ High Rating', fontsize=16)
plt.xlabel('Budget (USD, Log Scale)', fontsize=12)
plt.ylabel('IMDB Rating', fontsize=12)

# 移动图例到旁边
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='ROI (Ratio)')
plt.tight_layout()
plt.show()

# ==========================================
# 分析三：验证“内容情绪” (Sentiment Analysis)
# 逻辑：模型显示 sentiment 有用，我们看看到底是悲剧分高还是喜剧分高。
# ==========================================
if 'overview_sentiment' in df.columns:
    print("正在生成图表 3: 情绪对评分的影响...")
    
    # 1. 将情绪分数分箱 (Binning)
    # -1 到 -0.1 为负面（黑暗/悲伤），-0.1 到 0.1 为中性，0.1 到 1 为正面（快乐）
    bins = [-1, -0.1, 0.1, 1]
    labels = ['Negative (Dark/Serious)', 'Neutral', 'Positive (Light/Happy)']
    df['Tone'] = pd.cut(df['overview_sentiment'], bins=bins, labels=labels)
    
    # 2. 可视化：箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Tone', y='IMDB_Rating', data=df, palette="coolwarm")
    
    plt.title('Tone & Quality: Do "Dark" Movies Get Higher Ratings?', fontsize=16)
    plt.ylabel('IMDB Rating', fontsize=12)
    plt.xlabel('Movie Tone (Based on Overview Sentiment)', fontsize=12)
    plt.tight_layout()
    plt.show()
else:
    print("跳过图表 3：数据集中缺少 'overview_sentiment' 列。")

print("\n✅ 所有图表生成完毕！")