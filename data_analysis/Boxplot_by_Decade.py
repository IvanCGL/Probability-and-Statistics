import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LOAD_DIR = "../dataset/IMDB TMDB Movie Metadata Big Dataset (1M).csv"

try:
    df = pd.read_csv(LOAD_DIR)

except Exception as e:

    print(f"Load csv file fails.")

# 1. 数据清洗
df_run = df.dropna(subset=['runtime', 'release_year']).copy()
df_run = df_run[df_run['runtime'] > 0] # 剔除时长为0的错误数据

# 2. 关键步骤：创建“年代”列
df_run['decade'] = (df_run['release_year'] // 10) * 10

# 3. 过滤年份（保持一致性，只看 1910 - 2025）
df_run = df_run[(df_run['decade'] >= 1910) & (df_run['decade'] <= 2020)]

# 4. 绘图
plt.figure(figsize=(14, 8))

# 使用 Seaborn 画箱线图
# x=年代, y=时长
sns.boxplot(data=df_run, x='decade', y='runtime', 
            palette="Blues",  # 使用渐变蓝色
            width=0.6,        # 箱子宽度
            showfliers=False) # 是否显示离群点
                              # False = 不显示极端值（让图表更清晰，聚焦主流）
                              # True = 显示所有黑点（可以看到那些 200分钟+ 的电影）

# 5. 添加辅助线和装饰
plt.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90 min (Standard)')
plt.axhline(y=120, color='g', linestyle='--', alpha=0.5, label='120 min (Epic)')

plt.title('Distribution of Movie Runtimes per Decade', fontsize=16)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Runtime (Minutes)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, axis='y', alpha=0.3)

# 限制 Y 轴范围（如果不隐藏离群点，这行很有用）
# 大部分电影都在 300分钟以内，限制一下防止被个别 10小时的艺术片压缩画面
plt.ylim(0, 200) 

plt.show()