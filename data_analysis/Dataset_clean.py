import pandas as pd
import ast

# ==========================================
# 1. 配置参数
# ==========================================
LOAD_PATH = "./dataset/IMDB TMDB Movie Metadata Big Dataset (1M).csv"
SAVE_PATH = "./dataset/IMDB_Feature_Films_Cleaned.csv"

# 筛选标准
START_YEAR = 1910
END_YEAR = 2025
MIN_RUNTIME = 60  # 定义长片：大于 60 分钟

print(f"正在加载原始数据: {LOAD_PATH} ...")
try:
    df = pd.read_csv(LOAD_PATH)
except FileNotFoundError:
    print("错误：找不到文件。")
    exit()

print(f"原始数据行数: {len(df)}")

# ==========================================
# 2. 基础清洗 (Basic Cleaning)
# ==========================================
# 剔除核心字段缺失的行
# 注意：这里我们保留 revenue, budget 等列，虽然可能有空值，但先不急着删，
# 后续做相关性分析时再单独处理那些列。这里只删核心元数据缺失的行。
cols_to_check = ['id', 'title', 'release_year', 'runtime', 'genres_list']
df = df.dropna(subset=cols_to_check)

# 类型转换
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')

# ==========================================
# 3. 执行三大过滤标准 (Filtering)
# ==========================================

# A. 时间过滤 (Time Filter)
df = df[(df['release_year'] >= START_YEAR) & (df['release_year'] <= END_YEAR)]
print(f"时间过滤后 ({START_YEAR}-{END_YEAR}): {len(df)} 行")

# B. 时长过滤 (Runtime Filter - Remove Shorts)
# 只保留长片 (> 60 min)
df = df[df['runtime'] > MIN_RUNTIME]
print(f"时长过滤后 (>{MIN_RUNTIME} min): {len(df)} 行")

# C. 题材过滤 (Genre Filter - Remove 'Unknown')
# 这一步稍微复杂一点，因为 genres_list 是字符串，我们需要先解析
print("正在解析并过滤 Unknown 题材...")

# 1. 格式修复：将字符串 "['Action']" 转为列表
if isinstance(df['genres_list'].iloc[0], str):
    df['genres_list'] = df['genres_list'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

# 2. 过滤逻辑：
# 如果列表里包含 'Unknown'，或者列表为空，剔除该行
def is_valid_genre(g_list):
    if not isinstance(g_list, list): return False
    if len(g_list) == 0: return False
    if 'Unknown' in g_list: return False
    return True

df = df[df['genres_list'].apply(is_valid_genre)]
print(f"剔除 Unknown 题材后: {len(df)} 行")

# ==========================================
# 4. 保存结果 (Saving)
# ==========================================
print(f"正在保存清洗后的长片数据集至: {SAVE_PATH} ...")

df.to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')

print("\n✅ 数据清洗完成！")
print(f"最终数据集包含 {len(df)} 部长篇电影。")
print("你可以使用这个文件进行 Phase 2 (Correlation) 和 Phase 3 (Prediction) 的分析了。")