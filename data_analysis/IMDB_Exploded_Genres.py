import pandas as pd
import ast
import os

# ==========================================
# 1. 配置参数
# ==========================================
LOAD_PATH = "IMDB TMDB Movie Metadata Big Dataset (1M).csv"
SAVE_PATH = "IMDB_Genres_Lite.csv"  # 更改文件名以示区分

# ==========================================
# 2. 加载与预处理
# ==========================================
print(f"正在加载数据: {LOAD_PATH} ...")

try:
    # 为了加快读取速度，如果文件非常大，可以在读取时只读取需要的列
    # 注意：这里假设源文件里叫 'id', 'title', 'genres_list'
    # 如果源文件列名不同（例如叫 'original_title'），请在 usecols 中修改
    cols_to_use = ['id', 'title', 'genres_list', 'runtime', 'release_year'] 
    df = pd.read_csv(LOAD_PATH, usecols=cols_to_use)
    
except ValueError as e:
    print(f"列名错误: {e}")
    print("请检查你的CSV文件列名是否包含 'id', 'title', 'genres_list'")
    exit()
except FileNotFoundError:
    print("错误：找不到文件。")
    exit()

print(f"原始行数: {len(df)}")

# 清洗：删除任意一列为空的行
df_clean = df.dropna().copy()

# 格式修复：将 "['Action', 'Drama']" 字符串转为列表
print("正在解析题材列表...")
if isinstance(df_clean['genres_list'].iloc[0], str):
    df_clean['genres_list'] = df_clean['genres_list'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

# ==========================================
# 3. 核心步骤：炸开 (Explode)
# ==========================================
print("正在炸开数据 (Exploding)...")

# 炸开列表
df_exploded = df_clean.explode('genres_list')

# ==========================================
# 4. 筛选列与重命名
# ==========================================
df_final = df_exploded[['id', 'title', 'genres_list', 'runtime', 'release_year']].copy()

# 将 'genres_list' 重命名为 'genre'，符合你的要求
df_final.rename(columns={'genres_list': 'genre'}, inplace=True)

# (可选) 再次剔除 genre 为 Unknown 的行
# df_final = df_final[df_final['genre'] != 'Unknown']

print(f"最终保存行数: {len(df_final)}")

# ==========================================
# 5. 保存
# ==========================================
print(f"正在保存至: {SAVE_PATH} ...")

df_final.to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')

print("✅ 保存成功！")