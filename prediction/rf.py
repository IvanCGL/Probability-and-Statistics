import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb  # 需要安装: pip install xgboost

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 1. 数据加载与基础清洗
# ==========================================
# 假设我们要用包含所有列的大表，或者你已经merge好的表
# 这里请替换为你包含上述所有字段的文件路径
LOAD_PATH = "./dataset/IMDB_Feature_Films_Cleaned.csv" 
# 注意：如果你的 cleaned 文件里没有 Director/Star 等列，你需要重新读取原始大文件并做一次清洗
# 为了演示，假设 df 已经包含了你列出的所有 columns
df = pd.read_csv(LOAD_PATH) 

# 步骤 A: 强制转换为数值类型
# 'coerce' 会将无法转换的字符串（如 "Not Rated", "N/A", 空字符串）统统变成 NaN
df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')

# 步骤 B: 剔除目标变量为空的行
# 这是解决报错的核心！
df = df.dropna(subset=['IMDB_Rating'])

# ==========================================
# 2. 高级特征工程 (The "Secret Sauce")
# ==========================================
print("正在构建高级特征...")

# --- A. 时间特征 ---
# 从 release_date 提取月份 (捕捉季节性)
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month

# --- B. 商业特征 ---
# 是否有主页 (1=有, 0=无)
df['has_homepage'] = df['homepage'].notna().astype(int)
# 投资回报率 (处理分母为0的情况)
df['roi'] = df.apply(lambda x: (x['revenue'] - x['budget']) / x['budget'] if x['budget'] > 1000 else 0, axis=1)

# --- C. "名气"特征 (Target Encoding) ---
# 这是一个非常强大的技巧：计算导演/演员的历史平均评分
# 注意：严谨的做法是在 Train Set 上计算映射到 Test Set，防止数据泄露。
# 这里为了代码简洁，演示全局计算（在做学术分析时通常可接受，但在严格预测比赛中需分开）

def calculate_reputation(df, col_name, target_col='vote_average'):
    # 计算每个人的平均分
    reputation = df.groupby(col_name)[target_col].mean()
    # 映射回原表，如果是一个新导演(没在库里)，就填全局平均分
    global_mean = df[target_col].mean()
    return df[col_name].map(reputation).fillna(global_mean)

# 对关键人物进行编码
# 假设你的列名是 'Director', 'Star1', 'Writer'
if 'Director' in df.columns:
    df['Director_Score'] = calculate_reputation(df, 'Director')
if 'Star1' in df.columns:
    df['Star1_Score'] = calculate_reputation(df, 'Star1')
if 'Writer' in df.columns:
    df['Writer_Score'] = calculate_reputation(df, 'Writer')

# --- D. 题材特征 (One-Hot) ---
# 再次处理 Genre
if isinstance(df['genres_list'].iloc[0], str):
    df['genres_list'] = df['genres_list'].apply(ast.literal_eval)

mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres_list'])
genres_df = pd.DataFrame(genres_encoded, columns=[f"Genre_{g}" for g in mlb.classes_], index=df.index)

# ==========================================
# 3. 准备训练数据
# ==========================================
# 挑选我们要扔给模型的所有特征
# 包含了：数值基础 + 题材 + 名气特征 + 情感 + 时间
feature_cols = [
    'runtime', 'budget', 'revenue', 'release_year', 'release_month', # 基础
    'roi', 'has_homepage', 'overview_sentiment',                     # 商业与情感
    'Director_Score', 'Star1_Score', 'Writer_Score'                  # 名气 (Key!)
]

# 确保列存在
selected_features = [c for c in feature_cols if c in df.columns]
X = pd.concat([df[selected_features], genres_df], axis=1)
y = df['IMDB_Rating'] # 或者 IMDB_Rating，看你想预测哪个

# (可选) 步骤 C: 剔除评分异常的行（例如 0分或超过10分，视数据情况而定）
# 有些数据集会用 -1 代表缺失
df = df[(df['IMDB_Rating'] > 0) & (df['IMDB_Rating'] <= 10)]

# 填充空值 (XGBoost其实可以自动处理空值，但填上更保险)
X = X.fillna(X.median())

# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. 模型升级: XGBoost Regressor
# ==========================================
print(f"正在训练 XGBoost (特征数量: {X.shape[1]})...")

# XGBoost 参数配置 (可以微调)
model = xgb.XGBRegressor(
    n_estimators=500,     # 树的数量
    learning_rate=0.05,   # 学习率
    max_depth=6,          # 树的深度 (防过拟合)
    subsample=0.8,        # 每次只用80%的数据
    colsample_bytree=0.8, # 每次只用80%的特征
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# 5. 评估与可视化
# ==========================================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n🚀 模型升级结果:")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# 特征重要性绘图
plt.figure(figsize=(12, 10))
# XGBoost 提供了非常方便的 plot_importance
# 但为了美观，我们手动画 Top 20
importances = model.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=False).head(20)

sns.barplot(x='Importance', y='Feature', data=feat_df, palette='magma')
plt.title('What REALLY drives Movie Ratings? (XGBoost Feature Importance)', fontsize=16)
plt.tight_layout()
plt.show()

# 预测对比图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='#8e44ad', s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('XGBoost Prediction Accuracy', fontsize=16)
plt.legend()
plt.show()