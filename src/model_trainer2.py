import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

# --- 1. 动态路径定位 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

# 设置数据和结果路径
DATA_DIR = os.path.join(BASE_DIR, "data", "solar_stations")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "plots")
MODEL_DIR = os.path.join(BASE_DIR, "results", "models")

# 确保输出目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 站点装机容量配置 (MW)
STATION_CAPACITY = {1: 50.0, 2: 130.0, 3: 30.0, 4: 130.0, 5: 110.0, 6: 35.0, 7: 30.0, 8: 30.0, 9: 30.0, 10: 130.0}


def load_and_preprocess():
    all_data = []
    # 获取目录下所有 Excel 文件
    files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))

    if not files:
        print(f"❌ 找不到数据！请检查目录是否存在: {DATA_DIR}")
        return None

    for file in files:
        filename = os.path.basename(file)
        try:
            # 使用正则匹配文件名中的数字作为站点ID，解决 "Solar station site X" 匹配问题
            match = re.search(r'site\s*(\d+)', filename, re.IGNORECASE)
            sid = int(match.group(1)) if match else 99

            df = pd.read_excel(file)
            cols = list(df.columns)
            # 自动映射前三列：时间、辐照度、功率
            df = df.rename(columns={cols[0]: 'datetime', cols[1]: 'irradiance', cols[2]: 'power'})

            # 数据类型转换与清洗
            df['power'] = pd.to_numeric(df['power'], errors='coerce')
            df['irradiance'] = pd.to_numeric(df['irradiance'], errors='coerce')
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime', 'power'])

            # 注入特征：站点ID、装机容量、小时、月份
            df['station_id'] = sid
            df['capacity'] = STATION_CAPACITY.get(sid, 30.0)
            df['hour'] = df['datetime'].dt.hour
            df['month'] = df['datetime'].dt.month

            all_data.append(df)
            print(f"✅ 成功提取站点 {sid}: {filename} (样本数: {len(df)})")
        except Exception as e:
            print(f"⚠️ 跳过文件 {filename}: {e}")

    if not all_data:
        raise ValueError("未能读取到任何有效数据，请检查 Excel 文件。")

    return pd.concat(all_data, ignore_index=True).fillna(0)


def train_and_plot(df):
    # 选择训练特征
    features = ['irradiance', 'station_id', 'capacity', 'month', 'hour']
    X, y = df[features].astype(float), df['power'].astype(float)

    # 按照任务书 70/15/15 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("🚀 正在训练 LightGBM 模型...")
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, importance_type='gain', random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)])

    # 保存模型
    joblib.dump(model, os.path.join(MODEL_DIR, "lgbm_solar_full_model.pkl"))

    # 预测并应用物理约束 (功率不为负)
    preds = np.clip(model.predict(X_test), 0, None)
    r2 = r2_score(y_test, preds)
    print(f"🏆 训练完成！R2 Score: {r2:.4f}")

    # --- 绘图 (任务书要求的可视化) ---
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Linux 兼容字体

    # 1. 特征重要性图
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, importance_type='gain', max_num_features=10)
    plt.title('Feature Importance (Gain)')
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"))

    # 2. 预测对比图 (前150个点)
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values[:150], label='Actual', alpha=0.8)
    plt.plot(preds[:150], label='Predicted', linestyle='--')
    plt.title(f"Prediction vs Actual (R2: {r2:.4f})")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "pred_vs_true.png"))

    # 3. 误差分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(preds - y_test, kde=True, color='orange')
    plt.title("Error Distribution")
    plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.png"))

    print(f"📊 图表已保存至: {RESULTS_DIR}")


if __name__ == "__main__":
    data = load_and_preprocess()
    if data is not None:
        train_and_plot(data)