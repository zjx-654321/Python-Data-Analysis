import pandas as pd
import numpy as np
import os
import sys
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split

# 1. 环境与路径配置
# 强制让 matplotlib 不依赖 GUI 界面（防止虚拟机报错）
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 确保能引用到你的 src 模块
sys.path.append(os.path.join(os.getcwd(), "src"))
try:
    from model_trainer import evaluate_model, error_analysis
except ImportError:
    print("⚠️ 警告：未找到 src/model_trainer.py 中的评估函数，将仅执行训练。")

DATA_DIR = "./data/solar_stations"
RESULTS_DIR = "./results"
MODEL_SAVE_PATH = "./results/models"

# 创建必要文件夹
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def run_full_training():
    print("🚜 正在扫描 8 个站点，准备处理 54 万行数据...")
    all_dfs = []

    # 检查路径是否存在
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误：路径 {DATA_DIR} 不存在！")
        return

    excel_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx')]

    # 2. 循环读取并清洗数据
    for file in excel_files:
        path = os.path.join(DATA_DIR, file)
        print(f"📖 正在读取: {file}...")

        # 读取前 3 列：时间、辐射度、功率
        df = pd.read_excel(path, usecols=[0, 1, 2])
        df.columns = ['time', 'irradiance', 'power']

        # 提取站点 ID 作为特征
        try:
            s_id = int(file.split('site ')[1].split(' ')[0])
        except:
            s_id = 0
        df['station_id'] = s_id
        all_dfs.append(df)

    # 合并全量数据
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ 数据加载完成，当前行数: {len(full_df)}")

    # 3. 【核心修复】处理 <NULL> 等异常值
    print("🧹 正在进行数据清洗（处理非数值字符串与空值）...")
    # 将所有的 "<NULL>" 替换为真正的空值
    full_df = full_df.replace("<NULL>", np.nan)
    # 将辐射度和功率中的空值/异常值填充为 0
    full_df[['irradiance', 'power']] = full_df[['irradiance', 'power']].apply(pd.to_numeric, errors='coerce').fillna(0)

    # 4. 特征工程
    print("🛠️ 正在提取时间特征...")
    full_df['time'] = pd.to_datetime(full_df['time'])
    full_df['hour'] = full_df['time'].dt.hour
    full_df['month'] = full_df['time'].dt.month

    features = ['irradiance', 'hour', 'month', 'station_id']
    X = full_df[features]
    y = full_df['power']

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. 训练 LightGBM
    print("🤖 正在启动 LightGBM 进行 54 万行量级建模...")
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # 6. 评估与保存
    print("\n📊 正在生成专业评估结果...")
    y_pred = model.predict(X_test)

    # 如果 model_trainer 可用，则调用专业分析
    if 'evaluate_model' in globals():
        evaluate_model(y_test, y_pred, [np.mean(y_train)] * len(y_test),
                       ["LightGBM_Full", "Baseline"], RESULTS_DIR)
        analysis_report = error_analysis(X_test, y_test, y_pred, model, RESULTS_DIR)
        print(f"📝 误差分析总结：\n{analysis_report}")

    # 保存模型
    model_file = os.path.join(MODEL_SAVE_PATH, "lgbm_solar_full_model.pkl")
    joblib.dump(model, model_file)
    print(f"\n✨ 恭喜！模型已封存至: {model_file}")
    print(f"📈 请查看 {RESULTS_DIR} 目录下的图表和报告。")


if __name__ == "__main__":
    run_full_training()