import pandas as pd
import numpy as np
import os
import re
import glob
from sklearn.model_selection import train_test_split


def load_and_clean_solar_data(data_path):
    """加载并清洗8个站点的光伏数据"""
    # 读取所有XLSX文件
    xlsx_files = glob.glob(f"{data_path}/*.xlsx")
    if not xlsx_files:
        raise FileNotFoundError(f"❌ 未在 {data_path} 找到XLSX文件！")

    df_list = []
    for file in xlsx_files:
        try:
            df = pd.read_excel(file, engine="openpyxl")
            # 从文件名提取站点ID
            site_match = re.search(r"site\s*(\d+)", file.lower())
            df["site_id"] = int(site_match.group(1)) if site_match else 0
            df_list.append(df)
            print(f"✅ 读取：{os.path.basename(file)}，行数：{len(df)}")
        except Exception as e:
            print(f"⚠️ 跳过：{os.path.basename(file)}，原因：{str(e)[:40]}")

    if not df_list:
        raise ValueError("❌ 所有文件读取失败！")
    df = pd.concat(df_list, ignore_index=True)

    # 列名映射（适配你的特殊列名）
    col_map = {
        "Time(year-month-day h:m:s)": "time",
        "Power (MW)": "load",
        "Total solar irradiance (W/m2)": "total_irradiance",
        "Air temperature  (°C) ": "temperature",
        "Atmosphere (hpa)": "atmosphere",
        "Relative humidity (%)": "humidity"
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # 检查关键列
    required_cols = ["time", "load", "site_id"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ 缺少关键列：{missing_cols}")

    # 数据清洗
    df = df.dropna(subset=["time", "load"])
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["time"])
    df["load"] = pd.to_numeric(df["load"], errors="coerce")
    df = df[(df["load"] >= 0) & (df["load"] <= df["load"].quantile(0.99))]

    # 提取时间特征
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.weekday

    print(
        f"✅ 清洗完成：总行数 {len(df)}，时间范围 {df['time'].min()} ~ {df['time'].max()}，站点 {sorted(df['site_id'].unique())}")
    return df


def split_data(df, site_id, test_size=0.15, val_size=0.15, random_state=42):
    """按时间拆分数据集（避免数据泄露）"""
    # 筛选指定站点
    df_site = df[df["site_id"] == site_id].copy()
    if len(df_site) < 1000:
        raise ValueError(f"❌ 站点{site_id}数据不足（仅{len(df_site)}行）")

    # 按时间排序
    df_site = df_site.sort_values("time").reset_index(drop=True)

    # 特征/标签分离
    features = ["hour", "weekday", "total_irradiance", "temperature", "atmosphere", "humidity"]
    features = [f for f in features if f in df_site.columns]
    X = df_site[features]
    y = df_site["load"]

    # 拆分训练+验证集 / 测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    # 拆分训练集 / 验证集
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, shuffle=False
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)