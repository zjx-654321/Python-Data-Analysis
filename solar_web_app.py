import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- 1. 配置与路径管理 ---
# 自动定位模型文件：确保指向项目根目录下的 results/models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "results", "models", "lgbm_solar_full_model.pkl")

# 站点装机容量配置
STATION_CAPACITY = {1: 50.0, 2: 130.0, 3: 30.0, 4: 130.0, 5: 110.0, 6: 35.0, 7: 30.0, 8: 30.0, 9: 30.0, 10: 130.0}

st.set_page_config(page_title="光伏功率预测系统", layout="wide")
st.title("☀️ 光伏功率 RAG+ML 联合决策系统")
st.markdown("**当前运行环境：本地 Web 推理引擎**")

# --- 2. 输入端 (侧边栏) ---
with st.sidebar:
    st.header("⚙️ 实时工况模拟")
    sid = st.selectbox("电站编号", list(STATION_CAPACITY.keys()))
    m = st.slider("月份", 1, 12, 6)
    h = st.slider("小时", 0, 23, 12)
    irr = st.number_input("核心辐射度 (W/m2)", value=500.0)

    # 额外参数（仅展示，不参与计算，因为模型训练时没用这些）
    st.info("💡 以下参数模型暂未作为核心特征：")
    temp = st.number_input("Air temperature (°C)", value=25.0)
    press = st.number_input("Atmosphere (hpa)", value=1013.0)
    hum = st.number_input("Relative humidity (%)", value=50.0)

# --- 3. 预测逻辑 (特征对齐修复) ---
if st.button("运行联合诊断"):
    if os.path.exists(MODEL_PATH):
        try:
            # 【关键修复】：必须与 model_trainer2.py 中的 features 顺序完全一致
            # 训练特征为：['irradiance', 'station_id', 'capacity', 'month', 'hour']
            features_list = ['irradiance', 'station_id', 'capacity', 'month', 'hour']

            # 构造输入 DataFrame
            input_data = pd.DataFrame([[
                float(irr),
                int(sid),
                float(STATION_CAPACITY[sid]),
                int(m),
                int(h)
            ]], columns=features_list)

            # 加载并预测
            model = joblib.load(MODEL_PATH)
            raw_pred = model.predict(input_data)[0]

            # 物理约束：不为负，不超过装机容量
            cap = STATION_CAPACITY[sid]
            prediction = np.clip(raw_pred, 0, cap)
            ratio = prediction / cap

            # 4. 结果展示
            st.divider()
            col1, col2 = st.columns(2)
            col1.metric("🤖 模型预测功率", f"{prediction:.2f} MW")

            with col2:
                if ratio > 0.8:
                    st.warning("⚡ 高出力状态：请监控逆变器温升")
                elif ratio < 0.1 and 7 < h < 18:
                    st.error("📉 异常低出力：建议检查组件遮挡或故障")
                else:
                    st.success("✅ 系统运行平稳")

            st.info(f"📊 决策支持：当前负荷率 {ratio * 100:.1f}% | 理论最大出力 {cap} MW")

        except Exception as e:
            st.error(f"预测出错：{str(e)}")
    else:
        st.error(f"找不到模型文件：{MODEL_PATH}，请先运行 model_trainer2.py 进行训练！")

# 5. 模型状态看板
st.sidebar.markdown("---")
st.sidebar.write(f"模型路径：`{MODEL_PATH}`")
if os.path.exists(MODEL_PATH):
    st.sidebar.success("模型状态：已就绪")
else:
    st.sidebar.error("模型状态：未找到")