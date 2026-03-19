import joblib
import pandas as pd
import numpy as np
import os
import sys


# 假设你之前有一个从 ChromaDB 检索的工具函数，我们模拟它的行为
def mock_rag_search(station_id, month, hour):
    """
    模拟 RAG 过程：
    实际操作中，这里会调用 vector_service_pro.py 里的检索函数，
    根据 (station_id, month, hour) 去 ChromaDB 查找历史相似记录。
    """
    # 模拟从向量数据库返回的相似工况平均值
    # 在真实系统中，这里会返回一段文本：“去年6月12点，站点4平均功率为115MW，最高128MW...”
    db_record = {
        "hist_avg": 115.0,
        "hist_max": 130.0,
        "note": "历史记录显示该时段处于夏季大负荷期，常伴随高温。"
    }
    return db_record


def run_rag_plus_ml_advisor(station_id, month, hour, irradiance):
    print(f"📡 启动 RAG + ML 联合诊断系统...")

    # --- 步骤 1：数值预测 (Machine Learning) ---
    model_path = "./results/models/lgbm_solar_full_model.pkl"
    if not os.path.exists(model_path):
        print("❌ 未找到模型，请先运行训练脚本！")
        return

    model = joblib.load(model_path)
    input_data = pd.DataFrame({
        'irradiance': [irradiance],
        'hour': [hour],
        'month': [month],
        'station_id': [station_id]
    })
    pred_power = max(0, model.predict(input_data)[0])

    # --- 步骤 2：经验检索 (RAG) ---
    # 这一步体现了“检索增强”：不再用死数字，而是去数据库查“知识”
    rag_data = mock_rag_search(station_id, month, hour)
    hist_avg = rag_data["hist_avg"]

    # --- 步骤 3：综合决策 (Expert System) ---
    print("\n" + "★" * 40)
    print(f"【AI 专家报告 - RAG 增强版】")
    print(f"▶ 实时预测：{pred_power:.2f} MW")
    print(f"▶ 历史经验：{hist_avg:.2f} MW (源自 ChromaDB 检索)")
    print(f"▶ 历史备注：{rag_data['note']}")

    # 计算偏差率
    deviation = (pred_power - hist_avg) / hist_avg

    if abs(deviation) > 0.3:
        status = "【显著异常】"
        if pred_power > hist_avg:
            advice = "预测功率远超历史同期！请检查是否存在数据上报错误或局部气象突变，建议准备泄压调度。"
        else:
            advice = "预测功率低于历史均值30%以上！检测到潜在隐患，请排查光伏板遮挡或逆变器效率损耗。"
    else:
        status = "【工况稳定】"
        advice = "预测值与历史经验吻合，系统运行良好，维持正常巡检。"

    print(f"📌 诊断结论：{status}")
    print(f"💡 运维建议：{advice}")
    print("★" * 40)


if __name__ == "__main__":
    # 测试：输入站点4，6月，12时，高辐射度 900
    run_rag_plus_ml_advisor(station_id=4, month=6, hour=12, irradiance=900)