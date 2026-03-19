import joblib
import pandas as pd
import numpy as np
import os
import chromadb
from langchain_community.embeddings import ZhipuAIEmbeddings

API_KEY = "043707adb8874d7a9484de631d4c1cc0.HJPvNtZh8ROTJHL3"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "results", "models", "lgbm_solar_full_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "debug_vector_db")


def get_rag_context(station_id, month, hour):
    try:
        zhipu_ef = ZhipuAIEmbeddings(api_key=API_KEY)
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name="solar_real_knowledge")

        # 【终极进化】：使用 $and 同时锁定月份和小时
        results = collection.query(
            query_embeddings=[zhipu_ef.embed_query(f"站点{station_id}在{month}月{hour}点的表现")],
            n_results=3,
            where={
                "$and": [
                    {"hour": {"$eq": int(hour)}},
                    {"month": {"$eq": int(month)}}
                ]
            }
        )
        if not results['metadatas'][0]: return None, f"未找到{month}月{hour}时的历史记录"
        avg_hist = np.mean([m['power'] for m in results['metadatas'][0]])
        return avg_hist, results['documents'][0][0]
    except Exception as e:
        return None, f"检索异常: {e}"


def run_expert_system(station_id, month, hour, irradiance):
    print(f"\n🚀 启动【双维度对齐版】决策引擎...")
    model = joblib.load(MODEL_PATH)

    # ML 预测
    input_df = pd.DataFrame({'irradiance': [irradiance], 'hour': [hour], 'month': [month], 'station_id': [station_id]})
    ml_pred = max(0, model.predict(input_df)[0])

    # RAG 检索（带双维度过滤）
    hist_val, hist_text = get_rag_context(station_id, month, hour)

    print("\n" + "█" * 60)
    print(f"┃ 任务：站点 {station_id} | {month}月 | {hour}:00")
    print(f"┃ ML预测: {ml_pred:.2f} MW")

    if hist_val is not None:
        print(f"┃ RAG经验: {hist_val:.2f} MW (同月同点对标)")
        print(f"┃ 历史详情: {hist_text}")
        # 偏差阈值设为 0.5
        diff = abs(ml_pred - hist_val) / (hist_val + 1e-6)
        status = "✅ 运行平稳" if diff < 0.5 else "⚠️ 异常预警"
    else:
        status = "⚪ 缺少数据"

    print(f"┃ 最终结论: {status}")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    # 现在你可以放心地查 6 月份了！
    run_expert_system(station_id=4, month=6, hour=12, irradiance=850)