import pandas as pd
import numpy as np
import chromadb
import os
from langchain_community.embeddings import ZhipuAIEmbeddings

# 配置信息
API_KEY = "043707adb8874d7a9484de631d4c1cc0.HJPvNtZh8ROTJHL3"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "solar_stations")
DB_PATH = os.path.join(BASE_DIR, "debug_vector_db")


def run_final_ingestion():
    print("🚀 正在构建【双维度过滤】黄金知识库...")
    zhipu_ef = ZhipuAIEmbeddings(api_key=API_KEY)
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection(name="solar_real_knowledge")
    except:
        pass
    collection = client.create_collection(name="solar_real_knowledge")

    excel_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx')]
    for file_name in excel_files:
        file_path = os.path.join(DATA_DIR, file_name)
        print(f"📡 正在处理: {file_name}...")
        try:
            raw_df = pd.read_excel(file_path)
            raw_df.replace('<NULL>', np.nan, inplace=True)
            df_clean = raw_df.dropna().copy()

            # 物理约束：剔除脏数据
            mask = (df_clean.iloc[:, 1] < 0.01) & (df_clean.iloc[:, 2] > 1.0)
            df_final = df_clean[~mask].copy()

            # 采样 500 条
            df_sample = df_final.sample(min(500, len(df_final)), random_state=42)

            documents, metadatas, ids = [], [], []
            for idx, row in df_sample.iterrows():
                dt = pd.to_datetime(row.iloc[0])
                doc = f"电站{file_name}记录：时间{row.iloc[0]}，辐射度{row.iloc[1]}，功率{row.iloc[2]}MW。"
                documents.append(doc)
                # 存入 hour 和 month 用于双维度过滤
                metadatas.append({
                    "power": float(row.iloc[2]),
                    "hour": int(dt.hour),
                    "month": int(dt.month)
                })
                ids.append(f"{file_name}_{idx}")

            # 分批上传
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                b_docs = documents[i:i + batch_size]
                vectors = zhipu_ef.embed_documents(b_docs)
                collection.add(embeddings=vectors, documents=b_docs,
                               metadatas=metadatas[i:i + batch_size], ids=ids[i:i + batch_size])
            print(f"✅ {file_name} 入库成功")
        except Exception as e:
            print(f"⚠️ {file_name} 失败: {e}")
    print(f"\n✨ 数据库已焕新: {DB_PATH}")


if __name__ == "__main__":
    run_final_ingestion()