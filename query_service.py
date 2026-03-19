import chromadb
from langchain_community.embeddings import ZhipuAIEmbeddings
import os

# 1. 配置（保持与入库脚本一致）
API_KEY = "043707adb8874d7a9484de631d4c1cc0.HJPvNtZh8ROTJHL3"
# 指向你刚才生成的那个“真大脑”文件夹
DB_PATH = "./solar_vector_db_pro"


def search_solar_data(question):
    print(f"🔍 正在检索与 '{question}' 最相关的电站记录...")

    # 初始化智谱嵌入模型
    zhipu_ef = ZhipuAIEmbeddings(api_key=API_KEY)

    # 连接本地数据库
    if not os.path.exists(DB_PATH):
        print(f"❌ 错误：找不到数据库文件夹 {DB_PATH}，请先运行入库脚本！")
        return

    client = chromadb.PersistentClient(path=DB_PATH)

    # 获取之前创建的集合
    try:
        collection = client.get_collection(name="solar_real_knowledge")
    except Exception:
        print("❌ 错误：数据库中没有名为 'solar_real_knowledge' 的集合")
        return

    # 2. 【核心步骤】将问题转化为向量
    query_vec = zhipu_ef.embed_query(question)

    # 3. 在 400 条记录中寻找最相似的前 3 条
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=3
    )

    # 4. 打印结果
    print("\n✅ 找到以下最匹配的历史数据：")
    print("=" * 60)
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        score = results['distances'][0][i]  # 距离越小，匹配度越高
        print(f"【匹配项 {i + 1}】(相似度得分: {score:.4f})")
        print(f" 内容: {doc}")
        print("-" * 60)


if __name__ == "__main__":
    # 既然你存了 8 个站点的样板，你可以试着问：
    # "告诉我关于站点4的信息" 或 "辐射度很高的时候功率是多少？"
    user_input = "站点1在中午的发电情况"
    search_solar_data(user_input)