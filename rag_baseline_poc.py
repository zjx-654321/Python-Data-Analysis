import chromadb
from langchain_community.embeddings import ZhipuAIEmbeddings

# 1. 粘贴你刚才发给我的这串 Key
api_key = "043707adb8874d7a9484de631d4c1cc0.HJPvNtZh8ROTJHL3"

# 2. 初始化
zhipu_ef = ZhipuAIEmbeddings(api_key=api_key)
client = chromadb.PersistentClient(path="./test_db")
collection = client.get_or_create_collection(name="solar_knowledge")

# 3. 执行测试
print("正在通过智谱云端进行向量化测试...")
test_text = "站点1在夏季中午12点容易达到功率峰值。"
test_vector = zhipu_ef.embed_query(test_text)

collection.add(
    embeddings=[test_vector],
    documents=[test_text],
    ids=["test_id_001"]
)

# 4. 检索测试
query_vector = zhipu_ef.embed_query("中午发电量高吗？")
results = collection.query(query_embeddings=[query_vector], n_results=1)

print("-" * 30)
if results['documents']:
    print("🎉 恭喜“xuan大工”！RAG 检索成功：", results['documents'][0])
else:
    print("❌ 检索失败，请检查代码逻辑。")
print("-" * 30)