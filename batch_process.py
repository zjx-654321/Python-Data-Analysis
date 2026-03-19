from src.utils import SolarKnowledgeHandler
import glob

handler = SolarKnowledgeHandler()
all_docs = []

# 一次性读取 data/solar_stations/ 下的所有 xlsx 文件
files = glob.glob("data/solar_stations/*.xlsx")
for file in files:
    print(f"正在处理: {file}")
    all_docs.extend(handler.get_chunks_from_excel(file))

# 构建统一的向量库
vector_store = handler.create_vector_store(all_docs)
print(f"成功将 {len(files)} 个电站数据存入数据库！")