from src.utils import SolarKnowledgeHandler
import glob

# 1. 实例化核心处理类
handler = SolarKnowledgeHandler()

# 2. 批量读取所有 Excel 文件
file_pattern = "data/solar_stations/*.xlsx"
files = glob.glob(file_pattern)

if not files:
    print(f"警告: 在 {file_pattern} 下没找到任何 .xlsx 文件，请检查路径！")
else:
    print(f"找到 {len(files)} 个电站文件，开始处理...")

    # 3. 整合所有文档
    all_docs = []
    for file in files:
        docs = handler.get_chunks_from_excel(file)
        all_docs.extend(docs)

    # 4. 构建向量库
    vector_store = handler.create_vector_store(all_docs)
    print(f"成功构建向量数据库，包含 {len(all_docs)} 个文档片段！")

    # 5. 执行一次模拟检索测试
    query = "电站发电量"  # 换成你 Excel 里实际存在的关键词
    results = vector_store.similarity_search(query, k=1)
    print("\n--- 检索测试结果 ---")
    print(results[0].page_content[:200])  # 打印前200个字符看看效果
