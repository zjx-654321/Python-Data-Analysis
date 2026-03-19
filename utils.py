import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings


class SolarKnowledgeHandler:
    def __init__(self):
        self.embeddings = FakeEmbeddings(size=768)

    def get_chunks_from_excel(self, file_path):
        """专门读取 Excel 文件的逻辑"""
        if not os.path.exists(file_path):
            return []

        # 使用 pandas 读取 Excel
        df = pd.read_excel(file_path).head(100)
        # 将每一行转为字符串作为文档内容
        content = df.to_string()
        doc = Document(page_content=content, metadata={"source": file_path})

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_documents([doc])

    def create_vector_store(self, docs):
        return FAISS.from_documents(docs, self.embeddings)