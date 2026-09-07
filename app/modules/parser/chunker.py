import uuid
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class DocumentChunker:
    def __init__(self, parent_chunk_size: int = 1000, child_chunk_size: int = 300, chunk_overlap: int = 50):
        # 父文档分块器（保证上下文完整性）
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        # 子文档分块器（保证向量检索的精准度）
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def load_and_split(self, file_path: str) -> Dict[str, any]:
        """加载文件并生成关联的父子块结构"""
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        raw_docs = loader.load()

        # 1. 切分为父块
        parent_docs = self.parent_splitter.split_documents(raw_docs)

        child_docs = []
        parent_doc_map = {}

        # 2. 为每个父块生成唯一 ID，并进一步切出子块
        for p_doc in parent_docs:
            parent_id = str(uuid.uuid4())
            p_doc.metadata["parent_id"] = parent_id
            parent_doc_map[parent_id] = p_doc.page_content

            # 对父块内容做二次切分作为子块
            sub_chunks = self.child_splitter.split_text(p_doc.page_content)
            for sub_text in sub_chunks:
                child_docs.append({
                    "child_id": str(uuid.uuid4()),
                    "parent_id": parent_id,
                    "content": sub_text,
                    "source": p_doc.metadata.get("source", file_path)
                })

        return {
            "parent_map": parent_doc_map,
            "child_chunks": child_docs
        }