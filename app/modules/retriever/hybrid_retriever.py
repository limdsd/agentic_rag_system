import os
import jieba
from typing import List, Dict
import chromadb
from rank_bm25 import BM25Okapi
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import settings

# 自动配置 HuggingFace 镜像源加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class HybridRetriever:
    def __init__(self, collection_name: str = "rag_knowledge_base"):
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

        # 使用本地轻量中文 Embedding 模型
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.bm25 = None
        self.child_chunks_store = []
        self.parent_map_store = {}

    def index_documents(self, parent_map: Dict[str, str], child_chunks: List[Dict]):
        """存储父文档，并将子块写入 Chroma 向量库与构建 BM25 索引"""
        self.parent_map_store.update(parent_map)
        self.child_chunks_store.extend(child_chunks)

        texts = [c["content"] for c in child_chunks]
        ids = [c["child_id"] for c in child_chunks]
        metadatas = [{"parent_id": c["parent_id"], "source": c["source"]} for c in child_chunks]
        embeddings = self.embedding_fn.embed_documents(texts)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        tokenized_corpus = [list(jieba.cut(c["content"])) for c in self.child_chunks_store]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _dense_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """稠密向量检索"""
        query_embed = self.embedding_fn.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embed],
            n_results=top_k
        )
        dense_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                dense_results.append({
                    "child_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "parent_id": results["metadatas"][0][i]["parent_id"],
                    "source": results["metadatas"][0][i]["source"],
                })
        return dense_results

    def _sparse_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 稀疏检索"""
        if not self.bm25:
            return []
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        sparse_results = [self.child_chunks_store[idx] for idx in top_indices if scores[idx] > 0]
        return sparse_results

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict]:
        """执行 RRF 混合多路召回，并映射回父文档完整上下文"""
        dense_res = self._dense_search(query, top_k=top_k * 2)
        sparse_res = self._sparse_search(query, top_k=top_k * 2)

        rrf_scores = {}
        candidate_docs = {}

        for rank, item in enumerate(dense_res):
            c_id = item["child_id"]
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            candidate_docs[c_id] = item

        for rank, item in enumerate(sparse_res):
            c_id = item["child_id"]
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            candidate_docs[c_id] = item

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        final_results = []
        for cid in sorted_ids:
            child_info = candidate_docs[cid]
            parent_id = child_info["parent_id"]
            parent_text = self.parent_map_store.get(parent_id, child_info["content"])
            final_results.append({
                "child_id": cid,
                "parent_id": parent_id,
                "child_content": child_info["content"],
                "full_context": parent_text,
                "source": child_info["source"],
                "score": rrf_scores[cid]
            })

        return final_results