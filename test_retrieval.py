from app.modules.parser.chunker import DocumentChunker
from app.modules.retriever.hybrid_retriever import HybridRetriever


def main():
    print("1. 正在解析文档并切分子块/父块...")
    chunker = DocumentChunker(parent_chunk_size=200, child_chunk_size=60)
    data = chunker.load_and_split("data/sample.txt")

    print(f"   生成父块数量: {len(data['parent_map'])}")
    print(f"   生成子块数量: {len(data['child_chunks'])}")

    print("\n2. 正在构建混合检索索引（向量入库 + BM25）...")
    retriever = HybridRetriever(collection_name="test_knowledge_base")
    retriever.index_documents(data["parent_map"], data["child_chunks"])

    print("\n3. 执行混合检索查询测试...")
    test_query = "NexusRAG 采用了什么算法解决专有名词召回问题？"
    results = retriever.retrieve(test_query, top_k=2)

    print(f"\n--- 查询结果 (Query: {test_query}) ---")
    for i, res in enumerate(results, 1):
        print(f"\n[候选 {i}] (RRF得分: {res['score']:.4f})")
        print(f"命中子块: {res['child_content']}")
        print(f"关联父文档完整上下文: {res['full_context']}")


if __name__ == "__main__":
    main()