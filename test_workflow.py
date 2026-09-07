from app.modules.parser.chunker import DocumentChunker
from app.modules.retriever.hybrid_retriever import HybridRetriever
from app.modules.agent.workflow import SelfRAGWorkflow

def main():
    print("1. 正在初始化知识库索引...")
    chunker = DocumentChunker(parent_chunk_size=200, child_chunk_size=60)
    data = chunker.load_and_split("data/sample.txt")

    retriever = HybridRetriever(collection_name="test_workflow_db")
    retriever.index_documents(data["parent_map"], data["child_chunks"])

    print("\n2. 初始化 Self-RAG 工作流引擎...")
    rag_engine = SelfRAGWorkflow(retriever=retriever)

    # 测试 1：文档内可查到的问题
    query_1 = "NexusRAG 是在什么时候由谁研发的？"
    print(f"\n================ 测试 1: [{query_1}] ================")
    res_1 = rag_engine.run(query_1)
    print(f"\n【最终回答】:\n{res_1['generation']}")
    print(f"【命中有效文档数】: {len(res_1['relevant_docs'])}")

    # 测试 2：文档内不存在的问题（验证反思与抗幻觉能力）
    query_2 = "NexusRAG 支持哪些数据库的集群部署？"
    print(f"\n================ 测试 2: [{query_2}] ================")
    res_2 = rag_engine.run(query_2)
    print(f"\n【最终回答】:\n{res_2['generation']}")
    print(f"【改写后 Query】: {res_2['transformed_query']}")
    print(f"【命中有效文档数】: {len(res_2['relevant_docs'])}")

if __name__ == "__main__":
    main()