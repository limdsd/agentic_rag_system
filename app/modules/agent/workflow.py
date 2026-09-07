from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from app.core.config import settings
from app.modules.agent.state import GraphState
from app.modules.retriever.hybrid_retriever import HybridRetriever


class SelfRAGWorkflow:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=0
        )
        self.app = self._build_graph()

    # ---------------- 节点定义 ---------------- #

    def retrieve_node(self, state: GraphState) -> Dict:
        """检索节点：根据当前查询检索候选文档"""
        query = state.get("transformed_query") or state["query"]
        docs = self.retriever.retrieve(query=query, top_k=3)
        return {"documents": docs}

    def grade_documents_node(self, state: GraphState) -> Dict:
        """评估节点：评估检索到的文档是否与查询相关"""
        query = state["query"]
        docs = state["documents"]
        relevant_docs = []

        prompt = PromptTemplate(
            template="""你是一个严格的文档相关性评估员。
请判断以下参考文档是否包含回答用户问题所需的线索或相关信息。
只需输出 'yes' 或 'no'。

用户问题: {query}
参考文档: {document}
评估结果:""",
            input_variables=["query", "document"]
        )
        grader_chain = prompt | self.llm

        for doc in docs:
            score = grader_chain.invoke({
                "query": query,
                "document": doc["full_context"]
            }).content.strip().lower()
            if "yes" in score:
                relevant_docs.append(doc)

        return {"relevant_docs": relevant_docs}

    def rewrite_query_node(self, state: GraphState) -> Dict:
        """改写节点：当文档不相关时改写 Query 以提高召回率"""
        query = state["query"]
        retry_count = state.get("retry_count", 0) + 1

        prompt = PromptTemplate(
            template="""你是一个搜索查询优化专家。初始查询未能检索到有效文档。
请分析用户意图，重写一个更具体、包含关键术语的搜索 Query。
只输出重写后的 Query 文本，不要有任何多余解释。

原始问题: {query}
优化后Query:""",
            input_variables=["query"]
        )
        rewriter_chain = prompt | self.llm
        response = rewriter_chain.invoke({"query": query})

        return {
            "transformed_query": response.content.strip(),
            "retry_count": retry_count
        }

    def generate_node(self, state: GraphState) -> Dict:
        """生成节点：基于相关文档生成带有来源引用的回答"""
        query = state["query"]
        docs = state.get("relevant_docs") or state["documents"]

        context_text = "\n\n".join([
            f"[来源 {i + 1}]: {d['full_context']}" for i, d in enumerate(docs)
        ])

        prompt = PromptTemplate(
            template="""你是一个企业级 AI 知识库助手。请严格根据以下提供的参考文档回答问题。
要求：
1. 必须完全基于参考文档内容，严禁胡编乱造（杜绝幻觉）。
2. 在陈述关键事实时，在句末标注对应的引用来源，例如 [来源 1]。
3. 若参考文档完全未提及答案，直接回答“根据当前知识库内容，无法回答该问题”。

参考文档：
{context}

用户问题: {query}
回答：""",
            input_variables=["context", "query"]
        )
        generator_chain = prompt | self.llm
        generation = generator_chain.invoke({"context": context_text, "query": query})
        return {"generation": generation.content}

    # ---------------- 条件边判定 ---------------- #

    def _decide_to_generate(self, state: GraphState) -> str:
        """决策逻辑：如果存在相关文档则进入生成，否则若重试未超限则改写 Query"""
        relevant_docs = state.get("relevant_docs", [])
        retry_count = state.get("retry_count", 0)

        if len(relevant_docs) > 0:
            return "generate"
        if retry_count < 1:  # 最多改写重试 1 次
            return "rewrite_query"
        return "generate"

    # ---------------- 构图 ---------------- #

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("grade_documents", self.grade_documents_node)
        workflow.add_node("rewrite_query", self.rewrite_query_node)
        workflow.add_node("generate", self.generate_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query"
            }
        )
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def run(self, query: str) -> Dict:
        """对外暴露的执行入口"""
        initial_state = {
            "query": query,
            "transformed_query": "",
            "documents": [],
            "relevant_docs": [],
            "generation": "",
            "retry_count": 0
        }
        return self.app.invoke(initial_state)