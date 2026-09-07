from typing import List, Dict, TypedDict

class GraphState(TypedDict):
    query: str                      # 用户原始查询
    transformed_query: str          # 改写后的查询
    documents: List[Dict]           # 检索出的文档列表
    relevant_docs: List[Dict]       # 经过评估判定为相关的文档
    generation: str                 # LLM 生成的最终回答
    retry_count: int                # 查询改写重试次数计数器