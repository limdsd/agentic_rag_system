import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.chat import ChatRequest, ChatResponse, DocumentCitation
from app.modules.parser.chunker import DocumentChunker
from app.modules.retriever.hybrid_retriever import HybridRetriever
from app.modules.agent.workflow import SelfRAGWorkflow

app = FastAPI(
    title="NexusRAG Agentic System",
    description="基于混合检索与 Self-RAG 反思机制的企业级知识库系统",
    version="1.0.0"
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局单例引擎初始化
retriever = HybridRetriever(collection_name="production_kb")
chunker = DocumentChunker(parent_chunk_size=500, child_chunk_size=150)
agent_engine = SelfRAGWorkflow(retriever=retriever)

UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/api/v1/documents/upload", summary="上传并索引知识库文档")
async def upload_document(file: UploadFile = File(...)):
    """接收 PDF/TXT 文件，执行父子切分并同步至向量库与 BM25"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 解析与索引构建
        parsed_data = chunker.load_and_split(file_path)
        retriever.index_documents(
            parent_map=parsed_data["parent_map"],
            child_chunks=parsed_data["child_chunks"]
        )

        return {
            "status": "success",
            "filename": file.filename,
            "parent_chunks_count": len(parsed_data["parent_map"]),
            "child_chunks_count": len(parsed_data["child_chunks"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件解析失败: {str(e)}")


@app.post("/api/v1/chat/completions", response_model=ChatResponse, summary="知识库智能问答")
async def chat_completion(request: ChatRequest):
    """执行包含反思与自省机制的问答工作流"""
    try:
        result = agent_engine.run(request.query)

        # 组装溯源卡片
        citations = [
            DocumentCitation(
                source=doc.get("source", "未知来源"),
                child_content=doc.get("child_content", ""),
                score=round(doc.get("score", 0.0), 4)
            )
            for doc in result.get("relevant_docs", [])
        ]

        return ChatResponse(
            query=result["query"],
            transformed_query=result.get("transformed_query"),
            answer=result["generation"],
            citations=citations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答生成异常: {str(e)}")


@app.get("/api/v1/health", summary="健康检查")
async def health_check():
    return {"status": "healthy"}