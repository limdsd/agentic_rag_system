# 🧠 NexusRAG — 自省式 Agentic RAG 系统

基于**混合检索 + Self-RAG 反思机制**的企业级知识库问答系统。底层由 LangGraph 状态机编排，FastAPI 提供后端服务，Streamlit 构建交互前端。

## ✨ 核心特性

- 🔍 **混合召回**：BM25 稀疏检索 + ChromaDB 向量稠密检索，通过 RRF（倒数排名融合）打分合并，解决专业术语召回不准的问题。
- 🧩 **父子文档切分（Parent-Document）**：小切片用于检索、大切片送入 LLM 上下文，避免长文问答的信息断层。
- 🔄 **Self-RAG 反思纠错**：基于 LangGraph 构建的查询改写 → 检索 → 反思自检 → 生成的多阶段工作流。
- 🧠 **可插拔 LLM**：默认接入 DeepSeek（兼容 OpenAI 协议），本地 `bge-small-zh-v1.5` 生成中文 Embedding。

## 🏗️ 架构

```
frontend.py (Streamlit)
        │ HTTP
        ▼
app/main.py (FastAPI)
        │
        ├── app/modules/parser/chunker.py        # 父子文档切分
        ├── app/modules/retriever/hybrid_retriever.py  # BM25 + 向量 + RRF
        └── app/modules/agent/workflow.py        # LangGraph Self-RAG 工作流
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env   # Windows: copy .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY（DeepSeek API Key）
```

### 3. 验证环境

```bash
python check_env.py
```

### 4. 启动后端

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. 启动前端（新终端）

```bash
streamlit run frontend.py
```

打开 http://localhost:8501 ，上传 PDF/TXT 文档即可开始问答。

## 📡 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/documents/upload` | 上传并索引文档（PDF/TXT） |
| `POST` | `/api/v1/chat/completions` | 知识库问答（含溯源引用） |
| `GET`  | `/api/v1/health` | 健康检查 |

## 📁 目录结构

```
├── app/
│   ├── main.py                 # FastAPI 入口
│   ├── core/config.py          # 配置（pydantic-settings）
│   ├── modules/
│   │   ├── parser/chunker.py           # 父子切分
│   │   ├── retriever/hybrid_retriever.py  # 混合检索
│   │   └── agent/{state,workflow}.py   # Self-RAG 工作流
│   └── schemas/chat.py         # 请求/响应模型
├── frontend.py                 # Streamlit 前端
├── data/
│   ├── sample.txt              # 示例文档
│   ├── chroma_db/              # 向量库（git 忽略）
│   └── uploads/                # 上传文档（git 忽略）
└── tests/
```

## ⚠️ 安全说明

- `.env` 已加入 `.gitignore`，请勿提交真实 API Key。
- 提交前请确认没有将密钥、向量库数据或用户上传文件纳入版本控制。
