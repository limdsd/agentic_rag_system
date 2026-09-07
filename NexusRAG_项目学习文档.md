# NexusRAG 自省式企业知识库系统 —— 小白学习文档

> 目标：让你能「看懂每一行代码」+「讲清楚这个项目」+「扛住 HR 和技术面提问」。
> 适合：对 AI 应用 / 大模型 / RAG 刚入门的同学。

---

## 目录

1. [这个项目到底在做什么（大白话）](#1-这个项目到底在做什么大白话)
2. [名词扫盲：先把术语搞懂](#2-名词扫盲先把术语搞懂)
3. [技术栈清单](#3-技术栈清单)
4. [整体架构图](#4-整体架构图)
5. [代码逐文件详解](#5-代码逐文件详解)
6. [一条提问的完整旅程（数据流）](#6-一条提问的完整旅程数据流)
7. [五大核心机制深度讲解](#7-五大核心机制深度讲解)
8. [如何跑起来这个项目](#8-如何跑起来这个项目)
9. [HR 会问的问题 + 参考答案](#9-hr-会问的问题--参考答案)
10. [简历写法](#10-简历写法)
11. 
12. 
13. 
14. 
15. 
16. 
17. 
18. 
19. 
20. 
21. 
22. 
23. 
24. 
25. 
26. 
27. 
28. 
29. 
30. 
31. 
32. 
33. 
34. 
35. 
36. 
37. [进一步学习路线](#11-进一步学习路线)

---

## 1. 这个项目到底在做什么（大白话）

**一句话**：做一个「企业知识库问答机器人」——你把公司的 PDF/文档丢进去，它就能基于这些文档准确回答员工的问题，并且**不会瞎编**。

**为什么需要它？** 直接问大模型（比如 ChatGPT）有两个致命问题：

1. **它不知道你公司的内部资料**——大模型的记忆截止在训练那天，你家内部的文档它根本没学过。
2. **它会「一本正经地胡说八道」**（术语叫「幻觉 / Hallucination」）——问它一个它不知道的事，它可能编一个看起来很真的答案。

这个项目就是来解决这两个问题的：**把文档变成大模型能查的「资料库」，回答时逼它只能照着资料答**。

---

## 2. 名词扫盲：先把术语搞懂

> 这些词后面会反复出现，先在这里一次看懂。

| 名词 | 大白话解释 |
|------|-----------|
| **LLM / 大模型** | 大语言模型，就是 ChatGPT 这类，本文用 DeepSeek |
| **RAG** | Retrieval-Augmented Generation（检索增强生成）。三步：**检索**相关资料 → 把资料塞给大模型 → **生成**答案。让大模型「先查资料再回答」 |
| **Agentic RAG** | 在 RAG 基础上加「大脑」：不只是查一次，还会**判断、反思、重试**，像智能体一样多步行动 |
| **Embedding / 向量** | 把一段文字变成一串数字（向量），语义相近的文字，向量也相近 |
| **向量数据库** | 专门存向量、并能快速找「最相近向量」的数据库，本文用 ChromaDB |
| **BM25** | 一种「关键词匹配」算法，字面词对上了就加分。擅长精确词、专有名词 |
| **RRF** | Reciprocal Rank Fusion（倒数排名融合），把多个检索结果按排名合并成一份 |
| **Chunk / 切片** | 把长文档切成一小段一小段，方便检索 |
| **父子文档** | 大切片（父）套小切片（子），小片用来精准检索，大片用来给大模型完整上下文 |
| **LangGraph** | 用「图/状态机」的方式编排大模型流程，节点之间可以判断、循环、重试 |
| **Self-RAG** | 「自我反思的 RAG」：检索后自己评估「资料有没有用」，没用就改写问题重来 |
| **幻觉** | 大模型编造不存在的事实 |
| **FastAPI** | Python 的 Web 后端框架，用来写接口 |
| **Streamlit** | Python 的快速前端框架，写界面 |

---

## 3. 技术栈清单

| 类别 | 用的技术 | 作用 |
|------|---------|------|
| 大模型 | DeepSeek（`deepseek-chat`） | 生成答案、评估、改写问题 |
| 编排框架 | LangChain + LangGraph | 组织大模型调用流程（状态机） |
| 后端 | FastAPI + Uvicorn | 提供 HTTP 接口 |
| 前端 | Streamlit | 聊天界面 + 文档上传 |
| 向量库 | ChromaDB | 存向量、做相似度检索 |
| 中文向量模型 | `BAAI/bge-small-zh-v1.5`（sentence-transformers） | 把中文变成向量 |
| 关键词检索 | rank-bm25 + jieba（中文分词） | 精确词召回 |
| 文档解析 | pypdf + langchain-text-splitters | 读 PDF/TXT、切块 |
| 配置 | pydantic-settings + python-dotenv | 读 `.env` 配置 |

依赖清单见 `requirements.txt`。

---

## 4. 整体架构图

```
┌──────────────┐  HTTP请求   ┌────────────────────────────────┐
│  Streamlit   │ ──────────▶ │          FastAPI 后端           │
│  (聊天界面)   │             │  POST /documents/upload         │
│              │             │  POST /chat/completions         │
│              │             │  GET  /health                   │
└──────────────┘             └───────────────┬────────────────┘
                                              │
                        ┌─────────────────────▼─────────────────────┐
                        │      Self-RAG 工作流（LangGraph 状态机）    │
                        │                                            │
                        │   retrieve ──▶ grade_documents             │
                        │      ▲              │                       │
                        │      │         有相关? ──yes──▶ generate    │
                        │      │              │no                     │
                        │      └─ rewrite_query（改写，最多1次）        │
                        └─────────────────────┬─────────────────────┘
                                              │
                        ┌─────────────────────▼─────────────────────┐
                        │      HybridRetriever 混合检索               │
                        │   ├─ 稠密检索：ChromaDB 向量（BGE 中文向量）   │
                        │   ├─ 稀疏检索：BM25 + jieba 分词             │
                        │   ├─ 融合：RRF 倒数排名融合                  │
                        │   └─ 回溯：子块 → 父块（完整上下文）           │
                        └───────────────────────────────────────────┘
```

---

## 5. 代码逐文件详解

> 项目文件结构：
> ```
> app/
> ├─ core/config.py              # 配置
> ├─ main.py                     # FastAPI 入口
> ├─ schemas/chat.py             # 请求/响应数据结构
> └─ modules/
>     ├─ parser/chunker.py       # 父子切分
>     ├─ retriever/hybrid_retriever.py  # 混合检索
>     └─ agent/
>         ├─ state.py            # 工作流状态定义
>         └─ workflow.py         # Self-RAG 工作流（核心）
> frontend.py                    # Streamlit 前端
> test_retrieval.py / test_workflow.py  # 测试脚本
> check_env.py                   # 环境连通性检查
> ```

### 5.1 `app/core/config.py` —— 配置中心

```python
class Settings(BaseSettings):
    OPENAI_API_KEY: str                              # DeepSeek 的 API Key
    OPENAI_BASE_URL: str = "https://api.deepseek.com" # 接口地址
    LLM_MODEL: str = "deepseek-chat"                 # 模型名
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"  # 中文向量模型
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"     # 向量库落盘路径
```

**要点**：
- `pydantic-settings` 会自动从 `.env` 文件读取这些变量，代码里不用写死密码。
- 这里用的是 **DeepSeek 的 OpenAI 兼容接口**：DeepSeek 官方提供了和 OpenAI 一样的调用格式，所以代码里用 `ChatOpenAI` 就能调 DeepSeek，只是把 `base_url` 换成了 DeepSeek 的地址。**这是一个很实用的省钱技巧**，值得记住。
- `BAAI/bge-small-zh-v1.5` 是智源（BAAI）开源的中文向量模型，「small」表示轻量，能跑在 CPU 上。

### 5.2 `app/modules/parser/chunker.py` —— 父子文档切分

核心思想一句话：**「小片负责查得准，大片负责讲得全」**。

```python
# 两个切分器：父块大（500字），子块小（150字），overlap=50 表示块之间重叠50字
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, ...)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=50, ...)
```

`load_and_split` 做了什么：
1. 根据后缀选 loader（PDF 用 `PyPDFLoader`，其他用 `TextLoader`）读文件。
2. 先切成**父块**（大块），每个父块生成一个 `parent_id`（UUID 唯一标识）。
3. 再把每个父块进一步切成**子块**（小块），每个子块记录自己属于哪个 `parent_id`。

**为什么 overlap（重叠）要有 50 字？** 防止一句话正好被切在块边界，前后断开了，重叠能保证边界信息不丢失。

**为什么父子要这样设计？**
- 如果只用小块检索：语义纯粹、匹配精准，但喂给大模型的上下文太短，可能答不完整。
- 如果只用大块检索：上下文完整，但一大段混着多个主题，向量被稀释，检索不准。
- 所以：**检索命中子块 → 通过 parent_id 找到父块 → 把父块完整内容给大模型**。两全其美。

### 5.3 `app/modules/retriever/hybrid_retriever.py` —— 混合检索（第二个核心）

这个类干四件事：**存文档、稠密检索、稀疏检索、RRF 融合**。

**(1) 初始化**：连 ChromaDB（持久化到本地 `./data/chroma_db`），加载中文向量模型（CPU、归一化）。

**(2) `index_documents` 存文档**：
- 把子块文本算成向量，写进 ChromaDB（带上 `parent_id`、`source` 元数据）。
- 把子块用 jieba 分词后，构建 BM25 索引。

**(3) 稠密检索 `_dense_search`**（语义检索）：
- 把问题也变成向量 → 在向量库里找最相近的 top_k 个子块。

**(4) 稀疏检索 `_sparse_search`**（关键词检索）：
- jieba 分词 → BM25 打分 → 返回分数最高的 top_k 个子块。

**(5) `retrieve` 融合（RRF）**：
- 两路各召回 `top_k * 2` 个（多召回一点，给融合留空间）。
- RRF 公式：`score = 1.0 / (rrf_k + rank + 1)`，`rrf_k=60`。
  - 意思是：排名第 1 加 `1/61`，排名第 2 加 `1/62`……只和**排名**有关，和具体分数无关。
- 把两路的 RRF 分数累加 → 排序 → 取 top_k。
- 最后把命中的子块**回溯成父块**（用 `parent_map_store` 查），输出完整上下文 + 来源 + 得分。

**为什么 RRF 用排名而不用原始分数？**
向量相似度（0~1 的余弦值）和 BM25 分数（可能几十上百）**量纲完全不一样**，直接相加不公平。RRF 只看排名，绕开量纲问题，简单又有效——这是面试常考点。

### 5.4 `app/modules/agent/state.py` —— 状态定义

```python
class GraphState(TypedDict):
    query: str              # 用户原始问题
    transformed_query: str  # 改写后的问题
    documents: List[Dict]   # 检索出的候选文档
    relevant_docs: List[Dict]  # 评估后判定「相关」的文档
    generation: str         # 最终答案
    retry_count: int        # 改写重试次数
```

**要点**：LangGraph 的每个「节点」接收这个 state、修改一部分、返回一个部分更新的字典，状态就在节点间流转。`TypedDict` 只是给这个字典定了「字段类型」。

### 5.5 `app/modules/agent/workflow.py` —— Self-RAG 工作流（最核心！）

这是整个项目**含金量最高**的文件，务必吃透。

**四个节点（node）**：

| 节点 | 干什么 |
|------|--------|
| `retrieve_node` | 用（改写后的）问题做混合检索，取 top3 |
| `grade_documents_node` | 让 LLM 逐篇判断「这篇文档和问题相关吗」，只输出 yes/no，筛选出相关文档 |
| `rewrite_query_node` | 让 LLM 把问题改写成「更具体、带关键词」的版本，重试次数 +1 |
| `generate_node` | 基于相关文档，严格照着资料生成答案，句末标注来源 |

**构图（`_build_graph`）**：

```
入口 → retrieve → grade_documents
                     │
        ┌────────────┼──────────────┐
     有相关文档      无相关 && 未超重试    无相关 && 已超重试
        │            │               │
        ▼            ▼               │
     generate    rewrite_query       │
                   │                 │
                   └──▶ retrieve ─────┘（再走一轮）
                        （重试超限后走 generate 兜底）
```

关键条件判断在 `_decide_to_generate`：

```python
if len(relevant_docs) > 0:      # 有相关文档 → 直接生成
    return "generate"
if retry_count < 1:             # 没相关文档且还没重试过 → 改写问题再来一次
    return "rewrite_query"
return "generate"               # 已经重试过了还是不行 → 兜底生成（会回答“无法回答”）
```

**三个「自省」动作总结**（这就是「Self-RAG」的 Self 体现在哪）：
1. **检索后评估**：不盲信检索结果，先过滤噪声。
2. **查询改写重试**：第一次没召回，就改写关键词再查一次。
3. **抗幻觉生成**：生成时强制「只基于文档 + 标来源 + 答不出就明说」。

**`run` 方法**：初始化 state → `self.app.invoke(initial_state)` 跑完整张图 → 返回结果。

### 5.6 `app/schemas/chat.py` —— 数据结构

- `ChatRequest`：入参，就一个 `query` 字段。
- `DocumentCitation`：引用信息（来源、子块内容、得分），用于前端「溯源卡片」。
- `ChatResponse`：返回（问题、改写后问题、答案、引用列表）。

**要点**：Pydantic 模型让 FastAPI 自动完成「入参校验 + 出参格式化 + 自动生成接口文档」。

### 5.7 `app/main.py` —— FastAPI 入口

三个接口：
1. `POST /api/v1/documents/upload`：接收文件 → 落盘 → 切分 → 建索引，返回父子块数量。
2. `POST /api/v1/chat/completions`：接收问题 → 跑工作流 → 组装成 `ChatResponse`（含引用）。
3. `GET /api/v1/health`：健康检查。

**几个工程细节**：
- `app.add_middleware(CORSMiddleware, ...)`：允许跨域，前端和后端分开部署时能通信。
- 全局单例：`retriever`、`chunker`、`agent_engine` 在启动时初始化一次，避免每次请求都重新加载模型。
- `UploadFile` 用 `shutil.copyfileobj` 流式写文件，省内存。

### 5.8 `frontend.py` —— Streamlit 前端

- 侧边栏：上传 PDF/TXT → 调后端 upload 接口 → 显示入库结果。
- 主界面：聊天框 → 调 chat 接口 → 显示答案 + 「📑 查看引用溯源文档」折叠卡片（能看到答案来自哪、匹配分多少）。

**要点**：前端本身没有 AI 逻辑，只是调后端 HTTP 接口，前后端解耦。

---

## 6. 一条提问的完整旅程（数据流）

以「NexusRAG 采用了什么算法解决专有名词召回问题？」为例：

1. **准备阶段**（上传文档时）：文档 → 切父子块 → 子块向量进 ChromaDB + 建 BM25 索引 + 存父块映射。
2. **用户提问** → 前端调 `POST /chat/completions`。
3. **检索** `retrieve_node`：混合检索（向量 + BM25 + RRF）召回 top3 子块，回溯成父块。
4. **评估** `grade_documents_node`：LLM 逐篇判断相关 → 筛选出 relevant_docs。
5. **分支**：
   - 有相关文档 → 跳到第 6 步生成。
   - 没相关文档 → **改写问题**（比如「NexusRAG 召回算法」→「NexusRAG 混合检索 BM25 RRF 专有名词召回」）→ 回到第 3 步再查一次。
6. **生成** `generate_node`：把相关父块作为 `[来源 1][来源 2]...` 拼进 Prompt，LLM 严格照资料回答并标引用。
7. **返回**：答案 + 引用列表 → 前端展示。

---

## 7. 五大核心机制深度讲解

> 这些是「面试能不能讲出深度」的关键，每个都值得你用自己的话复述一遍。

### 7.1 为什么「向量检索 + BM25」要一起用？

- **向量（稠密）检索**：靠语义。你问「怎么提高召回率」，它能找到写「改善检索效果」的段落——字面不同但意思接近。**短板**：遇到型号、编号、人名这类「专有名词」，语义相近性帮不上忙，可能漏掉。
- **BM25（稀疏）检索**：靠字面。你问「RRF 算法」，它能精确命中写「RRF」的段落。**短板**：不懂同义改写，问「召回」找不到写「recall」的。
- **结论**：一个懂「意思」，一个懂「字面」，互补。本项目 `sample.txt` 里写的「解决专有名词召回不准」正是为此设计。

### 7.2 RRF 为什么用「排名倒数」而不是「分数相加」？

- 向量相似度范围约 0~1，BM25 分数可以是 0~几十甚至更大，**量纲不同**，直接相加或加权不公平。
- RRF 只关心「你在这条路上排第几」：`1/(60+rank+1)`。排名越靠前贡献越大，两路排名都靠前的文档总分最高。
- 优点：无需调参、对不同打分器通用、简单鲁棒。

### 7.3 父子文档切分的意义

- 检索精度和生成完整度是一对矛盾：小块利于检索、大块利于生成。
- 父子结构用 `parent_id` 把两者关联，**检索用子块，生成用父块**，同时拿到两个好处。
- 这正是 LangChain 官方 `ParentDocumentRetriever` 背后的思想。

### 7.4 Self-RAG 比普通 RAG 强在哪？

- 普通 RAG：`检索 → 生成`，一条流水线，查不到就直接硬答（容易幻觉）。
- Self-RAG：`检索 → 评估 → （不相关则）改写 → 再检索 → 生成`，多了「评估」和「改写重试」两个自省环节，能自我纠错。
- 这就是从「RAG」到「Agentic RAG」的区别：**从一次性流水线，变成带反思的智能体**。

### 7.5 如何防幻觉（生成 Prompt 的巧妙设计）

看 `generate_node` 的 Prompt 三条规定：
1. 「必须完全基于参考文档，严禁胡编乱造」→ 强制 grounded。
2. 「陈述关键事实时句末标注 [来源 N]」→ 可溯源。
3. 「文档没提到就直接回答『根据当前知识库内容，无法回答该问题』」→ 拒答兜底。

这三条组合拳，是业界公认的 RAG 防幻觉基本范式。

---

## 8. 如何跑起来这个项目

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 .env（填入你的 DeepSeek API Key）
#    .env 里需要三样：
#    OPENAI_API_KEY=sk-xxx
#    OPENAI_BASE_URL=https://api.deepseek.com
#    LLM_MODEL=deepseek-chat

# 3. 检查环境连通性
python check_env.py        # 期望输出「环境就绪」

# 4. 启动后端
uvicorn app.main:app --reload

# 5. 另开一个终端启动前端
streamlit run frontend.py
```

- 后端地址：`http://127.0.0.1:8000`（自带接口文档 `/docs`）
- 前端地址：Streamlit 会自动打开浏览器。

> 注意：向量模型 `bge-small-zh-v1.5` 首次运行会从 HuggingFace 下载（代码里已配置 `hf-mirror.com` 国内镜像加速）。

---

## 9. HR 会问的问题 + 参考答案

> 分两类：**HR/项目面**（考察你「是不是真做过、表达清不清楚」）和**技术面**（考察深度）。

### 9.1 HR / 项目面常见问题

**Q1：请简单介绍一下你这个项目。**
> 答：这是一个「企业知识库问答系统」。用户把公司的 PDF 或文档上传进去，系统会把这些文档切块、建立索引；之后员工提问，系统先从文档里检索出相关内容，再让大模型基于这些内容生成答案，并且能给出答案的出处。核心解决两个问题：一是让大模型能用到公司内部资料，二是防止大模型瞎编。

**Q2：你在这个项目里主要负责什么？**
> 答：整个项目是独立完成的。我负责了从架构设计到代码实现的全部环节：文档切分模块、混合检索模块、基于 LangGraph 的 Self-RAG 工作流、以及 FastAPI 后端和 Streamlit 前端。

**Q3：这个项目有什么难点？你是怎么解决的？**
> 答：最大的难点是「检索不准导致答案不准」。我用了两个手段：一是**混合检索**（向量 + BM25 用 RRF 融合），解决专有名词召回不准；二是**Self-RAG 自省机制**，检索后让大模型评估相关性，不相关就改写问题重新检索，最后还通过严格 Prompt 防止幻觉。

**Q4：你从这个项目里学到了什么？**
> 答：学到了 RAG 的完整技术栈和工程落地思路，理解了「检索质量决定生成质量」这个核心，也学会了用状态机（LangGraph）把大模型调用编排成可自我纠错的多步流程。

**Q5：这个项目还有什么可以改进的地方？**
> 答：（主动说不足是加分项）比如 BM25 索引目前是内存态的，重启要重建；向量模型跑在 CPU 上，文档多时较慢；文档评估是逐篇串行调用大模型，可以并行化。后续可以加并发锁、持久化索引、多轮自适应检索等。

### 9.2 技术面试常见问题

**Q1：什么是 RAG？为什么不能直接把文档全塞给大模型？**
> 答：RAG 是检索增强生成，先从文档库检索相关内容，再让大模型基于这些内容回答。不能直接全塞，因为：①文档可能远超模型的上下文长度限制；②无关内容太多会稀释注意力、增加成本；③检索能提供「可溯源」的答案。

**Q2：向量检索和 BM25 有什么区别？**
> 答：向量检索靠语义相似度，能理解同义表达；BM25 靠关键词精确匹配，擅长专有名词。两者互补，所以用混合检索 + RRF 融合。

**Q3：RRF 是什么？为什么不用分数直接相加？**
> 答：RRF 是倒数排名融合，对每个文档在每路检索里的排名取倒数求和。因为向量相似度和 BM25 分数的量纲不同，直接相加不公平，用排名倒数能统一度量、免调参。

**Q4：父子文档切分解决什么问题？**
> 答：检索精度和生成完整度有矛盾——小块检索准但上下文短，大块上下文全但检索不准。父子结构让「小块检索、通过 parent_id 回溯大块生成」，两者兼得。

**Q5：Self-RAG 和普通 RAG 的区别？「Self」体现在哪？**
> 答：普通 RAG 是「检索→生成」单一路径；Self-RAG 增加了「检索后评估相关性」和「查询改写重试」两个自省环节，能自我纠错。Self 体现在系统会自己判断检索结果好不好、不好就自己改问题重来。

**Q6：如何防止大模型幻觉？**
> 答：三点——①Prompt 强制只基于检索文档回答；②要求标注引用来源；③文档没答案时明确拒答，而不是编造。

**Q7：LangGraph 在这个项目里起什么作用？**
> 答：用状态机（图）编排大模型流程。定义了 retrieve / grade / rewrite / generate 四个节点和条件边，让流程能根据「评估结果」动态决定走「生成」还是「改写重试」，而不是写死的线性代码。

**Q8：为什么用 DeepSeek 而不是 OpenAI？**
> 答：一是成本低，二是 DeepSeek 提供了 OpenAI 兼容接口，改一下 base_url 就能复用现有的 ChatOpenAI 代码，迁移成本几乎为零。

---

## 10. 简历写法

**项目标题**：NexusRAG 自省式企业知识库问答系统 | Python / LangChain / FastAPI

**项目描述（一段）**：
> 独立设计并实现基于 Self-RAG 反思机制与混合检索的企业级知识库问答系统，解决垂直知识库场景下「检索不准」与「生成幻觉」两大痛点。

**技术亮点（挑 3–4 条）**：
- 设计 BM25 + 向量双路召回，通过 RRF 倒数排名融合统一排序，提升专有名词与语义查询的召回率
- 实现 Parent-Document 父子文档切分，小切片精确检索、大切片回填上下文，缓解长文问答信息断层
- 基于 LangGraph 状态机构建 Self-RAG 多步工作流：检索 → 相关性评估 → 查询改写重试 → 带引用生成
- 通过 grounded 生成 Prompt + 来源标注 + 拒答兜底，有效抑制大模型幻觉
- 采用 FastAPI 异步后端 + Streamlit 前端，封装上传/问答/溯源接口，支持 ChromaDB 持久化与中文 BGE 向量模型

**技术栈关键词**：
`LangGraph · LangChain · FastAPI · ChromaDB · BM25 · RRF · sentence-transformers (BGE) · jieba · DeepSeek API · Streamlit`

---

## 11. 进一步学习路线（从本项目往外延伸）

1. **先复述**：能不看文档，把 5 个核心机制（混合检索、RRF、父子块、Self-RAG、防幻觉）各用一句话讲清楚。
2. **看 LangChain 官方概念**：`ParentDocumentRetriever`、`RAG` 文档，理解本项目的实现和官方方案的对应关系。
3. **进阶 Agentic**：了解「多轮自适应检索」「Self-Refine」「ReAct」「工具调用（Function Calling）」，理解本项目的 Self-RAG 是其中一种简化形态。
4. **进阶检索**：了解「多路召回 + 重排序（Reranker）」「多查询（Multi-Query）」「HyDE」等更高级的检索优化手段。
5. **进阶工程**：了解「文档增量更新」「并发安全」「索引持久化」「评测（RAGAS）」等生产级问题。
6. **横向对比**：了解 LangGraph 之外的编排框架（如 LlamaIndex、AutoGen），以及常见向量库（Milvus、FAISS、Qdrant）的差异。

---

*本文档配套项目源码，建议一边读文档一边打开对应文件对照代码，效果最佳。*
