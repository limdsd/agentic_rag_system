import streamlit as st
import requests

# 页面基础配置
st.set_page_config(
    page_title="NexusRAG 智能知识库",
    page_icon="🧠",
    layout="wide"
)

API_BASE = "http://127.0.0.1:8000"

st.title("🧠 NexusRAG - 自省式企业智能知识库")

# 侧边栏：文档管理
with st.sidebar:
    st.header("📂 知识库文档上传")
    uploaded_file = st.file_uploader("上传 PDF 或 TXT 文档", type=["pdf", "txt"])

    if uploaded_file and st.button("开始解析并入库", use_container_width=True):
        with st.spinner("正在执行父子切块与向量/BM25混合索引..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                res = requests.post(f"{API_BASE}/api/v1/documents/upload", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.success(
                        f"✅ 上传成功！\n- 父文档块: {data['parent_chunks_count']}\n- 检索子块: {data['child_chunks_count']}")
                else:
                    st.error(f"❌ 上传失败: {res.text}")
            except Exception as e:
                st.error(f"连接后端失败: {e}")

    st.markdown("---")
    st.markdown("**系统特性：**")
    st.markdown("- 🔍 **BM25 + 向量多路召回**")
    st.markdown("- 🧩 **父子文档切分 (Parent-Document)**")
    st.markdown("- 🔄 **LangGraph Self-RAG 反思纠错**")

# 主聊天界面
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("📑 查看引用溯源文档"):
                for idx, cite in enumerate(msg["citations"], 1):
                    st.markdown(f"**[引用 {idx}]** (来源: `{cite['source']}`, 匹配分: `{cite['score']}`)")
                    st.info(cite["child_content"])

# 处理用户提问
if prompt := st.chat_input("请输入您关于文档的疑问..."):
    # 渲染用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 请求后端处理
    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并进行事实自检..."):
            try:
                response = requests.post(
                    f"{API_BASE}/api/v1/chat/completions",
                    json={"query": prompt}
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    citations = data.get("citations", [])

                    st.markdown(answer)
                    if citations:
                        with st.expander("📑 查看引用溯源文档"):
                            for idx, cite in enumerate(citations, 1):
                                st.markdown(f"**[引用 {idx}]** (来源: `{cite['source']}`, 匹配分: `{cite['score']}`)")
                                st.info(cite["child_content"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })
                else:
                    st.error(f"生成失败: {response.text}")
            except Exception as e:
                st.error(f"后端未启动或连接异常: {e}")