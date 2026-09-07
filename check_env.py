from app.core.config import settings
from langchain_openai import ChatOpenAI


def test_connection():
    print(f"正在测试模型: {settings.LLM_MODEL}")
    print(f"接口地址: {settings.OPENAI_BASE_URL}")

    try:
        llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=0
        )
        res = llm.invoke("请回复四个字：环境就绪")
        print(f"\n✅ 响应成功: {res.content}")
    except Exception as e:
        print(f"\n❌ 连接失败: {e}")


if __name__ == "__main__":
    test_connection()