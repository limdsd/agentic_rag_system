from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str = "https://api.deepseek.com"
    LLM_MODEL: str = "deepseek-chat"  # DeepSeek 官方通用模型名

    # 本地中文 Embedding 模型名称（会自动从 HuggingFace/ModelScope 本地缓存）
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()