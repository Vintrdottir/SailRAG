from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"
    log_level: str = "INFO"

    opensearch_url: str = "http://localhost:9200"
    ollama_url: str = "http://localhost:11434"
    opensearch_index: str = "sailrag-chunks"
    data_dir: str = "/data"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_llm_model: str = "llama3.2:3b"


settings = Settings()
