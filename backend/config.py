from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "AI QC Insights Generator"
    APP_ENV: str = "development"

    # REQUIRED (no defaults)
    SECRET_KEY: str
    DATABASE_URL: str
    GOOGLE_API_KEY: str
    OPENAI_API_KEY: str

    # Optional
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    OPENAI_MODEL: str = "gpt-4o-mini"

    GOOGLE_AI_MODEL: str = "gemini-2.0-flash-exp"

    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    PHOENIX_HOST: str = "localhost"
    PHOENIX_PORT: int = 6006

    STATIC_OTP: str = "123456"
    OTP_EXPIRE_MINUTES: int = 10

    SIMULATOR_INTERVAL_SECONDS: float = 2.0
    FASTAPI_INGEST_URL: str = "http://localhost:8000/api/ingest"

    BACKEND_URL: str = "http://localhost:8000"
    WS_URL: str = "ws://localhost:8000/ws/dashboard"

    class Config:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        extra = "ignore"


settings = Settings()
