# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 全部給預設值，沒 .env 也能跑
    exchange: str = "binance"
    cors_origins: list[str] = ["*"]

    class Config:
        env_prefix = ""
        case_sensitive = False

settings = Settings()
