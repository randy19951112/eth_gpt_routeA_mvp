# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices

class Settings(BaseSettings):
    # 交易所預設
    exchange: str = "binance"

    # DB 連線字串：優先吃環境變數 DB_URL / DATABASE_URL，否則用預設 SQLite
    # **注意 4 條斜線** sqlite:////var/data/app.db 代表絕對路徑 /var/data/app.db
    db_url: str = Field(
        default="sqlite:////var/data/app.db",
        validation_alias=AliasChoices("DB_URL", "DATABASE_URL"),
    )

    model_config = SettingsConfigDict(
        env_file=".env",  # 本地可用 .env
        extra="ignore",
    )

settings = Settings()
