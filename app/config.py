from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # 讀 .env，大小寫不敏感
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # 對應到 .env 的變數
    exchange: str = Field(default="binance", alias="EXCHANGE")
    default_symbol: str = Field(default="ETH/USDT", alias="DEFAULT_SYMBOL")
    default_timeframe: str = Field(default="1h", alias="DEFAULT_TIMEFRAME")
    db_url: str = Field(default="sqlite:///./eth_agent.db", alias="DB_URL")

settings = Settings()
