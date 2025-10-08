import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application settings
    app_name: str = "CryptoCaller"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Streamlit settings
    streamlit_port: int = 8501
    streamlit_host: str = "0.0.0.0"
    
    # Database settings
    database_url: str = "sqlite:///data/cryptocaller.db"
    
    # API settings
    api_timeout: int = 30
    api_rate_limit: int = 10
    
    # Security settings
    secret_key: str = "your-secret-key-here"
    session_timeout: int = 3600
    
    # Exchange API keys (optional)
    binance_api_key: Optional[str] = None
    binance_secret_key: Optional[str] = None
    coinbase_api_key: Optional[str] = None
    coinbase_secret_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()