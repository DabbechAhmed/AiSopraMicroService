from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # AI Model Configuration
    model_name: str = "all-MiniLM-L6-v2"
    model_cache_dir: str = "./models"
    
    # Service Configuration
    service_name: str = "ai-recommendation-service"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Performance Configuration
    max_batch_size: int = 32
    similarity_threshold: float = 0.3
    
    class Config:
        env_file = ".env"

settings = Settings()