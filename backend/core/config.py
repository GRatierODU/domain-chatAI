from pydantic_settings import BaseSettings
from typing import Dict, List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # Application
    app_name: str = "AI Customer Service Chatbot"
    debug: bool = False
    
    # Models
    model_cache_dir: Path = Path("./models")
    use_gpu: bool = True
    device_map: str = "auto"
    
    # Model selection based on available resources
    reasoning_models: Dict = {
        "primary": "Qwen/QwQ-32B-Preview",
        "secondary": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
        "efficient": "microsoft/Phi-3-medium-128k-instruct"
    }
    
    vision_models: Dict = {
        "multimodal": "Qwen/Qwen2-VL-7B-Instruct",
        "layout": "microsoft/layoutlmv3-base",
        "table": "microsoft/table-transformer-detection",
        "ocr": "microsoft/trocr-large-printed"
    }
    
    # Crawler
    max_concurrent_crawlers: int = 5
    crawler_timeout: int = 30
    respect_robots_txt: bool = True
    
    # Vector Database
    chroma_persist_dir: Path = Path("./chroma_db")
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 64
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["*"]
    
    # Cache
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600
    
    # Performance
    max_context_length: int = 8192
    max_response_length: int = 1024
    batch_size: int = 4
    
    class Config:
        env_file = ".env"
        
settings = Settings()