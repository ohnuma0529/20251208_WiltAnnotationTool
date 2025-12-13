import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_DIR: str = "/media/HDD-6TB/Leaf_Images"
    CACHE_DIR: str = "/home/happyai2023/20251208_WiltAnnotationTool/fast_cache"
    OUTPUT_DIR: str = "/home/happyai2023/20251208_WiltAnnotationTool/output" 
    WORK_DIR: str = "/media/HDD-6TB/Wilt_Project_Work"
    
    # Filter Settings
    START_TIME_HOUR: int = 7
    END_TIME_HOUR: int = 17 # Exclusive (07:00 <= t < 17:00)
    DEFAULT_FREQUENCY: int = 10 # Minutes
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost",
        "*"
    ]

    class Config:
        env_file = ".env"

settings = Settings()
