from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("MODEL", "gpt-4o-mini")
    persist_dir: str = os.getenv("PERSIST_DIR", "vector_db")
    data_dir: str = os.getenv("DATA_DIR", "data")
    top_k: int = int(os.getenv("TOP_K", "6"))

settings = Settings()
