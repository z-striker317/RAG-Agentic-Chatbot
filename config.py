import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI # Best to use this directly for OpenRouter

load_dotenv()

class Config:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    MODEL_NAME = "openai/gpt-oss-120b:free" # Your free model
    
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    DEFAULT_URLS = [
        "https://github.com/gowthamrajk/SQL-Tutorials",
        "https://github.com/s-shemmee/SQL-101"
    ]

    @classmethod
    def get_llm(cls):
        """Initialize LLM using the logic from the SDK you found"""
        return ChatOpenAI(
            model=cls.MODEL_NAME,
            openai_api_key=cls.OPENROUTER_API_KEY,
            openai_api_base=cls.OPENROUTER_BASE_URL,
            # These are the headers from your image snippet
            default_headers={
                "HTTP-Referer": "http://localhost:8501", # Standard Streamlit port
                "X-Title": "RAG Document Search",
            },
            # This enables the reasoning logic you had earlier
            extra_body={"reasoning": {"enabled": True}}
        )