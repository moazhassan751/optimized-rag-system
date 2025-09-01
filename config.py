import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    DEFAULT_INDEX_NAME = "universal-rag"  # Recreated with 768 dimensions for production
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
    GEMINI_MODEL = "gemini-2.5-flash-lite"  # Optimized for speed and efficiency 
