"""
Configuration file for the RAG system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the RAG system."""
    
    # API Keys - Set these in your .env file
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional: for OpenAI embeddings
    
    # Model configurations
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")  # Hugging Face model
    # Alternative OpenAI embedding model: "text-embedding-3-small"
    
    # Directory paths
    DATA_DIR = Path("data")
    PERSIST_DIR = Path("storage")
    
    # Vector store settings
    VECTOR_STORE_DIM = 384  # Dimension for BAAI/bge-small-en-v1.5
    # For OpenAI text-embedding-3-small, use 1536
    
    # Chunk settings
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200
    
    # Retrieval settings
    SIMILARITY_TOP_K = 5
    
    # Response settings
    TEMPERATURE = 0.1
    MAX_TOKENS = 1024
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configurations are set."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY must be set in environment variables")
        
        # Create directories if they don't exist
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.PERSIST_DIR.mkdir(exist_ok=True)
        
        return True