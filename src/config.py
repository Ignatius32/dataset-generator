import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Configuration
    GS_API_KEY = os.getenv("GS_API_KEY")
    API_URL = os.getenv("API_URL", "https://api.gradientesur.com/functions/v1/embeddings")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")    # Chat API Configuration for dataset generation
    CHAT_API_URL = os.getenv("CHAT_API_URL", "https://api.gradientesur.com/functions/v1/chat/completions")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen3-1.7B")
      # Document Processing - Enhanced for better content processing
    CHUNK_SIZE = 1500  # Increased for better context
    CHUNK_OVERLAP = 300  # Increased overlap for better continuity
    MAX_TOKENS_PER_CHUNK = 1024  # Increased token limit
    MIN_CHUNK_SIZE = 100  # Minimum chunk size to avoid tiny chunks
    
    # API Token limits - Account for reasoning content in Qwen models
    DEFAULT_MAX_TOKENS = 500  # Increased default for reasoning models
    QUERY_GENERATION_TOKENS = 800  # For query generation tasks
    RESPONSE_GENERATION_TOKENS = 1000  # For response generation tasks
    
    # Vector Store
    VECTOR_DIMENSION = 768  # nomic-embed-text-v1.5 dimension
    
    # API Retry Configuration - Enhanced for robustness
    MAX_RETRIES = 5  # Increased retries
    BASE_RETRY_DELAY = 2.0  # Increased delay
    MAX_RETRY_DELAY = 32.0  # Maximum delay cap
    REQUEST_TIMEOUT = 60    # Increased timeout
    
    # Dataset Generation - Enhanced workflow
    QUERIES_PER_CHUNK = 5  # Generate more initial queries
    CANDIDATE_QUERIES_PER_CHUNK = 3  # Select best 3 from the 5
    MAX_RETRIEVED_DOCS = 7  # More context for responses
    JUDGE_QUERIES = True  # Enable query judging step
    
    # File Paths
    DEFAULT_INPUT_DIR = "data/documents"
    DEFAULT_VECTORSTORE_DIR = "data/vectorstore"
    DEFAULT_OUTPUT_FILE = "data/dpo_dataset.json"
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GS_API_KEY:
            raise ValueError("GS_API_KEY not found in environment variables")
        if not cls.API_URL:
            raise ValueError("API_URL not found in environment variables")
