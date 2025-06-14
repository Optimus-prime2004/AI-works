import logging
import os
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# --- General Configuration ---
LOG_FILE = 'ai_resume_pro.log'
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
DEFAULT_USER_ID = 1

# --- AI Model Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q4_K_M") # NEW: Switched to llama3
# Using a top-tier, high-performance model for embeddings.
# It's larger but provides superior accuracy.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5") # NEW: Upgraded model

# --- Vector Database Configuration (Milvus) ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "157.180.121.62")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
JOB_COLLECTION_NAME = "job_postings"

# --- Third-Party API Keys ---
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# --- SQLite Database Configuration ---
DB_PATH = os.getenv("DB_PATH", "ai_resume.db")

# --- Performance Configuration ---
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
OLLAMA_TIMEOUT = 120 # Increased for larger models like Llama3
REQUEST_TIMEOUT = 180 # Increased for more complex operations

# --- Logging Setup ---
log_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s'
)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger('ai_resume_pro')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

for lib in ['urllib3', 'requests', 'hpack', 'asyncio', 'pymilvus']: # Added pymilvus
    logging.getLogger(lib).setLevel(logging.WARNING)