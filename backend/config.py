import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Control panel for env vars
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_PATH=os.path.join(_BASE_DIR, "..", "chroma_db")
UPLOAD_PATH=os.path.join(_BASE_DIR, "..", "data", "uploads")
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=3
OLLAMA_BASE_URL="http://localhost:11434"