# PDF Analyzer Configuration

# Ollama Settings
OLLAMA_MODEL = "llama3.2"
OLLAMA_HOST = "http://localhost:11434"

# Text Processing Settings
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
DEFAULT_TOP_K = 5

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Supported file types
SUPPORTED_EXTENSIONS = [".pdf"]

# Cache directory for storing processed indexes
CACHE_DIR = "./cache"
