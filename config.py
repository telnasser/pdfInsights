import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 35 * 1024 * 1024  # 35 MB

# Vector database configuration
VECTOR_DB_PATH = os.path.join(BASE_DIR, 'vector_db')
VECTOR_DB_TYPE = 'faiss'  # Only 'faiss' is supported

# Embedding configuration
EMBEDDING_DIMENSION = 768  # Use 768 to match existing index dimensions
CLAUDE_EMBEDDING_MODEL = "claude-3-haiku-20240307"  # Widely available Claude model
EMBEDDING_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# When rebuilding the index, ensure the index dimension matches the embedding dimension
INDEX_DIMENSION = EMBEDDING_DIMENSION

# Retrieval configuration
TOP_K_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.3  # Lower threshold to get more results with cosine similarity
RERANKING_ENABLED = True
USE_COSINE_SIMILARITY = True  # Prefer cosine similarity over just topK

# LLM configuration
CLAUDE_LLM_MODEL = "claude-3-haiku-20240307"  # Widely available Claude model
LLM_API_KEY = EMBEDDING_API_KEY  # Use the same API key for both embedding and LLM

# Claude API configuration
API_TIMEOUT = 30  # Timeout in seconds for API calls

# Chunking configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_CHUNK_STRATEGY = "paragraph"  # Options: "sentence", "paragraph", "sliding"

# File handling configuration
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

# Knowledge Graph configuration
KNOWLEDGE_GRAPH_PATH = os.path.join(BASE_DIR, 'knowledge_graph')
ENTITY_RECOGNITION_CONFIDENCE = 0.7
MAX_ENTITIES_PER_DOCUMENT = 200
MAX_RELATIONSHIPS_PER_ENTITY = 50

# Multi-hop retrieval configuration
MULTIHOP_ENABLED = True
MULTIHOP_HOPS = 2                    # BFS depth for graph traversal
MULTIHOP_MAX_NEIGHBORS = 6           # Max neighbors to follow per node per hop
MULTIHOP_CHUNK_LIMIT = 60            # Max chunk indices collected via graph

# Query routing configuration
QUERY_ROUTING_ENABLED = True         # Enable keyword/vector/hybrid routing

# Vector store directory
if not os.path.exists(VECTOR_DB_PATH):
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Uploads directory
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Knowledge graph directory
if not os.path.exists(KNOWLEDGE_GRAPH_PATH):
    os.makedirs(KNOWLEDGE_GRAPH_PATH, exist_ok=True)