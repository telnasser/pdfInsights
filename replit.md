# Advanced RAG Application

A Flask-based Retrieval-Augmented Generation (RAG) system with intelligent query routing, multi-hop graph retrieval, and dual-mode search.

## Architecture

### Storage Layer
- **PostgreSQL** (via SQLAlchemy): Document metadata, chunk text, query history, and the knowledge graph (entities + relationships)
- **FAISS** (IndexFlatIP): Vector index for cosine-similarity semantic search
- **NetworkX + JSON files**: In-memory graph used for visualization; also serves as fallback when PostgreSQL graph tables are empty

### Entity Extraction (Knowledge Graph Ingestion)
- **Primary**: Claude Haiku via `KnowledgeGraph._extract_entities_llm()` — sends each chunk to the LLM with a structured prompt; extracts COMPANY, ROLE, TECHNOLOGY, DATE_RANGE, PRODUCT, DEGREE, SKILL, LOCATION, PERSON entities
- **Fallback**: spaCy `en_core_web_sm` NER + noun chunks (used if API unavailable)
- **Query-time**: always uses spaCy (fast, no API call needed for query entity extraction)
- **Cache**: SHA-256 hash → entity list, in-memory per `KnowledgeGraph` instance

### Core RAG Pipeline (`routes/query_routes.py` → `rag/retriever.py`)

```
Query
  │
  ▼
QueryRouter (rag/query_router.py)
  │   Rule-based scoring → KEYWORD | VECTOR | HYBRID
  │
  ▼
KnowledgeGraph.multi_hop_search_db (rag/knowledge_graph.py)
  │   BFS over PostgreSQL graph_entity / graph_relationship tables
  │   Falls back to NetworkX when DB tables are empty
  │   Returns chunk_indices for graph-matched entities (2 hops)
  │
  ├─► [KEYWORD mode] KeywordSearch (rag/keyword_search.py)
  │       PostgreSQL full-text search via to_tsvector / plainto_tsquery
  │
  ├─► [VECTOR mode]  VectorStore.search (rag/vector_store.py)
  │       FAISS cosine-similarity search with optional re-scoring
  │
  └─► [HYBRID mode]  Both keyword + vector, merged by blended score
  │
  ▼
Retriever._merge_results → de-duplicate, blend scores, top-k
  │
  ▼
Retriever._rerank_results (optional)
  │   Refined embedding blend (0.7 original + 0.3 context-enriched)
  │
  ▼
Generator.generate_response (rag/generator.py)
  │   Claude claude-3-5-sonnet-20241022
  │
  ▼
JSON response (includes routing mode + graph entity info)
```

### Key Files

| File | Responsibility |
|------|----------------|
| `rag/query_router.py` | Rule-based keyword vs vector vs hybrid decision |
| `rag/keyword_search.py` | PostgreSQL full-text search (BM25-style) |
| `rag/knowledge_graph.py` | spaCy entity extraction, NetworkX graph, multi-hop BFS, PostgreSQL sync |
| `rag/retriever.py` | Orchestrates the full pipeline |
| `rag/embeddings.py` | Claude API embeddings with TF-IDF fallback (768-dim) |
| `rag/vector_store.py` | FAISS IndexFlatIP with cosine similarity |
| `rag/generator.py` | Claude response generation |
| `models.py` | SQLAlchemy models: Document, Chunk, Query, QueryChunk, GraphEntity, GraphRelationship |
| `routes/query_routes.py` | `/query/ask`, `/query/search`, `/query/graph/multihop` |
| `routes/document_routes.py` | Upload, KG processing, `/document/graph/api/migrate` |
| `config.py` | All tunable constants |

## PostgreSQL Tables

- `document` — uploaded PDFs metadata
- `chunk` — text chunks with page numbers
- `query` — query history and responses
- `query_chunk` — query↔chunk relevance association
- `graph_entity` — knowledge graph nodes (name, type, doc_ids, chunk_indices)
- `graph_relationship` — knowledge graph edges (source, target, weight)

## Configuration (`config.py`)

| Setting | Default | Purpose |
|---------|---------|---------|
| `EMBEDDING_DIMENSION` | 768 | Vector size |
| `TOP_K_CHUNKS` | 5 | Results per query |
| `SIMILARITY_THRESHOLD` | 0.3 | Min cosine score |
| `RERANKING_ENABLED` | True | Blend-score reranking |
| `MULTIHOP_HOPS` | 2 | Graph BFS depth |
| `MULTIHOP_MAX_NEIGHBORS` | 6 | Neighbors per node per hop |
| `MULTIHOP_CHUNK_LIMIT` | 60 | Cap on graph-sourced chunks |

## API Endpoints

### Query
- `POST /query/ask` — Full RAG (retrieve + generate)
- `POST /query/search` — Retrieve only (no generation)
- `POST /query/graph/multihop` — Inspect multi-hop graph traversal for a query

### Documents
- `POST /document/upload` — Upload and index a PDF
- `GET /document/graph/<doc_id>` — View knowledge graph (processes + syncs to DB)
- `POST /document/graph/api/migrate` — Migrate NetworkX graph → PostgreSQL (one-time)

## Embedding Strategy

The system attempts to use the Anthropic Claude embeddings API. If unavailable, it falls back to:
1. TF-IDF with `max_features=10,000`
2. Random projection from 10k → 768 dimensions
3. L2 normalization for cosine similarity compatibility

## Knowledge Graph Migration

The existing NetworkX graph (6,618 entities, 160,684 relationships from JSON files) can be migrated to PostgreSQL at any time:

```bash
curl -X POST http://localhost:5000/document/graph/api/migrate \
  -H "Content-Type: application/json" -d '{}'
```

After migration, the SQL-backed `multi_hop_search_db` method is used instead of the in-memory NetworkX path, enabling efficient filtered multi-hop queries.
