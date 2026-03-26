# PDF Insights — Advanced RAG Application

A production-ready **Retrieval-Augmented Generation (RAG)** system built with Flask, PostgreSQL, FAISS, and a persistent Knowledge Graph. Upload PDF documents and ask natural-language questions; the system retrieves the most relevant passages using a hybrid of keyword search, semantic vector search, and multi-hop knowledge graph traversal, then synthesises an answer with Claude AI.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Component Overview](#2-component-overview)
3. [Data Models (ER Diagram)](#3-data-models-er-diagram)
4. [Document Ingestion Flow](#4-document-ingestion-flow)
5. [Query & Retrieval Flow](#5-query--retrieval-flow)
6. [Embedding Pipeline](#6-embedding-pipeline)
7. [Knowledge Graph Architecture](#7-knowledge-graph-architecture)
8. [Query Router Logic](#8-query-router-logic)
9. [Directory Structure](#9-directory-structure)
10. [Configuration Reference](#10-configuration-reference)
11. [API Endpoints](#11-api-endpoints)
12. [Setup & Running](#12-setup--running)
13. [Technology Stack](#13-technology-stack)

---

## 1. System Architecture

```mermaid
graph TB
    subgraph Browser["Browser / UI"]
        UI[Flask Templates<br/>Bootstrap + Feather Icons]
    end

    subgraph App["Flask Application (Gunicorn)"]
        DR[Document Routes<br/>/upload, /view, /graph]
        QR[Query Routes<br/>/ask, /search, /history]
    end

    subgraph RAG["RAG Pipeline"]
        DP[Document Processor<br/>PyPDF2 + pdfminer]
        CK[Text Chunker<br/>Paragraph Strategy]
        EMB[Embedding Generator<br/>TF-IDF → 768-dim via<br/>Random Projection]
        VS[Vector Store<br/>FAISS IndexFlatIP]
        KG[Knowledge Graph<br/>spaCy NER + NetworkX]
        RET[Retriever<br/>Orchestrator]
        QROUTER[Query Router<br/>Rule-based Scoring]
        KWS[Keyword Search<br/>PostgreSQL FTS]
        GEN[Generator<br/>Claude claude-3-haiku-20240307]
    end

    subgraph Storage["Persistent Storage"]
        PG[(PostgreSQL<br/>Documents · Chunks<br/>Queries · Graph)]
        FAISS[(FAISS Index<br/>vector_db/faiss_index)]
        TFIDF[(TF-IDF Vectorizer<br/>vector_db/tfidf_vectorizer.pkl)]
        UPLOAD[(PDF Files<br/>uploads/)]
    end

    Browser -->|HTTP| App
    DR --> DP
    DP --> CK
    CK --> EMB
    EMB --> VS
    EMB -->|fit + save| TFIDF
    VS -->|write| FAISS
    DP --> KG
    KG -->|sync entities & rels| PG
    CK -->|store chunks| PG
    DR -->|store metadata| PG

    QR --> RET
    RET --> QROUTER
    QROUTER -->|keyword| KWS
    QROUTER -->|vector| VS
    QROUTER -->|hybrid| KWS
    KWS -->|FTS| PG
    VS -->|read| FAISS
    EMB -->|load| TFIDF
    RET --> KG
    KG -->|multi-hop BFS| PG
    RET --> GEN
    GEN -->|Claude API| Claude[(Anthropic Claude)]
    GEN -->|store Q+A| PG
```

---

## 2. Component Overview

| Module | File | Responsibility |
|---|---|---|
| **Document Processor** | `rag/document_processor.py` | Extracts text from PDF (PyPDF2 + pdfminer fallback), generates MD5 document ID, delegates to chunker |
| **Text Chunker** | `rag/chunking.py` | Splits text into paragraph-based chunks; preserves paragraph boundaries; configurable size/overlap |
| **Embedding Generator** | `rag/embeddings.py` | Primary: Anthropic API (unavailable). Active: TF-IDF + random projection → 768-dim unit vectors; vectorizer persisted to disk |
| **Vector Store** | `rag/vector_store.py` | Wraps FAISS `IndexFlatIP` (inner product on L2-normalised vectors = cosine similarity); saves/loads from disk |
| **Knowledge Graph** | `rag/knowledge_graph.py` | Builds entity–relation graph with spaCy NER; persists nodes/edges to PostgreSQL; performs multi-hop BFS at query time |
| **Query Router** | `rag/query_router.py` | Rule-based scoring assigns `keyword`, `vector`, or `hybrid` mode per query |
| **Keyword Search** | `rag/keyword_search.py` | PostgreSQL full-text search with OR-based `to_tsquery`; ILIKE fallback |
| **Retriever** | `rag/retriever.py` | Orchestrates Router → Graph BFS → Keyword/Vector/Hybrid search → merge & rerank |
| **Generator** | `rag/generator.py` | Calls Claude claude-3-haiku-20240307 with retrieved context to produce final answer |
| **Visualization** | `rag/visualization.py` | Grid-layout chunk distribution chart; pyvis CDN-hosted knowledge graph HTML |
| **Document Routes** | `routes/document_routes.py` | Upload, view, delete, knowledge graph view, graph API endpoints |
| **Query Routes** | `routes/query_routes.py` | Ask, search, query history, multi-hop API |
| **Models** | `models.py` | SQLAlchemy ORM: Document, Chunk, Query, QueryChunk, GraphEntity, GraphRelationship |
| **Config** | `config.py` | Central constants: dimensions, thresholds, paths, model names |

---

## 3. Data Models (ER Diagram)

```mermaid
erDiagram
    DOCUMENT {
        string id PK "MD5 hash of filename"
        string filename
        string title
        datetime upload_date
        int num_pages
        int num_chunks
        int file_size
        string chunk_strategy
        int chunk_size
        int chunk_overlap
        bool kg_processed
        int kg_entity_count
        int kg_relationship_count
        float kg_processing_time
    }

    CHUNK {
        int id PK
        string document_id FK
        int chunk_index
        text text
        int page_num
        json chunk_metadata
    }

    QUERY {
        int id PK
        text query_text
        text response_text
        string document_id FK
        datetime timestamp
        int top_k
        float temperature
    }

    QUERY_CHUNK {
        int id PK
        int query_id FK
        int chunk_id FK
        float relevance_score
    }

    GRAPH_ENTITY {
        int id PK
        string name "unique, indexed"
        string entity_type "PERSON/ORG/DATE/GPE/etc"
        json doc_ids
        json chunk_indices
        int occurrence_count
    }

    GRAPH_RELATIONSHIP {
        int id PK
        int source_id FK
        int target_id FK
        int weight
        json doc_ids
        json chunk_indices
    }

    DOCUMENT ||--o{ CHUNK : "has"
    DOCUMENT ||--o{ QUERY : "scoped to"
    QUERY ||--o{ QUERY_CHUNK : "references"
    CHUNK ||--o{ QUERY_CHUNK : "used in"
    GRAPH_ENTITY ||--o{ GRAPH_RELATIONSHIP : "source of"
    GRAPH_ENTITY ||--o{ GRAPH_RELATIONSHIP : "target of"
```

---

## 4. Document Ingestion Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Upload Form
    participant DR as Document Route
    participant DP as DocumentProcessor
    participant CK as TextChunker
    participant EMB as EmbeddingGenerator
    participant VS as VectorStore (FAISS)
    participant KG as KnowledgeGraph
    participant DB as PostgreSQL

    User->>UI: Upload PDF
    UI->>DR: POST /upload (multipart)
    DR->>DP: process(pdf_path)

    DP->>DP: Extract text<br/>(pdfminer primary,<br/>PyPDF2 fallback)
    DP->>CK: chunk(text, strategy=paragraph)
    CK-->>DP: 14 paragraph chunks

    DP-->>DR: {doc_id, chunks[]}
    DR->>DB: INSERT Document + Chunks

    DR->>EMB: embed_chunks(chunks)
    Note over EMB: API fails → TF-IDF fallback
    EMB->>EMB: _refit_and_save(all_DB_chunks)<br/>vocab=1833, dim=768
    EMB-->>DR: embeddings[14 × 768]

    DR->>VS: add_document(doc_id, chunks+embeddings)
    VS->>VS: L2-normalise → FAISS.add()
    VS-->>DR: saved faiss.index

    DR->>KG: process_document(doc_id, text)
    KG->>KG: spaCy NER → entities + co-occurrences
    KG->>DB: UPSERT GraphEntity (433 nodes)
    KG->>DB: UPSERT GraphRelationship (11k+ edges)
    KG-->>DR: {entities, relationships}

    DR-->>UI: Redirect to /view/{doc_id}
    UI-->>User: Document view page
```

---

## 5. Query & Retrieval Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Query Form
    participant QR as Query Route /ask
    participant ROUTER as QueryRouter
    participant KG as KnowledgeGraph<br/>multi_hop_search_db
    participant KWS as KeywordSearch<br/>PostgreSQL FTS
    participant VS as VectorStore<br/>FAISS
    participant EMB as EmbeddingGenerator
    participant MERGE as Retriever<br/>merge & rerank
    participant GEN as Generator<br/>Claude API
    participant DB as PostgreSQL

    User->>UI: Type query
    UI->>QR: POST /ask {query, doc_id, top_k}

    QR->>ROUTER: route(query)
    Note over ROUTER: Score keyword signals:<br/>dates, proper nouns, who/where/when<br/>Score vector signals:<br/>explain, compare, summarize
    ROUTER-->>QR: mode=keyword (kw=1.0, vec=0.0)

    par Graph Expansion
        QR->>KG: multi_hop_search_db(query, hops=2)
        KG->>KG: spaCy NER + noun extraction
        KG->>DB: SELECT GraphEntity WHERE name ILIKE any term
        KG->>DB: BFS: outgoing + incoming rels (hop 1 → hop 2)
        KG-->>QR: {chunk_indices[4], entities[5]}
    and Keyword Search
        QR->>KWS: search(query, top_k=10)
        KWS->>KWS: Build OR tsquery:<br/>"where | did | tamer | work | 2019"
        KWS->>DB: to_tsvector @@ to_tsquery + ts_rank_cd
        KWS-->>QR: 5 ranked chunks
    and Vector Search (if hybrid/vector mode)
        QR->>EMB: embed_query(query)
        EMB->>EMB: Load TF-IDF from disk<br/>(same vocab=1833 as index)
        EMB-->>QR: query_vec[768]
        QR->>VS: search(query_vec, top_k=15)
        VS->>VS: FAISS inner product search<br/>+ cosine rerank
        VS-->>QR: top-k chunks
    end

    QR->>MERGE: merge(keyword, vector, graph)
    Note over MERGE: keyword×1.0, graph×0.6, vector×0.4<br/>De-duplicate by (doc_id, chunk_index)<br/>Rerank with blended cosine score
    MERGE-->>QR: top-5 chunks

    QR->>GEN: generate(query, chunks)
    GEN->>GEN: Build prompt with context
    GEN->>GEN: Claude claude-3-haiku-20240307
    GEN-->>QR: answer text

    QR->>DB: INSERT Query + QueryChunk associations
    QR-->>UI: {answer, sources, routing_info}
    UI-->>User: Answer + source chunks displayed
```

---

## 6. Embedding Pipeline

```mermaid
flowchart TD
    A[Input Texts] --> B{Anthropic API<br/>available?}

    B -->|Yes| C[client.embeddings.create<br/>claude-3-haiku-20240307]
    C --> D{Success?}
    D -->|Yes| E[API Embeddings<br/>dim = 768]
    D -->|No| F[Set use_fallback = True]

    B -->|No / Fallback| G{Vectorizer on disk?<br/>tfidf_vectorizer.pkl}
    F --> G

    G -->|Load| H[Load TF-IDF + Projection<br/>from disk]
    G -->|Missing / len>1 texts| I[Refit TF-IDF on<br/>ALL DB chunks + new texts]
    I --> J[Build Random Projection<br/>seed=42, shape vocab×768]
    J --> K[Save to disk]
    K --> L[TF-IDF.transform texts]
    H --> L

    L --> M[Dense matrix N × vocab]
    M --> N[Matrix multiply: dense @ projection<br/>→ N × 768]
    N --> O[L2 normalise each row]
    O --> P[Local Embeddings<br/>dim = 768, unit vectors]

    E --> Q[Return Embeddings]
    P --> Q

    style I fill:#f4a,stroke:#c22
    style K fill:#f4a,stroke:#c22
    style J fill:#f4a,stroke:#c22
```

**Key design decisions:**

| Property | Value |
|---|---|
| Target dimension | 768 |
| Primary method | Anthropic embeddings API (currently unavailable — `embeddings` attr missing) |
| Active method | TF-IDF (unigrams + bigrams, max 10 000 features) + random projection |
| Projection seed | `np.random.seed(42)` — deterministic across all workers |
| Similarity metric | Cosine (L2-normalised inner product via FAISS `IndexFlatIP`) |
| Persistence | `vector_db/tfidf_vectorizer.pkl` + `vector_db/tfidf_projection.pkl` |
| Refit trigger | Any document batch (len > 1 texts) forces refit on full DB corpus |
| Startup recovery | If vectorizer missing on disk but DB chunks exist → auto-rebuild FAISS index |

---

## 7. Knowledge Graph Architecture

```mermaid
flowchart LR
    subgraph Build["Graph Construction (per document)"]
        PDF[PDF Text] --> NER[spaCy en_core_web_sm<br/>Named Entity Recognition]
        NER --> COOC[Co-occurrence Analysis<br/>sliding window per chunk]
        COOC --> NODES[GraphEntity nodes<br/>PERSON, ORG, DATE,<br/>GPE, CONCEPT, ...]
        NODES --> EDGES[GraphRelationship edges<br/>weighted by co-occurrence count]
        EDGES --> NX[NetworkX in-memory<br/>MultiDiGraph]
        NX --> PG[(PostgreSQL<br/>graph_entity<br/>graph_relationship)]
        NX --> HTML[pyvis HTML<br/>cdn_resources=remote]
    end

    subgraph Query["Multi-Hop Query Traversal"]
        Q[User Query] --> NER2[spaCy NER<br/>+ noun extraction]
        NER2 --> ILIKE[ILIKE fuzzy match<br/>candidates → DB seeds]
        ILIKE --> BFS1[Hop 1: fetch matching entities<br/>→ collect chunk_indices]
        BFS1 --> RELS[Follow outgoing + incoming<br/>relationships by entity ID]
        RELS --> BFS2[Hop 2: fetch neighbor entities<br/>→ collect more chunk_indices]
        BFS2 --> CIDX[Aggregated chunk_indices]
        CIDX --> FETCH[Fetch Chunk rows from DB<br/>score = 0.5 neutral]
    end

    PG --> ILIKE
```

**Entity types extracted by spaCy:**

`PERSON` · `ORG` · `GPE` (geo-political entity) · `DATE` · `MONEY` · `PERCENT` · `CARDINAL` · `CONCEPT` (custom)

---

## 8. Query Router Logic

```mermaid
flowchart TD
    Q[Raw Query] --> KS[Keyword Signal Scoring]
    Q --> VS[Vector Signal Scoring]

    KS --> K1["who/where/when/which (+2)"]
    KS --> K2["quoted phrases (+2)"]
    KS --> K3["years e.g. 2019 (+1)"]
    KS --> K4["numbers/currency (+1)"]
    KS --> K5["ALL-CAPS abbreviations (+1)"]

    VS --> V1["explain/describe/summarise (+2)"]
    VS --> V2["compare/contrast/difference (+2)"]
    VS --> V3["why/how does (+1)"]
    VS --> V4["long query >8 words (+1)"]

    K1 & K2 & K3 & K4 & K5 --> KW_SCORE[keyword_score]
    V1 & V2 & V3 & V4 --> VEC_SCORE[vector_score]

    KW_SCORE --> NORM[Normalise to 0..1]
    VEC_SCORE --> NORM

    NORM --> DECIDE{Decide Mode}
    DECIDE -->|kw ≥ 0.6| KEYWORD[KEYWORD MODE<br/>PostgreSQL FTS × 1.0<br/>Graph chunks × 0.6<br/>Vector × 0.4]
    DECIDE -->|vec ≥ 0.6| VECTOR[VECTOR MODE<br/>FAISS search × 1.0<br/>Graph chunks × 0.6<br/>Keyword × 0.4]
    DECIDE -->|balanced| HYBRID[HYBRID MODE<br/>Vector × 0.7<br/>Keyword × 0.7<br/>Graph × 0.6]
```

---

## 9. Directory Structure

```
pdf-insights/
├── app.py                        # Flask app factory, startup re-index hook
├── main.py                       # Gunicorn entry point
├── config.py                     # Central configuration constants
├── models.py                     # SQLAlchemy ORM models
│
├── rag/                          # Core RAG pipeline modules
│   ├── chunking.py               # Paragraph / sentence / sliding chunkers
│   ├── document_processor.py     # PDF extraction + processing orchestrator
│   ├── embeddings.py             # TF-IDF + random projection embeddings (persistent)
│   ├── generator.py              # Claude LLM answer generation
│   ├── keyword_search.py         # PostgreSQL FTS with OR-tsquery + ILIKE fallback
│   ├── knowledge_graph.py        # spaCy NER, NetworkX graph, PostgreSQL sync, BFS
│   ├── query_router.py           # Rule-based keyword / vector / hybrid routing
│   ├── retriever.py              # Retrieval orchestrator: route → graph → search → rerank
│   ├── vector_store.py           # FAISS IndexFlatIP wrapper with persistence
│   └── visualization.py          # Chunk distribution + pyvis knowledge graph rendering
│
├── routes/
│   ├── document_routes.py        # Upload, view, delete, KG view, KG API
│   └── query_routes.py           # Ask, search, history, multi-hop API
│
├── templates/
│   ├── layout.html               # Base Bootstrap layout
│   ├── index.html                # Home / landing page
│   ├── upload.html               # PDF upload form
│   ├── document_view.html        # Document detail + visualization
│   ├── query.html                # Query interface + answer display
│   ├── knowledge_graph.html      # Interactive graph iframe page
│   └── error.html                # Error page
│
├── vector_db/
│   ├── faiss_index/
│   │   └── faiss.index           # Binary FAISS index (auto-rebuilt on startup)
│   ├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer (shared across workers)
│   └── tfidf_projection.pkl      # Random projection matrix seed=42 (shared)
│
├── uploads/                      # Uploaded PDF files (served via /pdf/<filename>)
├── knowledge_graph/              # Per-document JSON graph snapshots (legacy)
└── static/                       # CSS, JS, favicon assets
```

---

## 10. Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_DIMENSION` | `768` | Output dimension for all embeddings |
| `CLAUDE_LLM_MODEL` | `claude-3-haiku-20240307` | Claude model for answer generation |
| `CLAUDE_EMBEDDING_MODEL` | `claude-3-haiku-20240307` | Claude model (unavailable — TF-IDF active) |
| `TOP_K_CHUNKS` | `5` | Number of chunks returned to generator |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine score to include a vector result |
| `RERANKING_ENABLED` | `True` | Blend original score with context-aware cosine rerank |
| `USE_COSINE_SIMILARITY` | `True` | Re-score FAISS results with true cosine similarity |
| `DEFAULT_CHUNK_SIZE` | `1000` | Target characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `200` | Character overlap between consecutive chunks |
| `DEFAULT_CHUNK_STRATEGY` | `paragraph` | `paragraph` / `sentence` / `sliding` |
| `MULTIHOP_HOPS` | `2` | Graph BFS depth |
| `MULTIHOP_MAX_NEIGHBORS` | `6` | Max edges to follow per node per hop |
| `MULTIHOP_CHUNK_LIMIT` | `60` | Max chunk indices collected via graph |
| `MAX_FILE_SIZE` | `35 MB` | Maximum PDF upload size |
| `ENTITY_RECOGNITION_CONFIDENCE` | `0.7` | spaCy NER confidence threshold |

---

## 11. API Endpoints

### Document Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/upload` | Upload form |
| `POST` | `/upload` | Upload and process PDF |
| `GET` | `/view/<doc_id>` | Document detail + chunk visualisation |
| `GET` | `/pdf/<filename>` | Serve raw PDF file |
| `GET` | `/chunks/<doc_id>` | JSON list of all chunks for a document |
| `POST` | `/delete/<doc_id>` | Delete document and all associated data |
| `GET` | `/visualization/<doc_id>` | Chunk embedding distribution (grid layout) |
| `GET` | `/graph/<doc_id>` | Interactive knowledge graph (pyvis) |
| `GET` | `/graph/api/entities/<doc_id>` | JSON: entities for a document |
| `GET` | `/graph/api/search?q=<term>` | JSON: entity search in graph |
| `POST` | `/graph/api/migrate` | Sync all in-memory graphs to PostgreSQL |

### Query Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Query interface home |
| `POST` | `/ask` | Submit a question; returns answer + source chunks |
| `POST` | `/search` | Raw retrieval (no generation); returns ranked chunks |
| `GET` | `/documents` | JSON: list all uploaded documents |
| `GET` | `/history` | Query history list |
| `POST` | `/graph/multihop` | JSON: multi-hop graph expansion for a query |

---

## 12. Setup & Running

### Prerequisites

- Python 3.11+
- PostgreSQL (connection string in `DATABASE_URL`)
- `ANTHROPIC_API_KEY` (for Claude answer generation)

### Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname
ANTHROPIC_API_KEY=sk-ant-...
SESSION_SECRET=<random-secret>
```

### Install & Run

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start (development)
python main.py

# Start (production, as configured)
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

### First-Run Behaviour

On startup the application:
1. Creates all PostgreSQL tables via SQLAlchemy (`db.create_all()`)
2. Checks for `vector_db/tfidf_vectorizer.pkl` on disk
3. **If missing and chunks exist in the DB** → automatically rebuilds the TF-IDF vectorizer (fitted on all stored chunks) and regenerates the FAISS index — no manual re-upload required

---

## 13. Technology Stack

| Layer | Technology |
|---|---|
| **Web framework** | Flask 3.x + Flask-Bootstrap |
| **WSGI server** | Gunicorn (sync workers) |
| **Database ORM** | SQLAlchemy + Flask-SQLAlchemy |
| **Database** | PostgreSQL (full-text search, JSON columns, ACID) |
| **Vector index** | FAISS `IndexFlatIP` (Facebook AI Similarity Search) |
| **Embeddings** | TF-IDF + random projection (seed=42, dim=768) via scikit-learn |
| **NLP / NER** | spaCy `en_core_web_sm` |
| **Graph library** | NetworkX (in-memory) + pyvis (visualisation, CDN resources) |
| **LLM** | Anthropic Claude claude-3-haiku-20240307 |
| **PDF extraction** | pdfminer.six (primary) + PyPDF2 (fallback) |
| **Frontend** | Bootstrap 4, Feather Icons, Chart.js |
| **Language** | Python 3.11 |
