"""
Retriever — orchestrates the full retrieval pipeline.

Pipeline
────────
1. QueryRouter  → decide: KEYWORD | VECTOR | HYBRID
2. KnowledgeGraph.multi_hop_search_db  → entity-seeded chunk indices
3. Based on mode:
     KEYWORD  → KeywordSearch  (PostgreSQL FTS)
     VECTOR   → FAISS cosine-similarity search
     HYBRID   → both, merged by score
4. Graph-seeded chunks always appended (de-duplicated)
5. Optional reranking via refined embedding blend
"""

import logging
from typing import List, Dict, Any, Optional, Set

import numpy as np

from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.query_router import QueryRouter, SearchMode
from rag.keyword_search import KeywordSearch
from config import TOP_K_CHUNKS, RERANKING_ENABLED, SIMILARITY_THRESHOLD, USE_COSINE_SIMILARITY

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant document chunks for a given query using a three-stage
    pipeline: query routing → multi-hop graph expansion → search execution.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        knowledge_graph=None,
        top_k: int = TOP_K_CHUNKS,
        reranking: bool = RERANKING_ENABLED,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        use_cosine_similarity: bool = USE_COSINE_SIMILARITY,
    ):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.knowledge_graph = knowledge_graph
        self.top_k = top_k
        self.reranking = reranking
        self.similarity_threshold = similarity_threshold
        self.use_cosine_similarity = use_cosine_similarity

        self.router = QueryRouter()
        self.keyword_search = KeywordSearch()

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Main entry point.  Returns a ranked list of chunk dicts augmented with:
          score        – relevance score in [0, 1]
          search_type  – 'vector' | 'keyword' | 'graph'
          hop_info     – (optional) which graph hop sourced this chunk
        """
        if not query:
            logger.warning("Empty query")
            return []

        k = top_k if top_k is not None else self.top_k
        logger.info("Retriever.retrieve: query='%s' doc_id=%s top_k=%d", query[:80], doc_id, k)

        # ── Stage 1: route the query ───────────────────────────────────
        routing = self.router.route(query)
        mode: SearchMode = routing['mode']
        logger.info("QueryRouter decision: %s (kw=%.1f, vec=%.1f)",
                    mode.value, routing['keyword_score'], routing['vector_score'])

        # ── Stage 2: execute searches FIRST ───────────────────────────
        # Run search before graph so we can seed graph BFS from results.
        vector_results: List[Dict[str, Any]] = []
        keyword_results: List[Dict[str, Any]] = []

        if mode in (SearchMode.VECTOR, SearchMode.HYBRID):
            vector_results = self._vector_search(query, doc_id, k * 3)

        if mode in (SearchMode.KEYWORD, SearchMode.HYBRID):
            keyword_results = self.keyword_search.search(query, doc_id=doc_id, top_k=k * 2)

        # ── Stage 3a: graph expansion seeded by QUERY ─────────────────
        # Works when the query contains named entities (e.g. "Infoblox").
        graph_chunk_indices: List[int] = []
        graph_metadata: Dict[str, Any] = {'all_entities': []}
        if self.knowledge_graph is not None:
            try:
                graph_result = self.knowledge_graph.multi_hop_search_db(
                    query, hops=2, doc_id=doc_id
                )
                graph_chunk_indices = graph_result.get('chunk_indices', [])
                graph_metadata = graph_result
                logger.info("Graph (query-seeded): %d indices, entities=%s",
                            len(graph_chunk_indices),
                            [e['entity'] for e in graph_result.get('all_entities', [])[:5]])
            except Exception as exc:
                logger.error("Graph multi-hop (query) error: %s", exc, exc_info=True)

        # ── Stage 3b: graph expansion seeded by TOP SEARCH RESULTS ────
        # This is the key path for queries like "what streaming service did he
        # build?" — keyword/vector finds the Shahid.net chunk, then graph BFS
        # from entities IN that chunk surfaces related context chunks.
        if self.knowledge_graph is not None:
            top_initial = (vector_results + keyword_results)[:3]
            if top_initial:
                try:
                    combined_text = " ".join(r['text'][:500] for r in top_initial)
                    graph_result2 = self.knowledge_graph.multi_hop_search_db(
                        combined_text, hops=1, doc_id=doc_id
                    )
                    extra_indices = graph_result2.get('chunk_indices', [])
                    existing_set: Set[int] = set(graph_chunk_indices)
                    for idx in extra_indices:
                        if idx not in existing_set:
                            graph_chunk_indices.append(idx)
                            existing_set.add(idx)
                    extra_ents = graph_result2.get('all_entities', [])
                    logger.info("Graph (result-seeded): +%d indices via entities=%s",
                                len(extra_indices),
                                [e['entity'] for e in extra_ents[:5]])
                    graph_metadata.setdefault('all_entities', []).extend(extra_ents)
                except Exception as exc:
                    logger.error("Graph multi-hop (results) error: %s", exc, exc_info=True)

        # ── Stage 4: fetch graph-sourced chunks from DB ───────────────
        graph_chunks = self._fetch_chunks_by_indices(graph_chunk_indices, doc_id)

        # ── Stage 5: merge & de-duplicate ─────────────────────────────
        merged = self._merge_results(
            vector_results, keyword_results, graph_chunks, mode, k
        )

        # ── Stage 6: rerank ───────────────────────────────────────────
        if self.reranking and len(merged) > 1:
            try:
                merged = self._rerank_results(query, merged)
            except Exception as exc:
                logger.error("Reranking error: %s", exc, exc_info=True)

        # Annotate with routing information
        for chunk in merged:
            chunk['routing'] = {
                'mode': mode.value,
                'keyword_score': routing['keyword_score'],
                'vector_score': routing['vector_score'],
                'graph_entities': [e['entity'] for e in
                                   graph_metadata.get('all_entities', [])[:10]],
            }

        logger.info("Retriever: returning %d chunks (mode=%s)", len(merged), mode.value)
        return merged

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _vector_search(
        self, query: str, doc_id: Optional[str], k: int
    ) -> List[Dict[str, Any]]:
        """Run FAISS vector search and return results tagged as 'vector'."""
        try:
            query_embedding = self.embedding_generator.embed_query(query)
            if query_embedding is None:
                return []

            search_k = k * 5 if self.use_cosine_similarity else k
            results = self.vector_store.search(query_embedding, doc_id, search_k)

            if self.use_cosine_similarity and results:
                # Re-score using true cosine similarity
                for r in results:
                    try:
                        re = self.embedding_generator.generate_embeddings([r['text']])[0]
                        r['score'] = float(self._cosine_similarity(query_embedding, re))
                    except Exception:
                        pass
                results.sort(key=lambda x: x.get('score', 0), reverse=True)

            filtered = [r for r in results if r.get('score', 0) > self.similarity_threshold]
            filtered = filtered[:k]

            for r in filtered:
                r.setdefault('search_type', 'vector')

            return filtered
        except Exception as exc:
            logger.error("Vector search error: %s", exc, exc_info=True)
            return []

    def _fetch_chunks_by_indices(
        self, chunk_indices: List[int], doc_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Given a list of chunk_index values, fetch matching Chunk rows from
        PostgreSQL and return them in the standard chunk-dict format.
        """
        if not chunk_indices:
            return []

        try:
            from models import Chunk
            q = Chunk.query.filter(Chunk.chunk_index.in_(chunk_indices))
            if doc_id:
                q = q.filter(Chunk.document_id == doc_id)
            db_chunks = q.all()

            results = []
            for c in db_chunks:
                results.append({
                    'id': c.id,
                    'document_id': c.document_id,
                    'chunk_index': c.chunk_index,
                    'text': c.text,
                    'page_num': c.page_num,
                    'score': 0.3,       # graph-sourced; kept below keyword/vector baseline
                    'search_type': 'graph',
                })
            return results
        except Exception as exc:
            logger.error("_fetch_chunks_by_indices error: %s", exc, exc_info=True)
            return []

    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        graph_chunks: List[Dict[str, Any]],
        mode: SearchMode,
        k: int,
    ) -> List[Dict[str, Any]]:
        """
        Merge the three result streams, de-duplicate by (document_id, chunk_index),
        and return the top-k by blended score.
        """
        seen: Set[tuple] = set()
        merged: List[Dict[str, Any]] = []

        def _add(chunks: List[Dict[str, Any]], weight: float):
            for c in chunks:
                key = (c.get('document_id', ''), c.get('chunk_index', -1))
                if key not in seen:
                    seen.add(key)
                    c = dict(c)
                    c['blended_score'] = weight * float(c.get('score', 0))
                    merged.append(c)
                else:
                    # Boost the existing entry if we've seen it via another path
                    for m in merged:
                        mk = (m.get('document_id', ''), m.get('chunk_index', -1))
                        if mk == key:
                            m['blended_score'] = min(
                                1.0,
                                m.get('blended_score', 0) + weight * 0.3
                            )
                            break

        if mode == SearchMode.KEYWORD:
            _add(keyword_results, 1.0)
            _add(graph_chunks, 0.45)
            _add(vector_results, 0.4)
        elif mode == SearchMode.VECTOR:
            _add(vector_results, 1.0)
            _add(graph_chunks, 0.45)
            _add(keyword_results, 0.4)
        else:  # HYBRID
            _add(vector_results, 0.7)
            _add(keyword_results, 0.7)
            _add(graph_chunks, 0.45)

        merged.sort(key=lambda x: x.get('blended_score', 0), reverse=True)
        # Normalise score field to blended_score for downstream compatibility
        for m in merged:
            m['score'] = m.get('blended_score', m.get('score', 0))

        return merged[:k]

    def _rerank_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Refine ranking using a context-enriched query vector blended with
        original scores (0.7 original + 0.3 refined).
        """
        try:
            context_text = query + " " + " ".join(r['text'][:100] for r in results[:2])
            refined_emb = self.embedding_generator.embed_query(context_text)
            if refined_emb is None:
                return results

            for r in results:
                try:
                    r_emb = self.embedding_generator.generate_embeddings([r['text']])[0]
                    sim = float(self._cosine_similarity(refined_emb, r_emb))
                    r['score'] = 0.7 * r.get('score', 0) + 0.3 * sim
                    r['reranked'] = True
                except Exception:
                    pass

            results.sort(key=lambda x: x.get('score', 0), reverse=True)
        except Exception as exc:
            logger.error("_rerank_results error: %s", exc, exc_info=True)

        return results

    @staticmethod
    def _cosine_similarity(v1, v2) -> float:
        try:
            a = np.array(v1, dtype=np.float64)
            b = np.array(v2, dtype=np.float64)
            n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
            if n1 == 0 or n2 == 0:
                return 0.0
            return float(np.dot(a, b) / (n1 * n2))
        except Exception:
            return 0.0

    # ──────────────────────────────────────────────────────────────────
    # Legacy helper (kept for backward compatibility)
    # ──────────────────────────────────────────────────────────────────

    def retrieve_with_context_window(
        self,
        query: str,
        doc_id: Optional[str] = None,
        window_size: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks and augment each with *window_size* surrounding chunks
        for additional context.
        """
        initial_results = self.retrieve(query, doc_id)
        if not initial_results:
            return []

        if not doc_id and initial_results:
            doc_id = initial_results[0].get('document_id')
        if not doc_id:
            return initial_results

        generic_emb = self.embedding_generator.embed_query("document content")
        all_chunks = self.vector_store.search(generic_emb, doc_id, 100)
        all_chunks_sorted = sorted(all_chunks, key=lambda x: x.get('chunk_index', 0))
        chunk_positions = {c.get('chunk_index'): i for i, c in enumerate(all_chunks_sorted)}

        results_with_context = []
        for result in initial_results:
            pos = chunk_positions.get(result.get('chunk_index'))
            if pos is None:
                results_with_context.append(result)
                continue

            before = [all_chunks_sorted[i] for i in range(max(0, pos - window_size), pos)]
            after = [all_chunks_sorted[i]
                     for i in range(pos + 1, min(len(all_chunks_sorted), pos + window_size + 1))]

            ctx = ""
            for c in before:
                ctx += c['text'] + "\n\n"
            ctx += "--- RELEVANT CHUNK ---\n" + result['text'] + "\n--- END ---\n\n"
            for c in after:
                ctx += c['text'] + "\n\n"

            r = result.copy()
            r['text_with_context'] = ctx
            r['context_chunks'] = before + after
            results_with_context.append(r)

        return results_with_context
