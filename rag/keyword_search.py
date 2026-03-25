"""
Keyword Search — BM25-style retrieval backed by PostgreSQL full-text search.

PostgreSQL's `to_tsvector` / `plainto_tsquery` pipeline gives us:
  • Stemming and stop-word removal
  • `ts_rank_cd` for BM25-like relevance scoring
  • Optional document-level filtering

Results are returned in the same dict shape as FAISS vector results so
the retriever can blend them transparently.
"""

import logging
from typing import List, Dict, Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


class KeywordSearch:
    """
    Full-text keyword search over the ``chunk`` table using PostgreSQL's
    built-in text-search features.
    """

    # Language configuration for PostgreSQL text search
    TS_LANG = "english"

    def search(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Return up to *top_k* chunks that best match *query* using full-text
        search.  Each result dict contains the same keys as FAISS results plus
        ``search_type = 'keyword'``.

        Args:
            query:  Raw user query text.
            doc_id: When provided, restricts search to that document.
            top_k:  Maximum results to return.

        Returns:
            List of chunk dicts ordered by relevance (descending).
        """
        if not query or not query.strip():
            return []

        try:
            from app import db  # local import to avoid circular deps

            params: Dict[str, Any] = {
                "query": query.strip(),
                "lang": self.TS_LANG,
                "top_k": top_k,
            }

            doc_filter = ""
            if doc_id:
                doc_filter = "AND c.document_id = :doc_id"
                params["doc_id"] = doc_id

            sql = text(f"""
                SELECT
                    c.id,
                    c.document_id,
                    c.chunk_index,
                    c.text,
                    c.page_num,
                    ts_rank_cd(
                        to_tsvector(:lang, c.text),
                        plainto_tsquery(:lang, :query),
                        32
                    ) AS rank
                FROM chunk c
                WHERE
                    to_tsvector(:lang, c.text) @@
                    plainto_tsquery(:lang, :query)
                    {doc_filter}
                ORDER BY rank DESC
                LIMIT :top_k
            """)

            rows = db.session.execute(sql, params).fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row.id,
                    "document_id": row.document_id,
                    "chunk_index": row.chunk_index,
                    "text": row.text,
                    "page_num": row.page_num,
                    # Normalise to [0, 1] – ts_rank_cd returns values in [0, 1]
                    # already, but cap just in case.
                    "score": min(float(row.rank), 1.0),
                    "search_type": "keyword",
                })

            logger.info(
                "KeywordSearch: query='%s' doc_id=%s → %d results",
                query[:60], doc_id, len(results)
            )
            return results

        except Exception as exc:
            logger.error("KeywordSearch error: %s", exc, exc_info=True)
            return []

    def search_by_terms(
        self,
        terms: List[str],
        doc_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper that joins a list of entity/keyword terms with OR
        and performs a full-text search.  Useful for graph-expanded queries.
        """
        if not terms:
            return []
        combined = " | ".join(terms[:20])  # tsquery OR operator
        try:
            from app import db

            params: Dict[str, Any] = {
                "lang": self.TS_LANG,
                "query": combined,
                "top_k": top_k,
            }

            doc_filter = ""
            if doc_id:
                doc_filter = "AND c.document_id = :doc_id"
                params["doc_id"] = doc_id

            sql = text(f"""
                SELECT
                    c.id,
                    c.document_id,
                    c.chunk_index,
                    c.text,
                    c.page_num,
                    ts_rank_cd(
                        to_tsvector(:lang, c.text),
                        to_tsquery(:lang, :query),
                        32
                    ) AS rank
                FROM chunk c
                WHERE
                    to_tsvector(:lang, c.text) @@
                    to_tsquery(:lang, :query)
                    {doc_filter}
                ORDER BY rank DESC
                LIMIT :top_k
            """)

            rows = db.session.execute(sql, params).fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row.id,
                    "document_id": row.document_id,
                    "chunk_index": row.chunk_index,
                    "text": row.text,
                    "page_num": row.page_num,
                    "score": min(float(row.rank), 1.0),
                    "search_type": "keyword",
                })
            return results

        except Exception as exc:
            logger.error("KeywordSearch.search_by_terms error: %s", exc, exc_info=True)
            return []
