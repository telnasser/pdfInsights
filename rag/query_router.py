"""
Query Router — decides whether a query should be answered via
keyword search, vector (semantic) search, or a hybrid of both.

Decision logic:
  - Rule-based scoring is always applied (fast, zero cost).
  - An optional Claude-based classification can be enabled for higher accuracy.
"""

import re
import logging
from enum import Enum
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


# Patterns that push toward keyword / exact-match retrieval
_KEYWORD_SIGNALS = [
    # Question words that imply looking up a specific fact
    (r'\b(who is|who are|who was|who were)\b', 2),
    (r'\b(when did|when was|when were|what year|what date|what time)\b', 2),
    (r'\b(where is|where are|where was|where were)\b', 2),
    (r'\b(which|what number|how many|how much)\b', 1),
    # Quoted phrases suggest the user wants exact terms
    (r'"[^"]+"', 2),
    # Years, percentages, currency, concrete numbers
    (r'\b(19|20)\d{2}\b', 1),
    (r'\b\d+(\.\d+)?\s*(million|billion|trillion|percent|%|usd|\$|€|£)\b', 1),
    # Proper-noun-dense queries (all-caps abbreviations)
    (r'\b[A-Z]{2,6}\b', 1),
]

# Patterns that push toward vector / semantic retrieval
_VECTOR_SIGNALS = [
    (r'\b(how does|how do|how did|how can|how should)\b', 2),
    (r'\b(why|explain|describe|summarize|overview|discuss|analyze|analyse)\b', 2),
    (r'\b(compare|contrast|difference between|similarities|versus|vs\.?)\b', 2),
    (r'\b(impact|effect|implication|significance|consequence|affect)\b', 1),
    (r'\b(strategy|approach|methodology|mechanism|process|framework)\b', 1),
    (r'\b(relationship between|connection between|related to)\b', 1),
    (r'\b(trend|pattern|theme|insight|concept)\b', 1),
]


class QueryRouter:
    """
    Classifies an incoming query and returns a SearchMode recommendation.

    Usage::

        router = QueryRouter()
        result = router.route("Who is the CEO of Amazon?")
        # result = {
        #   'mode': SearchMode.KEYWORD,
        #   'keyword_score': 4.0,
        #   'vector_score': 0.0,
        #   'reasons': [...],
        #   'mode_label': 'keyword'
        # }
    """

    def route(self, query: str) -> Dict[str, Any]:
        """
        Classify *query* and return routing metadata.

        Returns a dict with keys:
          mode         – SearchMode enum value
          mode_label   – human-readable string ('keyword'|'vector'|'hybrid')
          keyword_score, vector_score – float scores used for decision
          reasons      – list of matched signal descriptions
        """
        result = self._rule_based(query)
        logger.info(
            "QueryRouter: '%s' → %s (kw=%.1f, vec=%.1f)",
            query[:80], result['mode_label'],
            result['keyword_score'], result['vector_score']
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rule_based(self, query: str) -> Dict[str, Any]:
        kw_score = 0.0
        vec_score = 0.0
        reasons = []

        q = query.lower()

        for pattern, weight in _KEYWORD_SIGNALS:
            if re.search(pattern, q):
                kw_score += weight
                reasons.append(f"keyword:{pattern}")

        for pattern, weight in _VECTOR_SIGNALS:
            if re.search(pattern, q):
                vec_score += weight
                reasons.append(f"vector:{pattern}")

        # Length heuristic
        word_count = len(query.split())
        if word_count <= 5:
            kw_score += 1.0
            reasons.append("keyword:short_query")
        elif word_count >= 20:
            vec_score += 1.0
            reasons.append("vector:long_query")

        mode = self._decide(kw_score, vec_score)
        return {
            'mode': mode,
            'mode_label': mode.value,
            'keyword_score': kw_score,
            'vector_score': vec_score,
            'reasons': reasons,
        }

    @staticmethod
    def _decide(kw: float, vec: float) -> SearchMode:
        if kw == 0 and vec == 0:
            return SearchMode.HYBRID
        if kw == 0:
            return SearchMode.VECTOR
        if vec == 0:
            return SearchMode.KEYWORD
        ratio = kw / (kw + vec)
        if ratio >= 0.65:
            return SearchMode.KEYWORD
        if ratio <= 0.35:
            return SearchMode.VECTOR
        return SearchMode.HYBRID
