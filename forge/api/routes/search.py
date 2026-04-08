"""Search API route with layered context + BM25 keyword boost."""
import logging
import re
from typing import Optional

from fastapi import APIRouter, Query

from forge.config import (
    BM25_MAX_KEYWORDS,
    BM25_MIN_KEYWORD_LEN,
    SEARCH_CANDIDATE_MULTIPLIER,
    SEARCH_DEFAULT_LIMIT,
)

router = APIRouter(prefix="/api", tags=["search"])
logger = logging.getLogger(__name__)

# Common English stop words to exclude from BM25 keywords
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "but", "and", "or", "if",
    "while", "about", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "it", "its", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their",
}


def _extract_keywords(query: str) -> list[str]:
    """Extract BM25-suitable keywords from a query string."""
    words = re.findall(r"\w+", query.lower())
    keywords = [
        w for w in words
        if len(w) >= BM25_MIN_KEYWORD_LEN and w not in _STOP_WORDS
    ]
    return keywords[:BM25_MAX_KEYWORDS]


@router.get("/search")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(SEARCH_DEFAULT_LIMIT, ge=1, le=100),
    expert: Optional[str] = Query(None, description="Filter by expert name"),
    max_tokens: int = Query(4000, ge=100, le=32000),
    layers: int = Query(3, ge=0, le=3, description="Max layer depth (0-3)"),
    compress: bool = Query(True, description="Apply AAAK compression to layer 2"),
):
    """Search the knowledge base with layered context and BM25 keyword boost.

    Returns both raw search results and an assembled layered context
    suitable for LLM consumption.
    """
    from forge.api.main import get_graph_engine
    from forge.core import embeddings, qdrant_client
    from forge.layers.engine import get_context

    engine = get_graph_engine()

    # --- Vector search function ---
    async def _search_fn(query: str, limit: int = 10, expert: str | None = None):
        vec = await embeddings.get_embedding(query)
        if vec is None:
            return []
        return qdrant_client.search_vectors(vec, limit=limit, expert=expert)

    # --- Deep search (more candidates, no compression) ---
    async def _deep_search_fn(query: str, limit: int = 5, expert: str | None = None):
        vec = await embeddings.get_embedding(query)
        if vec is None:
            return []
        candidates = limit * SEARCH_CANDIDATE_MULTIPLIER
        return qdrant_client.search_vectors(vec, limit=candidates, expert=expert)

    # --- Build layered context ---
    ctx = await get_context(
        query=q,
        max_tokens=max_tokens,
        max_layer=layers,
        expert_filter=expert,
        compress_chunks=compress,
        graph_engine=engine,
        search_fn=_search_fn,
        deep_search_fn=_deep_search_fn,
    )

    # --- BM25 keyword boost ---
    keywords = _extract_keywords(q)
    bm25_results: list[dict] = []
    if keywords:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            BM25Okapi = None

        if BM25Okapi is not None:
            # Get raw search results for BM25 scoring
            raw_results = await _search_fn(q, limit=limit * SEARCH_CANDIDATE_MULTIPLIER, expert=expert)
            if raw_results:
                corpus = [r.get("text", "") for r in raw_results]
                tokenized = [doc.lower().split() for doc in corpus]
                bm25 = BM25Okapi(tokenized)
                scores = bm25.get_scores(keywords)
                for i, result in enumerate(raw_results):
                    result["bm25_score"] = float(scores[i])
                    result["combined_score"] = (
                        result.get("score", 0.0) * 0.7 + float(scores[i]) * 0.3
                    )
                raw_results.sort(key=lambda r: r.get("combined_score", 0), reverse=True)
                bm25_results = raw_results[:limit]

    # --- Raw search results (without BM25) ---
    raw_results = await _search_fn(q, limit=limit, expert=expert)

    return {
        "query": q,
        "keywords": keywords,
        "results": bm25_results if bm25_results else raw_results,
        "context": {
            "layer_0": ctx.layer_0,
            "layer_1": ctx.layer_1,
            "layer_2": ctx.layer_2,
            "layer_3": ctx.layer_3,
            "total_tokens": ctx.total_tokens,
            "layers_used": ctx.layers_used,
            "experts_referenced": ctx.experts_referenced,
            "assembled": ctx.as_text(),
        },
    }
