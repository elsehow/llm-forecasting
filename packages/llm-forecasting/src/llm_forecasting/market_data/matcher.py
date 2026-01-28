"""Market matching: find prediction market for a given question text."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import litellm
from pydantic import BaseModel, Field

from llm_forecasting.market_data.models import Market, MarketMatch, MarketStatus

if TYPE_CHECKING:
    from llm_forecasting.market_data.storage import MarketDataStorage

logger = logging.getLogger(__name__)

# Common stopwords to filter out of keyword extraction
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "will", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "or", "and", "but",
    "if", "than", "that", "this", "these", "those", "it", "its", "what",
    "which", "who", "whom", "how", "when", "where", "why", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "same", "so", "can", "could", "would", "should", "may",
    "might", "must", "shall", "win", "winner", "winning", "best", "before",
    "after", "during", "between", "above", "below", "up", "down", "out",
    "into", "over", "under", "again", "further", "then", "once", "yes", "no",
}


def extract_keywords(question: str) -> list[str]:
    """Extract meaningful search keywords from a question.

    Extracts:
    - Capitalized proper nouns/titles (e.g., "One Battle", "Best Picture")
    - Years (e.g., "2026", "2027")
    - Multi-word phrases in quotes
    - Significant words (after filtering stopwords)

    Args:
        question: The question text to extract keywords from

    Returns:
        List of keywords for search, ordered by likely importance
    """
    keywords = []

    # Extract quoted phrases first
    quoted = re.findall(r'"([^"]+)"', question)
    keywords.extend(quoted)

    # Remove quotes for further processing
    clean_text = re.sub(r'"[^"]+"', '', question)

    # Extract years (4-digit numbers starting with 19 or 20)
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', clean_text)
    keywords.extend(years)

    # Extract capitalized sequences (proper nouns, titles)
    # Match 2+ consecutive capitalized words
    proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', clean_text)
    keywords.extend(proper_nouns)

    # Extract single capitalized words that aren't at sentence start
    # Look for capitals not preceded by period/start
    single_caps = re.findall(r'(?<![.?!]\s)(?<!^)\b([A-Z][a-z]+)\b', clean_text)
    # Filter out common sentence-start words that might slip through
    single_caps = [w for w in single_caps if w.lower() not in STOPWORDS]
    keywords.extend(single_caps)

    # Extract remaining significant words
    words = re.findall(r'\b\w+\b', clean_text.lower())
    significant = [w for w in words if len(w) > 3 and w not in STOPWORDS]

    # Add significant words not already captured
    existing_lower = {k.lower() for k in keywords}
    for word in significant:
        if word not in existing_lower:
            keywords.append(word)
            existing_lower.add(word)

    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)

    return unique_keywords[:10]  # Cap at 10 keywords


class LLMMatchResponse(BaseModel):
    """Structured output for LLM market matching."""

    best_match_index: int | None = Field(
        description="Index (0-based) of the best matching market, or null if none match"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the match (0.0 = no match, 1.0 = exact match)"
    )
    reasoning: str = Field(
        description="Brief explanation of why this market matches (or why none match)"
    )


async def _llm_rerank(
    question: str,
    candidates: list[Market],
    model: str = "anthropic/claude-sonnet-4-20250514",
) -> tuple[int | None, float, str]:
    """Use LLM to select the best matching market from candidates.

    Args:
        question: The target question to match
        candidates: List of candidate markets from keyword search
        model: LiteLLM model identifier

    Returns:
        Tuple of (best_index, confidence, reasoning)
        best_index is None if no suitable match found
    """
    if not candidates:
        return None, 0.0, "No candidates to evaluate"

    # Build candidate list for prompt
    candidate_list = "\n".join(
        f"{i}. {m.title} (platform: {m.platform}, probability: {m.current_probability:.0%})"
        for i, m in enumerate(candidates)
        if m.current_probability is not None
    )

    prompt = f"""Select the prediction market that best matches this question.

Target question: {question}

Candidate markets:
{candidate_list}

Rules:
- The market must be about the SAME event/outcome as the target question
- Minor wording differences are OK if the underlying question is the same
- Return null for best_match_index if NONE of the candidates match the target
- Confidence should reflect how certain you are this is the same question

Return the index of the best match (or null if none match) with your confidence level."""

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format=LLMMatchResponse,
        )

        content = response.choices[0].message.content
        parsed = LLMMatchResponse.model_validate_json(content)

        return parsed.best_match_index, parsed.confidence, parsed.reasoning

    except Exception as e:
        logger.warning(f"LLM rerank failed: {e}, falling back to first candidate")
        # Fallback: return first candidate with low confidence
        return 0, 0.3, f"LLM rerank failed ({e}), using first keyword match"


async def find_matching_market(
    question: str,
    storage: MarketDataStorage,
    platforms: list[str] | None = None,
    min_liquidity: float | None = None,
    use_llm_rerank: bool = True,
    llm_model: str = "anthropic/claude-sonnet-4-20250514",
) -> MarketMatch | None:
    """Find the best matching prediction market for a question.

    Uses a two-stage approach:
    1. Keyword search to find candidates
    2. LLM reranking to select the best match

    Args:
        question: The question text to match
        storage: MarketDataStorage instance with cached markets
        platforms: List of platforms to search (None = all)
        min_liquidity: Minimum liquidity filter
        use_llm_rerank: Whether to use LLM to rerank results (default True)
        llm_model: Model to use for reranking

    Returns:
        MarketMatch if a match is found, None otherwise
    """
    keywords = extract_keywords(question)
    if not keywords:
        logger.warning(f"No keywords extracted from question: {question}")
        return None

    logger.debug(f"Extracted keywords: {keywords}")

    # Extract quoted entities (highest priority - these are the key subjects)
    quoted_entities = re.findall(r'"([^"]+)"', question)

    # Search each platform
    all_candidates: list[Market] = []
    platforms_to_search = platforms or ["polymarket", "metaculus", "kalshi"]

    for platform in platforms_to_search:
        try:
            # FIRST: Search for quoted entities (highest priority)
            # These are usually the actual subjects of the question
            for entity in quoted_entities:
                entity_results = await storage.search_by_title(
                    keywords=[entity],
                    platform=platform,
                    min_liquidity=min_liquidity,
                    status=MarketStatus.OPEN,
                    limit=10,
                )
                all_candidates.extend(entity_results)

            # SECOND: Search with all keywords for broader coverage
            # Keep original order (quoted first, then years, then proper nouns)
            candidates = await storage.search_by_title(
                keywords=keywords[:5],
                platform=platform,
                min_liquidity=min_liquidity,
                status=MarketStatus.OPEN,
                limit=15,
            )
            all_candidates.extend(candidates)
        except Exception as e:
            logger.warning(f"Search failed for platform {platform}: {e}")

    if not all_candidates:
        logger.debug(f"No candidates found for keywords: {keywords}")
        return None

    # Deduplicate by (platform, id)
    seen = set()
    unique_candidates = []
    for m in all_candidates:
        key = (m.platform, m.id)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(m)

    # Filter to markets with probability
    candidates_with_prob = [m for m in unique_candidates if m.current_probability is not None]
    if not candidates_with_prob:
        logger.debug("No candidates with probability found")
        return None

    if use_llm_rerank and len(candidates_with_prob) > 1:
        # LLM reranking
        best_idx, confidence, reasoning = await _llm_rerank(
            question, candidates_with_prob, llm_model
        )

        if best_idx is None:
            logger.debug(f"LLM found no match: {reasoning}")
            return None

        if best_idx >= len(candidates_with_prob):
            logger.warning(f"LLM returned invalid index {best_idx}, using 0")
            best_idx = 0

        best_market = candidates_with_prob[best_idx]
        return MarketMatch(
            market=best_market,
            match_confidence=confidence,
            match_method="llm_rerank",
        )
    else:
        # Single candidate or no LLM - use first match with modest confidence
        best_market = candidates_with_prob[0]
        return MarketMatch(
            market=best_market,
            match_confidence=0.5,  # Conservative since no LLM verification
            match_method="keyword",
        )


def compute_validation_status(gap_pp: float) -> str:
    """Compute validation status based on gap magnitude.

    Args:
        gap_pp: Gap in percentage points (computed - market) * 100

    Returns:
        Status string
    """
    abs_gap = abs(gap_pp)
    if abs_gap <= 5:
        return "OK"
    elif abs_gap <= 15:
        return f"WARNING - gap >{5}pp"
    else:
        return f"REVIEW - gap >{15}pp"
