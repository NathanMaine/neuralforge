"""Token estimation and budget allocation for the layered context engine.

Provides heuristic token counting (4 chars ~= 1 token) and a
budget allocator that distributes a max-token budget across the
four context layers.
"""
import logging

logger = logging.getLogger(__name__)

# Rough ratio: 1 token ~= 4 characters for English text
_CHARS_PER_TOKEN = 4

# Default budget splits (as fractions of total budget)
_DEFAULT_SPLITS = {
    0: 0.05,   # Layer 0: identity (~5%)
    1: 0.20,   # Layer 1: graph context (~20%)
    2: 0.45,   # Layer 2: compressed chunks (~45%)
    3: 0.30,   # Layer 3: deep search (~30%)
}


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses the heuristic that 1 token ~= 4 characters for English text.
    Returns 0 for empty strings.

    Parameters
    ----------
    text:
        Input text to estimate.

    Returns
    -------
    int
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def allocate_budget(
    max_tokens: int,
    max_layer: int = 3,
    identity_tokens: int = 0,
    graph_tokens: int = 0,
) -> dict[int, int]:
    """Allocate a token budget across context layers.

    The allocator first satisfies layers 0 and 1 if actual content is
    known (via ``identity_tokens`` / ``graph_tokens``), then distributes
    the remainder to layers 2 and 3.

    Parameters
    ----------
    max_tokens:
        Total token budget.
    max_layer:
        Highest layer to include (0-3).
    identity_tokens:
        Actual tokens already used by layer 0 (identity).
    graph_tokens:
        Actual tokens already used by layer 1 (graph context).

    Returns
    -------
    dict[int, int]
        Mapping of layer number to allocated token budget.
    """
    if max_tokens <= 0:
        return {i: 0 for i in range(max_layer + 1)}

    budget: dict[int, int] = {}
    remaining = max_tokens

    # Layer 0: identity (small, fixed)
    if max_layer >= 0:
        if identity_tokens > 0:
            l0 = min(identity_tokens, remaining)
        else:
            l0 = int(remaining * _DEFAULT_SPLITS[0])
        budget[0] = l0
        remaining -= l0
    else:
        return budget

    # Layer 1: graph context
    if max_layer >= 1:
        if graph_tokens > 0:
            l1 = min(graph_tokens, remaining)
        else:
            l1 = int(remaining * _DEFAULT_SPLITS[1])
        budget[1] = l1
        remaining -= l1

    # Distribute remainder between layers 2 and 3
    if max_layer >= 2 and remaining > 0:
        if max_layer >= 3:
            # Split remaining: 60% to layer 2, 40% to layer 3
            l2 = int(remaining * 0.60)
            l3 = remaining - l2
            budget[2] = l2
            budget[3] = l3
        else:
            # All remaining to layer 2
            budget[2] = remaining

    return budget


def fits_budget(text: str, budget_tokens: int) -> bool:
    """Check whether a text fits within a token budget.

    Parameters
    ----------
    text:
        Text to check.
    budget_tokens:
        Maximum allowed tokens.

    Returns
    -------
    bool
        ``True`` if the estimated token count is within budget.
    """
    return estimate_tokens(text) <= budget_tokens


def truncate_to_budget(text: str, budget_tokens: int) -> str:
    """Truncate text to fit within a token budget.

    Truncates at word boundaries when possible, appending ``...``
    when truncation occurs.

    Parameters
    ----------
    text:
        Text to truncate.
    budget_tokens:
        Maximum allowed tokens.

    Returns
    -------
    str
        Truncated text, or original if it fits.
    """
    if fits_budget(text, budget_tokens):
        return text

    max_chars = budget_tokens * _CHARS_PER_TOKEN
    if max_chars <= 3:
        return "..."

    truncated = text[:max_chars - 3]
    # Try to break at a word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]

    return truncated + "..."
