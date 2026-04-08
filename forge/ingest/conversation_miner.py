"""Conversation miner -- extract knowledge from chat transcripts.

Supports Claude, ChatGPT, Slack, JSONL, and generic markdown formats.
Detects format automatically, normalizes to a common structure,
classifies chunks, extracts entities and graph edges, and ingests
via the batch upserter with PII scrubbing.
"""
import json
import logging
import re
from typing import Optional

from forge.core.utils import content_hash, now_iso
from forge.graph.models import EdgeSource, EdgeType
from forge.ingest.chunker import chunk_text
from forge.ingest.pii_scrubber import scrub_pii
from forge.ingest.upserter import ingest_chunks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

# Standard message structure after normalization
# {role: str, content: str, timestamp: Optional[str]}


def detect_format(text: str) -> str:
    """Detect the conversation format from raw text.

    Parameters
    ----------
    text:
        Raw conversation text.

    Returns
    -------
    str
        One of ``'claude'``, ``'chatgpt'``, ``'slack'``,
        ``'jsonl'``, ``'markdown'``.
    """
    if not text or not text.strip():
        return "markdown"

    # JSONL: lines of JSON objects
    lines = text.strip().splitlines()
    first_line = lines[0].strip()
    if first_line.startswith("{"):
        try:
            obj = json.loads(first_line)
            if isinstance(obj, dict):
                return "jsonl"
        except json.JSONDecodeError:
            pass

    # Claude format: "Human:" / "Assistant:" pairs
    if re.search(r"^\s*(Human|Assistant)\s*:", text, re.MULTILINE):
        return "claude"

    # ChatGPT format: "User:" / "ChatGPT:" or "GPT:" pairs
    if re.search(
        r"^\s*(User|ChatGPT|GPT|System)\s*:", text, re.MULTILINE
    ):
        return "chatgpt"

    # Slack format: timestamps with usernames like "<@U12345>" or "username [10:30 AM]"
    if re.search(
        r"<@[A-Z0-9]+>|^\w+\s*\[\d{1,2}:\d{2}", text, re.MULTILINE
    ):
        return "slack"

    return "markdown"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize_claude(text: str) -> list[dict]:
    """Parse Claude-style Human:/Assistant: conversation."""
    messages: list[dict] = []
    pattern = re.compile(r"^(Human|Assistant)\s*:\s*", re.MULTILINE)
    parts = pattern.split(text)

    # parts alternates: prefix, role, content, role, content, ...
    i = 1
    while i < len(parts) - 1:
        role = parts[i].strip().lower()
        content = parts[i + 1].strip()
        if role == "human":
            role = "user"
        elif role == "assistant":
            role = "assistant"
        if content:
            messages.append({"role": role, "content": content, "timestamp": None})
        i += 2

    return messages


def _normalize_chatgpt(text: str) -> list[dict]:
    """Parse ChatGPT-style User:/ChatGPT: conversation."""
    messages: list[dict] = []
    pattern = re.compile(
        r"^(User|ChatGPT|GPT|System)\s*:\s*", re.MULTILINE
    )
    parts = pattern.split(text)

    i = 1
    while i < len(parts) - 1:
        role = parts[i].strip().lower()
        content = parts[i + 1].strip()
        if role in ("chatgpt", "gpt"):
            role = "assistant"
        elif role == "user":
            role = "user"
        elif role == "system":
            role = "system"
        if content:
            messages.append({"role": role, "content": content, "timestamp": None})
        i += 2

    return messages


def _normalize_slack(text: str) -> list[dict]:
    """Parse Slack-style conversation with usernames and timestamps."""
    messages: list[dict] = []
    # Pattern: "username [HH:MM AM/PM]" or "<@USERID> text"
    pattern = re.compile(
        r"^(\w+)\s*\[(\d{1,2}:\d{2}\s*(?:AM|PM)?)\]\s*(.+?)$|"
        r"<@([A-Z0-9]+)>\s*(.+?)$",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        if match.group(1):
            username = match.group(1)
            timestamp = match.group(2)
            content = match.group(3).strip()
        else:
            username = match.group(4)
            timestamp = None
            content = match.group(5).strip()

        if content:
            messages.append({
                "role": "user",
                "content": content,
                "timestamp": timestamp,
                "username": username,
            })

    return messages


def _normalize_jsonl(text: str) -> list[dict]:
    """Parse JSONL conversation (one JSON object per line)."""
    messages: list[dict] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            role = obj.get("role", "user")
            content = obj.get("content", obj.get("text", obj.get("message", "")))
            if content:
                messages.append({
                    "role": role,
                    "content": str(content),
                    "timestamp": obj.get("timestamp"),
                })
        except json.JSONDecodeError:
            continue

    return messages


def _normalize_markdown(text: str) -> list[dict]:
    """Parse generic markdown text as a single-message conversation."""
    if not text or not text.strip():
        return []
    return [{"role": "user", "content": text.strip(), "timestamp": None}]


def normalize_conversation(text: str, fmt: Optional[str] = None) -> list[dict]:
    """Normalize raw conversation text into a list of message dicts.

    Parameters
    ----------
    text:
        Raw conversation text.
    fmt:
        Format override. If None, auto-detected via :func:`detect_format`.

    Returns
    -------
    list[dict]
        Each dict has ``role``, ``content``, and optionally ``timestamp``
        keys.
    """
    if not text or not text.strip():
        return []

    if fmt is None:
        fmt = detect_format(text)

    normalizers = {
        "claude": _normalize_claude,
        "chatgpt": _normalize_chatgpt,
        "slack": _normalize_slack,
        "jsonl": _normalize_jsonl,
        "markdown": _normalize_markdown,
    }

    normalizer = normalizers.get(fmt, _normalize_markdown)
    return normalizer(text)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Chunk type labels
CHUNK_TYPES = {
    "factual": "Factual information or data",
    "opinion": "Subjective opinion or recommendation",
    "instruction": "How-to, tutorial, or step-by-step guide",
    "question": "Question seeking information",
    "code": "Code snippet or technical implementation",
    "reference": "Citation or reference to external source",
    "meta": "Meta-conversation (greetings, acknowledgements)",
}

_CLASSIFICATION_PATTERNS = {
    "code": re.compile(r"```|def\s+\w+|class\s+\w+|import\s+\w+|function\s+\w+"),
    "question": re.compile(r"\?\s*$", re.MULTILINE),
    "instruction": re.compile(
        r"^\s*\d+[\.\)]\s|^step\s+\d+|^first,?\s|^then,?\s|you\s+should\s+|"
        r"make\s+sure\s+to|here'?s?\s+how",
        re.IGNORECASE | re.MULTILINE,
    ),
    "reference": re.compile(
        r"https?://|according\s+to|per\s+the|as\s+(stated|described)\s+in|"
        r"see\s+(also|the)|cited\s+in|source:",
        re.IGNORECASE,
    ),
    "meta": re.compile(
        r"^(hi|hello|hey|thanks|thank\s+you|bye|goodbye|sure|ok|okay|"
        r"you're\s+welcome|no\s+problem)\s*[.!]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
}


def classify_chunk(text: str) -> str:
    """Classify a text chunk into one of the known types.

    Parameters
    ----------
    text:
        The text chunk to classify.

    Returns
    -------
    str
        One of the keys from :data:`CHUNK_TYPES`.
    """
    if not text or not text.strip():
        return "meta"

    # Check patterns in priority order
    if _CLASSIFICATION_PATTERNS["code"].search(text):
        return "code"
    if _CLASSIFICATION_PATTERNS["meta"].search(text) and len(text.strip()) < 50:
        return "meta"
    if _CLASSIFICATION_PATTERNS["reference"].search(text):
        return "reference"
    if _CLASSIFICATION_PATTERNS["instruction"].search(text):
        return "instruction"
    if _CLASSIFICATION_PATTERNS["question"].search(text):
        return "question"

    return "factual"


# ---------------------------------------------------------------------------
# Entity and edge extraction
# ---------------------------------------------------------------------------

# Common entity patterns
_ENTITY_PATTERNS = {
    "tool": re.compile(
        r"\b(?:Python|JavaScript|TypeScript|Rust|Go|Docker|Kubernetes|"
        r"PyTorch|TensorFlow|React|Vue|FastAPI|Django|Flask|Git|"
        r"PostgreSQL|Redis|MongoDB|Kafka|Spark|Airflow|MLflow|"
        r"CUDA|Triton|Qdrant|NVIDIA|LangChain|LlamaIndex|"
        r"GPT|Claude|Gemini|Llama|Mistral|Qwen)\b",
        re.IGNORECASE,
    ),
    "concept": re.compile(
        r"\b(?:machine learning|deep learning|neural network|"
        r"natural language processing|NLP|computer vision|"
        r"reinforcement learning|transformer|attention mechanism|"
        r"fine[- ]tuning|embedding|vector database|RAG|"
        r"retrieval[- ]augmented generation|knowledge graph|"
        r"large language model|LLM|prompt engineering|"
        r"transfer learning|few[- ]shot|zero[- ]shot|"
        r"supervised learning|unsupervised learning|"
        r"gradient descent|backpropagation|convolution|"
        r"recurrent neural network|RNN|LSTM|GAN|VAE|"
        r"diffusion model|RLHF|DPO|LoRA|QLoRA)\b",
        re.IGNORECASE,
    ),
    "person": re.compile(
        r"\b(?:(?:Dr\.|Prof\.)\s+)?(?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+\b"
    ),
}


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract named entities from text.

    Parameters
    ----------
    text:
        Input text to scan.

    Returns
    -------
    dict[str, list[str]]
        Maps entity type to deduplicated list of entity names found.
    """
    entities: dict[str, list[str]] = {}

    for entity_type, pattern in _ENTITY_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # Deduplicate, preserving case of first occurrence
            seen: dict[str, str] = {}
            for m in matches:
                key = m.lower()
                if key not in seen:
                    seen[key] = m
            entities[entity_type] = list(seen.values())

    return entities


def extract_edges(
    messages: list[dict],
    creator: str = "conversation",
) -> list[dict]:
    """Extract potential graph edges from conversation messages.

    Looks for relationships between entities mentioned across messages,
    such as tools used with concepts, expert opinions, etc.

    Parameters
    ----------
    messages:
        Normalized conversation messages.
    creator:
        Creator/source label for edge attribution.

    Returns
    -------
    list[dict]
        Each dict has ``source``, ``target``, ``edge_type``,
        ``confidence``, ``evidence``, and ``source_label`` keys.
    """
    edges: list[dict] = []
    all_entities: dict[str, set[str]] = {}

    for msg in messages:
        msg_entities = extract_entities(msg.get("content", ""))
        for etype, names in msg_entities.items():
            if etype not in all_entities:
                all_entities[etype] = set()
            all_entities[etype].update(n.lower() for n in names)

    # Create edges between tools and concepts co-occurring
    tools = list(all_entities.get("tool", set()))
    concepts = list(all_entities.get("concept", set()))

    for tool in tools:
        for concept in concepts:
            edges.append({
                "source": tool,
                "target": concept,
                "edge_type": EdgeType.related_to.value,
                "confidence": 0.6,
                "evidence": f"Co-occurrence in conversation by {creator}",
                "source_label": EdgeSource.mined.value,
            })

    # Create edges between co-occurring tools
    for i, tool_a in enumerate(tools):
        for tool_b in tools[i + 1:]:
            edges.append({
                "source": tool_a,
                "target": tool_b,
                "edge_type": EdgeType.related_to.value,
                "confidence": 0.5,
                "evidence": f"Co-mentioned in conversation by {creator}",
                "source_label": EdgeSource.mined.value,
            })

    return edges


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def mine_conversation(
    text: str,
    creator: str = "conversation",
    title: str = "",
    source: str = "",
    fmt: Optional[str] = None,
) -> dict:
    """Full mining pipeline: normalize, classify, extract, scrub, ingest.

    Parameters
    ----------
    text:
        Raw conversation text.
    creator:
        Expert/creator name for the ingested chunks.
    title:
        Conversation title.
    source:
        Source file path or URL.
    fmt:
        Format override (auto-detected if None).

    Returns
    -------
    dict
        Summary with ``format``, ``messages``, ``chunks``,
        ``ingested``, ``entities``, ``edges``, and ``classifications``
        keys.
    """
    if not text or not text.strip():
        return {
            "format": "unknown",
            "messages": 0,
            "chunks": 0,
            "ingested": 0,
            "entities": {},
            "edges": 0,
            "classifications": {},
        }

    # Detect and normalize
    detected_format = fmt or detect_format(text)
    messages = normalize_conversation(text, fmt=detected_format)

    if not messages:
        return {
            "format": detected_format,
            "messages": 0,
            "chunks": 0,
            "ingested": 0,
            "entities": {},
            "edges": 0,
            "classifications": {},
        }

    # Combine message content for processing
    combined_text = "\n\n".join(msg["content"] for msg in messages)

    # PII scrub
    scrubbed_text, pii_counts = scrub_pii(combined_text)
    if pii_counts:
        logger.info("Scrubbed PII from conversation: %s", pii_counts)

    # Chunk
    chunks = chunk_text(scrubbed_text)

    # Classify each chunk
    classifications: dict[str, int] = {}
    for chunk in chunks:
        chunk_type = classify_chunk(chunk)
        classifications[chunk_type] = classifications.get(chunk_type, 0) + 1

    # Extract entities and edges
    entities = extract_entities(scrubbed_text)
    edges = extract_edges(messages, creator=creator)

    # Ingest
    ingested = 0
    if chunks:
        ingested = await ingest_chunks(
            chunks=chunks,
            creator=creator,
            title=title or f"Conversation ({detected_format})",
            source=source,
            source_type="conversation",
        )

    return {
        "format": detected_format,
        "messages": len(messages),
        "chunks": len(chunks),
        "ingested": ingested,
        "entities": entities,
        "edges": len(edges),
        "classifications": classifications,
    }
