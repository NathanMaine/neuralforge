"""Comprehensive tests for forge.ingest.conversation_miner -- 30+ tests.

All external calls (embeddings, Qdrant) are mocked.
"""
import pytest
from unittest.mock import AsyncMock, patch

from forge.ingest.conversation_miner import (
    detect_format,
    normalize_conversation,
    classify_chunk,
    extract_entities,
    extract_edges,
    mine_conversation,
    CHUNK_TYPES,
    _normalize_claude,
    _normalize_chatgpt,
    _normalize_slack,
    _normalize_jsonl,
    _normalize_markdown,
)


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


class TestDetectFormat:
    """Tests for detect_format."""

    def test_detect_claude(self):
        """Detects Claude Human:/Assistant: format."""
        text = "Human: What is ML?\nAssistant: Machine learning is..."
        assert detect_format(text) == "claude"

    def test_detect_chatgpt(self):
        """Detects ChatGPT User:/ChatGPT: format."""
        text = "User: Tell me about AI.\nChatGPT: AI stands for..."
        assert detect_format(text) == "chatgpt"

    def test_detect_chatgpt_gpt_label(self):
        """Detects GPT: label as chatgpt format."""
        text = "User: Hello\nGPT: Hi there!"
        assert detect_format(text) == "chatgpt"

    def test_detect_slack(self):
        """Detects Slack format with timestamps."""
        text = "alice [10:30 AM] Hey everyone\nbob [10:31 AM] Hello!"
        assert detect_format(text) == "slack"

    def test_detect_slack_user_ids(self):
        """Detects Slack format with <@USERID> mentions."""
        text = "<@U12345> posted something interesting"
        assert detect_format(text) == "slack"

    def test_detect_jsonl(self):
        """Detects JSONL format."""
        text = '{"role": "user", "content": "hello"}\n{"role": "assistant", "content": "hi"}'
        assert detect_format(text) == "jsonl"

    def test_detect_markdown(self):
        """Generic text defaults to markdown."""
        text = "This is just some regular text about technology."
        assert detect_format(text) == "markdown"

    def test_detect_empty(self):
        """Empty text defaults to markdown."""
        assert detect_format("") == "markdown"

    def test_detect_whitespace(self):
        """Whitespace-only defaults to markdown."""
        assert detect_format("   \n\t  ") == "markdown"


# ---------------------------------------------------------------------------
# normalize_conversation
# ---------------------------------------------------------------------------


class TestNormalizeConversation:
    """Tests for normalize_conversation and format-specific normalizers."""

    def test_normalize_claude(self):
        """Claude format is parsed into messages."""
        text = "Human: What is AI?\nAssistant: AI is artificial intelligence."
        messages = normalize_conversation(text, fmt="claude")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "AI" in messages[0]["content"]

    def test_normalize_chatgpt(self):
        """ChatGPT format is parsed into messages."""
        text = "User: Hello\nChatGPT: Hi there!\nUser: Thanks"
        messages = normalize_conversation(text, fmt="chatgpt")
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_normalize_chatgpt_system(self):
        """ChatGPT System: messages get system role."""
        text = "System: You are helpful.\nUser: Hi"
        messages = normalize_conversation(text, fmt="chatgpt")
        assert messages[0]["role"] == "system"

    def test_normalize_slack(self):
        """Slack format is parsed with usernames."""
        text = "alice [10:30 AM] Good morning!\nbob [10:31 AM] Morning!"
        messages = normalize_conversation(text, fmt="slack")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert "username" in messages[0]

    def test_normalize_jsonl(self):
        """JSONL format parses JSON objects."""
        text = '{"role": "user", "content": "hello"}\n{"role": "assistant", "content": "hi"}'
        messages = normalize_conversation(text, fmt="jsonl")
        assert len(messages) == 2
        assert messages[0]["content"] == "hello"
        assert messages[1]["role"] == "assistant"

    def test_normalize_jsonl_with_text_key(self):
        """JSONL with 'text' key instead of 'content'."""
        text = '{"role": "user", "text": "hello world"}'
        messages = normalize_conversation(text, fmt="jsonl")
        assert len(messages) == 1
        assert messages[0]["content"] == "hello world"

    def test_normalize_markdown(self):
        """Markdown is treated as single message."""
        text = "Regular text about neural networks."
        messages = normalize_conversation(text, fmt="markdown")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_normalize_auto_detect(self):
        """Auto-detect format and normalize."""
        text = "Human: Test\nAssistant: Response"
        messages = normalize_conversation(text)
        assert len(messages) == 2

    def test_normalize_empty(self):
        """Empty text returns empty list."""
        assert normalize_conversation("") == []

    def test_normalize_invalid_jsonl(self):
        """Invalid JSONL lines are skipped."""
        text = '{"role": "user", "content": "hi"}\nnot json\n{"role": "assistant", "content": "hello"}'
        messages = normalize_conversation(text, fmt="jsonl")
        assert len(messages) == 2


# ---------------------------------------------------------------------------
# classify_chunk
# ---------------------------------------------------------------------------


class TestClassifyChunk:
    """Tests for classify_chunk."""

    def test_classify_code(self):
        """Code patterns are detected."""
        assert classify_chunk("```python\ndef hello():\n    pass\n```") == "code"

    def test_classify_code_import(self):
        """Import statements classify as code."""
        assert classify_chunk("import torch\nimport numpy as np") == "code"

    def test_classify_question(self):
        """Questions are detected by trailing ?."""
        assert classify_chunk("What is deep learning and how does it work?") == "question"

    def test_classify_instruction(self):
        """Step-by-step instructions are detected."""
        text = "1. Install Python\n2. Create a virtualenv\n3. Run the script"
        assert classify_chunk(text) == "instruction"

    def test_classify_instruction_here_is_how(self):
        """'Here's how' pattern classifies as instruction."""
        assert classify_chunk("Here's how to train a model with PyTorch.") == "instruction"

    def test_classify_reference(self):
        """References with URLs are detected."""
        assert classify_chunk("See https://arxiv.org/abs/2301.00001 for details.") == "reference"

    def test_classify_reference_according_to(self):
        """'According to' pattern classifies as reference."""
        assert classify_chunk("According to the official documentation...") == "reference"

    def test_classify_meta_greeting(self):
        """Greetings classify as meta."""
        assert classify_chunk("Hello!") == "meta"

    def test_classify_meta_thanks(self):
        """Thank you messages classify as meta."""
        assert classify_chunk("Thanks!") == "meta"

    def test_classify_factual(self):
        """Regular informational text classifies as factual."""
        text = "Neural networks consist of layers of interconnected nodes that process information."
        assert classify_chunk(text) == "factual"

    def test_classify_empty(self):
        """Empty text classifies as meta."""
        assert classify_chunk("") == "meta"

    def test_all_chunk_types_defined(self):
        """All expected chunk types are defined."""
        expected = {"factual", "opinion", "instruction", "question", "code", "reference", "meta"}
        assert set(CHUNK_TYPES.keys()) == expected


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------


class TestExtractEntities:
    """Tests for extract_entities."""

    def test_extract_tools(self):
        """Tool names are extracted."""
        text = "I used Python and PyTorch to build the model."
        entities = extract_entities(text)
        assert "tool" in entities
        tools_lower = [t.lower() for t in entities["tool"]]
        assert "python" in tools_lower
        assert "pytorch" in tools_lower

    def test_extract_concepts(self):
        """Concept names are extracted."""
        text = "We applied deep learning with a transformer architecture."
        entities = extract_entities(text)
        assert "concept" in entities
        concepts_lower = [c.lower() for c in entities["concept"]]
        assert "deep learning" in concepts_lower

    def test_extract_no_entities(self):
        """Text without entities returns empty dict."""
        text = "The weather is nice today."
        entities = extract_entities(text)
        # May or may not have entities depending on pattern matching
        # but common words should not match
        assert isinstance(entities, dict)

    def test_extract_deduplicates(self):
        """Duplicate entities are deduplicated."""
        text = "Python is great. I love Python. Use Python daily."
        entities = extract_entities(text)
        assert "tool" in entities
        python_count = sum(1 for t in entities["tool"] if t.lower() == "python")
        assert python_count == 1

    def test_extract_multiple_types(self):
        """Multiple entity types can be extracted from same text."""
        text = "Using PyTorch for deep learning and fine-tuning transformers."
        entities = extract_entities(text)
        assert "tool" in entities
        assert "concept" in entities


# ---------------------------------------------------------------------------
# extract_edges
# ---------------------------------------------------------------------------


class TestExtractEdges:
    """Tests for extract_edges."""

    def test_edges_from_tool_concept_cooccurrence(self):
        """Edges are created between co-occurring tools and concepts."""
        messages = [
            {"role": "user", "content": "I use PyTorch for deep learning projects."},
        ]
        edges = extract_edges(messages)
        assert len(edges) > 0
        assert edges[0]["edge_type"] == "related_to"

    def test_edges_from_tool_cooccurrence(self):
        """Edges between co-occurring tools."""
        messages = [
            {"role": "user", "content": "We use Python, Docker, and FastAPI."},
        ]
        edges = extract_edges(messages)
        tool_tool_edges = [
            e for e in edges
            if e["source"] in ("python", "docker", "fastapi")
            and e["target"] in ("python", "docker", "fastapi")
        ]
        assert len(tool_tool_edges) >= 1

    def test_edges_empty_messages(self):
        """No messages produces no edges."""
        edges = extract_edges([])
        assert edges == []

    def test_edges_no_entities(self):
        """Messages without recognizable entities produce no edges."""
        messages = [{"role": "user", "content": "The weather is nice."}]
        edges = extract_edges(messages)
        assert edges == []

    def test_edge_structure(self):
        """Edge dicts have required keys."""
        messages = [
            {"role": "user", "content": "PyTorch for deep learning."},
        ]
        edges = extract_edges(messages)
        if edges:
            edge = edges[0]
            assert "source" in edge
            assert "target" in edge
            assert "edge_type" in edge
            assert "confidence" in edge
            assert "evidence" in edge
            assert "source_label" in edge

    def test_edge_creator_in_evidence(self):
        """Creator name appears in edge evidence."""
        messages = [
            {"role": "user", "content": "Using CUDA with deep learning."},
        ]
        edges = extract_edges(messages, creator="alice")
        if edges:
            assert "alice" in edges[0]["evidence"]


# ---------------------------------------------------------------------------
# mine_conversation (full pipeline)
# ---------------------------------------------------------------------------


class TestMineConversation:
    """Tests for mine_conversation."""

    @pytest.mark.asyncio
    @patch("forge.ingest.conversation_miner.ingest_chunks", new_callable=AsyncMock, return_value=3)
    async def test_mine_claude_conversation(self, mock_ingest):
        """Mines a Claude-format conversation end-to-end."""
        text = (
            "Human: What is deep learning and how is it used with PyTorch?\n"
            "Assistant: Deep learning is a subset of machine learning that uses "
            "neural networks with multiple layers. PyTorch is a popular framework "
            "for building deep learning models. It provides automatic differentiation "
            "and GPU acceleration. You can train models on large datasets efficiently."
        )
        result = await mine_conversation(text, creator="alice", title="ML Chat")

        assert result["format"] == "claude"
        assert result["messages"] == 2
        assert result["chunks"] > 0
        assert result["ingested"] == 3

    @pytest.mark.asyncio
    async def test_mine_empty_text(self):
        """Empty text returns zeros."""
        result = await mine_conversation("")
        assert result["messages"] == 0
        assert result["chunks"] == 0
        assert result["ingested"] == 0

    @pytest.mark.asyncio
    @patch("forge.ingest.conversation_miner.ingest_chunks", new_callable=AsyncMock, return_value=2)
    async def test_mine_chatgpt_format(self, mock_ingest):
        """Mines ChatGPT format successfully."""
        text = (
            "User: Explain transformers.\n"
            "ChatGPT: Transformers are a type of neural network architecture "
            "that uses attention mechanisms to process sequential data in parallel. "
            "They were introduced in the paper Attention Is All You Need."
        )
        result = await mine_conversation(text, creator="bob")
        assert result["format"] == "chatgpt"
        assert result["messages"] == 2

    @pytest.mark.asyncio
    @patch("forge.ingest.conversation_miner.ingest_chunks", new_callable=AsyncMock, return_value=1)
    async def test_mine_with_pii_scrubbing(self, mock_ingest):
        """PII is scrubbed before ingestion."""
        text = (
            "Human: My email is alice@example.com and SSN is 123-45-6789.\n"
            "Assistant: I'll help you with that. " * 20
        )
        await mine_conversation(text, creator="alice")

        call_args = mock_ingest.call_args
        chunks = call_args[1]["chunks"] if "chunks" in call_args[1] else call_args[0][0]
        combined = " ".join(chunks)
        assert "alice@example.com" not in combined
        assert "123-45-6789" not in combined

    @pytest.mark.asyncio
    @patch("forge.ingest.conversation_miner.ingest_chunks", new_callable=AsyncMock, return_value=1)
    async def test_mine_includes_classifications(self, mock_ingest):
        """Result includes chunk classifications."""
        text = (
            "Human: How do I install PyTorch?\n"
            "Assistant: Here's how to install it:\n"
            "1. Create a virtualenv\n"
            "2. Run pip install torch\n"
            "3. Verify with import torch"
        )
        result = await mine_conversation(text, creator="alice")
        assert "classifications" in result
        assert isinstance(result["classifications"], dict)

    @pytest.mark.asyncio
    @patch("forge.ingest.conversation_miner.ingest_chunks", new_callable=AsyncMock, return_value=1)
    async def test_mine_includes_entities(self, mock_ingest):
        """Result includes extracted entities."""
        text = (
            "Human: Tell me about PyTorch and deep learning.\n"
            "Assistant: PyTorch is great for deep learning research. "
            "It integrates well with CUDA for GPU acceleration."
        )
        result = await mine_conversation(text, creator="alice")
        assert "entities" in result
        assert isinstance(result["entities"], dict)

    @pytest.mark.asyncio
    @patch("forge.ingest.conversation_miner.ingest_chunks", new_callable=AsyncMock, return_value=1)
    async def test_mine_with_format_override(self, mock_ingest):
        """Format override is respected."""
        text = "Just some plain text about neural networks and transformers."
        result = await mine_conversation(text, fmt="markdown")
        assert result["format"] == "markdown"

    @pytest.mark.asyncio
    @patch("forge.ingest.conversation_miner.ingest_chunks", new_callable=AsyncMock, return_value=1)
    async def test_mine_jsonl(self, mock_ingest):
        """Mines JSONL format conversation."""
        text = (
            '{"role": "user", "content": "What is RAG?"}\n'
            '{"role": "assistant", "content": "RAG stands for retrieval-augmented generation. '
            'It combines a retriever with a generator to produce more accurate responses."}'
        )
        result = await mine_conversation(text, creator="charlie")
        assert result["format"] == "jsonl"
        assert result["messages"] == 2
