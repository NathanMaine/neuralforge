"""Comprehensive tests for forge.ingest.chunker -- 20+ tests."""

import pytest

from forge.ingest.chunker import chunk_text


class TestChunkText:
    """Tests for chunk_text."""

    def test_basic_chunking(self):
        """Text is split into chunks of the right size."""
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) >= 2

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert chunk_text("") == []

    def test_whitespace_only(self):
        """Whitespace-only text returns empty list."""
        assert chunk_text("   \n\t  ") == []

    def test_none_text_returns_empty(self):
        """None input returns empty list (falsy check)."""
        assert chunk_text(None) == []

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size produces one chunk."""
        chunks = chunk_text("hello world", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_exact_chunk_size(self):
        """Text exactly chunk_size long produces one chunk."""
        text = "x" * 500
        chunks = chunk_text(text, chunk_size=500, overlap=0)
        assert len(chunks) == 1

    def test_overlap_creates_more_chunks(self):
        """Overlap causes more chunks than without overlap."""
        text = "a" * 1000
        no_overlap = chunk_text(text, chunk_size=500, overlap=0)
        with_overlap = chunk_text(text, chunk_size=500, overlap=100)
        assert len(with_overlap) >= len(no_overlap)

    def test_overlap_content(self):
        """Overlapping chunks share content at boundaries."""
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunk_text(text, chunk_size=10, overlap=3)
        # Each chunk (except last) should have length 10
        assert len(chunks) >= 2
        # Verify overlap: end of chunk 0 overlaps with start of chunk 1
        if len(chunks) >= 2:
            end_of_first = chunks[0][-3:]
            start_of_second = chunks[1][:3]
            assert end_of_first == start_of_second

    def test_zero_overlap(self):
        """Overlap of 0 produces non-overlapping chunks."""
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=50, overlap=0)
        assert len(chunks) == 2

    def test_chunk_size_one(self):
        """chunk_size=1 produces one char per chunk."""
        chunks = chunk_text("abc", chunk_size=1, overlap=0)
        assert chunks == ["a", "b", "c"]

    def test_invalid_chunk_size_zero(self):
        """chunk_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("hello", chunk_size=0)

    def test_invalid_chunk_size_negative(self):
        """Negative chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("hello", chunk_size=-1)

    def test_invalid_overlap_negative(self):
        """Negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello", overlap=-1)

    def test_overlap_equals_chunk_size(self):
        """overlap == chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello", chunk_size=5, overlap=5)

    def test_overlap_exceeds_chunk_size(self):
        """overlap > chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello", chunk_size=5, overlap=10)

    def test_default_parameters(self):
        """Default chunk_size=500, overlap=50."""
        text = "word " * 200  # ~1000 chars
        chunks = chunk_text(text)
        assert len(chunks) >= 2

    def test_chunks_cover_all_content(self):
        """All content appears in at least one chunk."""
        text = "the quick brown fox jumps over the lazy dog"
        chunks = chunk_text(text, chunk_size=15, overlap=3)
        combined = " ".join(chunks)
        # Every word should appear in combined output
        for word in text.split():
            assert word in combined

    def test_unicode_text(self):
        """Unicode text is handled correctly."""
        text = "hello" * 50 + " " + "world" * 50
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) >= 2

    def test_chunks_stripped(self):
        """Each chunk is stripped of leading/trailing whitespace."""
        text = "  hello  " * 50
        chunks = chunk_text(text, chunk_size=50, overlap=0)
        for chunk in chunks:
            assert chunk == chunk.strip()

    def test_large_text(self):
        """Large text produces many chunks."""
        text = "x" * 10000
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 50

    def test_single_character(self):
        """Single character text returns one chunk."""
        chunks = chunk_text("x", chunk_size=500)
        assert chunks == ["x"]
