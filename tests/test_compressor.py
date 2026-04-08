"""Tests for AAAK fact-preserving compression -- 25+ tests.

Covers all compression levels, filler removal, abbreviations,
code/URL protection, confidence qualifiers, attribution preservation,
deduplication, ratios, and edge cases.
"""
import pytest

from forge.layers.compressor import (
    compress,
    compression_ratio,
    estimate_savings,
)


# ===================================================================
# Level 0 -- no compression
# ===================================================================

class TestLevel0:
    """Level 0 should be pure passthrough."""

    def test_level_0_passthrough(self):
        text = "This is a test. It is important to note that nothing changes."
        assert compress(text, level=0) == text

    def test_level_0_empty_string(self):
        assert compress("", level=0) == ""

    def test_level_0_whitespace_only(self):
        assert compress("   ", level=0) == "   "

    def test_negative_level_passthrough(self):
        text = "basically, this should not change"
        assert compress(text, level=-1) == text


# ===================================================================
# Level 1 -- filler removal + abbreviations
# ===================================================================

class TestLevel1Filler:
    """Level 1 should remove filler words and phrases."""

    def test_removes_basically(self):
        result = compress("This is basically correct.", level=1)
        assert "basically" not in result.lower()
        assert "correct" in result

    def test_removes_essentially(self):
        result = compress("It is essentially a wrapper.", level=1)
        assert "essentially" not in result.lower()
        assert "wrapper" in result

    def test_removes_it_is_important_to_note_that(self):
        result = compress(
            "It is important to note that GPUs are fast.", level=1
        )
        assert "important to note" not in result.lower()
        assert "GPUs are fast" in result

    def test_removes_in_order_to(self):
        result = compress("In order to train the model, use GPUs.", level=1)
        assert "in order to" not in result.lower()
        assert "train" in result

    def test_removes_needless_to_say(self):
        result = compress("Needless to say, this is correct.", level=1)
        assert "needless to say" not in result.lower()
        assert "correct" in result

    def test_preserves_factual_content(self):
        text = "The model achieves 95.3% accuracy on ImageNet."
        result = compress(text, level=1)
        assert "95.3%" in result
        assert "ImageNet" in result

    def test_multiple_fillers_removed(self):
        text = "Basically, it is important to note that obviously this works."
        result = compress(text, level=1)
        assert "basically" not in result.lower()
        assert "important to note" not in result.lower()
        assert "obviously" not in result.lower()
        assert "works" in result


class TestLevel1Abbreviations:
    """Level 1 should apply standard abbreviations."""

    def test_for_example(self):
        result = compress("For example, PyTorch supports CUDA.", level=1)
        assert "e.g." in result
        assert "PyTorch" in result

    def test_for_instance(self):
        result = compress("For instance, use Adam optimizer.", level=1)
        assert "e.g." in result

    def test_machine_learning(self):
        result = compress("Machine learning is a subset of AI.", level=1)
        assert "ML" in result

    def test_natural_language_processing(self):
        result = compress("Natural language processing requires data.", level=1)
        assert "NLP" in result

    def test_large_language_model(self):
        result = compress("A large language model can generate text.", level=1)
        assert "LLM" in result

    def test_preserves_existing_abbreviations(self):
        result = compress("Use ML and NLP techniques.", level=1)
        assert "ML" in result
        assert "NLP" in result


# ===================================================================
# Code and URL protection
# ===================================================================

class TestCodeProtection:
    """Code blocks and inline code should never be compressed."""

    def test_code_block_preserved(self):
        text = "Basically, use this:\n```python\nimport torch\nbasically = True\n```"
        result = compress(text, level=1)
        assert "```python" in result
        assert "import torch" in result
        assert "basically = True" in result

    def test_inline_code_preserved(self):
        text = "Use `basically_flag = True` to enable it."
        result = compress(text, level=1)
        assert "`basically_flag = True`" in result

    def test_url_preserved(self):
        text = "Basically, visit https://pytorch.org/docs for info."
        result = compress(text, level=1)
        assert "https://pytorch.org/docs" in result

    def test_multiple_urls_preserved(self):
        text = "See https://a.com and https://b.com for details."
        result = compress(text, level=1)
        assert "https://a.com" in result
        assert "https://b.com" in result


# ===================================================================
# Confidence qualifier preservation
# ===================================================================

class TestConfidenceQualifiers:
    """Confidence qualifiers must be preserved."""

    def test_may_preserved(self):
        result = compress("This may cause issues with large batches.", level=1)
        assert "may" in result

    def test_likely_preserved(self):
        result = compress("This is likely the best approach.", level=1)
        assert "likely" in result

    def test_approximately_context_preserved(self):
        result = compress("The latency is approximately 10ms.", level=1)
        # "approximately" is abbreviated to "approx." at level 1
        assert "approx." in result or "approximately" in result

    def test_possibly_preserved(self):
        result = compress("This could possibly fail under load.", level=1)
        assert "possibly" in result


# ===================================================================
# Attribution preservation
# ===================================================================

class TestAttributionPreservation:
    """Expert attribution phrases must not be mangled."""

    def test_according_to_preserved(self):
        text = "According to Alice Smith the model is accurate."
        result = compress(text, level=1)
        assert "According to Alice Smith" in result

    def test_as_noted_by_preserved(self):
        text = "As noted by Bob Jones this approach works."
        result = compress(text, level=1)
        assert "as noted by Bob Jones" in result.lower() or "As noted by Bob Jones" in result


# ===================================================================
# Level 2 -- deduplication + aggressive trimming
# ===================================================================

class TestLevel2:
    """Level 2 adds sentence deduplication and whitespace trimming."""

    def test_deduplicates_sentences(self):
        text = "The model is fast. The model is accurate. The model is fast."
        result = compress(text, level=2)
        assert result.lower().count("the model is fast") == 1

    def test_trims_excessive_whitespace(self):
        text = "Line 1.\n\n\n\n\nLine 2."
        result = compress(text, level=2)
        assert "\n\n\n" not in result

    def test_level_2_also_removes_filler(self):
        text = "Basically, this is a test. Obviously, it works."
        result = compress(text, level=2)
        assert "basically" not in result.lower()
        assert "obviously" not in result.lower()

    def test_level_2_preserves_code(self):
        text = "Basically:\n```\nx = 1\nbasically = True\n```"
        result = compress(text, level=2)
        assert "basically = True" in result


# ===================================================================
# Compression ratio and statistics
# ===================================================================

class TestCompressionStats:
    """Tests for compression_ratio and estimate_savings."""

    def test_ratio_no_compression(self):
        assert compression_ratio("hello", "hello") == 1.0

    def test_ratio_full_compression(self):
        assert compression_ratio("hello", "") == 0.0

    def test_ratio_empty_original(self):
        assert compression_ratio("", "anything") == 1.0

    def test_ratio_partial(self):
        ratio = compression_ratio("abcdefghij", "abcde")
        assert ratio == 0.5

    def test_estimate_savings_keys(self):
        stats = estimate_savings("hello world", "hello")
        assert "original_chars" in stats
        assert "compressed_chars" in stats
        assert "saved_chars" in stats
        assert "ratio" in stats
        assert "pct_saved" in stats

    def test_estimate_savings_values(self):
        stats = estimate_savings("abcdefghij", "abcde")
        assert stats["original_chars"] == 10
        assert stats["compressed_chars"] == 5
        assert stats["saved_chars"] == 5
        assert stats["ratio"] == 0.5
        assert stats["pct_saved"] == 50.0

    def test_estimate_savings_empty(self):
        stats = estimate_savings("", "")
        assert stats["pct_saved"] == 0.0

    def test_real_compression_reduces_size(self):
        text = (
            "It is important to note that basically, for example, "
            "machine learning is essentially a subset of artificial intelligence. "
            "In order to train a model, needless to say, you need data."
        )
        result = compress(text, level=1)
        ratio = compression_ratio(text, result)
        assert ratio < 0.85  # At least 15% reduction
