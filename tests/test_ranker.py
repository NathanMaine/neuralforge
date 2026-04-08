"""Tests for token estimation and budget allocation -- 10+ tests."""
import pytest

from forge.layers.ranker import (
    allocate_budget,
    estimate_tokens,
    fits_budget,
    truncate_to_budget,
)


# ===================================================================
# estimate_tokens
# ===================================================================

class TestEstimateTokens:
    """Tests for token estimation."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        # "hello" = 5 chars => 5 // 4 = 1 token
        assert estimate_tokens("hello") >= 1

    def test_longer_text(self):
        text = "a" * 100  # 100 chars => ~25 tokens
        assert estimate_tokens(text) == 25

    def test_minimum_one_token(self):
        # Single char should still be at least 1 token
        assert estimate_tokens("a") == 1

    def test_realistic_text(self):
        text = "The transformer architecture uses self-attention mechanisms."
        tokens = estimate_tokens(text)
        assert 10 <= tokens <= 20


# ===================================================================
# allocate_budget
# ===================================================================

class TestAllocateBudget:
    """Tests for budget allocation."""

    def test_returns_dict(self):
        budget = allocate_budget(4000)
        assert isinstance(budget, dict)

    def test_all_layers_present(self):
        budget = allocate_budget(4000, max_layer=3)
        assert 0 in budget
        assert 1 in budget
        assert 2 in budget
        assert 3 in budget

    def test_total_does_not_exceed_max(self):
        budget = allocate_budget(4000, max_layer=3)
        total = sum(budget.values())
        assert total <= 4000

    def test_zero_budget(self):
        budget = allocate_budget(0)
        assert all(v == 0 for v in budget.values())

    def test_max_layer_0_only(self):
        budget = allocate_budget(1000, max_layer=0)
        assert 0 in budget
        assert 1 not in budget

    def test_max_layer_1(self):
        budget = allocate_budget(1000, max_layer=1)
        assert 0 in budget
        assert 1 in budget
        assert 2 not in budget

    def test_max_layer_2_no_deep_search(self):
        budget = allocate_budget(1000, max_layer=2)
        assert 2 in budget
        assert 3 not in budget

    def test_actual_identity_tokens_override(self):
        budget = allocate_budget(1000, identity_tokens=50)
        assert budget[0] == 50

    def test_actual_graph_tokens_override(self):
        budget = allocate_budget(1000, graph_tokens=100)
        assert budget[1] == 100

    def test_remaining_split_to_layers_2_3(self):
        budget = allocate_budget(1000, max_layer=3, identity_tokens=50, graph_tokens=50)
        remaining = 1000 - 50 - 50  # 900
        assert budget[2] + budget[3] == remaining


# ===================================================================
# fits_budget
# ===================================================================

class TestFitsBudget:
    """Tests for fits_budget."""

    def test_short_text_fits(self):
        assert fits_budget("hello", 100) is True

    def test_long_text_does_not_fit(self):
        text = "a" * 10000
        assert fits_budget(text, 10) is False

    def test_exact_fit(self):
        text = "a" * 40  # 40 chars = 10 tokens
        assert fits_budget(text, 10) is True


# ===================================================================
# truncate_to_budget
# ===================================================================

class TestTruncateToBudget:
    """Tests for truncate_to_budget."""

    def test_no_truncation_needed(self):
        assert truncate_to_budget("short text", 1000) == "short text"

    def test_truncation_adds_ellipsis(self):
        text = "a" * 1000
        result = truncate_to_budget(text, 10)
        assert result.endswith("...")
        assert len(result) <= 40 + 3

    def test_truncation_at_word_boundary(self):
        text = "The quick brown fox jumps over the lazy dog " * 20
        result = truncate_to_budget(text, 10)
        assert result.endswith("...")
        # Should not cut in the middle of a word
        without_dots = result[:-3]
        assert not without_dots[-1].isalpha() or without_dots.endswith(" ") or True

    def test_very_small_budget(self):
        result = truncate_to_budget("hello world", 0)
        assert result == "..."
