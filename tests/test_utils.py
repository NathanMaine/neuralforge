"""Comprehensive tests for forge.core.utils — 25+ tests."""
import re
from datetime import datetime

import pytest

from forge.core.utils import slugify, content_hash, now_iso, human_size, estimate_tokens


class TestSlugify:
    """Tests for the slugify function."""

    def test_basic_text(self):
        assert slugify("Hello World") == "hello-world"

    def test_uppercase(self):
        assert slugify("ALL CAPS TEXT") == "all-caps-text"

    def test_special_characters(self):
        assert slugify("Hello! @World #2024") == "hello-world-2024"

    def test_multiple_spaces(self):
        assert slugify("hello   world") == "hello-world"

    def test_leading_trailing_spaces(self):
        assert slugify("  hello world  ") == "hello-world"

    def test_consecutive_hyphens(self):
        assert slugify("hello---world") == "hello-world"

    def test_underscores_to_hyphens(self):
        assert slugify("hello_world_test") == "hello-world-test"

    def test_empty_string(self):
        assert slugify("") == ""

    def test_only_special_chars(self):
        assert slugify("!@#$%^&*()") == ""

    def test_mixed_unicode_and_ascii(self):
        result = slugify("cafe & restaurant")
        assert result == "cafe-restaurant"

    def test_numbers_preserved(self):
        assert slugify("version 2.0 release") == "version-20-release"

    def test_leading_hyphens_stripped(self):
        assert slugify("---hello") == "hello"

    def test_trailing_hyphens_stripped(self):
        assert slugify("hello---") == "hello"

    def test_tabs_converted(self):
        assert slugify("hello\tworld") == "hello-world"


class TestContentHash:
    """Tests for the content_hash function."""

    def test_returns_string(self):
        assert isinstance(content_hash("test"), str)

    def test_returns_32_char_hex(self):
        result = content_hash("test")
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_inputs_different_hashes(self):
        assert content_hash("hello") != content_hash("world")

    def test_empty_string(self):
        result = content_hash("")
        assert len(result) == 32

    def test_known_md5(self):
        # MD5 of "test" is well-known
        assert content_hash("test") == "098f6bcd4621d373cade4e832627b4f6"


class TestNowIso:
    """Tests for the now_iso function."""

    def test_returns_string(self):
        assert isinstance(now_iso(), str)

    def test_parseable_iso_format(self):
        result = now_iso()
        # Should not raise
        parsed = datetime.fromisoformat(result)
        assert isinstance(parsed, datetime)

    def test_contains_date_components(self):
        result = now_iso()
        # Should contain year, month, day separated by hyphens
        assert re.match(r"\d{4}-\d{2}-\d{2}", result)

    def test_contains_time_component(self):
        result = now_iso()
        assert "T" in result


class TestHumanSize:
    """Tests for the human_size function."""

    def test_zero_bytes(self):
        assert human_size(0) == "0 B"

    def test_bytes(self):
        assert human_size(500) == "500 B"

    def test_one_kb(self):
        result = human_size(1024)
        assert "KB" in result

    def test_one_mb(self):
        result = human_size(1024 * 1024)
        assert "MB" in result

    def test_one_gb(self):
        result = human_size(1024 ** 3)
        assert "GB" in result

    def test_1_5_gb(self):
        result = human_size(int(1.5 * 1024 ** 3))
        assert "1.5 GB" == result

    def test_one_tb(self):
        result = human_size(1024 ** 4)
        assert "TB" in result

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            human_size(-1)

    def test_large_tb(self):
        result = human_size(5 * 1024 ** 4)
        assert "TB" in result


class TestEstimateTokens:
    """Tests for the estimate_tokens function."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_whitespace_only(self):
        assert estimate_tokens("   ") == 0

    def test_single_word(self):
        result = estimate_tokens("hello")
        assert result >= 1

    def test_returns_int(self):
        assert isinstance(estimate_tokens("hello world test"), int)

    def test_longer_text_more_tokens(self):
        short = estimate_tokens("hello")
        long = estimate_tokens("hello world this is a longer sentence with many words")
        assert long > short

    def test_minimum_one_for_nonempty(self):
        assert estimate_tokens("a") >= 1

    def test_rough_ratio(self):
        text = "the quick brown fox jumps over the lazy dog"
        tokens = estimate_tokens(text)
        word_count = len(text.split())
        # Should be roughly 1.33x word count
        assert tokens == int(word_count * 1.33)
