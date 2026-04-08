"""Comprehensive tests for forge.ingest.pii_scrubber — 25+ tests."""

import pytest

from forge.ingest.pii_scrubber import scrub_pii, detect_pii, PII_PATTERNS


# ---------------------------------------------------------------------------
# scrub_pii
# ---------------------------------------------------------------------------


class TestScrubPii:
    """Tests for scrub_pii."""

    # -- SSN --

    def test_scrub_ssn(self):
        """SSN pattern is redacted."""
        text = "My SSN is 123-45-6789."
        scrubbed, counts = scrub_pii(text)
        assert "123-45-6789" not in scrubbed
        assert "[REDACTED]" in scrubbed
        assert counts["ssn"] == 1

    def test_scrub_multiple_ssns(self):
        """Multiple SSNs are all redacted."""
        text = "SSN 123-45-6789 and 987-65-4321"
        scrubbed, counts = scrub_pii(text)
        assert "123-45-6789" not in scrubbed
        assert "987-65-4321" not in scrubbed
        assert counts["ssn"] == 2

    # -- Email --

    def test_scrub_email(self):
        """Email address is redacted."""
        text = "Contact me at alice@example.com for details."
        scrubbed, counts = scrub_pii(text)
        assert "alice@example.com" not in scrubbed
        assert counts["email"] == 1

    def test_scrub_email_with_plus(self):
        """Email with + addressing is redacted."""
        text = "Email: user+tag@domain.org"
        scrubbed, counts = scrub_pii(text)
        assert "user+tag@domain.org" not in scrubbed
        assert counts["email"] == 1

    def test_scrub_email_with_dots(self):
        """Email with dots in local part is redacted."""
        text = "Send to first.last@company.co.uk"
        scrubbed, counts = scrub_pii(text)
        assert "first.last@company.co.uk" not in scrubbed

    # -- Phone --

    def test_scrub_phone_dashes(self):
        """Phone number with dashes is redacted."""
        text = "Call 555-123-4567"
        scrubbed, counts = scrub_pii(text)
        assert "555-123-4567" not in scrubbed
        assert counts["phone"] == 1

    def test_scrub_phone_dots(self):
        """Phone number with dots is redacted."""
        text = "Phone: 555.123.4567"
        scrubbed, counts = scrub_pii(text)
        assert "555.123.4567" not in scrubbed

    def test_scrub_phone_parens(self):
        """Phone with parentheses is redacted."""
        text = "Call (555) 123-4567"
        scrubbed, counts = scrub_pii(text)
        assert "(555) 123-4567" not in scrubbed

    def test_scrub_phone_with_country_code(self):
        """Phone with +1 country code is redacted."""
        text = "Reach me at +1-555-123-4567"
        scrubbed, counts = scrub_pii(text)
        assert "555-123-4567" not in scrubbed

    # -- Credit Card --

    def test_scrub_credit_card_spaces(self):
        """Credit card with spaces is redacted."""
        text = "Card: 4111 1111 1111 1111"
        scrubbed, counts = scrub_pii(text)
        assert "4111 1111 1111 1111" not in scrubbed
        assert counts["credit_card"] == 1

    def test_scrub_credit_card_dashes(self):
        """Credit card with dashes is redacted."""
        text = "Card: 4111-1111-1111-1111"
        scrubbed, counts = scrub_pii(text)
        assert "4111-1111-1111-1111" not in scrubbed
        assert counts["credit_card"] == 1

    def test_scrub_credit_card_no_separator(self):
        """Credit card without separators is redacted."""
        text = "Card: 4111111111111111"
        scrubbed, counts = scrub_pii(text)
        assert "4111111111111111" not in scrubbed

    # -- IP Address --

    def test_scrub_ip_address(self):
        """IP address is redacted."""
        text = "Server at 192.168.1.100"
        scrubbed, counts = scrub_pii(text)
        assert "192.168.1.100" not in scrubbed
        assert counts["ip_address"] == 1

    def test_scrub_ip_localhost(self):
        """Localhost IP is redacted."""
        text = "Connect to 127.0.0.1"
        scrubbed, counts = scrub_pii(text)
        assert "127.0.0.1" not in scrubbed

    def test_scrub_ip_max_values(self):
        """IP with max octet values is redacted."""
        text = "IP: 255.255.255.255"
        scrubbed, counts = scrub_pii(text)
        assert "255.255.255.255" not in scrubbed

    # -- Mixed --

    def test_scrub_mixed_pii(self):
        """Multiple PII types in one text are all redacted."""
        text = (
            "Name: Alice, SSN: 123-45-6789, "
            "Email: alice@example.com, Phone: 555-123-4567"
        )
        scrubbed, counts = scrub_pii(text)
        assert "123-45-6789" not in scrubbed
        assert "alice@example.com" not in scrubbed
        assert "555-123-4567" not in scrubbed
        assert counts.get("ssn", 0) >= 1
        assert counts.get("email", 0) >= 1
        assert counts.get("phone", 0) >= 1

    def test_scrub_clean_text(self):
        """Text with no PII returns unchanged text and empty counts."""
        text = "This is a clean document about machine learning."
        scrubbed, counts = scrub_pii(text)
        assert scrubbed == text
        assert counts == {}

    def test_scrub_custom_replacement(self):
        """Custom replacement string is used."""
        text = "SSN: 123-45-6789"
        scrubbed, counts = scrub_pii(text, replacement="***")
        assert "123-45-6789" not in scrubbed
        assert "***" in scrubbed
        assert "[REDACTED]" not in scrubbed

    def test_scrub_empty_text(self):
        """Empty text returns empty text and no counts."""
        scrubbed, counts = scrub_pii("")
        assert scrubbed == ""
        assert counts == {}

    def test_scrub_preserves_non_pii(self):
        """Non-PII content is preserved after scrubbing."""
        text = "Important: SSN 123-45-6789 was leaked."
        scrubbed, _ = scrub_pii(text)
        assert "Important:" in scrubbed
        assert "was leaked." in scrubbed


# ---------------------------------------------------------------------------
# detect_pii
# ---------------------------------------------------------------------------


class TestDetectPii:
    """Tests for detect_pii."""

    def test_detect_ssn(self):
        """SSN is detected."""
        found = detect_pii("SSN: 123-45-6789")
        assert "ssn" in found
        assert "123-45-6789" in found["ssn"]

    def test_detect_email(self):
        """Email is detected."""
        found = detect_pii("Contact: bob@test.com")
        assert "email" in found
        assert "bob@test.com" in found["email"]

    def test_detect_phone(self):
        """Phone is detected."""
        found = detect_pii("Call 555-123-4567")
        assert "phone" in found

    def test_detect_credit_card(self):
        """Credit card is detected."""
        found = detect_pii("Card: 4111 1111 1111 1111")
        assert "credit_card" in found

    def test_detect_ip(self):
        """IP address is detected."""
        found = detect_pii("Host: 10.0.4.93")
        assert "ip_address" in found
        assert "10.0.4.93" in found["ip_address"]

    def test_detect_clean_text(self):
        """Clean text returns empty dict."""
        found = detect_pii("No personal information here.")
        assert found == {}

    def test_detect_multiple_of_same_type(self):
        """Multiple matches of same type are all returned."""
        found = detect_pii("IPs: 10.0.0.1 and 10.0.0.2")
        assert "ip_address" in found
        assert len(found["ip_address"]) == 2

    def test_detect_does_not_modify(self):
        """detect_pii does not modify the original text."""
        text = "SSN: 123-45-6789"
        detect_pii(text)
        assert "123-45-6789" in text


# ---------------------------------------------------------------------------
# PII_PATTERNS
# ---------------------------------------------------------------------------


class TestPiiPatterns:
    """Tests for the PII_PATTERNS dict."""

    def test_all_expected_types(self):
        """All expected PII types are defined."""
        expected = {"ssn", "email", "phone", "credit_card", "ip_address"}
        assert set(PII_PATTERNS.keys()) == expected

    def test_patterns_are_compiled(self):
        """All patterns are compiled regex objects."""
        import re
        for name, pattern in PII_PATTERNS.items():
            assert isinstance(pattern, re.Pattern), f"{name} is not compiled"
