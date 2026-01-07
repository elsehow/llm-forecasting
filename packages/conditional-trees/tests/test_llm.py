"""Tests for LLM utilities including batch API validation."""

import pytest

from conditional_trees.llm import (
    validate_custom_id,
    BatchValidationError,
    LLMRequest,
    CUSTOM_ID_PATTERN,
)


class TestCustomIdValidation:
    """Tests for batch API custom_id validation."""

    def test_valid_simple_id(self):
        """Simple alphanumeric IDs should be valid."""
        validate_custom_id("abc123")
        validate_custom_id("question1")
        validate_custom_id("us_gdp_2040")

    def test_valid_with_underscore(self):
        """Underscores should be valid separators."""
        validate_custom_id("question__scenario")
        validate_custom_id("us_gdp_2040__ai_transform")

    def test_valid_with_hyphen(self):
        """Hyphens should be valid separators."""
        validate_custom_id("question-scenario")
        validate_custom_id("us-gdp-2040")

    def test_invalid_colon(self):
        """Colons should be rejected."""
        with pytest.raises(BatchValidationError) as exc:
            validate_custom_id("question:scenario")
        assert "Invalid custom_id" in str(exc.value)
        assert "question:scenario" in str(exc.value)

    def test_invalid_space(self):
        """Spaces should be rejected."""
        with pytest.raises(BatchValidationError):
            validate_custom_id("question scenario")

    def test_invalid_special_chars(self):
        """Special characters should be rejected."""
        invalid_ids = [
            "question.scenario",
            "question/scenario",
            "question@scenario",
            "question#scenario",
            "question$scenario",
        ]
        for invalid_id in invalid_ids:
            with pytest.raises(BatchValidationError):
                validate_custom_id(invalid_id)

    def test_empty_string(self):
        """Empty strings should be rejected."""
        with pytest.raises(BatchValidationError):
            validate_custom_id("")

    def test_too_long(self):
        """IDs over 64 characters should be rejected."""
        long_id = "a" * 65
        with pytest.raises(BatchValidationError):
            validate_custom_id(long_id)

    def test_max_length_valid(self):
        """IDs of exactly 64 characters should be valid."""
        max_id = "a" * 64
        validate_custom_id(max_id)  # Should not raise


class TestLLMRequest:
    """Tests for LLMRequest dataclass validation."""

    def test_valid_request(self):
        """Valid requests should be created successfully."""
        req = LLMRequest(
            custom_id="test_request",
            system="You are a helpful assistant.",
            user="Hello",
        )
        assert req.custom_id == "test_request"

    def test_invalid_custom_id_rejected(self):
        """Requests with invalid custom_ids should fail on creation."""
        with pytest.raises(BatchValidationError):
            LLMRequest(
                custom_id="test:request",  # Invalid colon
                system="You are a helpful assistant.",
                user="Hello",
            )


class TestCustomIdPattern:
    """Tests for the regex pattern itself."""

    def test_pattern_matches_valid(self):
        """Pattern should match valid IDs."""
        valid_ids = [
            "abc",
            "ABC",
            "abc123",
            "abc_def",
            "abc-def",
            "abc_def_123",
            "a" * 64,
        ]
        for id in valid_ids:
            assert CUSTOM_ID_PATTERN.match(id), f"Should match: {id}"

    def test_pattern_rejects_invalid(self):
        """Pattern should reject invalid IDs."""
        invalid_ids = [
            "",
            "abc:def",
            "abc def",
            "abc.def",
            "a" * 65,
            "abc@def",
        ]
        for id in invalid_ids:
            assert not CUSTOM_ID_PATTERN.match(id), f"Should not match: {id}"
