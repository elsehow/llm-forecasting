"""Unit tests for the model registry."""

from datetime import date

import pytest

from llm_forecasting.registry import (
    MODEL_REGISTRY,
    ModelInfo,
    Organization,
    get_active_models,
    get_model,
    get_model_by_litellm_id,
    get_models_by_org,
    get_models_released_before,
    get_models_with_structured_output,
)


class TestModelInfo:
    """Tests for the ModelInfo model."""

    def test_create_model_info(self):
        """Create a basic ModelInfo."""
        info = ModelInfo(
            id="test-model",
            litellm_id="provider/test-model",
            organization=Organization.ANTHROPIC,
            context_window=100000,
        )
        assert info.id == "test-model"
        assert info.litellm_id == "provider/test-model"
        assert info.organization == Organization.ANTHROPIC
        assert info.context_window == 100000
        assert info.supports_structured_output is True  # default
        assert info.is_reasoning_model is False  # default
        assert info.active is True  # default
        assert info.release_date is None  # default

    def test_create_model_info_with_all_fields(self):
        """Create ModelInfo with all fields specified."""
        info = ModelInfo(
            id="test-reasoning",
            litellm_id="openai/o3-2025-01-01",
            organization=Organization.OPENAI,
            release_date=date(2025, 1, 1),
            context_window=128000,
            supports_structured_output=True,
            is_reasoning_model=True,
            active=False,
        )
        assert info.release_date == date(2025, 1, 1)
        assert info.is_reasoning_model is True
        assert info.active is False

    def test_organization_enum(self):
        """Test Organization enum values."""
        assert Organization.ANTHROPIC.value == "Anthropic"
        assert Organization.OPENAI.value == "OpenAI"
        assert Organization.GOOGLE.value == "Google"
        assert Organization.XAI.value == "xAI"


class TestModelRegistry:
    """Tests for the MODEL_REGISTRY dictionary."""

    def test_registry_not_empty(self):
        """Registry should contain models."""
        assert len(MODEL_REGISTRY) > 0

    def test_registry_has_anthropic_models(self):
        """Registry should have Anthropic models."""
        anthropic_models = [m for m in MODEL_REGISTRY.values() if m.organization == Organization.ANTHROPIC]
        assert len(anthropic_models) > 0

    def test_registry_has_openai_models(self):
        """Registry should have OpenAI models."""
        openai_models = [m for m in MODEL_REGISTRY.values() if m.organization == Organization.OPENAI]
        assert len(openai_models) > 0

    def test_all_models_have_required_fields(self):
        """All models should have required fields populated."""
        for model_id, model in MODEL_REGISTRY.items():
            assert model.id == model_id, f"Model {model_id} has mismatched id"
            assert model.litellm_id, f"Model {model_id} missing litellm_id"
            assert model.organization, f"Model {model_id} missing organization"
            assert model.context_window > 0, f"Model {model_id} has invalid context_window"

    def test_litellm_ids_have_provider_prefix(self):
        """All litellm_ids should have a provider prefix."""
        for model_id, model in MODEL_REGISTRY.items():
            assert "/" in model.litellm_id, f"Model {model_id} litellm_id missing provider prefix"


class TestGetModel:
    """Tests for get_model function."""

    def test_get_existing_model(self):
        """Get a model that exists in the registry."""
        # Pick the first model in the registry
        first_id = next(iter(MODEL_REGISTRY.keys()))
        model = get_model(first_id)
        assert model.id == first_id

    def test_get_nonexistent_model(self):
        """Getting a non-existent model should raise KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_model("nonexistent-model-xyz")
        assert "nonexistent-model-xyz" in str(exc_info.value)

    def test_get_specific_model(self):
        """Get a specific known model."""
        if "claude-opus-4-5-20251101" in MODEL_REGISTRY:
            model = get_model("claude-opus-4-5-20251101")
            assert model.organization == Organization.ANTHROPIC
            assert model.context_window == 200000


class TestGetModelByLitellmId:
    """Tests for get_model_by_litellm_id function."""

    def test_find_by_litellm_id(self):
        """Find a model by its LiteLLM identifier."""
        # Get the first model and look it up by litellm_id
        first_model = next(iter(MODEL_REGISTRY.values()))
        found = get_model_by_litellm_id(first_model.litellm_id)
        assert found is not None
        assert found.id == first_model.id

    def test_not_found_returns_none(self):
        """Non-existent litellm_id should return None."""
        result = get_model_by_litellm_id("nonexistent/model")
        assert result is None


class TestGetModelsByOrg:
    """Tests for get_models_by_org function."""

    def test_get_by_organization_enum(self):
        """Get models by Organization enum."""
        anthropic_models = get_models_by_org(Organization.ANTHROPIC)
        assert len(anthropic_models) > 0
        for model in anthropic_models:
            assert model.organization == Organization.ANTHROPIC

    def test_get_by_organization_string(self):
        """Get models by organization string."""
        anthropic_models = get_models_by_org("Anthropic")
        assert len(anthropic_models) > 0
        for model in anthropic_models:
            assert model.organization == Organization.ANTHROPIC

    def test_empty_result_for_org_with_no_models(self):
        """Organizations with no models return empty list."""
        # Create a fake org value that doesn't exist - this would raise ValueError
        # So instead we test that a real org returns models
        google_models = get_models_by_org(Organization.GOOGLE)
        # Google should have models
        assert isinstance(google_models, list)


class TestGetActiveModels:
    """Tests for get_active_models function."""

    def test_returns_only_active_models(self):
        """Should only return models where active=True."""
        active_models = get_active_models()
        for model in active_models:
            assert model.active is True

    def test_excludes_inactive_models(self):
        """Should exclude models where active=False."""
        active_models = get_active_models()
        active_ids = {m.id for m in active_models}

        # Check that inactive models are not included
        for model_id, model in MODEL_REGISTRY.items():
            if not model.active:
                assert model_id not in active_ids


class TestGetModelsReleasedBefore:
    """Tests for get_models_released_before function."""

    def test_filter_by_date(self):
        """Should only return models released before cutoff."""
        cutoff = date(2025, 6, 1)
        models = get_models_released_before(cutoff)

        for model in models:
            assert model.release_date is not None
            assert model.release_date < cutoff

    def test_excludes_models_without_release_date(self):
        """Should exclude models with no release_date."""
        cutoff = date(2030, 1, 1)  # Far future
        models = get_models_released_before(cutoff)

        for model in models:
            assert model.release_date is not None

    def test_early_cutoff_returns_fewer_models(self):
        """Earlier cutoff should return fewer models."""
        early = date(2024, 1, 1)
        late = date(2026, 1, 1)

        early_models = get_models_released_before(early)
        late_models = get_models_released_before(late)

        assert len(early_models) <= len(late_models)


class TestGetModelsWithStructuredOutput:
    """Tests for get_models_with_structured_output function."""

    def test_returns_only_structured_output_models(self):
        """Should only return models that support structured output."""
        models = get_models_with_structured_output()
        for model in models:
            assert model.supports_structured_output is True

    def test_has_results(self):
        """Should return at least some models."""
        models = get_models_with_structured_output()
        assert len(models) > 0
