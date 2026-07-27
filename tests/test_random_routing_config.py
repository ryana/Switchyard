# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``BackendFormat``, ``LlmTarget`` and ``RandomRoutingConfig``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from switchyard.lib.backends.llm_target import (
    BackendFormat,
    LlmTarget,
)
from switchyard.lib.profiles.random_routing import RandomRoutingConfig

# ---------------------------------------------------------------------------
# BackendFormat enum
# ---------------------------------------------------------------------------


class TestBackendFormat:
    def test_auto_value(self):
        assert BackendFormat.AUTO.value == "auto"

    def test_openai_value(self):
        assert BackendFormat.OPENAI.value == "openai"

    def test_anthropic_value(self):
        assert BackendFormat.ANTHROPIC.value == "anthropic"

    def test_responses_value(self):
        assert BackendFormat.RESPONSES.value == "responses"

    def test_is_rust_value_object(self):
        assert not isinstance(BackendFormat.OPENAI, str)
        assert str(BackendFormat.OPENAI) == "openai"

    def test_equals_string(self):
        assert BackendFormat.AUTO == "auto"
        assert BackendFormat.OPENAI == "openai"
        assert BackendFormat.RESPONSES == "responses"
        assert BackendFormat.ANTHROPIC == "anthropic"

    def test_constructible_from_string(self):
        assert BackendFormat("auto") == BackendFormat.AUTO
        assert BackendFormat("openai") == BackendFormat.OPENAI
        assert BackendFormat("responses") == BackendFormat.RESPONSES
        assert BackendFormat("anthropic") == BackendFormat.ANTHROPIC

    def test_unknown_value_rejected(self):
        with pytest.raises(ValueError):
            BackendFormat("bedrock")


# ---------------------------------------------------------------------------
# LlmTarget
# ---------------------------------------------------------------------------


class TestTierConstruction:
    def test_minimal_construction(self):
        tier = LlmTarget(model="m")
        assert tier.model == "m"
        assert tier.id == "default"
        assert tier.format is BackendFormat.AUTO  # Rust LlmTarget default
        assert tier.api_key is None
        assert tier.base_url is None
        assert tier.timeout is None
        assert tier.input_modalities is None

    def test_all_fields(self):
        tier = LlmTarget(
            model="claude-opus-4-6",
            format=BackendFormat.ANTHROPIC,
            input_modalities=["text", "image"],
            api_key="sk-ant-xxx",
            base_url="https://api.anthropic.com",
            timeout=30.0,
        )
        assert tier.model == "claude-opus-4-6"
        assert tier.format is BackendFormat.ANTHROPIC
        assert tier.input_modalities == ["text", "image"]
        assert tier.api_key == "sk-ant-xxx"
        assert tier.base_url == "https://api.anthropic.com"
        assert tier.timeout == 30.0


class TestTierValidation:
    def test_empty_model_rejected(self):
        with pytest.raises(ValueError, match="model id"):
            LlmTarget(model="")


class TestTierFrozenSemantics:
    def test_model_is_immutable(self):
        tier = LlmTarget(model="m")
        with pytest.raises(AttributeError):
            tier.model = "other"  # type: ignore[misc]

    def test_backend_format_is_immutable(self):
        tier = LlmTarget(model="m")
        with pytest.raises(AttributeError):
            tier.format = BackendFormat.ANTHROPIC  # type: ignore[misc]

    def test_equality_by_fields(self):
        t1 = LlmTarget(
            model="m", format=BackendFormat.OPENAI, api_key="k",
        )
        t2 = LlmTarget(
            model="m", format=BackendFormat.OPENAI, api_key="k",
        )
        assert t1 == t2

    def test_inequality_on_format_difference(self):
        t1 = LlmTarget(model="m", format=BackendFormat.OPENAI)
        t2 = LlmTarget(model="m", format=BackendFormat.ANTHROPIC)
        assert t1 != t2


# ---------------------------------------------------------------------------
# RandomRoutingConfig
# ---------------------------------------------------------------------------


def _tier(model: str = "m") -> LlmTarget:
    return LlmTarget(model=model)


class TestConfigConstruction:
    def test_minimal_construction(self):
        cfg = RandomRoutingConfig(
            strong=_tier("s"), weak=_tier("w"),
        fallback_target_on_evict="strong")
        assert cfg.strong.model == "s"
        assert cfg.weak.model == "w"
        assert cfg.strong_probability == 0.5  # default

    def test_custom_probability(self):
        cfg = RandomRoutingConfig(
            strong=_tier("s"), weak=_tier("w"), strong_probability=0.7,
        fallback_target_on_evict="strong")
        assert cfg.strong_probability == 0.7


class TestConfigValidation:
    @pytest.mark.parametrize("p", [-0.1, -1.0, 1.1, 2.0, float("inf")])
    def test_out_of_range_probability_rejected(self, p: float):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            RandomRoutingConfig(
                strong=_tier(), weak=_tier(), strong_probability=p,
            fallback_target_on_evict="strong")

    @pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_valid_probability_accepted(self, p: float):
        cfg = RandomRoutingConfig(
            strong=_tier(), weak=_tier(), strong_probability=p,
        fallback_target_on_evict="strong")
        assert cfg.strong_probability == p


class TestConfigFrozenSemantics:
    def test_strong_is_immutable(self):
        cfg = RandomRoutingConfig(strong=_tier("a"), weak=_tier("b"), fallback_target_on_evict="strong")
        with pytest.raises(ValidationError):
            cfg.strong = _tier("other")  # type: ignore[misc]

    def test_strong_probability_is_immutable(self):
        cfg = RandomRoutingConfig(strong=_tier(), weak=_tier(), fallback_target_on_evict="strong")
        with pytest.raises(ValidationError):
            cfg.strong_probability = 0.9  # type: ignore[misc]


class TestTargetSerialization:
    def test_model_dump_uses_rust_shape(self):
        target = LlmTarget(
            id="target",
            model="m",
            format=BackendFormat.OPENAI,
            input_modalities=["text"],
            api_key="k",
            base_url="https://example.test/v1",
            timeout=3.0,
        )

        assert target.model_dump() == {
            "id": "target",
            "model": "m",
            "format": "openai",
            "input_modalities": ["text"],
            "endpoint": {
                "api_key": "k",
                "base_url": "https://example.test/v1",
                "timeout_secs": 3.0,
            },
        }
