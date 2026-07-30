# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pricing for direct OpenAI models used by Switchyard demos."""

from __future__ import annotations

from switchyard.cli.launchers.cost_estimator import MODEL_PRICING, estimate_model_cost

_GPT_5_4_MINI_KEYS = (
    "gpt-5.4-mini",
    "gpt-5.4-mini-2026-03-17",
)


def test_gpt_5_4_mini_keys_use_openai_list_price() -> None:
    for key in _GPT_5_4_MINI_KEYS:
        price = MODEL_PRICING[key]
        assert price.input == 0.75
        assert price.cached == 0.075
        assert price.output == 4.50
        assert price.cache_write == price.input


def test_gpt_5_4_mini_estimate_includes_cache_discount() -> None:
    result = estimate_model_cost(
        "gpt-5.4-mini-2026-03-17",
        prompt_tokens=2_000_000,
        cached_tokens=1_000_000,
        completion_tokens=1_000_000,
    )

    assert result["total_cost"] == 5.325
