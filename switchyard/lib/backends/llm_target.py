# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust-owned LLM target configuration helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from switchyard_rust.components import BackendFormat, EndpointConfig, LlmTarget

_DISABLE_THINKING_MODEL_FRAGMENTS = ("nemotron-3-super",)


def coerce_llm_target(value: object, *, default_id: str) -> LlmTarget:
    """Build a Rust ``LlmTarget`` from a target object or legacy mapping."""
    if isinstance(value, LlmTarget):
        if value.id == "default" and default_id != "default":
            return LlmTarget(
                id=default_id,
                model=value.model,
                format=value.format,
                input_modalities=value.input_modalities,
                endpoint=value.endpoint,
                extra_body=value.extra_body,
                extra_headers=value.extra_headers,
            )
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, dict):
        raise TypeError(f"expected LlmTarget or dict, got {type(value).__name__}")

    data: dict[str, Any] = dict(value)
    target_id = str(data.pop("id", default_id))
    model = data.pop("model", None)
    if not isinstance(model, str):
        raise TypeError("LlmTarget.model must be a string")

    target_format = data.pop("format", data.pop("backend_format", BackendFormat.OPENAI))
    input_modalities = data.pop("input_modalities", None)
    endpoint = data.pop("endpoint", None)
    base_url = data.pop("base_url", None)
    api_key = data.pop("api_key", None)
    timeout_secs = data.pop("timeout_secs", data.pop("timeout", None))
    extra_body = data.pop("extra_body", None)
    extra_headers = data.pop("extra_headers", None)
    data.pop("tuning", None)
    if data:
        unknown = ", ".join(sorted(data))
        raise ValueError(f"unknown LlmTarget fields: {unknown}")

    return LlmTarget(
        id=target_id,
        model=model,
        format=target_format,
        input_modalities=input_modalities,
        endpoint=endpoint,
        base_url=base_url,
        api_key=api_key,
        timeout_secs=timeout_secs,
        extra_body=extra_body,
        extra_headers=extra_headers,
    )


def llm_target_with_format(target: LlmTarget, target_format: BackendFormat) -> LlmTarget:
    """Return ``target`` with a resolved backend format."""
    return LlmTarget(
        id=target.id,
        model=target.model,
        format=target_format,
        input_modalities=target.input_modalities,
        endpoint=target.endpoint,
        extra_body=target.extra_body,
        extra_headers=target.extra_headers,
    )


def llm_target_with_runtime_defaults(target: LlmTarget) -> LlmTarget:
    """Return ``target`` with Switchyard's safe per-model runtime defaults."""
    if target.extra_body:
        return target
    if not _should_disable_thinking(target.model):
        return target
    return LlmTarget(
        id=target.id,
        model=target.model,
        format=target.format,
        input_modalities=target.input_modalities,
        endpoint=target.endpoint,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        extra_headers=target.extra_headers,
    )


def _should_disable_thinking(model: str) -> bool:
    normalized = model.lower()
    return any(fragment in normalized for fragment in _DISABLE_THINKING_MODEL_FRAGMENTS)


__all__ = [
    "BackendFormat",
    "EndpointConfig",
    "LlmTarget",
    "coerce_llm_target",
    "llm_target_with_format",
    "llm_target_with_runtime_defaults",
]
