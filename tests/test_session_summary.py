# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for session_summary._format_summary."""

from switchyard.cli.launchers.session_summary import _format_summary


def _snapshot(
    total_requests: int = 0,
    total_errors: int = 0,
    prompt: int = 0,
    completion: int = 0,
    cached: int = 0,
    models: dict | None = None,
    image_compression: dict | None = None,
) -> dict:
    snapshot = {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "total_tokens": {"prompt": prompt, "completion": completion, "cached": cached},
        "models": models or {},
    }
    if image_compression is not None:
        snapshot["image_compression"] = image_compression
    return snapshot


def test_empty_snapshot_returns_empty_string():
    assert _format_summary(_snapshot()) == ""


def test_summary_shows_request_and_token_counts():
    snap = _snapshot(total_requests=5, prompt=1000, completion=200)
    out = _format_summary(snap)
    assert "5" in out
    assert "1,000" in out
    assert "200" in out


def test_summary_shows_errors_when_present():
    snap = _snapshot(total_requests=3, total_errors=1, prompt=100, completion=50)
    out = _format_summary(snap)
    assert "1 error" in out


def test_summary_omits_errors_when_zero():
    snap = _snapshot(total_requests=3, prompt=100, completion=50)
    out = _format_summary(snap)
    assert "error" not in out


def test_summary_shows_cached_tokens_when_nonzero():
    snap = _snapshot(total_requests=1, prompt=500, completion=100, cached=200)
    out = _format_summary(snap)
    assert "200" in out
    assert "cached" in out


def test_summary_omits_cached_line_when_zero():
    snap = _snapshot(total_requests=1, prompt=500, completion=100, cached=0)
    out = _format_summary(snap)
    assert "cached" not in out


def test_summary_shows_per_model_row():
    snap = _snapshot(
        total_requests=2,
        prompt=800,
        completion=300,
        models={
            "nvidia/some-provider/my-model": {
                "calls": 2,
                "prompt_tokens": 800,
                "completion_tokens": 300,
                "cached_tokens": 0,
            }
        },
    )
    out = _format_summary(snap)
    assert "my-model" in out
    assert "2 req" in out


def test_summary_shows_cost_for_known_model():
    snap = _snapshot(
        total_requests=1,
        prompt=1_000_000,
        completion=1_000_000,
        models={
            "openai/openai/gpt-5.2": {
                "calls": 1,
                "prompt_tokens": 1_000_000,
                "completion_tokens": 1_000_000,
                "cached_tokens": 0,
            }
        },
    )
    out = _format_summary(snap)
    # gpt-5.2: $1.75 input + $14.00 output = $15.75 per 1M tokens each
    assert "$15.75" in out


def test_summary_omits_cost_for_unknown_model():
    snap = _snapshot(
        total_requests=1,
        prompt=100,
        completion=50,
        models={
            "unknown-provider/unknown-model": {
                "calls": 1,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cached_tokens": 0,
            }
        },
    )
    out = _format_summary(snap)
    assert "$" not in out


def test_summary_shows_image_compression_savings():
    snap = _snapshot(
        total_requests=2,
        prompt=1000,
        completion=50,
        image_compression={
            "images_seen": 3,
            "images_optimized": 3,
            "bytes_before": 3 * 1024 * 1024,
            "bytes_after": 300 * 1024,
        },
    )
    out = _format_summary(snap)

    assert "3/3 optimized" in out
    assert "3.0 MB → 300.0 KB" in out
    assert "90.2% saved" in out
