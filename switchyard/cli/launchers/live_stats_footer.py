# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared live token-usage footer for launcher TUI sessions.

One layout for every routing strategy: an aggregate row across all requests
plus one indented row per active outbound model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from switchyard.cli.launchers.cost_estimator import estimate_model_cost
from switchyard.cli.launchers.proxy_health_monitor import ProxyHealthMonitor
from switchyard.cli.launchers.stats_source import StatsSource

FOOTER_ROWS = 2


class LiveStatsFooter:
    """Live stats footer: aggregate row plus one row per active model."""

    def __init__(
        self,
        stats: StatsSource,
        model: str,
        health: ProxyHealthMonitor,
        *,
        strategy_label: str | None = None,
        image_compression: bool = False,
    ) -> None:
        self._stats = stats
        self._default_model_short = model.rsplit("/", 1)[-1]
        self._health = health
        self._strategy_label = strategy_label
        self._image_compression = image_compression
        # Ordered list of models seen in traffic so far. Grows as new tiers
        # receive their first request; order is first-seen, which keeps the
        # display stable across renders.
        self._seen_models: list[str] = []
        self._seen_set: set[str] = set()

    @property
    def height(self) -> int:
        """Current footer height, including image stats when enabled."""
        image_rows = int(self._image_compression)
        return FOOTER_ROWS - 1 + max(1, len(self._seen_models)) + image_rows

    def as_footer_fn(self) -> Callable[[int], list[tuple[str, int]]]:
        return self.render

    def render(self, cols: int) -> list[tuple[str, int]]:  # noqa: ARG002
        self._health.poll()
        snapshot = self._stats.snapshot_sync()
        rows = [self._aggregate_row(snapshot)]
        if self._image_compression:
            rows.append(_image_compression_row(_mapping(snapshot, "image_compression")))
        return [*rows, *self._tier_rows(snapshot)]

    def _aggregate_row(self, snapshot: Mapping[str, object]) -> tuple[str, int]:
        totals = _mapping(snapshot, "total_tokens")
        req = _int(snapshot, "total_requests")
        errs = _int(snapshot, "total_errors")
        prompt = _int(totals, "prompt")
        completion = _int(totals, "completion")
        cached = _int(totals, "cached")
        h_str, h_w = self._health.indicator

        req_label = f"{req:,} req" + (f" ({errs} err)" if errs else "")
        strategy_part = f" [{self._strategy_label}]" if self._strategy_label else ""
        prefix = f" switchyard{strategy_part} · {req_label}"
        styled = (
            "\x1b[2m switchyard\x1b[0m"
            + (f"\x1b[2m [{self._strategy_label}]\x1b[0m" if self._strategy_label else "")
            + f"\x1b[2m · {req_label}\x1b[0m"
        )

        in_p, in_s = f" · {prompt:,} in", f" · \x1b[96m{prompt:,}\x1b[0m in"
        out_p, out_s = f"  {completion:,} out", f"  \x1b[92m{completion:,}\x1b[0m out"
        cache_p = cache_s = ""
        if cached:
            cache_p = f"  {cached:,} cached"
            cache_s = f"  \x1b[33m{cached:,}\x1b[0m cached"
        cost = _snapshot_cost(snapshot)
        cost_p = cost_s = ""
        if cost:
            cost_p = f" · ${cost:.4f}"
            cost_s = f" · \x1b[95m${cost:.4f}\x1b[0m"

        line = " " + h_str + styled + in_s + out_s + cache_s + cost_s
        width = 1 + h_w + len(prefix) + len(in_p) + len(out_p) + len(cache_p) + len(cost_p)
        return (line, width)

    def _tier_rows(
        self, snapshot: Mapping[str, object],
    ) -> list[tuple[str, int]]:
        """Return one row per model that has received traffic.

        Before traffic lands, a placeholder uses the launch model. Once traffic
        arrives, rows grow in first-seen order and never shrink.
        """
        models = _mapping(snapshot, "models")
        if not models:
            return [_model_row(
                self._default_model_short,
                calls=0,
                errors=0,
                prompt=0,
                completion=0,
                cached=0,
            )]

        for m in models:
            if m not in self._seen_set:
                self._seen_models.append(m)
                self._seen_set.add(m)

        rows = []
        for m in self._seen_models:
            md = _mapping(models, m)
            rows.append(_model_row(
                m,
                calls=_int(md, "calls"),
                errors=_int(md, "errors"),
                prompt=_int(md, "prompt_tokens"),
                completion=_int(md, "completion_tokens"),
                cached=_int(md, "cached_tokens"),
            ))
        return rows


def _model_row(
    model: str,
    *,
    calls: int,
    errors: int,
    prompt: int,
    completion: int,
    cached: int,
) -> tuple[str, int]:
    short = model.rsplit("/", 1)[-1]
    req_label = f"{calls:,} req" + (f" ({errors} err)" if errors else "")
    plain = f"    {short}  {req_label} · {prompt:,} in  {completion:,} out"
    styled = (
        f"\x1b[2m    \x1b[0m"
        f"\x1b[1m{short}\x1b[0m  "
        f"\x1b[2m{req_label} · \x1b[0m"
        f"\x1b[96m{prompt:,}\x1b[0m in  "
        f"\x1b[92m{completion:,}\x1b[0m out"
    )
    if cached:
        plain += f"  {cached:,} cached"
        styled += f"  \x1b[33m{cached:,}\x1b[0m cached"
    return (styled, len(plain))


def _image_compression_row(snapshot: Mapping[str, object]) -> tuple[str, int]:
    images_seen = _int(snapshot, "images_seen")
    if images_seen == 0:
        plain = "    image compression · waiting for inline images"
        return (f"\x1b[2m{plain}\x1b[0m", len(plain))

    images_optimized = _int(snapshot, "images_optimized")
    bytes_before = _int(snapshot, "bytes_before")
    bytes_after = _int(snapshot, "bytes_after")
    saved_percent = (bytes_before - bytes_after) / bytes_before * 100 if bytes_before else 0.0
    plain = (
        f"    image compression · {images_optimized}/{images_seen} optimized · "
        f"{_format_bytes(bytes_before)} → {_format_bytes(bytes_after)} · "
        f"{saved_percent:.1f}% saved"
    )
    styled = (
        "\x1b[2m    image compression · \x1b[0m"
        f"\x1b[96m{images_optimized}/{images_seen}\x1b[0m optimized · "
        f"{_format_bytes(bytes_before)} → {_format_bytes(bytes_after)} · "
        f"\x1b[92m{saved_percent:.1f}% saved\x1b[0m"
    )
    return (styled, len(plain))


def _snapshot_cost(snapshot: Mapping[str, object]) -> float:
    total = 0.0
    for model, raw_model in _mapping(snapshot, "models").items():
        model_stats = (
            cast(Mapping[str, object], raw_model) if isinstance(raw_model, Mapping) else {}
        )
        total += estimate_model_cost(
            model,
            prompt_tokens=_int(model_stats, "prompt_tokens"),
            completion_tokens=_int(model_stats, "completion_tokens"),
            cached_tokens=_int(model_stats, "cached_tokens"),
            cache_creation_tokens=_int(model_stats, "cache_creation_tokens"),
        )["total_cost"]
    return total


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value / (1024 * 1024):.1f} MB"


def _mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = data.get(key)
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    return value if isinstance(value, int) else 0
