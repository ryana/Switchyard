#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a small paired image-compression demo through Switchyard.

The script generates a deterministic 4K operations dashboard, asks five
vision questions with the screenshot resent on every turn, and compares the
same workload with image compression disabled and enabled.

Usage:
    uv run --with openai --with pillow python benchmark/run_image_compression_demo.py
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from PIL import Image, ImageDraw, ImageFont

from switchyard.cli.launchers.native_server import NativeServer
from switchyard_rust.server import compress_image

_MODEL = "gpt-5.4-mini-2026-03-17"
_INPUT_USD_PER_MILLION = 0.75
_CACHED_INPUT_USD_PER_MILLION = 0.075
_OUTPUT_USD_PER_MILLION = 4.50
_PRICING_URL = "https://developers.openai.com/api/docs/models/gpt-5.4-mini"
_SYSTEM_PROMPT = (
    "You are answering a visual benchmark. Inspect the latest dashboard screenshot. "
    "Return only the requested answer, without explanation."
)
_DEMO_CONFIG = """schema_version = 1

[llm_clients.openai]
format = "openai_chat"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

[targets.vision]
id = "gpt-5.4-mini-2026-03-17"
llm_client = "openai"

[routes.vision]
id = "vision"
type = "passthrough"
target = "vision"
"""


@dataclass(frozen=True)
class Question:
    """One visual question and its accepted answer fragments."""

    prompt: str
    required_fragments: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Usage:
    """Provider-reported token usage for one turn."""

    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class TurnResult:
    """Recorded output for one model turn."""

    turn: int
    question: str
    answer: str
    passed: bool
    latency_ms: float
    usage: Usage
    estimated_cost_usd: float
    compression: dict[str, int] | None


@dataclass(frozen=True)
class ConditionResult:
    """Aggregate result for one compression condition."""

    condition: str
    forwarded_image: str
    forwarded_width: int
    forwarded_height: int
    forwarded_bytes_per_image: int
    turns: list[TurnResult]


_QUESTIONS = (
    Question(
        prompt="How many KPI cards are in the top row? Answer only the number.",
        required_fragments=(("4", "four"),),
    ),
    Question(
        prompt="What P95 latency is displayed? Answer only the value and unit.",
        required_fragments=(("842",), ("ms", "milliseconds")),
    ),
    Question(
        prompt=("Which region has the highest error rate? Answer only the region and percentage."),
        required_fragments=(("us-west-2", "us west 2", "uswest2"), ("3.8",)),
    ),
    Question(
        prompt=(
            "What release is current, and what percentage is in the canary? "
            "Answer only the version and percentage."
        ),
        required_fragments=(("v2.14.7", "2.14.7"), ("25",)),
    ),
    Question(
        prompt=(
            "One KPI card has a mismatched unit. Name the KPI, the displayed unit, "
            "and the correct unit."
        ),
        required_fragments=(
            ("success rate",),
            ("ms", "milliseconds"),
            ("percent", "%"),
        ),
    ),
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _draw_dashboard(output_path: Path) -> bytes:
    """Render the deterministic dashboard fixture and return its PNG bytes."""
    width, height = 3840, 2160
    image = Image.new("RGB", (width, height), "#07111f")
    draw = ImageDraw.Draw(image)

    white = "#f5f7fb"
    muted = "#91a0b7"
    panel = "#101d2f"
    panel_alt = "#14243a"
    border = "#263a56"
    green = "#52d69b"
    amber = "#f4bd62"
    red = "#ff6b79"
    blue = "#62a8ff"

    draw.text((140, 100), "SignalDeck", font=_font(92), fill=white)
    draw.text((140, 215), "Production overview", font=_font(40), fill=muted)
    draw.rounded_rectangle((3080, 110, 3700, 245), radius=64, fill="#123d35")
    draw.ellipse((3140, 155, 3174, 189), fill=green)
    draw.text((3205, 142), "All systems monitored", font=_font(34), fill=green)

    card_y1, card_y2 = 350, 780
    card_gap = 48
    card_width = (3560 - 3 * card_gap) // 4
    cards = (
        ("P95 latency", "842 ms", "+31% vs baseline", red),
        ("Success rate", "99.92 ms", "unit mismatch", amber),
        ("Requests / min", "18,420", "+8.4% today", green),
        ("Active incidents", "2", "1 high severity", red),
    )
    for index, (label, value, note, accent) in enumerate(cards):
        x1 = 140 + index * (card_width + card_gap)
        x2 = x1 + card_width
        draw.rounded_rectangle(
            (x1, card_y1, x2, card_y2),
            radius=34,
            fill=panel,
            outline=border,
            width=4,
        )
        draw.rectangle((x1, card_y1, x1 + 12, card_y2), fill=accent)
        draw.text((x1 + 65, card_y1 + 58), label, font=_font(38), fill=muted)
        draw.text((x1 + 65, card_y1 + 148), value, font=_font(76), fill=white)
        draw.text((x1 + 65, card_y1 + 315), note, font=_font(32), fill=accent)

    left = (140, 850, 2390, 2020)
    right_top = (2450, 850, 3700, 1350)
    right_bottom = (2450, 1410, 3700, 2020)
    for bounds in (left, right_top, right_bottom):
        draw.rounded_rectangle(bounds, radius=34, fill=panel, outline=border, width=4)

    draw.text((210, 920), "Regional health", font=_font(52), fill=white)
    draw.text((210, 995), "Last 15 minutes", font=_font(30), fill=muted)
    columns = (230, 940, 1370, 1780)
    for x, heading in zip(columns, ("Region", "Traffic", "Error rate", "P95"), strict=True):
        draw.text((x, 1115), heading, font=_font(31), fill=muted)
    draw.line((210, 1170, 2320, 1170), fill=border, width=3)

    rows = (
        ("us-east-1", "12,480 rpm", "0.4%", "118 ms", green),
        ("us-west-2", "3,920 rpm", "3.8%", "842 ms", red),
        ("eu-west-1", "2,020 rpm", "0.7%", "164 ms", green),
    )
    for index, row in enumerate(rows):
        y = 1240 + index * 225
        if index == 1:
            draw.rounded_rectangle((205, y - 35, 2325, y + 130), radius=24, fill="#331c2a")
        region, traffic, error_rate, latency, accent = row
        draw.ellipse((235, y + 18, 267, y + 50), fill=accent)
        for x, value in zip(
            (290, columns[1], columns[2], columns[3]),
            (region, traffic, error_rate, latency),
            strict=True,
        ):
            draw.text((x, y), value, font=_font(39), fill=white)

    draw.text((2520, 920), "Deployment", font=_font(50), fill=white)
    draw.text((2520, 1020), "Current release", font=_font(31), fill=muted)
    draw.text((2520, 1080), "v2.14.7", font=_font(65), fill=blue)
    draw.text((3130, 1020), "Canary", font=_font(31), fill=muted)
    draw.text((3130, 1080), "25%", font=_font(65), fill=amber)
    draw.rounded_rectangle((2520, 1220, 3630, 1255), radius=18, fill=panel_alt)
    draw.rounded_rectangle((2520, 1220, 2798, 1255), radius=18, fill=amber)

    draw.text((2520, 1480), "Incident focus", font=_font(50), fill=white)
    draw.rounded_rectangle((2520, 1580, 3630, 1920), radius=26, fill="#261c2b")
    draw.text((2580, 1640), "INC-2041", font=_font(32), fill=red)
    draw.text((2580, 1710), "Elevated checkout errors", font=_font(43), fill=white)
    draw.text((2580, 1790), "Primary region: us-west-2", font=_font(31), fill=muted)
    draw.text((2580, 1850), "Owner: Edge Reliability", font=_font(31), fill=muted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path.read_bytes()


def _data_url(payload: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"


def _normalize(value: str) -> str:
    value = value.lower().replace("%", " percent ")
    return " ".join(re.sub(r"[^a-z0-9.]+", " ", value).split())


def _answer_passes(answer: str, question: Question) -> bool:
    normalized = _normalize(answer)
    return all(
        any(_normalize(option) in normalized for option in alternatives)
        for alternatives in question.required_fragments
    )


def _usage(response: ChatCompletion) -> Usage:
    if response.usage is None:
        raise RuntimeError("provider response did not include token usage")
    details = response.usage.prompt_tokens_details
    cached_tokens = details.cached_tokens if details and details.cached_tokens else 0
    return Usage(
        prompt_tokens=response.usage.prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=response.usage.completion_tokens,
    )


def _cost(usage: Usage) -> float:
    fresh_tokens = max(usage.prompt_tokens - usage.cached_tokens, 0)
    return (
        fresh_tokens * _INPUT_USD_PER_MILLION
        + usage.cached_tokens * _CACHED_INPUT_USD_PER_MILLION
        + usage.completion_tokens * _OUTPUT_USD_PER_MILLION
    ) / 1_000_000


def _image_details(payload: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(payload)) as image:
        return cast(tuple[int, int], image.size)


def _write_demo_config(output_dir: Path) -> Path:
    config = output_dir / "openai-demo.toml"
    config.write_text(_DEMO_CONFIG)
    return config


def _compression_stats(snapshot: Mapping[str, object]) -> dict[str, int]:
    raw = snapshot.get("image_compression")
    if not isinstance(raw, Mapping):
        return {}
    return {key: value for key, value in raw.items() if isinstance(key, str) and isinstance(value, int)}


def _compression_delta(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, int]:
    before_stats = _compression_stats(before)
    after_stats = _compression_stats(after)
    return {
        key: max(value - before_stats.get(key, 0), 0)
        for key, value in after_stats.items()
    }


async def _run_condition(
    condition: str,
    source_data_url: str,
    source_bytes: bytes,
    output_dir: Path,
    max_patch_tokens: int,
) -> ConditionResult:
    server = NativeServer(
        _write_demo_config(output_dir),
        image_compression=condition == "on",
        image_max_patch_tokens=max_patch_tokens,
    )
    client = AsyncOpenAI(api_key="switchyard", base_url=f"{server.base_url}/v1", timeout=120)

    history: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": _SYSTEM_PROMPT}
    ]
    turns: list[TurnResult] = []
    try:
        for turn_number, question in enumerate(_QUESTIONS, start=1):
            user_message: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": [
                    {"type": "text", "text": question.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": source_data_url, "detail": "high"},
                    },
                ],
            }
            before = server.stats.snapshot_sync()
            started = time.perf_counter()
            response = await client.chat.completions.create(
                model="vision",
                messages=[*history, user_message],
                max_completion_tokens=100,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            after = server.stats.snapshot_sync()
            answer = response.choices[0].message.content
            if answer is None:
                raise RuntimeError("provider response did not contain message content")

            provider_usage = _usage(response)
            compression = _compression_delta(before, after) if condition == "on" else None
            turns.append(
                TurnResult(
                    turn=turn_number,
                    question=question.prompt,
                    answer=answer,
                    passed=_answer_passes(answer, question),
                    latency_ms=round(latency_ms, 1),
                    usage=provider_usage,
                    estimated_cost_usd=_cost(provider_usage),
                    compression=compression,
                )
            )
            history.extend((user_message, {"role": "assistant", "content": answer}))
    finally:
        await client.close()
        server.close()

    forwarded_bytes = source_bytes
    suffix = "png"
    if condition == "on":
        compressed, _ = compress_image(source_bytes, max_patch_tokens=max_patch_tokens)
        forwarded_bytes = bytes(compressed)
        suffix = "webp"
    forwarded_path = output_dir / f"forwarded-{condition}.{suffix}"
    forwarded_path.write_bytes(forwarded_bytes)
    forwarded_width, forwarded_height = _image_details(forwarded_bytes)
    return ConditionResult(
        condition=condition,
        forwarded_image=forwarded_path.name,
        forwarded_width=forwarded_width,
        forwarded_height=forwarded_height,
        forwarded_bytes_per_image=len(forwarded_bytes),
        turns=turns,
    )


def _sum_usage(result: ConditionResult) -> Usage:
    return Usage(
        prompt_tokens=sum(turn.usage.prompt_tokens for turn in result.turns),
        cached_tokens=sum(turn.usage.cached_tokens for turn in result.turns),
        completion_tokens=sum(turn.usage.completion_tokens for turn in result.turns),
    )


def _sum_cost(result: ConditionResult) -> float:
    return sum(turn.estimated_cost_usd for turn in result.turns)


def _total_forwarded_bytes(result: ConditionResult, source_bytes: int) -> int:
    if result.condition == "off":
        return source_bytes * sum(range(1, len(result.turns) + 1))
    return sum((turn.compression or {}).get("bytes_after", 0) for turn in result.turns)


def _total_patch_estimate(
    result: ConditionResult,
    source_size: tuple[int, int],
) -> int:
    if result.condition == "off":
        width, height = source_size
        patches = math.ceil(width / 32) * math.ceil(height / 32)
        return patches * sum(range(1, len(result.turns) + 1))
    return sum(
        (turn.compression or {}).get("estimated_patch_tokens_after", 0) for turn in result.turns
    )


def _percentage_change(before: float, after: float) -> float:
    return (after - before) / before * 100 if before else 0.0


def _write_results(
    output_dir: Path,
    source_bytes: bytes,
    results: list[ConditionResult],
    max_patch_tokens: int,
) -> None:
    source_size = _image_details(source_bytes)
    payload = {
        "model": _MODEL,
        "pricing_usd_per_million_tokens": {
            "input": _INPUT_USD_PER_MILLION,
            "cached_input": _CACHED_INPUT_USD_PER_MILLION,
            "output": _OUTPUT_USD_PER_MILLION,
            "source": _PRICING_URL,
        },
        "max_patch_tokens": max_patch_tokens,
        "source": {
            "image": "source-dashboard.png",
            "width": source_size[0],
            "height": source_size[1],
            "bytes": len(source_bytes),
        },
        "conditions": [asdict(result) for result in results],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    rows: list[str] = []
    for result in results:
        usage = _sum_usage(result)
        passed = sum(turn.passed for turn in result.turns)
        mean_latency = sum(turn.latency_ms for turn in result.turns) / len(result.turns)
        rows.append(
            f"| {result.condition} | {passed}/{len(result.turns)} | "
            f"{usage.prompt_tokens:,} | {usage.cached_tokens:,} | "
            f"{usage.completion_tokens:,} | ${_sum_cost(result):.5f} | "
            f"{mean_latency:,.0f} | "
            f"{_total_forwarded_bytes(result, len(source_bytes)):,} | "
            f"{_total_patch_estimate(result, source_size):,} |"
        )

    off, on = results
    cost_change = _percentage_change(_sum_cost(off), _sum_cost(on))
    token_change = _percentage_change(
        _sum_usage(off).prompt_tokens,
        _sum_usage(on).prompt_tokens,
    )
    latency_change = _percentage_change(
        sum(turn.latency_ms for turn in off.turns),
        sum(turn.latency_ms for turn in on.turns),
    )
    report = f"""# Image Compression Demo Results

Model: `{_MODEL}`

Compression budget: `{max_patch_tokens}` estimated 32x32 patches per image

Pricing: [OpenAI GPT-5.4 mini]({_PRICING_URL})

| Compression | Quality | Input tokens | Cached input | Output tokens | Est. cost | Mean latency (ms) | Cumulative image bytes | Switchyard patch estimate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Compression changed provider-reported input tokens by {token_change:+.1f}%, estimated total cost by
{cost_change:+.1f}%, and total wall-clock latency by {latency_change:+.1f}%.

## Forwarded Image

- Off: `{off.forwarded_width}x{off.forwarded_height}`, {off.forwarded_bytes_per_image:,} bytes
- On: `{on.forwarded_width}x{on.forwarded_height}`, {on.forwarded_bytes_per_image:,} bytes

## Answers

"""
    for result in results:
        report += f"### Compression {result.condition}\n\n"
        for turn in result.turns:
            mark = "PASS" if turn.passed else "FAIL"
            report += f"{turn.turn}. **{mark}** — {turn.question}\n   - `{turn.answer.strip()}`\n"
        report += "\n"
    (output_dir / "report.md").write_text(report)


async def _run(output_dir: Path, max_patch_tokens: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_demo_config(output_dir)
    source_path = output_dir / "source-dashboard.png"
    source_bytes = _draw_dashboard(source_path)
    source_data_url = _data_url(source_bytes)

    results = [
        await _run_condition(
            condition,
            source_data_url,
            source_bytes,
            output_dir,
            max_patch_tokens,
        )
        for condition in ("off", "on")
    ]
    _write_results(output_dir, source_bytes, results, max_patch_tokens)
    print((output_dir / "report.md").read_text())
    print(f"Artifacts: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = (
        Path(".pytest_cache")
        / "image-compression-demo"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Artifact directory (default: a timestamped .pytest_cache path)",
    )
    parser.add_argument(
        "--max-patch-tokens",
        type=int,
        default=576,
        help="Switchyard image patch budget for the compression-on condition",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate the dashboard fixture without making model requests",
    )
    args = parser.parse_args()

    if args.max_patch_tokens < 1:
        parser.error("--max-patch-tokens must be at least 1")
    if args.generate_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config_path = _write_demo_config(args.output_dir)
        fixture_path = args.output_dir / "source-dashboard.png"
        _draw_dashboard(fixture_path)
        print(f"Fixture: {fixture_path.resolve()}")
        print(f"Config: {config_path.resolve()}")
        return
    if "OPENAI_API_KEY" not in os.environ:
        parser.error("OPENAI_API_KEY must be set")
    asyncio.run(_run(args.output_dir, args.max_patch_tokens))


if __name__ == "__main__":
    main()
