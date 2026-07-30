<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Interactive Image Compression Demo Playbook

## Goal

Have two short Codex conversations about the same 4K dashboard:

1. a normal Switchyard single-model session;
2. the same session with `--image-compression`.

Watch tokens and estimated cost update in the terminal. In the compressed session, also watch the
inline-image payload shrink. When Codex exits, Switchyard leaves the final comparison numbers in the
terminal.

The scripted paired runner remains available as repeatable supporting validation, but it is not the
main presentation.

## Prerequisites

From the repository root:

```bash
uv sync
export OPENAI_API_KEY="..."
```

The demo uses `gpt-5.4-mini-2026-03-17` through OpenAI. Switchyard's cost table uses the current
GPT-5.4 mini prices: $0.75 per million input tokens, $0.075 per million cached input tokens, and
$4.50 per million output tokens. Verify prices before presenting old results:

<https://developers.openai.com/api/docs/models/gpt-5.4-mini>

## 1. Generate the Dashboard

```bash
DEMO_DIR="$PWD/.pytest_cache/image-compression-demo/interactive"
DEMO_IMAGE="$DEMO_DIR/source-dashboard.png"
DEMO_CONFIG="$DEMO_DIR/openai-demo.toml"

uv run --with openai --with pillow python benchmark/run_image_compression_demo.py \
  --generate-only \
  --output-dir "$DEMO_DIR"
```

On macOS, preview it before the demo:

```bash
open "$DEMO_IMAGE"
```

The fixture is a deterministic 3840x2160 operations dashboard. It has four KPI cards, regional
health data, deployment details, and one deliberately incorrect unit.

## 2. Chat With Codex: Compression Off

```bash
uv run switchyard launch codex \
  --model vision \
  --config "$DEMO_CONFIG" \
  -- \
  -c 'model_reasoning_effort="none"' \
  --no-alt-screen \
  "Inspect this dashboard without using tools or changing files. How many KPI cards are in the top row? Answer only the number." \
  --image "$DEMO_IMAGE"
```

The reasoning override is required for this pinned mini model because Codex supplies function tools
and the model's Chat Completions path rejects function tools combined with non-`none` reasoning.

Ask these follow-ups in the same Codex session:

```text
What P95 latency is displayed? Answer only the value and unit.

Which region has the highest error rate? Answer only the region and percentage.

What release is current, and what percentage is in the canary? Answer only the version and percentage.

One KPI card has a mismatched unit. Name the KPI, the displayed unit, and the correct unit.
```

The Switchyard footer updates after each response with request count, input/output/cache tokens, and
estimated cost. Exit Codex after the fifth answer and leave the final Switchyard session summary
visible in the terminal.

## 3. Chat With Codex: Compression On

Run the same command with `--image-compression`:

```bash
uv run switchyard launch codex \
  --model vision \
  --config "$DEMO_CONFIG" \
  --image-compression \
  -- \
  -c 'model_reasoning_effort="none"' \
  --no-alt-screen \
  "Inspect this dashboard without using tools or changing files. How many KPI cards are in the top row? Answer only the number." \
  --image "$DEMO_IMAGE"
```

Ask the same four follow-ups in the same order. The footer now includes an additional row similar to:

```text
image compression · 5/5 optimized · 1.1 MB → 83.0 KB · 92.5% saved
```

Exit Codex after the fifth answer. The session summary includes:

- provider-reported input, output, and cached tokens;
- estimated cost for the pinned model;
- cumulative inline images optimized;
- cumulative image bytes before and after compression.

## 4. Compare

Keep both terminal summaries visible. The useful live-demo result has three parts:

- Codex answers the five questions correctly in both sessions;
- the compressed session reports fewer provider input tokens and lower estimated cost;
- the compressed session shows the actual inline payload reduction.

Do not make a latency claim from one pair of interactive sessions. Provider prompt caching and
normal conversation variance can change absolute cost, so show cached tokens and describe this as a
demo rather than a controlled benchmark.

## Repeatable Supporting Run

Use the automated pair when you want fixed prompts, raw usage, preserved forwarded images, and an
exact 5/5 score:

```bash
uv run --with openai --with pillow python benchmark/run_image_compression_demo.py \
  --output-dir .pytest_cache/image-compression-demo/paired-576
```

That command writes `report.md`, `results.json`, `forwarded-off.png`, and `forwarded-on.webp`. It is
useful before a presentation or after changing the native compression path; the interactive Codex flow above is
the demo itself.

To explore where quality fails:

```bash
uv run --with openai --with pillow python benchmark/run_image_compression_demo.py \
  --max-patch-tokens 256 \
  --output-dir .pytest_cache/image-compression-demo/paired-256
```

The larger, repeatable follow-up is the
[visual-regression detection and repair proposal](image_compression_demo_proposal.md).
