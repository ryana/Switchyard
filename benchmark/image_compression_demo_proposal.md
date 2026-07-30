<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Image Compression Demo Proposal

## Status

Future demo proposal for the inline-image compression work. This is deliberately separate from the
smaller exploratory example used to iterate on the feature today.

## Goal

Demonstrate that Switchyard can reduce the provider-reported inference cost of image-heavy agent
workloads without materially reducing task quality or adding unacceptable latency.

The demo should make three effects visible:

1. resizing reduces provider-billed visual input tokens;
2. WebP encoding reduces request bytes, even when dimensions and visual-token cost do not change;
3. an aggressive visual-token budget eventually degrades task accuracy.

## Recommended Demo: Visual Regression Detection and Repair

Build a polished local website and a second version containing controlled visual regressions. Render
baseline and regressed screenshots at desktop and mobile breakpoints, then ask a vision-capable model
to compare each pair.

The detection phase returns structured JSON:

```json
{
  "page": "checkout",
  "changed": true,
  "regressions": [
    {
      "region": "order summary",
      "category": "overflow",
      "severity": "high",
      "description": "The total price is clipped at the mobile breakpoint."
    }
  ]
}
```

Each case has known ground truth. Candidate regressions should include:

- spacing and alignment;
- wrong colors or typography;
- missing or duplicated elements;
- clipping and overflow;
- desktop-only and mobile-only defects;
- one subtle defect that establishes the quality-failure boundary.

After the cost-quality comparison, give the detected regressions to a coding agent and let it repair
the site. The repair phase is the visual finale, not the primary benchmark, because variable code
generation and tool use can obscure the image-cost signal.

## Why This Workload

- Baseline/current pairs create multiple large images per task.
- Regression labels make answer quality objectively scoreable.
- Website screenshots are recognizable and visually compelling in a live presentation.
- The task resembles screenshot-driven coding and browser-agent workflows.
- The same fixture can test conservative and aggressive compression budgets.

## Experiment Matrix

Use the same model snapshot, prompts, image order, output schema, and decoding settings for every
condition.

| Condition | Switchyard configuration | Purpose |
|---|---|---|
| Off | No image processor | Establish quality, cost, bytes, and latency |
| WebP only | `max_patch_tokens=None` | Isolate transport-byte savings |
| Conservative | `max_patch_tokens=2304` | Preserve high visual fidelity |
| Balanced | `max_patch_tokens=1024` | Find a likely operating point |
| Default | `max_patch_tokens=576` | Evaluate the current default |
| Aggressive | `max_patch_tokens=256` | Expose the quality-failure boundary |

Send images as inline base64 content. The current processor intentionally leaves remote URLs and
animated images unchanged.

Run the detection benchmark more than once per condition and randomize condition order. Keep the
repair phase separate and run it only for the baseline and the best compression operating point.

## Measurements

### Primary

- provider-reported input tokens and calculated input cost;
- total inference cost, split into input and output;
- regression precision, recall, and F1;
- exact-match accuracy for page, region, category, and severity.

### Secondary

- Switchyard `bytes_before`, `bytes_after`, and `bytes_saved`;
- Switchyard estimated patch tokens before and after;
- end-to-end latency and time to first token;
- output-token count;
- coding-agent repair success, verified by rerendering and comparing screenshots;
- test and build results after the repair.

Provider-reported usage is the cost source of truth. Switchyard currently estimates 32×32 patches,
while providers apply their own tokenization and resizing rules. For example, Claude documents
28×28 visual tokens and model-specific resolution caps:

<https://platform.claude.com/docs/en/build-with-claude/vision#resolution-and-token-cost>

## Success Criteria

Identify at least one compression setting that:

- reduces provider-reported visual input cost by at least 40%;
- keeps regression-detection F1 within 5 percentage points of the uncompressed baseline;
- adds less than 15% to end-to-end latency;
- preserves successful repair of the representative site regressions.

Also identify the first aggressive setting where quality fails. A credible quality boundary is more
useful than presenting compression as universally free.

## Presentation

Show four synchronized views:

1. the baseline and regressed screenshots;
2. the actual image forwarded upstream with compression off or on;
3. the model's structured regression report and repaired page;
4. a live cost-quality chart with bytes, provider tokens, cost, latency, and F1.

The central result should be a Pareto curve rather than one favorable before/after example.

## Risks and Controls

- **Provider tokenization differs from the Switchyard estimate.** Use response usage and verified
  model pricing for the headline calculation.
- **Some providers normalize images to a fixed resolution.** Verify that the selected model's billed
  usage changes with dimensions before building the demo around it.
- **Lossy encoding can obscure small text or subtle defects.** Preserve the actual forwarded images
  as run artifacts and inspect quality failures.
- **Prompt caching can change multi-turn input prices.** Report cached and uncached input separately,
  or use independent single-turn detection requests for the primary comparison.
- **Coding trajectories are nondeterministic.** Score detection independently and treat repair cost
  as a secondary end-to-end result.
- **WebP support varies by upstream.** Verify the selected provider and model before the benchmark.

## Proposed Artifacts

When this proposal is implemented, keep the following under `benchmark/`:

- deterministic website fixtures and regression manifest;
- screenshot renderer with pinned viewport sizes;
- benchmark runner and fixed prompts;
- per-request raw usage and Switchyard compression metadata;
- scored result JSON;
- a generated cost-quality chart;
- a short, reproducible demo command.

## Deferred Next Step

Do not build the full fixture while informally evaluating the processor. First use a smaller
multi-turn image task to verify that inline payload rewriting, provider usage accounting, and visible
answer quality behave as expected. Promote that harness into this benchmark only after those basics
are confirmed.
