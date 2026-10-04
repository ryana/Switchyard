# Race JEV against an LLM

This route starts the normal LLM request and asks JEV to select a complete tool
call at the same time. It holds streamed output for up to 400 ms by default. A
valid JEV choice with enough confidence wins if it arrives before the LLM finishes
and before the deadline. Switchyard then drops the unfinished LLM request.

The LLM wins immediately if it finishes first. A JEV error, low score, `NONE`
choice, or deadline releases the buffered LLM output and continues the original
stream. Once released, that stream cannot be replaced by JEV. Requests without
eligible tools go straight to the LLM.

## Run locally

Set `JEV_KEY` and `INFERENCE_HUB_KEY` in the environment, then run from the
repository root:

```sh
cargo run -p switchyard-server -- \
  --config examples/jev-race/routes.toml --host 127.0.0.1 --port 4010
```

Point an OpenAI-compatible client at `http://127.0.0.1:4010/v1`. Use model
`sy-race` for the race or `sy-baseline` for ordinary LLM requests. `sy-user` is a
separate passthrough alias for a benchmark's user simulator. Both streamed and
buffered responses are supported. The example uses the same inference model on
all three routes.

To check the complete HTTP path without provider credentials, build the binary
and run the controlled smoke test. It starts and stops its own local proxy and
providers, checks both response modes, and retains the requests and results:

```sh
cargo build -p switchyard-server --locked
uv run --no-project python examples/jev-race/smoke.py \
  --server target/debug/switchyard-server --output tmp/jev-race-smoke
```

Use a new output directory for each run. The check includes early reasoning
chunks, buffer release, unknown JEV choices, provider failures, and an observed
upstream connection close when JEV wins.

The `[routes.<name>.jev_race]` table explicitly enables the behavior. It is
supported on passthrough routes without subagent routing. Other routes keep their
existing behavior. `enabled = false` disables JEV while retaining optional request
recording, so a baseline can use the same capture path. A disabled race does not
require a JEV credential.

## Which tools qualify

Every argument must have a finite set of supported values: string, integer, or
boolean enums; booleans; or integers with an explicit minimum and maximum. Calls
with no arguments can qualify. Optional arguments may be omitted. Unsupported
schemas and open text are excluded. The full choice set must fit within 254
complete calls plus `NONE`; it is never partially truncated.

JEV sees the conversation and the full tool menu, not just the eligible tools.
Its selected choice probability must meet `threshold`, which defaults to `0.90`.
The returned choice and probability distribution are validated before use.

`observed_retail_ids = true` opts into the existing tau-bench retail lookup rules.
They turn IDs from earlier tool results into candidate arguments for
`get_user_details`, `get_order_details`, and `get_product_details`. These rules
never read the benchmark database or future messages. They are off by default.

## Measurements and limits

The optional `audit_directory` saves per-request records. It is relative to the
server's working directory. The example writes under ignored `tmp/jev-race/`.
Records include request content, JEV choices, responses, winner, timing, and
available usage. These files can contain private conversation data. API keys and
authentication headers are excluded or redacted.

Compare the baseline and race with the same tasks and model settings. Measure
whole-task success and elapsed time, including any extra turns caused by a wrong
choice. Holding output can delay the first text or reasoning shown to a person;
it is not free for an interactive chat UI.

Canceling the client request does not prove the inference server stopped working.
An interrupted LLM stream may never provide usage. Keep that usage unknown and
report JEV usage separately. Tokens delivered to an agent, provider-reported
generated tokens, and billed tokens are different measurements.

A JEV response reports JEV's own input and output usage. It includes
`x-switchyard-winner: jev`, `x-switchyard-usage-source: jev`, and
`x-switchyard-canceled-llm-usage: unknown` headers. Those counts do not include
the canceled LLM request.
