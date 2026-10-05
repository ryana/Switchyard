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

`observed_domain_ids = true` separately opts into airline and telecom ID choices.
Airline supports `get_user_details`, `get_reservation_details`, and
`cancel_reservation`. Telecom supports `get_customer_by_id`, `get_details_by_id`,
`get_data_usage`, `resume_line`, `enable_roaming`, `disable_roaming`, and
`send_payment_request`. Every required argument must be available. The adapter
reads named IDs and reservation/line/bill arrays only from prior non-error JSON
tool results; it does not extract them from prose or user text. Schema constraints
still apply. This supplies candidates, not authorization: the decision must still
obey authentication, confirmation, and action-order policies.

JSON Schema `default` remains an annotation. For an optional Boolean, omission,
`false`, and `true` remain distinct choices. Generated tool-call IDs contain only
letters, digits, and underscores, allowing clients such as ToolSandbox to use
them as variable names without changing tool semantics.

## Compare identical requests

Set `observe_only = true` in the race table on a separate diagnostic route. Both
services finish and the route always returns the normal model response unchanged.
No winning branch cancels the other. JEV has a separate `service_timeout_ms` operational HTTP timeout (default 30,000 ms);
streaming diagnostic requests are rejected before starting either service.

Record `normal_complete_ms` and `jev_decision_ms` from each audit summary. The
diagnostic HTTP duration includes waiting for both and is not either service's
response time. Use a zero threshold to retain all valid choices, then compare
scores during analysis. `NONE`, malformed answers,
errors, and predictions arriving after the deadline remain in the captures.

Complete tool-call agreement is separate from correctness. Audit policy and
action ordering independently; an LLM response or passing benchmark grade is not
proof that a candidate action was allowed. Repeated draws of one input do not
provide independent correctness examples.

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


## Wait for both and select at an assumed latency

For a non-streaming timing experiment, use a separate route with:

```toml
[routes.race.jev_race]
evaluation_wait_both = true
disable_hold = true
assumed_jev_latency_ms = 200
service_timeout_ms = 30000
threshold = 0.90
```

Use the actual route table name from your configuration. `evaluation_wait_both`
and `observe_only` are mutually exclusive. Both reject streaming before sending
requests. Evaluation waits for both complete responses even when the normal
model finishes first. It selects a valid, score-qualified JEV call only when the
assumed total JEV latency is strictly less than actual `normal_complete_ms`.
Ties select the normal model. If the normal model fails, a qualified JEV call may
recover the request with reason `evaluation_normal_error_qualified_jev`.
Otherwise the normal error is returned. NONE, a low score, invalid responses,
and service errors have distinct `jev_outcome` fields. Existing tool eligibility
and choice limits remain unchanged.

All times share the audit request origin. The assumed latency includes the
whole JEV path. It is not added to normal fallback time. Actual collection wall
time is `elapsed_ms`; it includes both requests and is not a speedup measurement.
Both response payloads and usage remain available. No losing branch is canceled
in this mode. JEV-selected diagnostic responses omit the canceled-LLM-usage header.
Recomputing a different latency on saved turns is a model on that recorded
trajectory; it does not measure the success of an unexecuted conversation.

`disable_hold = true` maps to Rust `deadline: None`. The normal live race still
returns whichever qualifying branch completes first and cancels the other.
Its default remains `max_hold_ms = 400`; disabling the hold does not enable
wait-both behavior. Operational HTTP timeouts are independent of this hold.

Captures add `jev_headers_ms`, body completion `jev_response_ms`, returned
`jev_model`, and named timing/correlation response headers. Header capture uses
an allowlist and existing secret redaction. `normal_headers_ms` for aggregate
clients marks receipt of the complete aggregate, not network header arrival.
