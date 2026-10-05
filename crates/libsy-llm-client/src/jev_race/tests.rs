// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use switchyard_protocol::{LlmResponseChunk, ToolDefinition, Usage, text_request, text_response};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

struct Dropped(Arc<AtomicBool>);
impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct FakeNormal {
    delay: Duration,
    chunks: Option<Vec<(Duration, LlmResponseStreamEvent)>>,
    dropped: Arc<AtomicBool>,
}

#[async_trait]
impl RoutedLlmClient for FakeNormal {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        let guard = Dropped(self.dropped.clone());
        tokio::time::sleep(self.delay).await;
        let body = if let Some(chunks) = &self.chunks {
            let chunks = chunks.clone();
            LlmResponse::Stream(Box::pin(async_stream::stream! {
                let _guard = guard;
                for (delay, event) in chunks {
                    if !delay.is_zero() { tokio::time::sleep(delay).await; }
                    yield Ok(event);
                }
            }))
        } else {
            drop(guard);
            let mut response = text_response(Some("normal-returned".into()), "normal answer");
            response.usage.output_tokens = Some(2);
            LlmResponse::Agg(response)
        };
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("normal-request"));
        Ok(Response {
            llm_response: body,
            metadata: request.metadata,
            upstream_headers: headers,
        })
    }
}

fn request(stream: bool) -> Request {
    let mut llm_request = text_request(Some("normal".into()), "Choose the action");
    llm_request.stream = stream;
    llm_request.tools = vec![ToolDefinition {
        name: "choose".into(),
        description: Some("Choose action".into()),
        parameters: json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}),
        strict: None,
    }];
    Request {
        llm_request,
        metadata: Some(Metadata {
            session_id: Some("test-session".into()),
            ..Metadata::default()
        }),
        ..Request::default()
    }
}

fn decision(selected: &str, probability: f64) -> Value {
    json!({"model":"jev-test", "answers":{"next_call":{"type":"choice","choice":selected,"confidence":probability,
        "probabilities":{"call_000": if selected=="call_000" {probability} else {0.0},
            "call_001": if selected=="call_001" {probability} else {0.0},
            "NONE": if selected=="NONE" {1.0} else {1.0-probability}}}},
        "usage":{"input_tokens":123,"output_tokens":9}})
}

#[test]
fn generated_call_id_is_a_valid_python_identifier() {
    let response = jev_response(
        &request(false),
        &CompleteCall {
            name: "choose".into(),
            arguments: json!({"ok": true}),
            description: "Choose".into(),
            origin: "native".into(),
        },
        &decision("call_001", 0.99),
        "1791106242027159000-5",
    );
    let LlmResponse::Agg(aggregate) = response.llm_response else {
        panic!("aggregate expected")
    };
    let ContentBlock::ToolCall(call) = &aggregate.outputs[0].content[0] else {
        panic!("tool call expected")
    };
    assert!(call.id.starts_with("call_"));
    assert!(
        call.id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
    );
}

async fn server(delay: u64, response: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(response)
                .set_delay(Duration::from_millis(delay)),
        )
        .mount(&server)
        .await;
    server
}

fn directory() -> PathBuf {
    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tmp/jev-race-tests");
    std::fs::create_dir_all(&root).unwrap();
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let path = root.join(format!("{}-{id}-{sequence}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    path
}

fn read_summary(root: &std::path::Path) -> Value {
    let directory = std::fs::read_dir(root)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    serde_json::from_slice(&std::fs::read(directory.join("summary.json")).unwrap()).unwrap()
}

fn client(server: &MockServer, normal: FakeNormal, deadline: u64, audit: PathBuf) -> JevRaceClient {
    JevRaceClient::new(
        Arc::new(normal),
        JevRaceConfig {
            endpoint: format!("{}/v1/systemone", server.uri()),
            deadline: Some(Duration::from_millis(deadline)),
            audit_directory: Some(audit),
            ..JevRaceConfig::default()
        },
        "test-key".into(),
    )
    .unwrap()
}

fn text_event(value: &str) -> LlmResponseStreamEvent {
    LlmResponseStreamEvent::preserved(
        "openai_chat",
        json!({"original":value}),
        vec![LlmResponseChunk::TextDelta {
            index: 0,
            text: value.into(),
        }],
    )
}

#[tokio::test]
async fn diagnostic_finishes_both_requests_in_either_arrival_order() {
    for (jev_delay, normal_delay) in [(5, 65), (65, 5)] {
        let server = server(jev_delay, decision("call_001", 0.99)).await;
        let audit = directory();
        let mut client = client(
            &server,
            FakeNormal {
                delay: Duration::from_millis(normal_delay),
                chunks: None,
                dropped: Arc::new(AtomicBool::new(false)),
            },
            20,
            audit.clone(),
        );
        client.config.observe_only = true;
        let response = client.call(request(false)).await.unwrap();
        assert_eq!(response.upstream_headers["x-request-id"], "normal-request");
        let result = response.llm_response.into_agg().await.unwrap();
        assert_eq!(
            result.outputs[0].content,
            text_response(None, "normal answer").outputs[0].content
        );
        let summary = read_summary(&audit);
        assert_eq!(summary["winner"], "llm");
        assert_eq!(summary["reason"], "observe_only");
        assert_eq!(summary["observe_only"], true);
        assert_eq!(summary["jev_answer"]["choice"], "call_001");
        assert_eq!(summary["normal_usage"]["output_tokens"], 2);
        assert_eq!(summary["jev_usage"]["output_tokens"], 9);
        assert_eq!(summary["normal_cancel_requested"], false);
        assert_eq!(summary["jev_cancel_requested"], false);
        let jev_ms = summary["jev_decision_ms"].as_f64().unwrap();
        let normal_ms = summary["normal_complete_ms"].as_f64().unwrap();
        assert_eq!(jev_ms < normal_ms, jev_delay < normal_delay);
    }
}

#[tokio::test]
async fn diagnostic_rejects_streaming_before_starting_upstream_work() {
    let server = server(0, decision("call_001", 0.99)).await;
    let dropped = Arc::new(AtomicBool::new(false));
    let mut client = client(
        &server,
        FakeNormal {
            delay: Duration::ZERO,
            chunks: None,
            dropped: dropped.clone(),
        },
        400,
        directory(),
    );
    client.config.observe_only = true;
    assert!(client.call(request(true)).await.is_err());
    assert!(!dropped.load(Ordering::SeqCst));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn jev_can_win_after_normal_first_token_but_before_completion() {
    let server = server(30, decision("call_001", 0.99)).await;
    let audit = directory();
    let dropped = Arc::new(AtomicBool::new(false));
    let client = client(
        &server,
        FakeNormal {
            delay: Duration::ZERO,
            chunks: Some(vec![
                (Duration::ZERO, text_event("first")),
                (Duration::from_secs(2), text_event("late")),
            ]),
            dropped: dropped.clone(),
        },
        400,
        audit.clone(),
    );
    let response = client.call(request(true)).await.unwrap();
    assert_eq!(
        response.upstream_headers["x-switchyard-usage-source"],
        "jev"
    );
    assert_eq!(
        response.upstream_headers["x-switchyard-canceled-llm-usage"],
        "unknown"
    );
    assert_eq!(
        response.metadata.as_ref().unwrap().session_id.as_deref(),
        Some("test-session")
    );
    let result = response.llm_response.into_agg().await.unwrap();
    assert_eq!(result.usage.input_tokens, Some(123));
    assert_eq!(result.usage.output_tokens, Some(9));
    assert!(
        matches!(&result.outputs[0].content[0], ContentBlock::ToolCall(call) if call.name=="choose" && call.arguments==json!({"ok":true}))
    );
    assert!(dropped.load(Ordering::SeqCst));
    let summary = read_summary(&audit);
    assert_eq!(summary["winner"], "jev");
    assert!(summary["normal_first_output_ms"].is_number());
    assert!(summary["normal_usage"].is_null());
    assert_eq!(summary["jev_usage"]["output_tokens"], 9);
    assert!(summary["normal_backend_stopped"].is_null());
}

#[tokio::test]
async fn deadline_applies_while_consuming_and_fallback_preserves_exact_events() {
    let server = server(500, decision("call_001", 0.99)).await;
    let audit = directory();
    let events = vec![
        text_event("a"),
        text_event("b"),
        text_event("c"),
        LlmResponseChunk::Usage(Usage {
            output_tokens: Some(3),
            ..Usage::default()
        })
        .into(),
    ];
    let client = client(
        &server,
        FakeNormal {
            delay: Duration::ZERO,
            chunks: Some(
                events
                    .iter()
                    .cloned()
                    .map(|event| (Duration::from_millis(30), event))
                    .collect(),
            ),
            dropped: Arc::new(AtomicBool::new(false)),
        },
        70,
        audit.clone(),
    );
    let started = Instant::now();
    let response = client.call(request(true)).await.unwrap();
    assert!(started.elapsed() >= Duration::from_millis(70));
    assert!(started.elapsed() < Duration::from_millis(350));
    assert_eq!(response.upstream_headers["x-request-id"], "normal-request");
    assert_eq!(
        response.metadata.as_ref().unwrap().session_id.as_deref(),
        Some("test-session")
    );
    let LlmResponse::Stream(mut stream) = response.llm_response else {
        panic!("stream required")
    };
    let mut actual = Vec::new();
    while let Some(event) = stream.next().await {
        actual.push(event.unwrap());
    }
    assert_eq!(actual, events);
    let summary = read_summary(&audit);
    assert_eq!(summary["reason"], "deadline");
    assert_eq!(summary["winner"], "llm");
    assert_eq!(summary["jev_cancel_requested"], true);
    assert_eq!(summary["normal_usage"]["output_tokens"], 3);
    assert!(summary["buffer_hold_ms"].as_f64().unwrap() > 10.0);
    assert!(
        summary["first_forwarded_output_ms"].as_f64().unwrap()
            >= summary["selected_ms"].as_f64().unwrap()
    );
}

#[tokio::test]
async fn ready_normal_delta_burst_does_not_starve_jev() {
    let server = server(0, decision("call_001", 0.99)).await;
    let audit = directory();
    let client = client(
        &server,
        FakeNormal {
            delay: Duration::ZERO,
            chunks: Some(
                (0..10_000)
                    .map(|_| (Duration::ZERO, text_event("x")))
                    .collect(),
            ),
            dropped: Arc::new(AtomicBool::new(false)),
        },
        1_500,
        audit.clone(),
    );
    let result = client
        .call(request(true))
        .await
        .unwrap()
        .llm_response
        .into_agg()
        .await
        .unwrap();
    assert_eq!(result.model.as_deref(), Some("jev-test"));
    let summary = read_summary(&audit);
    assert_eq!(summary["winner"], "jev");
    assert!(summary["normal_first_output_ms"].is_number());
    assert!(summary["normal_complete_ms"].is_null());
}

#[tokio::test]
async fn missing_jev_usage_falls_back_instead_of_manufacturing_zero_usage() {
    let mut answer = decision("call_001", 0.99);
    answer.as_object_mut().unwrap().remove("usage");
    let server = server(0, answer).await;
    let audit = directory();
    let client = client(
        &server,
        FakeNormal {
            delay: Duration::from_millis(30),
            chunks: None,
            dropped: Arc::new(AtomicBool::new(false)),
        },
        400,
        audit.clone(),
    );
    let result = client
        .call(request(false))
        .await
        .unwrap()
        .llm_response
        .into_agg()
        .await
        .unwrap();
    assert_eq!(result.model.as_deref(), Some("normal-returned"));
    let summary = read_summary(&audit);
    assert_eq!(summary["reason"], "jev_error");
    assert!(summary["jev_error"].as_str().unwrap().contains("usage"));
}

#[tokio::test]
async fn opaque_conversation_continuation_does_not_call_jev() {
    let server = server(0, decision("call_001", 0.99)).await;
    let audit = directory();
    let client = client(
        &server,
        FakeNormal {
            delay: Duration::from_millis(5),
            chunks: None,
            dropped: Arc::new(AtomicBool::new(false)),
        },
        400,
        audit.clone(),
    );
    let mut request = request(false);
    request.raw_request = Some(json!({"previous_response_id":"private-context"}));
    let result = client
        .call(request)
        .await
        .unwrap()
        .llm_response
        .into_agg()
        .await
        .unwrap();
    assert_eq!(result.model.as_deref(), Some("normal-returned"));
    assert_eq!(server.received_requests().await.unwrap().len(), 0);
    assert_eq!(read_summary(&audit)["reason"], "opaque_conversation_state");
}

#[tokio::test]
async fn complete_normal_response_wins_and_preserves_aggregate() {
    let server = server(300, decision("call_001", 0.99)).await;
    let audit = directory();
    let client = client(
        &server,
        FakeNormal {
            delay: Duration::from_millis(15),
            chunks: None,
            dropped: Arc::new(AtomicBool::new(false)),
        },
        400,
        audit.clone(),
    );
    let response = client.call(request(false)).await.unwrap();
    assert_eq!(
        response
            .llm_response
            .into_agg()
            .await
            .unwrap()
            .model
            .as_deref(),
        Some("normal-returned")
    );
    let summary = read_summary(&audit);
    assert_eq!(summary["reason"], "normal_complete");
    assert_eq!(summary["normal_usage"]["output_tokens"], 2);
    assert!(
        summary["normal_complete_ms"].as_f64().unwrap() <= summary["selected_ms"].as_f64().unwrap()
    );
}

#[tokio::test]
async fn abstention_is_permanent_and_normal_request_is_unchanged() {
    let server = server(5, decision("NONE", 1.0)).await;
    let audit = directory();
    let client = client(
        &server,
        FakeNormal {
            delay: Duration::from_millis(50),
            chunks: None,
            dropped: Arc::new(AtomicBool::new(false)),
        },
        400,
        audit.clone(),
    );
    let result = client
        .call(request(false))
        .await
        .unwrap()
        .llm_response
        .into_agg()
        .await
        .unwrap();
    assert_eq!(result.model.as_deref(), Some("normal-returned"));
    assert_eq!(
        read_summary(&audit)["reason"],
        "jev_abstained_or_below_cutoff"
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let payload: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(payload["questions"]["next_call"]["criteria"]["NONE"].is_string());
    assert_eq!(
        payload["state"]["messages"][0]["content"],
        "Choose the action"
    );
}

#[tokio::test]
async fn malformed_prediction_is_recorded_and_falls_back() {
    let server = server(5, json!({"model":"bad", "answers":{}})).await;
    let audit = directory();
    let client = client(
        &server,
        FakeNormal {
            delay: Duration::from_millis(40),
            chunks: None,
            dropped: Arc::new(AtomicBool::new(false)),
        },
        400,
        audit.clone(),
    );
    client.call(request(false)).await.unwrap();
    let summary = read_summary(&audit);
    assert_eq!(summary["reason"], "jev_error");
    assert!(summary["jev_error"].is_string());
}

#[tokio::test]
async fn caller_disconnect_cancels_owned_pending_request() {
    let server = server(500, decision("call_001", 0.99)).await;
    let audit = directory();
    let dropped = Arc::new(AtomicBool::new(false));
    let client = client(
        &server,
        FakeNormal {
            delay: Duration::from_secs(2),
            chunks: None,
            dropped: dropped.clone(),
        },
        800,
        audit.clone(),
    );
    let task = tokio::spawn(async move { client.call(request(true)).await });
    tokio::time::sleep(Duration::from_millis(35)).await;
    task.abort();
    let _ = task.await;
    tokio::task::yield_now().await;
    assert!(dropped.load(Ordering::SeqCst));
    let summary = read_summary(&audit);
    assert_eq!(summary["finish_reason"], "caller_disconnected");
    assert_eq!(summary["normal_cancel_requested"], true);
    assert_eq!(summary["jev_cancel_requested"], true);
}

#[tokio::test]
async fn downstream_drop_closes_remaining_stream() {
    let server = server(500, decision("call_001", 0.99)).await;
    let audit = directory();
    let dropped = Arc::new(AtomicBool::new(false));
    let client = client(
        &server,
        FakeNormal {
            delay: Duration::ZERO,
            chunks: Some(vec![(Duration::from_secs(2), text_event("late"))]),
            dropped: dropped.clone(),
        },
        20,
        audit.clone(),
    );
    let response = client.call(request(true)).await.unwrap();
    drop(response);
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(read_summary(&audit)["finish_reason"], "caller_disconnected");
}

#[tokio::test]
async fn upstream_in_band_error_is_forwarded_without_jev_replacement() {
    let server = server(100, decision("call_001", 0.99)).await;
    let audit = directory();
    let client = client(
        &server,
        FakeNormal {
            delay: Duration::ZERO,
            chunks: Some(vec![(
                Duration::ZERO,
                LlmResponseChunk::DecodeError {
                    message: "bad chunk".into(),
                }
                .into(),
            )]),
            dropped: Arc::new(AtomicBool::new(false)),
        },
        400,
        audit.clone(),
    );
    let result = client
        .call(request(true))
        .await
        .unwrap()
        .llm_response
        .into_agg()
        .await;
    assert!(result.is_err());
    let summary = read_summary(&audit);
    assert_eq!(summary["reason"], "normal_stream_error");
    assert!(summary["normal_complete_ms"].is_null());
}

#[tokio::test]
async fn disabled_recording_needs_no_key_and_redacts_capture_values() {
    let audit = directory();
    let client = JevRaceClient::new(
        Arc::new(FakeNormal {
            delay: Duration::ZERO,
            chunks: None,
            dropped: Arc::new(AtomicBool::new(false)),
        }),
        JevRaceConfig {
            enabled: false,
            audit_directory: Some(audit.clone()),
            ..JevRaceConfig::default()
        },
        String::new(),
    )
    .unwrap()
    .with_redaction_keys(vec!["provider-secret".into()]);
    let mut request = request(false);
    request.raw_request = Some(json!({"api_key":"other-secret", "value":"provider-secret"}));
    client.call(request).await.unwrap();
    let path = std::fs::read_dir(&audit)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    for file in std::fs::read_dir(path).unwrap() {
        let body = std::fs::read_to_string(file.unwrap().path()).unwrap();
        assert!(!body.contains("provider-secret"));
        assert!(!body.contains("other-secret"));
    }
    assert_eq!(read_summary(&audit)["reason"], "disabled");
}

#[tokio::test]
async fn evaluation_retains_both_and_selects_by_assumed_latency_in_both_arrival_orders() {
    for (jev_delay, normal_delay, assumed, winner) in [
        (5, 450, 200, "jev"),
        (450, 5, 1, "jev"),
        (70, 5, 200, "llm"),
    ] {
        let server = server(jev_delay, decision("call_001", 0.99)).await;
        let audit = directory();
        let mut client = client(
            &server,
            FakeNormal {
                delay: Duration::from_millis(normal_delay),
                chunks: None,
                dropped: Arc::new(AtomicBool::new(false)),
            },
            1,
            audit.clone(),
        );
        client.config.evaluation_wait_both = true;
        client.config.deadline = None;
        client.config.assumed_jev_latency = Duration::from_millis(assumed);
        client.call(request(false)).await.unwrap();
        let summary = read_summary(&audit);
        assert_eq!(summary["winner"], winner);
        assert!(summary["normal_complete_ms"].is_number());
        assert!(summary["jev_response_ms"].is_number());
        assert_eq!(summary["normal_usage"]["output_tokens"], 2);
        assert_eq!(summary["jev_usage"]["output_tokens"], 9);
        assert_eq!(summary["normal_cancel_requested"], false);
        assert_eq!(summary["jev_cancel_requested"], false);
        assert!(summary["max_hold_ms"].is_null());
        let capture = std::fs::read_dir(&audit)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        for name in [
            "normal-aggregate.json",
            "jev-response.json",
            "returned-aggregate.json",
        ] {
            assert!(capture.join(name).exists(), "missing {name}");
        }
    }
}

#[tokio::test]
async fn evaluation_distinguishes_none_low_score_invalid_and_service_error() {
    for (body, outcome) in [
        (decision("NONE", 1.0), "none"),
        (decision("call_001", 0.7), "below_cutoff"),
        (json!({}), "invalid_response"),
    ] {
        let server = server(5, body).await;
        let audit = directory();
        let mut client = client(
            &server,
            FakeNormal {
                delay: Duration::from_millis(30),
                chunks: None,
                dropped: Arc::new(AtomicBool::new(false)),
            },
            1,
            audit.clone(),
        );
        client.config.evaluation_wait_both = true;
        client.config.assumed_jev_latency = Duration::ZERO;
        client.call(request(false)).await.unwrap();
        let summary = read_summary(&audit);
        assert_eq!(summary["winner"], "llm");
        assert_eq!(summary["jev_outcome"], outcome);
    }
    let server = server(100, decision("call_001", 0.99)).await;
    let audit = directory();
    let mut client = client(
        &server,
        FakeNormal {
            delay: Duration::ZERO,
            chunks: None,
            dropped: Arc::new(AtomicBool::new(false)),
        },
        1,
        audit.clone(),
    );
    client.config.evaluation_wait_both = true;
    client.config.service_timeout = Duration::from_millis(10);
    client.call(request(false)).await.unwrap();
    let summary = read_summary(&audit);
    assert_eq!(summary["jev_outcome"], "service_error");
    assert_eq!(summary["jev_cancel_requested"], false);
}

struct FailedNormal;
#[async_trait]
impl RoutedLlmClient for FailedNormal {
    async fn call(&self, _: Request) -> Result<Response, LlmClientError> {
        Err(LlmClientError::General("normal failed".into()))
    }
}

#[tokio::test]
async fn evaluation_recovers_normal_failure_only_with_qualified_jev() {
    for (body, recovered) in [
        (decision("call_001", 0.99), true),
        (decision("NONE", 1.0), false),
        (json!({}), false),
    ] {
        let server = server(20, body).await;
        let audit = directory();
        let client = JevRaceClient::new(
            Arc::new(FailedNormal),
            JevRaceConfig {
                endpoint: format!("{}/v1/systemone", server.uri()),
                evaluation_wait_both: true,
                audit_directory: Some(audit.clone()),
                ..JevRaceConfig::default()
            },
            "secret".into(),
        )
        .unwrap();
        assert_eq!(client.call(request(false)).await.is_ok(), recovered);
        let summary = read_summary(&audit);
        assert!(summary["normal_error"].is_string());
        assert_eq!(
            summary["reason"],
            if recovered {
                "evaluation_normal_error_qualified_jev"
            } else {
                "evaluation_normal_error"
            }
        );
        assert_eq!(summary["jev_cancel_requested"], false);
    }
}

#[tokio::test]
async fn disabled_hold_keeps_live_race_selection_and_allows_late_jev() {
    let server = server(450, decision("call_001", 0.99)).await;
    let audit = directory();
    let mut client = client(
        &server,
        FakeNormal {
            delay: Duration::from_millis(900),
            chunks: None,
            dropped: Arc::new(AtomicBool::new(false)),
        },
        400,
        audit.clone(),
    );
    client.config.deadline = None;
    client.call(request(false)).await.unwrap();
    assert_eq!(read_summary(&audit)["winner"], "jev");
}

#[tokio::test]
async fn evaluation_rejects_streaming_before_any_requests() {
    let server = server(0, decision("call_001", 0.99)).await;
    let dropped = Arc::new(AtomicBool::new(false));
    let mut client = client(
        &server,
        FakeNormal {
            delay: Duration::ZERO,
            chunks: None,
            dropped: dropped.clone(),
        },
        400,
        directory(),
    );
    client.config.evaluation_wait_both = true;
    assert!(client.call(request(true)).await.is_err());
    assert!(!dropped.load(Ordering::SeqCst));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn evaluation_records_safe_headers_and_invalid_json() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("server-timing", "inference;dur=12")
                .insert_header("x-request-id", "secret")
                .insert_header("set-cookie", "credential-cookie")
                .insert_header("x-api-key", "credential-key")
                .set_body_string("not JSON"),
        )
        .mount(&server)
        .await;
    let audit = directory();
    let client = JevRaceClient::new(
        Arc::new(FakeNormal {
            delay: Duration::from_millis(10),
            chunks: None,
            dropped: Arc::new(AtomicBool::new(false)),
        }),
        JevRaceConfig {
            endpoint: server.uri(),
            evaluation_wait_both: true,
            audit_directory: Some(audit.clone()),
            ..JevRaceConfig::default()
        },
        "secret".into(),
    )
    .unwrap();
    client.call(request(false)).await.unwrap();
    let summary = read_summary(&audit);
    assert_eq!(summary["jev_outcome"], "invalid_response");
    assert!(
        summary["jev_headers_ms"].as_f64().unwrap() <= summary["jev_response_ms"].as_f64().unwrap()
    );
    assert_eq!(
        summary["jev_response_headers"]["server-timing"],
        "inference;dur=12"
    );
    assert_eq!(
        summary["jev_response_headers"]["x-request-id"],
        "[REDACTED]"
    );
    assert!(summary["jev_response_headers"].get("set-cookie").is_none());
    assert!(summary["jev_response_headers"].get("x-api-key").is_none());
}

#[test]
fn rejects_conflicting_modes_and_zero_operational_timeout() {
    for config in [
        JevRaceConfig {
            observe_only: true,
            evaluation_wait_both: true,
            ..JevRaceConfig::default()
        },
        JevRaceConfig {
            service_timeout: Duration::ZERO,
            ..JevRaceConfig::default()
        },
    ] {
        assert!(JevRaceClient::new(Arc::new(FailedNormal), config, "secret".into()).is_err());
    }
}
