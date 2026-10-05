// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde_json::{Value, json};
use switchyard_protocol::{
    AggLlmResponse, LlmClientError, LlmResponseChunk, LlmResponseStreamEvent, Request,
    ResponseAccumulator,
};
use tokio::time::Instant;

use super::JevRaceConfig;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

pub(super) struct Audit {
    pub id: String,
    pub started: Instant,
    pub stream_requested: bool,
    directory: Option<PathBuf>,
    redaction_keys: Vec<String>,
    state: Mutex<State>,
}

struct State {
    events: Option<File>,
    summary: Value,
    accumulator: Option<ResponseAccumulator>,
    finished: bool,
}

impl Audit {
    pub fn new(
        config: &JevRaceConfig,
        request: &Request,
        redaction_keys: Vec<String>,
    ) -> Result<Arc<Self>, LlmClientError> {
        let started = Instant::now();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| LlmClientError::General(error.to_string()))?
            .as_nanos();
        let id = format!("{nanos}-{}", NEXT_ID.fetch_add(1, Ordering::Relaxed));
        let directory = config.audit_directory.as_ref().map(|root| root.join(&id));
        let events = if let Some(directory) = &directory {
            std::fs::create_dir(directory).map_err(io_error)?;
            Some(
                OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(directory.join("events.jsonl"))
                    .map_err(io_error)?,
            )
        } else {
            None
        };
        let metadata = request.metadata.as_ref();
        let audit = Arc::new(Self {
            id: id.clone(),
            started,
            stream_requested: request.llm_request.stream,
            directory,
            redaction_keys,
            state: Mutex::new(State {
                events,
                summary: json!({"request_id": id, "enabled": config.enabled,
                    "observe_only": config.observe_only,
                    "evaluation_wait_both": config.evaluation_wait_both,
                    "assumed_jev_latency_ms": config.assumed_jev_latency.as_secs_f64() * 1000.0,
                    "service_timeout_ms": config.service_timeout.as_secs_f64() * 1000.0,
                    "started_unix_seconds": nanos as f64 / 1_000_000_000.0,
                    "threshold": config.threshold, "max_hold_ms": config.deadline.map(|value| value.as_secs_f64() * 1000.0),
                    "model_requested": request.llm_request.model,
                    "correlation_id": metadata.and_then(|m| m.correlation_id.as_deref()),
                    "session_id": metadata.and_then(|m| m.session_id.as_deref()),
                    "turn_id": metadata.and_then(|m| m.turn_id.as_deref()),
                    "winner": null, "reason": null, "selected_ms": null,
                    "normal_first_event_ms": null, "normal_first_output_ms": null,
                    "normal_last_output_ms": null, "normal_complete_ms": null,
                    "first_forwarded_event_ms": null, "first_forwarded_output_ms": null,
                    "buffer_hold_ms": null,
                    "normal_usage": null, "jev_usage": null,
                    "normal_cancel_requested": false, "jev_cancel_requested": false,
                    "normal_backend_stopped": null, "jev_backend_stopped": null,
                    "normal_error": null, "jev_error": null, "finished": false}),
                accumulator: Some(ResponseAccumulator::new()),
                finished: false,
            }),
        });
        audit.event("request_started", json!({}))?;
        Ok(audit)
    }

    fn clean(&self, value: &Value) -> Value {
        match value {
            Value::String(text) => {
                let mut text = text.clone();
                for key in &self.redaction_keys {
                    text = text.replace(key, "[REDACTED]");
                }
                Value::String(text)
            }
            Value::Array(items) => {
                Value::Array(items.iter().map(|value| self.clean(value)).collect())
            }
            Value::Object(items) => Value::Object(
                items
                    .iter()
                    .map(|(key, value)| {
                        let sensitive = matches!(
                            key.to_ascii_lowercase().as_str(),
                            "authorization"
                                | "proxy-authorization"
                                | "x-api-key"
                                | "api-key"
                                | "api_key"
                                | "cookie"
                                | "set-cookie"
                        );
                        (
                            key.clone(),
                            if sensitive {
                                json!("[REDACTED]")
                            } else {
                                self.clean(value)
                            },
                        )
                    })
                    .collect(),
            ),
            value => value.clone(),
        }
    }

    pub fn write_json(&self, filename: &str, value: &Value) -> Result<(), LlmClientError> {
        if let Some(directory) = &self.directory {
            let value = self.clean(value);
            let bytes = serde_json::to_vec_pretty(&value)
                .map_err(|error| LlmClientError::General(error.to_string()))?;
            std::fs::write(directory.join(filename), bytes).map_err(io_error)?;
        }
        Ok(())
    }

    pub fn event(&self, name: &str, value: Value) -> Result<(), LlmClientError> {
        let millis = self.started.elapsed().as_secs_f64() * 1000.0;
        let value = self.clean(&value);
        let mut state = self.state.lock();
        match name {
            "normal_cancel_requested" | "jev_cancel_requested" => state.summary[name] = json!(true),
            "normal_error" | "jev_error" => {
                state.summary[name] = value["error"].clone();
                state.summary[format!("{name}_ms")] = json!(millis);
                if name == "jev_error" {
                    state.summary["jev_outcome"] = value["outcome"].clone();
                }
            }
            "jev_headers" => {
                state.summary["jev_headers_ms"] = json!(millis);
                state.summary["jev_response_headers"] = value["headers"].clone();
                state.summary["jev_http_version"] = value["http_version"].clone();
            }
            "jev_started" => state.summary["jev_started_ms"] = json!(millis),
            "jev_http_response" => state.summary["jev_response_ms"] = json!(millis),
            "jev_decision" => {
                state.summary["jev_decision_ms"] = json!(millis);
                state.summary["jev_outcome"] = value["outcome"].clone();
                state.summary["jev_model"] = value["model"].clone();
                state.summary["jev_answer"] = value["answer"].clone();
                state.summary["jev_usage"] = value["usage"].clone();
            }
            "normal_headers" => state.summary["normal_headers_ms"] = json!(millis),
            "normal_started" => state.summary["normal_started_ms"] = json!(millis),
            _ => {}
        }
        if let Some(file) = &mut state.events {
            let bytes =
                serde_json::to_vec(&json!({"elapsed_ms": millis, "event": name, "data": value}))
                    .map_err(|error| LlmClientError::General(error.to_string()))?;
            file.write_all(&bytes)
                .and_then(|()| file.write_all(b"\n"))
                .and_then(|()| file.flush())
                .map_err(io_error)?;
        }
        Ok(())
    }

    pub fn select(&self, winner: &str, reason: &str) -> Result<(), LlmClientError> {
        {
            let mut state = self.state.lock();
            state.summary["winner"] = json!(winner);
            state.summary["reason"] = json!(reason);
            state.summary["selected_ms"] = json!(self.started.elapsed().as_secs_f64() * 1000.0);
        }
        self.event(
            "winner_selected",
            json!({"winner": winner, "reason": reason}),
        )
    }

    pub fn normal_event(&self, event: &LlmResponseStreamEvent) -> Result<(), LlmClientError> {
        {
            let millis = self.started.elapsed().as_secs_f64() * 1000.0;
            let mut state = self.state.lock();
            if state.summary["normal_first_event_ms"].is_null() {
                state.summary["normal_first_event_ms"] = json!(millis);
            }
            for chunk in event.normalized() {
                match chunk {
                    LlmResponseChunk::MessageStart {
                        model: Some(model), ..
                    } => state.summary["normal_model"] = json!(model),
                    LlmResponseChunk::Usage(usage) => state.summary["normal_usage"] = json!(usage),
                    LlmResponseChunk::TextDelta { .. }
                    | LlmResponseChunk::ToolCallDelta { .. }
                    | LlmResponseChunk::ReasoningDelta { .. }
                    | LlmResponseChunk::ReasoningDetailsDelta { .. } => {
                        if state.summary["normal_first_output_ms"].is_null() {
                            state.summary["normal_first_output_ms"] = json!(millis);
                        }
                        state.summary["normal_last_output_ms"] = json!(millis);
                    }
                    LlmResponseChunk::DecodeError { message }
                    | LlmResponseChunk::StreamError { message } => {
                        state.summary["normal_error"] = json!(message)
                    }
                    _ => {}
                }
                if let Some(accumulator) = &mut state.accumulator {
                    accumulator.push(chunk.clone());
                }
            }
        }
        self.event("normal_stream_event", json!(event))
    }

    pub fn normal_complete(&self) -> Result<(), LlmClientError> {
        let aggregate = {
            let mut state = self.state.lock();
            if state.summary["normal_error"].is_null() {
                state.summary["normal_complete_ms"] =
                    json!(self.started.elapsed().as_secs_f64() * 1000.0);
            }
            state.accumulator.take().map(ResponseAccumulator::finish)
        };
        if let Some(aggregate) = aggregate {
            // Exact provider events are in events.jsonl. This convenience view
            // has the protocol accumulator's documented multi-output limitation.
            self.write_json("normal-reconstructed.json", &json!(aggregate))?;
        }
        self.event("normal_complete", json!({}))
    }

    pub fn normal_aggregate(&self, aggregate: &AggLlmResponse) -> Result<(), LlmClientError> {
        {
            let mut state = self.state.lock();
            if state.summary["normal_complete_ms"].is_null() {
                state.summary["normal_complete_ms"] =
                    json!(self.started.elapsed().as_secs_f64() * 1000.0);
            }
            state.summary["normal_model"] = json!(aggregate.model);
            state.summary["normal_usage"] = json!(aggregate.usage);
        }
        self.write_json("normal-aggregate.json", &json!(aggregate))?;
        self.event("normal_complete", json!({}))
    }

    pub fn normal_complete_ms(&self) -> Option<f64> {
        self.state.lock().summary["normal_complete_ms"].as_f64()
    }

    pub fn normal_is_complete(&self) -> bool {
        !self.state.lock().summary["normal_complete_ms"].is_null()
    }

    pub fn forwarded(&self, event: &LlmResponseStreamEvent) -> Result<(), LlmClientError> {
        let millis = self.started.elapsed().as_secs_f64() * 1000.0;
        {
            let mut state = self.state.lock();
            if state.summary["first_forwarded_event_ms"].is_null() {
                state.summary["first_forwarded_event_ms"] = json!(millis);
                if let Some(first) = state.summary["normal_first_event_ms"].as_f64()
                    && state.summary["winner"] == "llm"
                {
                    state.summary["buffer_hold_ms"] = json!((millis - first).max(0.0));
                }
            }
            if state.summary["first_forwarded_output_ms"].is_null()
                && event.normalized().iter().any(|chunk| {
                    matches!(
                        chunk,
                        LlmResponseChunk::TextDelta { .. }
                            | LlmResponseChunk::ToolCallDelta { .. }
                            | LlmResponseChunk::ReasoningDelta { .. }
                            | LlmResponseChunk::ReasoningDetailsDelta { .. }
                    )
                })
            {
                state.summary["first_forwarded_output_ms"] = json!(millis);
            }
        }
        self.event("forwarded_stream_event", json!(event))
    }

    fn finish(&self, reason: &str) -> Result<(), LlmClientError> {
        let summary = {
            let mut state = self.state.lock();
            if state.finished {
                return Ok(());
            }
            state.finished = true;
            state.summary["finished"] = json!(true);
            state.summary["finish_reason"] = json!(reason);
            state.summary["elapsed_ms"] = json!(self.started.elapsed().as_secs_f64() * 1000.0);
            if reason == "caller_disconnected" {
                if state.summary["normal_complete_ms"].is_null() {
                    state.summary["normal_cancel_requested"] = json!(true);
                }
                if state.summary["jev_response_ms"].is_null()
                    && !state.summary["jev_started_ms"].is_null()
                {
                    state.summary["jev_cancel_requested"] = json!(true);
                }
            }
            state.summary.clone()
        };
        self.write_json("summary.json", &summary)
    }
}

fn io_error(error: std::io::Error) -> LlmClientError {
    LlmClientError::General(format!("JEV audit write failed: {error}"))
}

pub(super) struct AuditGuard {
    pub audit: Arc<Audit>,
}

impl AuditGuard {
    pub fn new(audit: Arc<Audit>) -> Self {
        Self { audit }
    }
    pub fn finish(self, reason: &str) -> Result<(), LlmClientError> {
        self.audit.finish(reason)
    }
}

impl Drop for AuditGuard {
    fn drop(&mut self) {
        if let Err(error) = self.audit.finish("caller_disconnected") {
            tracing::error!(%error, "Could not finish JEV race audit");
        }
    }
}
