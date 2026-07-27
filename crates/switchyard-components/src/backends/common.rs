// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for native backend implementations.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use serde_json::{Map, Value};
use switchyard_core::{ChatRequestType, Result, SwitchyardError};
use switchyard_translation::{TranslationEngine, WireFormat};

pub(crate) enum ParsedSseFrame {
    /// Frame contained a JSON payload.
    Json(Value),
    /// Frame contained the provider's terminal marker.
    Done,
    /// Frame had no data payload.
    Empty,
}

/// Returns the shared translation engine used by native backends.
pub(crate) fn shared_translation_engine() -> Arc<TranslationEngine> {
    static ENGINE: OnceLock<Arc<TranslationEngine>> = OnceLock::new();
    Arc::clone(ENGINE.get_or_init(|| Arc::new(TranslationEngine::default())))
}

/// Builds a reqwest client with validated optional timeout.
pub(crate) fn build_reqwest_client(
    backend_name: &str,
    timeout_secs: Option<f64>,
) -> Result<reqwest::Client> {
    validate_timeout_secs(backend_name, timeout_secs)?;
    let mut builder = reqwest::Client::builder();
    if let Some(timeout_secs) = timeout_secs {
        builder = builder.timeout(Duration::from_secs_f64(timeout_secs));
    }
    builder.build().map_err(|error| {
        SwitchyardError::InvalidConfig(format!(
            "failed to build {backend_name} HTTP client: {error}"
        ))
    })
}

/// Validates timeout values before they reach reqwest.
pub(crate) fn validate_timeout_secs(backend_name: &str, timeout_secs: Option<f64>) -> Result<()> {
    if let Some(timeout_secs) = timeout_secs {
        if !timeout_secs.is_finite() || timeout_secs <= 0.0 {
            return Err(SwitchyardError::InvalidConfig(format!(
                "{backend_name} target timeout_secs must be finite and positive, got {timeout_secs:?}"
            )));
        }
    }
    Ok(())
}

/// Maps a Switchyard request type to its wire format.
pub(crate) fn request_wire_format(request_type: ChatRequestType) -> WireFormat {
    match request_type {
        ChatRequestType::OpenAiChat => WireFormat::OpenAiChat,
        ChatRequestType::OpenAiResponses => WireFormat::OpenAiResponses,
        ChatRequestType::Anthropic => WireFormat::AnthropicMessages,
    }
}

/// Sets or creates the JSON `model` field for an outbound provider request.
pub(crate) fn set_json_model(body: &mut Value, model: &str) {
    match body {
        Value::Object(object) => {
            object.insert("model".to_string(), Value::String(model.to_string()));
        }
        other => {
            let mut object = Map::new();
            object.insert("model".to_string(), Value::String(model.to_string()));
            *other = Value::Object(object);
        }
    }
}

/// Removes image blocks from provider message content in an outbound request.
///
/// Only message and tool-result content is visited, so JSON Schema fields such
/// as `{"type": "image"}` remain untouched. Empty content arrays become empty
/// strings to keep the provider message shape valid.
pub(crate) fn strip_image_content(body: &mut Value, request_type: ChatRequestType) {
    let Some(body) = body.as_object_mut() else {
        return;
    };
    let container = match request_type {
        ChatRequestType::OpenAiResponses => "input",
        ChatRequestType::OpenAiChat | ChatRequestType::Anthropic => "messages",
    };

    if let Some(value) = body.get_mut(container) {
        strip_message_items(value);
    }
    if request_type == ChatRequestType::Anthropic {
        if let Some(system) = body.get_mut("system") {
            strip_image_blocks(system);
        }
    }
}

// Visits each provider message or Responses input item without changing the container shape.
fn strip_message_items(value: &mut Value) {
    let Value::Array(items) = value else {
        return;
    };
    items.retain(|item| !is_image_block(item));
    for item in items {
        strip_nested_content(item);
    }
}

// Removes image objects only from arrays used as message content.
fn strip_image_blocks(value: &mut Value) {
    let Value::Array(items) = value else {
        return;
    };
    items.retain(|item| !is_image_block(item));
    for item in items.iter_mut() {
        strip_nested_content(item);
    }
    if items.is_empty() {
        *value = Value::String(String::new());
    }
}

// Descends through remaining content blocks so Anthropic tool results are covered.
fn strip_nested_content(value: &mut Value) {
    let Value::Object(object) = value else {
        return;
    };
    if let Some(content) = object.get_mut("content") {
        strip_image_blocks(content);
    }
}

// Recognizes image block spellings accepted across supported request formats.
fn is_image_block(value: &Value) -> bool {
    matches!(
        value.get("type").and_then(Value::as_str),
        Some("image" | "image_url" | "input_image")
    )
}

/// Drains one complete SSE frame from the buffer when a boundary is present.
pub(crate) fn drain_next_sse_frame(
    buffer: &mut Vec<u8>,
    backend_name: &str,
) -> Result<Option<String>> {
    let Some((index, separator_len)) = next_sse_boundary(buffer) else {
        return Ok(None);
    };
    let frame = decode_sse_frame(&buffer[..index], backend_name)?;
    buffer.drain(..index + separator_len);
    Ok(Some(frame))
}

/// Decodes one raw SSE frame as UTF-8.
pub(crate) fn decode_sse_frame(frame: &[u8], backend_name: &str) -> Result<String> {
    std::str::from_utf8(frame)
        .map(str::to_string)
        .map_err(|error| {
            SwitchyardError::Upstream(format!(
                "{backend_name} stream emitted invalid UTF-8 frame: {error}"
            ))
        })
}

/// Returns whether the buffer has any non-whitespace bytes.
pub(crate) fn has_non_whitespace_bytes(buffer: &[u8]) -> bool {
    buffer.iter().any(|byte| !byte.is_ascii_whitespace())
}

/// Parses data lines from one SSE frame into JSON, terminal, or empty states.
pub(crate) fn parse_json_sse_frame(
    frame: &str,
    backend_name: &str,
    done_marker: Option<&str>,
) -> Result<ParsedSseFrame> {
    let mut data_lines = Vec::new();
    for line in frame.lines() {
        // SSE comments and blank lines do not contribute data.
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_string());
        }
    }

    if data_lines.is_empty() {
        return Ok(ParsedSseFrame::Empty);
    }

    let data = data_lines.join("\n");
    if done_marker.is_some_and(|marker| data.trim() == marker) {
        return Ok(ParsedSseFrame::Done);
    }

    let value = serde_json::from_str::<Value>(&data).map_err(|error| {
        SwitchyardError::Upstream(format!(
            "{backend_name} stream emitted invalid JSON frame: {error}"
        ))
    })?;
    Ok(ParsedSseFrame::Json(value))
}

/// Finds the next CRLF or LF SSE frame boundary.
fn next_sse_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    match (find_bytes(buffer, b"\r\n\r\n"), find_bytes(buffer, b"\n\n")) {
        (Some(crlf), Some(lf)) if crlf < lf => Some((crlf, 4)),
        (Some(_), Some(lf)) => Some((lf, 2)),
        (Some(crlf), None) => Some((crlf, 4)),
        (None, Some(lf)) => Some((lf, 2)),
        (None, None) => None,
    }
}

/// Finds a byte needle inside a byte haystack.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    // Multi-byte UTF-8 split across network chunks should wait for a full frame.
    #[test]
    fn buffers_incomplete_utf8_until_a_complete_sse_frame_arrives() -> Result<()> {
        let mut buffer = b"data: {\"text\":\"".to_vec();
        let multibyte = "é".as_bytes();
        buffer.extend_from_slice(&multibyte[..1]);
        assert!(drain_next_sse_frame(&mut buffer, "test")?.is_none());

        buffer.extend_from_slice(&multibyte[1..]);
        buffer.extend_from_slice(b"\"}\n\n");

        let Some(frame) = drain_next_sse_frame(&mut buffer, "test")? else {
            return Err(SwitchyardError::Other(
                "complete SSE frame should be drained".to_string(),
            ));
        };
        let ParsedSseFrame::Json(value) = parse_json_sse_frame(&frame, "test", None)? else {
            return Err(SwitchyardError::Other(
                "SSE frame should parse as JSON".to_string(),
            ));
        };
        assert_eq!(value, json!({"text": "é"}));
        assert!(buffer.is_empty());
        Ok(())
    }

    #[test]
    fn strips_openai_chat_images_without_touching_tool_schemas() {
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "keep"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
                ]
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "inspect",
                    "parameters": {"type": "object", "properties": {"kind": {"type": "image"}}}
                }
            }]
        });

        strip_image_content(&mut body, ChatRequestType::OpenAiChat);

        assert_eq!(
            body["messages"][0]["content"],
            json!([{"type": "text", "text": "keep"}])
        );
        assert_eq!(
            body["tools"][0]["function"]["parameters"]["properties"]["kind"]["type"],
            "image"
        );
    }

    #[test]
    fn strips_responses_images_and_keeps_input_container_valid() {
        let mut body = json!({
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:image/png;base64,abc"}]
            }]
        });

        strip_image_content(&mut body, ChatRequestType::OpenAiResponses);

        assert_eq!(body["input"][0]["content"], "");
        assert!(body["input"].is_array());
    }

    #[test]
    fn strips_anthropic_images_from_system_messages_and_tool_results() {
        let mut body = json!({
            "system": [
                {"type": "text", "text": "system"},
                {"type": "image", "source": {"type": "base64", "data": "system-image"}}
            ],
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "message-image"}},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": [
                            {"type": "text", "text": "keep"},
                            {"type": "image", "source": {"type": "base64", "data": "tool-image"}}
                        ]
                    }
                ]
            }]
        });

        strip_image_content(&mut body, ChatRequestType::Anthropic);

        assert_eq!(body["system"], json!([{"type": "text", "text": "system"}]));
        assert_eq!(
            body["messages"][0]["content"],
            json!([{
                "type": "tool_result",
                "tool_use_id": "tool-1",
                "content": [{"type": "text", "text": "keep"}]
            }])
        );
    }
}
