// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded resizing and WebP encoding for inline request images.

use std::io::Cursor;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::codecs::gif::GifDecoder;
use image::codecs::png::PngDecoder;
use image::codecs::webp::WebPDecoder;
use image::imageops::FilterType;
use image::{AnimationDecoder, DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};
use serde::Serialize;
use serde_json::Value;
use switchyard_protocol::{ContentBlock, ImageSource, Request};

use crate::{ServerError, ServerResult};

const PATCH_PIXELS: u32 = 32;
const WEBP_MEDIA_TYPE: &str = "image/webp";

/// Conservative limits and encoder settings for inline-image compression.
#[derive(Clone, Copy, Debug)]
pub struct ImageCompressionConfig {
    /// Maximum estimated 32x32 image patches after resizing, or no resize limit.
    pub max_patch_tokens: Option<u32>,
    /// Lossy WebP quality in the inclusive range 1..=100.
    pub webp_quality: u8,
    /// libwebp encoding method in the inclusive range 0..=6.
    pub webp_method: u8,
    /// Maximum decoded source payload size.
    pub max_input_bytes: usize,
    /// Maximum source-image pixel count.
    pub max_source_pixels: u64,
}

impl Default for ImageCompressionConfig {
    fn default() -> Self {
        Self {
            max_patch_tokens: Some(576),
            webp_quality: 80,
            webp_method: 4,
            max_input_bytes: 20 * 1024 * 1024,
            max_source_pixels: 40_000_000,
        }
    }
}

impl ImageCompressionConfig {
    pub(crate) fn validate(self) -> ServerResult<Self> {
        if self.max_patch_tokens == Some(0) {
            return Err(ServerError::new("max_patch_tokens must be at least 1"));
        }
        if !(1..=100).contains(&self.webp_quality) {
            return Err(ServerError::new("webp_quality must be between 1 and 100"));
        }
        if self.webp_method > 6 {
            return Err(ServerError::new("webp_method must be between 0 and 6"));
        }
        if self.max_input_bytes == 0 {
            return Err(ServerError::new("max_input_bytes must be at least 1"));
        }
        if self.max_source_pixels == 0 {
            return Err(ServerError::new("max_source_pixels must be at least 1"));
        }
        Ok(self)
    }
}

/// Aggregate inline-image compression counters exposed through server stats.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ImageCompressionStats {
    /// Inline images examined.
    pub images_seen: u64,
    /// Images replaced by smaller WebP payloads.
    pub images_optimized: u64,
    /// Animated images deliberately passed through unchanged.
    pub animated_images_skipped: u64,
    /// Total inline bytes before compression.
    pub bytes_before: u64,
    /// Total inline bytes forwarded after compression.
    pub bytes_after: u64,
    /// Total inline bytes removed by compression.
    pub bytes_saved: u64,
    /// Estimated 32x32 patches before compression.
    pub estimated_patch_tokens_before: u64,
    /// Estimated 32x32 patches after compression.
    pub estimated_patch_tokens_after: u64,
    /// Estimated 32x32 patches removed from inline images.
    pub estimated_patch_tokens_saved: u64,
}

impl ImageCompressionStats {
    pub(crate) fn add(&mut self, other: Self) {
        self.images_seen = self.images_seen.saturating_add(other.images_seen);
        self.images_optimized = self.images_optimized.saturating_add(other.images_optimized);
        self.animated_images_skipped = self
            .animated_images_skipped
            .saturating_add(other.animated_images_skipped);
        self.bytes_before = self.bytes_before.saturating_add(other.bytes_before);
        self.bytes_after = self.bytes_after.saturating_add(other.bytes_after);
        self.bytes_saved = self.bytes_saved.saturating_add(other.bytes_saved);
        self.estimated_patch_tokens_before = self
            .estimated_patch_tokens_before
            .saturating_add(other.estimated_patch_tokens_before);
        self.estimated_patch_tokens_after = self
            .estimated_patch_tokens_after
            .saturating_add(other.estimated_patch_tokens_after);
        self.estimated_patch_tokens_saved = self
            .estimated_patch_tokens_saved
            .saturating_add(other.estimated_patch_tokens_saved);
    }

    fn record(&mut self, outcome: &CompressionOutcome) {
        self.images_seen = self.images_seen.saturating_add(1);
        self.images_optimized = self
            .images_optimized
            .saturating_add(u64::from(outcome.output.is_some()));
        self.animated_images_skipped = self
            .animated_images_skipped
            .saturating_add(u64::from(outcome.animated));
        self.bytes_before = self
            .bytes_before
            .saturating_add(outcome.source_bytes as u64);
        self.bytes_after = self.bytes_after.saturating_add(outcome.output_bytes as u64);
        self.bytes_saved = self.bytes_saved.saturating_add(
            (outcome.source_bytes as u64).saturating_sub(outcome.output_bytes as u64),
        );
        let patches_before = patch_tokens(outcome.source_size);
        let patches_after = patch_tokens(outcome.output_size);
        self.estimated_patch_tokens_before = self
            .estimated_patch_tokens_before
            .saturating_add(patches_before);
        self.estimated_patch_tokens_after = self
            .estimated_patch_tokens_after
            .saturating_add(patches_after);
        self.estimated_patch_tokens_saved = self
            .estimated_patch_tokens_saved
            .saturating_add(patches_before.saturating_sub(patches_after));
    }
}

/// One image compressed through the same implementation used by request handling.
pub struct CompressedImage {
    /// Original bytes when WebP is not smaller, otherwise the optimized payload.
    pub payload: Vec<u8>,
    /// Counters for this image.
    pub stats: ImageCompressionStats,
}

struct CompressionOutcome {
    source_bytes: usize,
    output_bytes: usize,
    source_size: (u32, u32),
    output_size: (u32, u32),
    output: Option<Vec<u8>>,
    animated: bool,
}

/// Compress one image payload for benchmarks and language bindings.
pub fn compress_image_payload(
    payload: &[u8],
    config: ImageCompressionConfig,
) -> ServerResult<CompressedImage> {
    let outcome = compress_payload(payload, config.validate()?)?;
    let mut stats = ImageCompressionStats::default();
    stats.record(&outcome);
    Ok(CompressedImage {
        payload: outcome.output.unwrap_or_else(|| payload.to_vec()),
        stats,
    })
}

/// Optimize all recognized inline images without blocking the async runtime.
pub(crate) async fn optimize_request(
    mut request: Request,
    config: ImageCompressionConfig,
) -> ServerResult<(Request, ImageCompressionStats)> {
    let config = config.validate()?;
    tokio::task::spawn_blocking(move || {
        let mut stats = ImageCompressionStats::default();
        for instruction in &mut request.llm_request.instructions {
            optimize_blocks(&mut instruction.content, config, &mut stats)?;
        }
        for message in &mut request.llm_request.messages {
            optimize_blocks(&mut message.content, config, &mut stats)?;
        }
        if stats.images_optimized > 0 {
            request.llm_request.preservation.requests.clear();
        }
        Ok((request, stats))
    })
    .await
    .map_err(|error| ServerError::new(format!("image compression task failed: {error}")))?
}

fn optimize_blocks(
    blocks: &mut [ContentBlock],
    config: ImageCompressionConfig,
    stats: &mut ImageCompressionStats,
) -> ServerResult<()> {
    for block in blocks {
        match block {
            ContentBlock::Image { source } => {
                if let Some(outcome) = optimize_source(source, config)? {
                    stats.record(&outcome);
                }
            }
            ContentBlock::ToolResult(result) => {
                optimize_blocks(&mut result.content, config, stats)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn optimize_source(
    source: &mut ImageSource,
    config: ImageCompressionConfig,
) -> ServerResult<Option<CompressionOutcome>> {
    match source {
        ImageSource::Url { url, .. } => {
            let Some(payload) = decode_data_url(url, config.max_input_bytes)? else {
                return Ok(None);
            };
            let outcome = compress_payload(&payload, config)?;
            if let Some(output) = &outcome.output {
                *url = data_url(output);
            }
            Ok(Some(outcome))
        }
        ImageSource::Base64 { media_type, data } => {
            let payload = decode_base64(data, config.max_input_bytes)?;
            let outcome = compress_payload(&payload, config)?;
            if let Some(output) = &outcome.output {
                *media_type = Some(WEBP_MEDIA_TYPE.to_string());
                *data = BASE64.encode(output);
            }
            Ok(Some(outcome))
        }
        ImageSource::Raw(raw) => optimize_raw_source(raw, config),
    }
}

fn optimize_raw_source(
    raw: &mut Value,
    config: ImageCompressionConfig,
) -> ServerResult<Option<CompressionOutcome>> {
    let Some(source) = raw.as_object_mut() else {
        return Ok(None);
    };
    if source.get("type").and_then(Value::as_str) == Some("base64") {
        let data = source.get("data").and_then(Value::as_str).ok_or_else(|| {
            ServerError::new("inline image base64 source must contain string data")
        })?;
        let payload = decode_base64(data, config.max_input_bytes)?;
        let outcome = compress_payload(&payload, config)?;
        if let Some(output) = &outcome.output {
            source.insert(
                "media_type".to_string(),
                Value::String(WEBP_MEDIA_TYPE.to_string()),
            );
            source.insert("data".to_string(), Value::String(BASE64.encode(output)));
        }
        return Ok(Some(outcome));
    }

    for key in ["url", "image_url"] {
        let Some(url) = source.get(key).and_then(Value::as_str) else {
            continue;
        };
        let Some(payload) = decode_data_url(url, config.max_input_bytes)? else {
            continue;
        };
        let outcome = compress_payload(&payload, config)?;
        if let Some(output) = &outcome.output {
            source.insert(key.to_string(), Value::String(data_url(output)));
        }
        return Ok(Some(outcome));
    }
    Ok(None)
}

fn decode_data_url(url: &str, max_input_bytes: usize) -> ServerResult<Option<Vec<u8>>> {
    if !url.starts_with("data:image/") {
        return Ok(None);
    }
    let Some((header, encoded)) = url.split_once(',') else {
        return Err(ServerError::new(
            "inline image data URL must contain base64 data",
        ));
    };
    if !header
        .split(';')
        .skip(1)
        .any(|part| part.eq_ignore_ascii_case("base64"))
    {
        return Err(ServerError::new(
            "inline image data URL must contain base64 data",
        ));
    }
    decode_base64(encoded, max_input_bytes).map(Some)
}

fn decode_base64(encoded: &str, max_input_bytes: usize) -> ServerResult<Vec<u8>> {
    let max_encoded_bytes = max_input_bytes
        .saturating_add(2)
        .saturating_div(3)
        .saturating_mul(4);
    if encoded.len() > max_encoded_bytes {
        return Err(ServerError::new(format!(
            "inline image exceeds the {max_input_bytes}-byte input limit"
        )));
    }
    let payload = BASE64
        .decode(encoded)
        .map_err(|_| ServerError::new("inline image contains invalid base64 data"))?;
    if payload.len() > max_input_bytes {
        return Err(ServerError::new(format!(
            "inline image is {} bytes; limit is {max_input_bytes}",
            payload.len()
        )));
    }
    Ok(payload)
}

fn data_url(payload: &[u8]) -> String {
    format!("data:{WEBP_MEDIA_TYPE};base64,{}", BASE64.encode(payload))
}

fn compress_payload(
    payload: &[u8],
    config: ImageCompressionConfig,
) -> ServerResult<CompressionOutcome> {
    if payload.len() > config.max_input_bytes {
        return Err(ServerError::new(format!(
            "inline image is {} bytes; limit is {}",
            payload.len(),
            config.max_input_bytes
        )));
    }
    let (format, source_size) = image_format_and_size(payload, config)?;
    if is_animated(payload, format, config)? {
        return Ok(CompressionOutcome {
            source_bytes: payload.len(),
            output_bytes: payload.len(),
            source_size,
            output_size: source_size,
            output: None,
            animated: true,
        });
    }

    let reader = image_reader(payload, config)?;
    let mut decoder = reader
        .into_decoder()
        .map_err(|error| image_error("could not be decoded", error))?;
    let orientation = decoder
        .orientation()
        .map_err(|error| image_error("orientation could not be read", error))?;
    let mut image = DynamicImage::from_decoder(decoder)
        .map_err(|error| image_error("could not be decoded", error))?;
    image.apply_orientation(orientation);
    let source_size = (image.width(), image.height());
    let target_size = config.max_patch_tokens.map_or(source_size, |budget| {
        fit_size_to_patch_budget(source_size, budget)
    });
    if target_size != source_size {
        image = image.resize_exact(target_size.0, target_size.1, FilterType::Lanczos3);
    }

    let output = encode_webp(&image, config)?;
    if output.len() >= payload.len() {
        return Ok(CompressionOutcome {
            source_bytes: payload.len(),
            output_bytes: payload.len(),
            source_size,
            output_size: source_size,
            output: None,
            animated: false,
        });
    }
    Ok(CompressionOutcome {
        source_bytes: payload.len(),
        output_bytes: output.len(),
        source_size,
        output_size: target_size,
        output: Some(output),
        animated: false,
    })
}

fn image_reader(
    payload: &[u8],
    config: ImageCompressionConfig,
) -> ServerResult<ImageReader<Cursor<&[u8]>>> {
    let mut reader = ImageReader::new(Cursor::new(payload))
        .with_guessed_format()
        .map_err(|error| image_error("format could not be read", error))?;
    let mut limits = Limits::default();
    limits.max_alloc = Some(config.max_source_pixels.saturating_mul(4));
    reader.limits(limits);
    Ok(reader)
}

fn image_format_and_size(
    payload: &[u8],
    config: ImageCompressionConfig,
) -> ServerResult<(ImageFormat, (u32, u32))> {
    let reader = image_reader(payload, config)?;
    let format = reader
        .format()
        .ok_or_else(|| ServerError::new("inline image format is not supported"))?;
    let size = reader
        .into_dimensions()
        .map_err(|error| image_error("dimensions could not be read", error))?;
    let pixels = u64::from(size.0).saturating_mul(u64::from(size.1));
    if pixels > config.max_source_pixels {
        return Err(ServerError::new(format!(
            "inline image has {pixels} pixels; limit is {}",
            config.max_source_pixels
        )));
    }
    Ok((format, size))
}

fn is_animated(
    payload: &[u8],
    format: ImageFormat,
    config: ImageCompressionConfig,
) -> ServerResult<bool> {
    match format {
        ImageFormat::Gif => {
            let mut decoder = GifDecoder::new(Cursor::new(payload))
                .map_err(|error| image_error("could not be decoded", error))?;
            let mut limits = Limits::default();
            limits.max_alloc = Some(config.max_source_pixels.saturating_mul(4));
            decoder
                .set_limits(limits)
                .map_err(|error| image_error("exceeds decoder limits", error))?;
            let mut frames = decoder.into_frames();
            let first = frames
                .next()
                .transpose()
                .map_err(|error| image_error("animation could not be decoded", error))?;
            if first.is_none() {
                return Err(ServerError::new("inline image could not be decoded"));
            }
            frames
                .next()
                .transpose()
                .map(|frame| frame.is_some())
                .map_err(|error| image_error("animation could not be decoded", error))
        }
        ImageFormat::Png => PngDecoder::new(Cursor::new(payload))
            .and_then(|decoder| decoder.is_apng())
            .map_err(|error| image_error("animation metadata could not be read", error)),
        ImageFormat::WebP => WebPDecoder::new(Cursor::new(payload))
            .map(|decoder| decoder.has_animation())
            .map_err(|error| image_error("animation metadata could not be read", error)),
        _ => Ok(false),
    }
}

fn encode_webp(image: &DynamicImage, config: ImageCompressionConfig) -> ServerResult<Vec<u8>> {
    let mut webp_config = webp::WebPConfig::new()
        .map_err(|_| ServerError::new("WebP encoder configuration could not be initialized"))?;
    webp_config.lossless = 0;
    webp_config.quality = f32::from(config.webp_quality);
    webp_config.method = i32::from(config.webp_method);
    webp_config.alpha_compression = 1;

    let memory = if image.color().has_alpha() {
        let rgba = image.to_rgba8();
        webp::Encoder::from_rgba(rgba.as_raw(), rgba.width(), rgba.height())
            .encode_advanced(&webp_config)
    } else {
        let rgb = image.to_rgb8();
        webp::Encoder::from_rgb(rgb.as_raw(), rgb.width(), rgb.height())
            .encode_advanced(&webp_config)
    }
    .map_err(|error| ServerError::new(format!("inline image WebP encoding failed: {error:?}")))?;
    Ok(memory.to_vec())
}

fn patch_tokens(size: (u32, u32)) -> u64 {
    u64::from(size.0.div_ceil(PATCH_PIXELS))
        .saturating_mul(u64::from(size.1.div_ceil(PATCH_PIXELS)))
}

fn fit_size_to_patch_budget(size: (u32, u32), max_patch_tokens: u32) -> (u32, u32) {
    if patch_tokens(size) <= u64::from(max_patch_tokens) {
        return size;
    }
    let (width, height) = size;
    let longest = width.max(height);
    let mut low = 1;
    let mut high = longest;
    let mut best = (1, 1);
    while low <= high {
        let candidate_longest = low + (high - low) / 2;
        let candidate = (
            (u64::from(width) * u64::from(candidate_longest) / u64::from(longest)).max(1) as u32,
            (u64::from(height) * u64::from(candidate_longest) / u64::from(longest)).max(1) as u32,
        );
        if patch_tokens(candidate) <= u64::from(max_patch_tokens) {
            best = candidate;
            low = candidate_longest.saturating_add(1);
        } else {
            high = candidate_longest.saturating_sub(1);
        }
    }
    best
}

fn image_error(context: &str, error: impl std::fmt::Display) -> ServerError {
    ServerError::new(format!("inline image {context}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use image::{Frame, ImageBuffer, ImageEncoder, Rgb, Rgba, RgbaImage};
    use serde_json::json;
    use switchyard_protocol::{FormatId, LlmRequest, Message, PreservationMetadata, Role};

    fn jpeg(width: u32, height: u32) -> ServerResult<Vec<u8>> {
        let image = ImageBuffer::from_fn(width, height, |x, y| {
            Rgb([
                (x.wrapping_mul(17).wrapping_add(y.wrapping_mul(3))) as u8,
                (x.wrapping_mul(5).wrapping_add(y.wrapping_mul(11))) as u8,
                (x.wrapping_mul(13).wrapping_add(y.wrapping_mul(7))) as u8,
            ])
        });
        let mut output = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 95)
            .write_image(
                image.as_raw(),
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .map_err(|error| image_error("test fixture could not be encoded", error))?;
        Ok(output)
    }

    fn request(source: ImageSource) -> Request {
        Request {
            llm_request: LlmRequest {
                messages: vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Image { source }],
                }],
                preservation: PreservationMetadata {
                    requests: BTreeMap::from([(
                        FormatId::new("openai_chat"),
                        json!({"model": "vision", "messages": []}),
                    )]),
                    ..PreservationMetadata::default()
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    fn only_image_source(request: &Request) -> ServerResult<&ImageSource> {
        let Some(ContentBlock::Image { source }) = request
            .llm_request
            .messages
            .first()
            .and_then(|message| message.content.first())
        else {
            return Err(ServerError::new("test request has no image"));
        };
        Ok(source)
    }

    fn optimized_payload(source: &ImageSource) -> ServerResult<Vec<u8>> {
        let encoded = match source {
            ImageSource::Url { url, .. } => url
                .strip_prefix("data:image/webp;base64,")
                .ok_or_else(|| ServerError::new("expected optimized WebP data URL"))?,
            ImageSource::Base64 { media_type, data } => {
                if media_type.as_deref() != Some(WEBP_MEDIA_TYPE) {
                    return Err(ServerError::new("expected optimized WebP media type"));
                }
                data
            }
            ImageSource::Raw(raw) => {
                if raw.get("media_type").and_then(Value::as_str) != Some(WEBP_MEDIA_TYPE) {
                    return Err(ServerError::new("expected optimized raw WebP media type"));
                }
                raw.get("data")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ServerError::new("expected optimized raw WebP data"))?
            }
        };
        BASE64
            .decode(encoded)
            .map_err(|error| ServerError::new(format!("test WebP base64 is invalid: {error}")))
    }

    fn two_frame_gif() -> ServerResult<Vec<u8>> {
        let mut output = Vec::new();
        let frames = [
            Frame::new(RgbaImage::from_pixel(16, 16, Rgba([255, 0, 0, 255]))),
            Frame::new(RgbaImage::from_pixel(16, 16, Rgba([0, 0, 255, 255]))),
        ];
        image::codecs::gif::GifEncoder::new(&mut output)
            .encode_frames(frames)
            .map_err(|error| image_error("test GIF could not be encoded", error))?;
        Ok(output)
    }

    #[test]
    fn compresses_to_the_patch_budget() -> ServerResult<()> {
        let source = jpeg(512, 256)?;
        let config = ImageCompressionConfig {
            max_patch_tokens: Some(16),
            ..ImageCompressionConfig::default()
        };

        let compressed = compress_image_payload(&source, config)?;
        let decoded = image::load_from_memory_with_format(&compressed.payload, ImageFormat::WebP)
            .map_err(|error| image_error("test output could not be decoded", error))?;

        assert_eq!(compressed.stats.images_seen, 1);
        assert_eq!(compressed.stats.images_optimized, 1);
        assert_eq!(compressed.stats.estimated_patch_tokens_before, 128);
        assert!(compressed.stats.estimated_patch_tokens_after <= 16);
        assert!(decoded.width().saturating_mul(decoded.height()) < 512 * 256);
        Ok(())
    }

    #[tokio::test]
    async fn optimizes_url_base64_and_raw_image_sources() -> ServerResult<()> {
        let source = jpeg(512, 256)?;
        let encoded = BASE64.encode(&source);
        let sources = [
            ImageSource::Url {
                url: format!("data:image/jpeg;base64,{encoded}"),
                detail: Some("high".to_string()),
            },
            ImageSource::Base64 {
                media_type: Some("image/jpeg".to_string()),
                data: encoded.clone(),
            },
            ImageSource::Raw(json!({
                "type": "base64",
                "media_type": "image/jpeg",
                "data": encoded,
            })),
        ];
        let config = ImageCompressionConfig {
            max_patch_tokens: Some(16),
            ..ImageCompressionConfig::default()
        };

        let mut outputs = Vec::new();
        for source in sources {
            let (optimized, stats) = optimize_request(request(source), config).await?;
            outputs.push(optimized_payload(only_image_source(&optimized)?)?);
            assert_eq!(stats.images_seen, 1);
            assert_eq!(stats.images_optimized, 1);
            assert!(stats.bytes_saved > 0);
            assert!(stats.estimated_patch_tokens_after <= 16);
            assert!(optimized.llm_request.preservation.requests.is_empty());
        }
        assert_eq!(outputs[0], outputs[1]);
        assert_eq!(outputs[1], outputs[2]);
        Ok(())
    }

    #[tokio::test]
    async fn leaves_remote_urls_and_exact_replay_untouched() -> ServerResult<()> {
        let original = "https://example.test/image.png";
        let (optimized, stats) = optimize_request(
            request(ImageSource::Url {
                url: original.to_string(),
                detail: None,
            }),
            ImageCompressionConfig::default(),
        )
        .await?;

        assert_eq!(stats.images_seen, 0);
        assert!(matches!(
            only_image_source(&optimized)?,
            ImageSource::Url { url, .. } if url == original
        ));
        assert!(!optimized.llm_request.preservation.requests.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn rejects_invalid_base64() {
        let result = optimize_request(
            request(ImageSource::Base64 {
                media_type: Some("image/jpeg".to_string()),
                data: "not-valid***".to_string(),
            }),
            ImageCompressionConfig::default(),
        )
        .await;

        assert!(matches!(result, Err(error) if error.to_string().contains("invalid base64")));
    }

    #[tokio::test]
    async fn enforces_byte_and_pixel_limits() -> ServerResult<()> {
        let source = jpeg(512, 256)?;
        let encoded = BASE64.encode(&source);
        let byte_limited = optimize_request(
            request(ImageSource::Base64 {
                media_type: Some("image/jpeg".to_string()),
                data: encoded.clone(),
            }),
            ImageCompressionConfig {
                max_input_bytes: source.len().saturating_sub(1),
                ..ImageCompressionConfig::default()
            },
        )
        .await;
        assert!(matches!(byte_limited, Err(error) if error.to_string().contains("limit")));

        let pixel_limited = optimize_request(
            request(ImageSource::Base64 {
                media_type: Some("image/jpeg".to_string()),
                data: encoded,
            }),
            ImageCompressionConfig {
                max_source_pixels: 512 * 256 - 1,
                ..ImageCompressionConfig::default()
            },
        )
        .await;
        assert!(matches!(pixel_limited, Err(error) if error.to_string().contains("pixels")));
        Ok(())
    }

    #[tokio::test]
    async fn skips_animated_images() -> ServerResult<()> {
        let source = two_frame_gif()?;
        let encoded = BASE64.encode(&source);
        let (optimized, stats) = optimize_request(
            request(ImageSource::Base64 {
                media_type: Some("image/gif".to_string()),
                data: encoded,
            }),
            ImageCompressionConfig::default(),
        )
        .await?;

        assert_eq!(stats.images_seen, 1);
        assert_eq!(stats.images_optimized, 0);
        assert_eq!(stats.animated_images_skipped, 1);
        assert_eq!(stats.bytes_saved, 0);
        assert!(!optimized.llm_request.preservation.requests.is_empty());
        let ImageSource::Base64 { data, .. } = only_image_source(&optimized)? else {
            return Err(ServerError::new("expected base64 GIF source"));
        };
        let after = BASE64
            .decode(data)
            .map_err(|error| ServerError::new(format!("test GIF base64 is invalid: {error}")))?;
        assert_eq!(after, source);
        Ok(())
    }

    #[test]
    fn keeps_an_already_minimal_webp() -> ServerResult<()> {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(1, 1, Rgb([255, 0, 0])));
        let config = ImageCompressionConfig {
            max_patch_tokens: None,
            ..ImageCompressionConfig::default()
        };
        let source = encode_webp(&image, config)?;

        let compressed = compress_image_payload(&source, config)?;

        assert_eq!(compressed.payload, source);
        assert_eq!(compressed.stats.images_optimized, 0);
        assert_eq!(compressed.stats.bytes_saved, 0);
        assert_eq!(compressed.stats.estimated_patch_tokens_saved, 0);
        Ok(())
    }

    #[test]
    fn rejects_invalid_bounds() {
        let error = ImageCompressionConfig {
            max_patch_tokens: Some(0),
            ..ImageCompressionConfig::default()
        }
        .validate();
        assert!(matches!(error, Err(error) if error.to_string().contains("at least 1")));
    }

    #[test]
    fn fits_landscape_and_portrait_images_within_budget() {
        for size in [(512, 256), (256, 512), (1, 4096), (4096, 1)] {
            let fitted = fit_size_to_patch_budget(size, 16);
            assert!(patch_tokens(fitted) <= 16, "{size:?} -> {fitted:?}");
        }
    }
}
