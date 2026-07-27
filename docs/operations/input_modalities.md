# Input Modalities

Coding-agent routes can mix models with different input capabilities. For
example, one tier may accept screenshots while another accepts only text.
Declare each target's accepted input modalities with `input_modalities` so
Switchyard can remove unsupported content before calling the selected model.

## Configure target capabilities

`input_modalities` is an optional allowlist. It accepts these values:

| Value | Content |
|---|---|
| `text` | Text-bearing messages and content blocks |
| `image` | Images supplied by URL, file reference, or inline data |
| `audio` | Audio supplied by URL, file reference, or inline data |
| `video` | Video supplied by URL, file reference, or inline data |
| `file` | Files and documents not represented by another modality |

This route preserves image-bearing content for Claude Opus and sends a
text-only trajectory to Nemotron:

```yaml
defaults:
  api_key: ${NVIDIA_API_KEY}
  base_url: https://inference-api.nvidia.com/v1
  format: responses

routes:
  coding-agent:
    type: random_routing
    strong:
      model: azure/anthropic/claude-opus-4-7
      input_modalities: [text, image]
    weak:
      model: nvidia/nvidia/nemotron-3-super-v3
      input_modalities: [text]
    strong_probability: 0.5
    fallback_target_on_evict: strong
```

Programmatic profiles use the same string literals:

```python
from switchyard import LlmTarget

text_only = LlmTarget(
    model="nvidia/nvidia/nemotron-3-super-v3",
    input_modalities=["text"],
)
```

Unknown literals are configuration errors. Omitting `input_modalities` means
the target's capabilities are unknown, so Switchyard preserves the existing
pass-through behavior. An explicit list is authoritative: content in a
recognized modality that is absent from the list is removed.

## Request behavior

Switchyard applies the allowlist after routing and format translation:

1. The route selects a target.
2. Switchyard translates or clones the request into that target's wire format.
3. Recognized content blocks for unsupported modalities are removed from the
   target-local outbound request.
4. The filtered request is sent to the selected model.

The original inbound request is not mutated. If Switchyard retries the turn
against a target that accepts more modalities, that target receives the
complete original trajectory.

Filtering covers message content in OpenAI Chat Completions, OpenAI Responses,
and Anthropic Messages requests, including nested Anthropic tool-result
content. Recognized provider spellings include:

| Modality | Recognized content block types |
|---|---|
| Text | `text`, `input_text`, `output_text`, `refusal` |
| Image | `image`, `image_url`, `input_image` |
| Audio | `audio`, `input_audio` |
| Video | `video`, `input_video` |
| File | `file`, `input_file`, `document` |

Switchyard only filters message and tool-result content. It does not inspect
arbitrary JSON fields such as tool schemas. Unknown provider-extension block
types are preserved by the filtering step. If all blocks are removed from one
message, its content becomes an empty string so the message container remains
structurally valid.

## Scope

`input_modalities` does not automatically discover model capabilities and does
not influence which target the router selects. It records operator knowledge
and filters the request after target selection. Modality-aware target
selection or a reject/reroute policy can build on the same target metadata in
a later change.

If an upstream returns an error such as `not a multimodal model`, verify that
the selected target declares `input_modalities: [text]` and that the route is
using a native OpenAI, Responses, or Anthropic backend.
