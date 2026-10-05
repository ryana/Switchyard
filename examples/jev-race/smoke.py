# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise a real Switchyard process against controlled HTTP providers."""

from __future__ import annotations

import argparse
import json
import os
import select
import socket
import subprocess
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def free_port() -> int:
    """Reserve an unused local port briefly for the child server."""
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Provider(BaseHTTPRequestHandler):
    """Serve deterministic JEV decisions and slow normal-model responses."""

    protocol_version = "HTTP/1.1"
    calls: list[dict[str, Any]] = []
    disconnected: list[str] = []
    disconnect_evidence: list[dict[str, Any]] = []
    active_request: dict[str, Any] | None = None
    response_complete = False

    def log_message(self, _format: str, *args: Any) -> None:
        """Keep the capture free of request headers and credentials."""

    def handle(self) -> None:
        """Record a peer closing an idle keepalive connection during cancellation."""
        try:
            super().handle()
        except ConnectionResetError:
            self.record_disconnect("connection_reset")

    def record_disconnect(self, observation: str) -> None:
        """Associate an observed socket close with the request on that connection."""
        if self.active_request is not None:
            self.disconnected.append(self.active_request["case"] + (":jev" if self.active_request["jev"] else ":llm"))
            self.disconnect_evidence.append({**self.active_request, "observation": observation,
                                             "before_response_complete": not self.response_complete})
        self.close_connection = True

    def delay(self, seconds: float) -> bool:
        """Wait for model work while observing a peer FIN or reset directly."""
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            readable, _, _ = select.select([self.connection], [], [], remaining)
            if readable:
                try:
                    peek = self.connection.recv(1, socket.MSG_PEEK)
                except ConnectionResetError:
                    self.record_disconnect("socket_peek_reset")
                    return False
                if not peek:
                    self.record_disconnect("socket_peek_eof")
                    return False
                # Pipelined bytes are not a disconnect and remain unread.
                time.sleep(min(remaining, 0.005))
        return True

    def do_POST(self) -> None:
        """Model enough of each API to verify the actual proxy transport."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        jev = self.path == "/v1/systemone"
        state = body["state"] if jev else body
        case = state["messages"][0]["content"]
        self.active_request = {"case": case, "jev": jev, "stream": bool(state.get("stream"))}
        self.response_complete = False
        self.calls.append({"case": case, "jev": jev, "body": body})
        try:
            if jev:
                self.jev(body, case)
            else:
                self.llm(body, case)
        except (BrokenPipeError, ConnectionResetError):
            # This is the expected, explicitly recorded result of transport cancellation.
            self.record_disconnect("response_write_failed")

    def send_json(self, body: dict[str, Any], status: int = 200) -> None:
        """Send a complete JSON response."""
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
        self.wfile.flush()
        self.response_complete = True

    def jev(self, body: dict[str, Any], case: str) -> None:
        """Return a choice, refusal, error, or late result based on the test case."""
        if not self.delay(0.8 if case in {"deadline", "llm_first"} else 0.05):
            return
        if case == "jev_error":
            self.send_json({"error": "controlled failure"}, 503)
            return
        criteria = body["questions"]["next_call"]["criteria"]
        choice = next(k for k, v in criteria.items() if isinstance(v, dict) and v["arguments"] == {"enabled": True})
        probabilities = dict.fromkeys(criteria, 0.0)
        probabilities[choice] = 0.99
        probabilities["NONE"] = 0.01
        if case == "none":
            choice = "NONE"
            probabilities = {key: float(key == "NONE") for key in criteria}
        if case == "low_confidence":
            probabilities[choice], probabilities["NONE"] = 0.6, 0.4
        if case == "invalid_choice":
            choice = "invented"
        self.send_json({"model": "jev-test", "usage": {"input_tokens": 20, "output_tokens": 4}, "answers": {"next_call": {
            "type": "choice", "choice": choice, "confidence": 0.99,
            "probabilities": probabilities,
        }}})

    def llm(self, body: dict[str, Any], case: str) -> None:
        """Stream early reasoning and finish later, or return a buffered answer."""
        usage = {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
        if not body.get("stream"):
            if not self.delay(0.02 if case == "llm_first" else 0.6):
                return
            self.send_json({"id": "normal", "object": "chat.completion", "model": "mock-llm",
                            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                                "role": "assistant", "content": "LLM answer"}}], "usage": usage})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        chunks = [
            {"role": "assistant", "content": ""},
            {"reasoning_content": "thinking"},
            {"content": "LLM answer"},
        ]
        for index, delta in enumerate(chunks):
            if not self.delay(0.005 if case == "llm_first" or index < 2 else 0.6):
                return
            event = {"id": "normal", "object": "chat.completion.chunk", "model": "mock-llm",
                     "preserved_test_field": index,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()
        for event in [
            {"id": "normal", "object": "chat.completion.chunk", "model": "mock-llm",
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"id": "normal", "object": "chat.completion.chunk", "model": "mock-llm",
             "choices": [], "usage": usage},
        ]:
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.response_complete = True
        self.close_connection = True


def main() -> None:
    """Start the providers and proxy, assert outcomes, and retain evidence."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    threading.Thread(target=provider.serve_forever, daemon=True).start()
    port = free_port()
    config = output / "routes.toml"
    config.write_text(f'''schema_version = 1
[llm_clients.mock]
format = "openai_chat"
base_url = "http://127.0.0.1:{provider.server_port}/v1"
max_retries = 0
failure_cooldown_ms = 0
[targets.mock]
id = "mock-llm"
llm_client = "mock"
[routes.race]
id = "race"
type = "passthrough"
target = "mock"
[routes.race.jev_race]
endpoint = "http://127.0.0.1:{provider.server_port}/v1/systemone"
api_key_env = "SWITCHYARD_SMOKE_KEY"
threshold = 0.9
max_hold_ms = 400
audit_directory = "{output / 'audit'}"
''')
    results = []
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            [str(args.server.resolve()), "--config", str(config), "--host", "127.0.0.1", "--port", str(port)],
            stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "SWITCHYARD_SMOKE_KEY": "controlled-test-key"},
        )
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(f"Switchyard exited; inspect {output / 'server.log'}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("Switchyard startup timed out")
            for case in ["jev_wins", "llm_first", "none", "low_confidence", "jev_error", "invalid_choice", "deadline", "no_tools"]:
                for streaming in (False, True):
                    request = {"model": "race", "stream": streaming,
                               "messages": [{"role": "user", "content": case}]}
                    if case != "no_tools":
                        request["tools"] = [{"type": "function", "function": {
                            "name": "set_flag", "parameters": {"type": "object", "properties": {
                                "enabled": {"type": "boolean"}}, "required": ["enabled"], "additionalProperties": False}}}]
                    session = f"{case}-{'stream' if streaming else 'buffered'}"
                    start = time.perf_counter()
                    first = None
                    with urlopen(Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(request).encode(),
                                         headers={"Content-Type": "application/json", "x-switchyard-session-id": session}), timeout=10) as response:
                        headers = dict(response.headers.items())
                        if streaming:
                            events = []
                            for line in response:
                                if line.startswith(b"data: "):
                                    first = first if first is not None else time.perf_counter() - start
                                    if line.strip() != b"data: [DONE]":
                                        events.append(json.loads(line[6:]))
                            body = events
                        else:
                            body = json.load(response)
                    elapsed = time.perf_counter() - start
                    results.append({"session_id": session, "elapsed_seconds": elapsed,
                                    "first_event_seconds": first, "headers": headers, "response": body})
                    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                    text = json.dumps(body)
                    if case == "jev_wins":
                        headers = {key.lower(): value for key, value in headers.items()}
                        assert headers.get("x-switchyard-winner") == "jev", (session, headers)
                        assert headers.get("x-switchyard-usage-source") == "jev", (session, headers)
                        assert headers.get("x-switchyard-canceled-llm-usage") == "unknown", (session, headers)
                        usage = [event["usage"] for event in body if event.get("usage")] if streaming else [body["usage"]]
                        assert len(usage) == 1, (session, usage)
                        assert usage[0]["prompt_tokens"] == 20 and usage[0]["completion_tokens"] == 4 and usage[0]["total_tokens"] == 24, (session, usage)
                        assert "set_flag" in text and "LLM answer" not in text and "thinking" not in text, (session, body)
                        if streaming:
                            calls: dict[int, dict[str, str]] = {}
                            finishes = []
                            for event in events:
                                for choice in event.get("choices", []):
                                    if choice.get("finish_reason"):
                                        finishes.append(choice["finish_reason"])
                                    for call in choice.get("delta", {}).get("tool_calls", []):
                                        target = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                                        for key in ("name", "arguments"):
                                            target[key] += call.get("function", {}).get(key, "")
                            assert finishes == ["tool_calls"], (session, finishes)
                            functions = list(calls.values())
                        else:
                            assert body["choices"][0]["finish_reason"] == "tool_calls", session
                            functions = [call["function"] for call in body["choices"][0]["message"]["tool_calls"]]
                        assert len(functions) == 1 and functions[0]["name"] == "set_flag", (session, functions)
                        assert json.loads(functions[0]["arguments"]) == {"enabled": True}, (session, functions)
                        assert elapsed < 0.4, (session, elapsed)
                    else:
                        assert "LLM answer" in text and "set_flag" not in text, (session, body)
                        if streaming:
                            assert "thinking" in text, session
                            assert [event["preserved_test_field"] for event in events if "preserved_test_field" in event] == [0, 1, 2], session
                    if case == "deadline" and streaming:
                        assert first is not None and 0.32 < first < 0.6, (session, first)
                    if case in {"none", "low_confidence", "jev_error", "invalid_choice"} and streaming:
                        assert first is not None and first < 0.3, (session, first)
            time.sleep(0.85)
            (output / "providers.json").write_text(json.dumps({"calls": Provider.calls, "disconnects": Provider.disconnected,
                                                               "disconnect_evidence": Provider.disconnect_evidence}, indent=2) + "\n")
            assert not any(call["jev"] for call in Provider.calls if call["case"] == "no_tools")
            assert any(event["case"] == "jev_wins" and not event["jev"] and event["stream"] and event["before_response_complete"]
                       for event in Provider.disconnect_evidence), "No observed normal-stream disconnect before completion"
            print(json.dumps({"passed": len(results), "output": str(output), "disconnects": Provider.disconnected}))
        finally:
            (output / "providers.json").write_text(json.dumps({"calls": Provider.calls, "disconnects": Provider.disconnected,
                                                               "disconnect_evidence": Provider.disconnect_evidence}, indent=2) + "\n")
            process.terminate()
            process.wait(timeout=10)
            provider.shutdown()
            provider.server_close()


if __name__ == "__main__":
    main()
