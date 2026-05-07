#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error as urllib_error
from urllib import request as urllib_request


APP_NAME = "CatSDK"
APP_TAGLINE = "CatSDK — local LM Studio chat"
DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
SYSTEM_PROMPT = (
    f"You are {APP_NAME}, a fast local assistant running through LM Studio. "
    "Be concise, practical, and execution-focused."
)


@dataclass
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    lm_studio_base_url: str = DEFAULT_LM_STUDIO_BASE_URL
    lm_studio_api_key: str = "lm-studio"
    default_temperature: float = 0.65
    default_max_tokens: int = 768
    request_timeout_seconds: float = 240.0


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def apply_apple_silicon_optimizations(config: AppConfig) -> AppConfig:
    if is_apple_silicon():
        config.default_temperature = 0.60
        config.default_max_tokens = min(config.default_max_tokens, 640)
        config.request_timeout_seconds = max(config.request_timeout_seconds, 240.0)
    return config


class LMStudioClient:
    """OpenAI-compatible client tuned for LM Studio's localhost server."""

    def __init__(self, base_url: str, timeout: float, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # LM Studio accepts any token; sending one avoids 401s on stricter builds.
        self.api_key = api_key or "lm-studio"

    def _request(
        self,
        path: str,
        method: str = "GET",
        payload: dict | None = None,
    ) -> dict:
        body = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": f"{APP_NAME}/1.0",
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")

        req = urllib_request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LM Studio HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(
                "Cannot connect to LM Studio. In LM Studio, open the "
                "'Developer' / 'Local Server' tab, click 'Start Server', "
                f"and confirm the base URL is {self.base_url}."
            ) from exc

    def ping(self) -> dict:
        """Light probe: returns model count and a friendly status."""
        try:
            data = self._request("/models", method="GET")
            count = len(data.get("data", []) or [])
            return {
                "ok": True,
                "models_loaded": count,
                "base_url": self.base_url,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "base_url": self.base_url}

    def list_models(self) -> list[str]:
        data = self._request("/models", method="GET")
        models = []
        for item in data.get("data", []):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                models.append(model_id)
        return models

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = self._request("/chat/completions", method="POST", payload=payload)
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("LM Studio returned no completion choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        return {
            "content": content,
            "model": data.get("model", model),
            "usage": data.get("usage", {}),
        }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>CatSDK</title>
  <style>
    :root {
      --bg-1: #010409;
      --bg-2: #061023;
      --bg-3: #0a1833;
      --panel: rgba(5, 12, 26, 0.85);
      --panel-border: #163b66;
      --text: #69b3ff;
      --text-dim: #3f7db8;
      --accent: #2d8cff;
      --accent-2: #4ba6ff;
      --bubble-user: #0d2342;
      --bubble-ai: #07172d;
      --error: #ff6f91;
    }

    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      height: 100%;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "SF Pro Text",
        "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(1200px 700px at 10% -20%, #0f2d55, transparent 65%),
                  linear-gradient(180deg, var(--bg-1), var(--bg-2) 45%, var(--bg-3));
    }

    .shell {
      max-width: 1080px;
      margin: 18px auto;
      height: calc(100vh - 36px);
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      background: var(--panel);
      backdrop-filter: blur(6px);
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.45);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--panel-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .title {
      font-weight: 700;
      letter-spacing: 0.2px;
      color: var(--accent-2);
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .badge {
      font-size: 11px;
      border: 1px solid #2a5488;
      border-radius: 999px;
      padding: 3px 8px;
      color: #7ab8ff;
      background: #091a31;
    }

    .controls {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--text-dim);
      font-size: 12px;
    }

    select, input {
      border: 1px solid #1f4c81;
      background: #050e1c;
      color: var(--text);
      border-radius: 8px;
      padding: 7px 8px;
      outline: none;
      min-width: 110px;
    }

    .messages {
      flex: 1;
      overflow: auto;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .msg {
      width: fit-content;
      max-width: 82%;
      border: 1px solid #1a3f6a;
      border-radius: 12px;
      padding: 10px 12px;
      white-space: pre-wrap;
      line-height: 1.42;
      word-wrap: break-word;
    }

    .msg.user {
      align-self: flex-end;
      background: var(--bubble-user);
      color: #8ec7ff;
      border-color: #245588;
    }

    .msg.ai {
      align-self: flex-start;
      background: var(--bubble-ai);
      color: #72b6ff;
    }

    .composer {
      border-top: 1px solid var(--panel-border);
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
    }

    textarea {
      resize: none;
      height: 86px;
      border-radius: 10px;
      border: 1px solid #1f4c81;
      background: #040c18;
      color: #7ec1ff;
      padding: 10px;
      outline: none;
      font: inherit;
    }

    button {
      border: 1px solid #2a639f;
      background: linear-gradient(180deg, #14539a, #0f3f75);
      color: #e7f3ff;
      font-weight: 600;
      border-radius: 10px;
      min-width: 110px;
      cursor: pointer;
    }

    button:disabled { opacity: 0.55; cursor: wait; }
    .status { margin-top: 6px; font-size: 12px; color: var(--text-dim); }
    .status.error { color: var(--error); }
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <div class="title">
        CatSDK
        <span class="badge" id="chipBadge">localhost</span>
      </div>
      <div class="controls">
        <label>Model
          <select id="modelSelect"></select>
        </label>
        <label>Temp
          <input id="tempInput" type="number" min="0" max="2" step="0.05" value="0.60" />
        </label>
        <label>Max tok
          <input id="tokensInput" type="number" min="64" max="4096" step="32" value="640" />
        </label>
      </div>
    </div>
    <div id="messages" class="messages"></div>
    <div class="composer">
      <textarea id="prompt" placeholder="Message CatSDK (LM Studio localhost)..."></textarea>
      <button id="sendBtn">Send</button>
    </div>
    <div id="status" class="status" style="padding: 0 12px 12px;"></div>
  </div>
  <script>
    const messagesEl = document.getElementById("messages");
    const modelSelect = document.getElementById("modelSelect");
    const tempInput = document.getElementById("tempInput");
    const tokensInput = document.getElementById("tokensInput");
    const promptEl = document.getElementById("prompt");
    const sendBtn = document.getElementById("sendBtn");
    const statusEl = document.getElementById("status");
    const chipBadge = document.getElementById("chipBadge");

    const state = { history: [] };

    function setStatus(text, isError = false) {
      statusEl.textContent = text || "";
      statusEl.classList.toggle("error", !!isError);
    }

    function appendMessage(role, content) {
      const div = document.createElement("div");
      div.className = `msg ${role === "user" ? "user" : "ai"}`;
      div.textContent = content;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    async function loadModels() {
      setStatus("Loading local LM Studio models...");
      try {
        const res = await fetch("/api/models");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to fetch models");

        modelSelect.innerHTML = "";
        for (const model of data.models) {
          const opt = document.createElement("option");
          opt.value = model;
          opt.textContent = model;
          modelSelect.appendChild(opt);
        }

        if (!data.models.length) {
          setStatus("LM Studio responded but no models are loaded. Load a model, then refresh.", true);
        } else {
          setStatus(`Connected to LM Studio • ${data.models.length} model(s) found`);
        }

        chipBadge.textContent = data.apple_silicon ? "M-chip optimized" : "localhost";
      } catch (err) {
        setStatus(err.message || String(err), true);
      }
    }

    async function sendMessage() {
      const text = promptEl.value.trim();
      if (!text) return;
      const model = modelSelect.value;
      if (!model) {
        setStatus("Pick a model first.", true);
        return;
      }

      sendBtn.disabled = true;
      appendMessage("user", text);
      state.history.push({ role: "user", content: text });
      promptEl.value = "";
      setStatus("Thinking...");

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            message: text,
            history: state.history.slice(0, -1),
            temperature: Number(tempInput.value || 0.6),
            max_tokens: Number(tokensInput.value || 640)
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Chat request failed");

        appendMessage("assistant", data.reply);
        state.history.push({ role: "assistant", content: data.reply });
        const tok = data.usage && data.usage.total_tokens
          ? ` • ${data.usage.total_tokens} tokens`
          : "";
        setStatus(`Model: ${data.model}${tok}`);
      } catch (err) {
        setStatus(err.message || String(err), true);
      } finally {
        sendBtn.disabled = false;
        promptEl.focus();
      }
    }

    sendBtn.addEventListener("click", sendMessage);
    promptEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    loadModels();
  </script>
</body>
</html>
"""


def build_messages(history: list[dict], message: str) -> list[dict[str, str]]:
    compiled: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            compiled.append({"role": role, "content": content})
    compiled.append({"role": "user", "content": message})
    return compiled


def create_handler(config: AppConfig):
    client = LMStudioClient(
        base_url=config.lm_studio_base_url,
        timeout=config.request_timeout_seconds,
        api_key=config.lm_studio_api_key,
    )

    class CatSDKHandler(BaseHTTPRequestHandler):
        server_version = f"{APP_NAME}/1.0"

        def _send_json(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_html(self, html: str) -> None:
            raw = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                parsed = json.loads(body.decode("utf-8"))
                if isinstance(parsed, dict):
                    return parsed
                return {}
            except json.JSONDecodeError:
                return {}

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send_html(HTML)
                return

            if self.path == "/api/models":
                try:
                    models = client.list_models()
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "models": models,
                            "apple_silicon": is_apple_silicon(),
                        },
                    )
                except Exception as exc:
                    self._send_json(
                        HTTPStatus.BAD_GATEWAY,
                        {"error": str(exc)},
                    )
                return

            if self.path == "/api/lmstudio":
                self._send_json(HTTPStatus.OK, client.ping())
                return

            if self.path == "/health":
                self._send_json(HTTPStatus.OK, {"status": "ok", "app": APP_NAME})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return

            payload = self._read_json()
            model = str(payload.get("model", "")).strip()
            message = str(payload.get("message", "")).strip()
            if not model:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Model is required."})
                return
            if not message:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Message cannot be empty."},
                )
                return

            history = payload.get("history", [])
            if not isinstance(history, list):
                history = []

            try:
                temperature = float(
                    payload.get("temperature", config.default_temperature),
                )
            except (TypeError, ValueError):
                temperature = config.default_temperature

            try:
                max_tokens = int(payload.get("max_tokens", config.default_max_tokens))
            except (TypeError, ValueError):
                max_tokens = config.default_max_tokens

            try:
                response = client.chat(
                    model=model,
                    messages=build_messages(history, message),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "reply": response["content"],
                        "model": response["model"],
                        "usage": response["usage"],
                    },
                )
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})

        def log_message(self, fmt: str, *args) -> None:
            return

    return CatSDKHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=APP_NAME.lower(),
        description=f"{APP_TAGLINE} (OpenAI-compatible LM Studio client).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument(
        "--lm-studio-url",
        default=os.environ.get("LM_STUDIO_BASE_URL", DEFAULT_LM_STUDIO_BASE_URL),
        help="LM Studio OpenAI-compatible base URL (defaults to env LM_STUDIO_BASE_URL or %(default)s)",
    )
    parser.add_argument(
        "--lm-studio-key",
        default=os.environ.get("LM_STUDIO_API_KEY", "lm-studio"),
        help="Bearer token sent to LM Studio (LM Studio accepts any value).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_apple_silicon_optimizations(
        AppConfig(
            host=args.host,
            port=args.port,
            lm_studio_base_url=args.lm_studio_url,
            lm_studio_api_key=args.lm_studio_key,
        ),
    )

    # Probe LM Studio at boot so the user gets a clear, immediate verdict in the terminal.
    probe_client = LMStudioClient(
        base_url=config.lm_studio_base_url,
        timeout=min(config.request_timeout_seconds, 8.0),
        api_key=config.lm_studio_api_key,
    )
    probe = probe_client.ping()

    print(f"{APP_NAME} running: http://{config.host}:{config.port}")
    print(f"LM Studio endpoint: {config.lm_studio_base_url}")
    if probe.get("ok"):
        print(f"LM Studio: connected • {probe['models_loaded']} model(s) loaded")
    else:
        print(f"LM Studio: NOT reachable — {probe.get('error', 'unknown error')}")
        print("Tip: open LM Studio → 'Developer' / 'Local Server' → Start Server.")

    handler = create_handler(config)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n{APP_NAME} stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
