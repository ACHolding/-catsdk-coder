#!/usr/bin/env python3
"""
CatSDK — single-file local LM Studio playground.

Built on Ralph AI as the foundation, this one .py file ships with the original
blue-hued CatSDK chat and adds every major agent paradigm in the open-source
ecosystem:

  * Chat       — original CatSDK conversational UI (preserved)
  * Ralph      — Geoffrey Huntley's autonomous Claude-Code loop, ported to
                 LM Studio: PROMPT.md + fix_plan.md driven, dual-condition
                 exit gate, rate limiter, circuit breaker, session continuity,
                 wake / sleep cycles, .ralph/ project layout
  * Sandbox    — ChatGPT-style code interpreter (Python + shell + file
                 artifacts inside an isolated workspace)
  * Engineer   — gpt-engineer style: describe a project, generate a full
                 multi-file codebase
  * AutoGPT    — AutoGPT 0.4.7 think -> plan -> act -> observe -> reflect
                 loop with a JSON tool surface (read/write files, run code,
                 fetch URLs, remember/recall, finish) and wake/sleep
  * Manus      — Manus.im-style long-horizon task agent that produces a
                 plan, executes step-by-step and emits deliverables

Pure standard library, no external deps. Tuned for Apple Silicon M-series
running LM Studio, but works against anything exposing the OpenAI /v1 surface
on localhost.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


# ===========================================================================
#  Constants & defaults
# ===========================================================================

APP_NAME = "CatSDK"
APP_TAGLINE = "CatSDK — local LM Studio playground (Chat • Ralph • Sandbox • Engineer • AutoGPT • Manus)"
APP_VERSION = "2.0.0"
DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

CHAT_SYSTEM_PROMPT = (
    f"You are {APP_NAME}, a fast local assistant running through LM Studio. "
    "Be concise, practical, and execution-focused."
)

# Ralph layout (Ralph is the foundation)
RALPH_DIR_NAME = ".ralph"
RALPH_LOG_FILE = "ralph.log"
RALPH_STATUS_FILE = "status.json"
RALPH_SESSION_FILE = ".ralph_session"
RALPH_EDITABLE_FILES = ["PROMPT.md", "fix_plan.md", "AGENT.md", ".ralphrc"]

# Per-feature workspaces under <root>/.catsdk/<feature>/
WORKSPACES_DIR_NAME = ".catsdk"

DEFAULT_PROMPT_MD = """# CatRalph PROMPT.md

You are an autonomous coding agent running inside a Ralph-style loop.
Each loop you receive this PROMPT.md, the current `fix_plan.md`, and your
prior responses (session continuity).

## Every loop
1. Pick the highest-priority unchecked task in `fix_plan.md`.
2. Describe the ONE concrete change you would make.
3. If you produce code, fence it and name the target file.
4. Track which tasks are done.

## REQUIRED — end every response with a RALPH_STATUS block

```
RALPH_STATUS:
  STATUS: IN_PROGRESS | COMPLETE | BLOCKED
  EXIT_SIGNAL: true | false
  PROGRESS_NOTE: <one short sentence>
```

Dual-condition exit gate:
  * EXIT_SIGNAL: true ONLY when every task in fix_plan.md is done.
  * If still working, EXIT_SIGNAL: false (even if a phase finished).
  * CatRalph also detects natural completion phrases but will not exit
    unless EXIT_SIGNAL is also true.

Be concise. One step per loop. Meow.
"""

DEFAULT_FIX_PLAN_MD = """# fix_plan.md

Prioritized tasks for CatRalph. Tick boxes as work completes.

- [ ] Replace this with your real first task
- [ ] Add a second concrete task
- [ ] Add a third concrete task
"""

DEFAULT_AGENT_MD = """# AGENT.md

Build, test, and run instructions CatRalph can reference.

## Build
(fill in)

## Test
(fill in)

## Run
(fill in)
"""

DEFAULT_RALPHRC = """# .ralphrc — CatRalph project configuration
PROJECT_NAME="my-project"
MAX_CALLS_PER_HOUR=100
TEMPERATURE=0.6
MAX_TOKENS=1024
CB_NO_PROGRESS_THRESHOLD=3
CB_SAME_ERROR_THRESHOLD=5
CB_COOLDOWN_MINUTES=30
SESSION_CONTINUITY=true
SESSION_EXPIRY_HOURS=24
COMPLETION_INDICATORS_REQUIRED=2
SLEEP_AFTER_IDLE_LOOPS=3
WAKE_INTERVAL_SECONDS=300
"""


# ===========================================================================
#  Config
# ===========================================================================


@dataclass
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 8765

    # LM Studio
    lm_studio_base_url: str = DEFAULT_LM_STUDIO_BASE_URL
    lm_studio_api_key: str = "lm-studio"
    request_timeout_seconds: float = 240.0
    default_temperature: float = 0.65
    default_max_tokens: int = 768

    # Ralph loop
    inter_loop_seconds: float = 1.0
    max_calls_per_hour: int = 100
    cb_no_progress_threshold: int = 3
    cb_same_error_threshold: int = 5
    cb_cooldown_minutes: int = 30
    session_continuity: bool = True
    session_expiry_hours: int = 24
    history_char_budget: int = 24000
    completion_indicators_required: int = 2

    # Wake / sleep
    sleep_after_idle_loops: int = 3
    wake_interval_seconds: int = 300

    # Project root that holds .ralph/ and .catsdk/
    project_root: Path = field(default_factory=lambda: Path.cwd())


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def apply_apple_silicon_optimizations(config: AppConfig) -> AppConfig:
    if is_apple_silicon():
        config.default_temperature = 0.60
        config.default_max_tokens = min(config.default_max_tokens, 640)
        config.request_timeout_seconds = max(config.request_timeout_seconds, 240.0)
        config.inter_loop_seconds = max(config.inter_loop_seconds, 1.0)
    return config


# ===========================================================================
#  LM Studio client (OpenAI-compatible /v1)
# ===========================================================================


class LMStudioClient:
    """OpenAI-compatible client tuned for LM Studio's localhost server."""

    def __init__(self, base_url: str, timeout: float, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # LM Studio accepts any token; sending one avoids 401s on stricter builds.
        self.api_key = api_key or "lm-studio"

    def _request(self, path: str, method: str = "GET", payload: dict | None = None,
                 timeout: float | None = None) -> dict:
        body = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": f"{APP_NAME}/{APP_VERSION}",
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
            with urllib_request.urlopen(req, timeout=timeout or self.timeout) as resp:
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
        try:
            data = self._request("/models", method="GET", timeout=5.0)
            count = len(data.get("data", []) or [])
            return {"ok": True, "models_loaded": count, "base_url": self.base_url}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "base_url": self.base_url}

    def list_models(self) -> list[str]:
        data = self._request("/models", method="GET")
        models: list[str] = []
        for item in data.get("data", []):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                models.append(model_id)
        return models

    def chat(self, *, model: str, messages: list[dict[str, str]],
             temperature: float, max_tokens: int) -> dict:
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


# ===========================================================================
#  Workspace helpers
# ===========================================================================


def workspaces_root(config: AppConfig) -> Path:
    root = config.project_root / WORKSPACES_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def feature_dir(config: AppConfig, name: str) -> Path:
    d = workspaces_root(config) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_join(base: Path, rel: str) -> Path:
    """Reject path traversal — any rel that escapes `base` raises ValueError."""
    rel = rel.lstrip("/").lstrip("\\")
    target = (base / rel).resolve()
    base_resolved = base.resolve()
    if base_resolved != target and base_resolved not in target.parents:
        raise ValueError(f"Refusing path outside workspace: {rel}")
    return target


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===========================================================================
#  Original CatSDK chat — preserved verbatim in spirit
# ===========================================================================


def build_chat_messages(history: list[dict], message: str) -> list[dict[str, str]]:
    compiled: list[dict[str, str]] = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            compiled.append({"role": role, "content": content})
    compiled.append({"role": "user", "content": message})
    return compiled


# ===========================================================================
#  Sandbox — ChatGPT-style code interpreter
# ===========================================================================


@dataclass
class ExecResult:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    artifacts: list[str]


class Sandbox:
    """Executes Python and shell commands inside an isolated workspace dir.

    Files written by executed code under the workspace become 'artifacts'
    that the UI can list, view, and download. Each session shares its
    workspace so successive runs build on each other (matches ChatGPT).
    """

    PYTHON_TIMEOUT_SECONDS = 120
    SHELL_TIMEOUT_SECONDS = 60
    OUTPUT_CHAR_LIMIT = 200_000

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def list_artifacts(self) -> list[dict]:
        items: list[dict] = []
        for p in sorted(self.workspace.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.workspace).as_posix()
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                items.append({"name": rel, "size": size})
        return items

    def read_artifact(self, rel: str) -> tuple[bytes, str]:
        target = safe_join(self.workspace, rel)
        if not target.is_file():
            raise FileNotFoundError(rel)
        suffix = target.suffix.lower()
        mime = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp",
            ".html": "text/html; charset=utf-8", ".json": "application/json",
            ".csv": "text/csv; charset=utf-8", ".txt": "text/plain; charset=utf-8",
            ".md": "text/markdown; charset=utf-8", ".py": "text/x-python; charset=utf-8",
            ".pdf": "application/pdf",
        }.get(suffix, "application/octet-stream")
        return target.read_bytes(), mime

    def write_artifact(self, rel: str, data: bytes) -> None:
        target = safe_join(self.workspace, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def reset(self) -> None:
        with self._lock:
            for p in self.workspace.rglob("*"):
                if p.is_file():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            for p in sorted(self.workspace.rglob("*"), reverse=True):
                if p.is_dir():
                    try:
                        p.rmdir()
                    except OSError:
                        pass

    def _snapshot(self) -> dict[str, float]:
        snap: dict[str, float] = {}
        for p in self.workspace.rglob("*"):
            if p.is_file():
                try:
                    snap[p.relative_to(self.workspace).as_posix()] = p.stat().st_mtime
                except OSError:
                    pass
        return snap

    def _capture(self, before: dict[str, float]) -> list[str]:
        artifacts: list[str] = []
        for p in self.workspace.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(self.workspace).as_posix()
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if rel not in before or mtime > before[rel] + 1e-6:
                artifacts.append(rel)
        return sorted(artifacts)

    def run_python(self, code: str) -> ExecResult:
        with self._lock:
            before = self._snapshot()
            start = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-c", code],
                    cwd=str(self.workspace),
                    capture_output=True,
                    text=True,
                    timeout=self.PYTHON_TIMEOUT_SECONDS,
                )
                stdout = proc.stdout[: self.OUTPUT_CHAR_LIMIT]
                stderr = proc.stderr[: self.OUTPUT_CHAR_LIMIT]
                exit_code = proc.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = ""
                if isinstance(exc.stdout, str):
                    stdout = exc.stdout[: self.OUTPUT_CHAR_LIMIT]
                stderr = f"[sandbox] python timed out after {self.PYTHON_TIMEOUT_SECONDS}s"
                exit_code = 124
            duration_ms = int((time.time() - start) * 1000)
            artifacts = self._capture(before)
            return ExecResult(
                ok=exit_code == 0, exit_code=exit_code,
                stdout=stdout, stderr=stderr,
                duration_ms=duration_ms, artifacts=artifacts,
            )

    def run_shell(self, command: str) -> ExecResult:
        with self._lock:
            before = self._snapshot()
            start = time.time()
            try:
                proc = subprocess.run(
                    command, shell=True,
                    cwd=str(self.workspace),
                    capture_output=True, text=True,
                    timeout=self.SHELL_TIMEOUT_SECONDS,
                    executable="/bin/bash" if os.path.exists("/bin/bash") else None,
                )
                stdout = proc.stdout[: self.OUTPUT_CHAR_LIMIT]
                stderr = proc.stderr[: self.OUTPUT_CHAR_LIMIT]
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                stdout = ""
                stderr = f"[sandbox] shell timed out after {self.SHELL_TIMEOUT_SECONDS}s"
                exit_code = 124
            duration_ms = int((time.time() - start) * 1000)
            artifacts = self._capture(before)
            return ExecResult(
                ok=exit_code == 0, exit_code=exit_code,
                stdout=stdout, stderr=stderr,
                duration_ms=duration_ms, artifacts=artifacts,
            )


# ===========================================================================
#  Tool registry — shared by AutoGPT, Manus and any agent feature
# ===========================================================================


class ToolRegistry:
    """JSON-friendly tool surface modelled after AutoGPT 0.4.7."""

    def __init__(self, workspace: Path, sandbox: Sandbox) -> None:
        self.workspace = workspace
        self.sandbox = sandbox
        self.notes: list[str] = []

    def spec(self) -> list[dict]:
        return [
            {"name": "write_file", "args": {"path": "string", "content": "string"},
             "desc": "Create or overwrite a text file inside the workspace."},
            {"name": "read_file", "args": {"path": "string"},
             "desc": "Read a text file from the workspace."},
            {"name": "list_files", "args": {},
             "desc": "List every file in the workspace."},
            {"name": "delete_file", "args": {"path": "string"},
             "desc": "Delete a file inside the workspace."},
            {"name": "run_python", "args": {"code": "string"},
             "desc": "Execute Python in the sandbox; returns stdout/stderr."},
            {"name": "run_shell", "args": {"command": "string"},
             "desc": "Execute a shell command in the sandbox."},
            {"name": "web_fetch", "args": {"url": "string"},
             "desc": "HTTP GET a URL and return the first 8KB of text."},
            {"name": "remember", "args": {"note": "string"},
             "desc": "Append a note to long-term memory the agent can re-read."},
            {"name": "recall", "args": {},
             "desc": "Return every note the agent has written so far."},
            {"name": "finish", "args": {"summary": "string"},
             "desc": "Signal the goal is complete and stop the loop."},
        ]

    def call(self, name: str, args: dict[str, Any]) -> str:
        try:
            handler = getattr(self, f"_t_{name}", None)
            if handler is None:
                return f"ERROR: unknown tool '{name}'"
            return handler(args or {})
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    def _t_write_file(self, args: dict) -> str:
        rel = str(args.get("path", "")).strip()
        content = args.get("content", "")
        if not rel:
            return "ERROR: path is required"
        if not isinstance(content, str):
            content = json.dumps(content)
        target = safe_join(self.workspace, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"OK: wrote {rel} ({len(content)} chars)"

    def _t_read_file(self, args: dict) -> str:
        rel = str(args.get("path", "")).strip()
        if not rel:
            return "ERROR: path is required"
        target = safe_join(self.workspace, rel)
        if not target.is_file():
            return f"ERROR: not a file: {rel}"
        return target.read_text(encoding="utf-8", errors="replace")[:16_000]

    def _t_list_files(self, args: dict) -> str:
        items = [p.relative_to(self.workspace).as_posix()
                 for p in sorted(self.workspace.rglob("*")) if p.is_file()]
        return "\n".join(items) if items else "(empty workspace)"

    def _t_delete_file(self, args: dict) -> str:
        rel = str(args.get("path", "")).strip()
        if not rel:
            return "ERROR: path is required"
        target = safe_join(self.workspace, rel)
        if not target.exists():
            return f"ERROR: not found: {rel}"
        target.unlink()
        return f"OK: deleted {rel}"

    def _t_run_python(self, args: dict) -> str:
        code = str(args.get("code", ""))
        if not code:
            return "ERROR: code is required"
        result = self.sandbox.run_python(code)
        body = f"exit={result.exit_code} time={result.duration_ms}ms"
        if result.stdout:
            body += "\n--- stdout ---\n" + result.stdout[:8_000]
        if result.stderr:
            body += "\n--- stderr ---\n" + result.stderr[:4_000]
        if result.artifacts:
            body += "\n--- new files ---\n" + "\n".join(result.artifacts)
        return body

    def _t_run_shell(self, args: dict) -> str:
        cmd = str(args.get("command", ""))
        if not cmd:
            return "ERROR: command is required"
        result = self.sandbox.run_shell(cmd)
        body = f"exit={result.exit_code} time={result.duration_ms}ms"
        if result.stdout:
            body += "\n--- stdout ---\n" + result.stdout[:8_000]
        if result.stderr:
            body += "\n--- stderr ---\n" + result.stderr[:4_000]
        return body

    def _t_web_fetch(self, args: dict) -> str:
        url = str(args.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            return "ERROR: url must be http(s)"
        req = urllib_request.Request(url, headers={"User-Agent": f"{APP_NAME}/{APP_VERSION}"})
        try:
            with urllib_request.urlopen(req, timeout=20) as resp:
                raw = resp.read(8 * 1024)
                charset = resp.headers.get_content_charset() or "utf-8"
                return raw.decode(charset, errors="replace")
        except Exception as exc:
            return f"ERROR: web_fetch failed: {exc}"

    def _t_remember(self, args: dict) -> str:
        note = str(args.get("note", "")).strip()
        if not note:
            return "ERROR: note is required"
        self.notes.append(note)
        return f"OK: stored note #{len(self.notes)}"

    def _t_recall(self, args: dict) -> str:
        if not self.notes:
            return "(no notes)"
        return "\n".join(f"{i+1}. {n}" for i, n in enumerate(self.notes))

    def _t_finish(self, args: dict) -> str:
        return "FINISH:" + str(args.get("summary", "")).strip()


# ===========================================================================
#  Rate limiter & circuit breaker (Ralph foundation)
# ===========================================================================


class RateLimiter:
    """Sliding 1-hour window of call timestamps."""

    def __init__(self, calls_per_hour: int) -> None:
        self.calls_per_hour = max(1, int(calls_per_hour))
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def update_limit(self, calls_per_hour: int) -> None:
        with self._lock:
            self.calls_per_hour = max(1, int(calls_per_hour))

    def _evict(self, now: float) -> None:
        cutoff = now - 3600.0
        while self._calls and self._calls[0] < cutoff:
            self._calls.popleft()

    def snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            self._evict(now)
            used = len(self._calls)
            remaining = max(0, self.calls_per_hour - used)
            reset_in = 0
            if self._calls and used >= self.calls_per_hour:
                reset_in = max(0, int(3600 - (now - self._calls[0])))
            return {"calls_used": used, "calls_limit": self.calls_per_hour,
                    "calls_remaining": remaining, "reset_seconds": reset_in}

    def acquire(self) -> tuple[bool, int]:
        with self._lock:
            now = time.time()
            self._evict(now)
            if len(self._calls) >= self.calls_per_hour:
                wait = max(1, int(3600 - (now - self._calls[0])))
                return False, wait
            self._calls.append(now)
            return True, 0


class CircuitBreaker:
    """CLOSED -> OPEN -> HALF_OPEN -> CLOSED with cooldown."""

    CLOSED, OPEN, HALF_OPEN = "CLOSED", "OPEN", "HALF_OPEN"

    def __init__(self, no_progress_threshold: int, same_error_threshold: int,
                 cooldown_minutes: int) -> None:
        self.no_progress_threshold = no_progress_threshold
        self.same_error_threshold = same_error_threshold
        self.cooldown_seconds = max(0, int(cooldown_minutes)) * 60
        self.state = self.CLOSED
        self.opened_at: float | None = None
        self.last_error = ""
        self.same_error_count = 0
        self.no_progress_count = 0
        self.last_signature = ""
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.state = self.CLOSED
            self.opened_at = None
            self.last_error = ""
            self.same_error_count = 0
            self.no_progress_count = 0
            self.last_signature = ""

    def snapshot(self) -> dict:
        with self._lock:
            cooldown_left = 0
            if self.state == self.OPEN and self.opened_at is not None:
                cooldown_left = max(0, int(self.cooldown_seconds - (time.time() - self.opened_at)))
            return {
                "state": self.state,
                "no_progress_count": self.no_progress_count,
                "same_error_count": self.same_error_count,
                "last_error": self.last_error[:200],
                "cooldown_seconds_left": cooldown_left,
            }

    def can_call(self) -> bool:
        with self._lock:
            if self.state == self.CLOSED or self.state == self.HALF_OPEN:
                return True
            if self.state == self.OPEN and self.opened_at is not None:
                if time.time() - self.opened_at >= self.cooldown_seconds:
                    self.state = self.HALF_OPEN
                    return True
            return False

    def record_success(self, signature: str) -> None:
        with self._lock:
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                self.opened_at = None
            if signature and signature == self.last_signature:
                self.no_progress_count += 1
                if self.no_progress_count >= self.no_progress_threshold:
                    self.state = self.OPEN
                    self.opened_at = time.time()
            else:
                self.no_progress_count = 0
            self.last_signature = signature
            # Successful call resets error counter
            self.same_error_count = 0
            self.last_error = ""

    def record_error(self, error: str) -> None:
        with self._lock:
            err = (error or "").strip()
            if err == self.last_error:
                self.same_error_count += 1
            else:
                self.same_error_count = 1
                self.last_error = err
            if self.same_error_count >= self.same_error_threshold:
                self.state = self.OPEN
                self.opened_at = time.time()


# ===========================================================================
#  Response analyzer — parses RALPH_STATUS + completion indicators
# ===========================================================================


class ResponseAnalyzer:
    COMPLETION_PHRASES = [
        "all tasks complete", "all tasks are complete",
        "project is complete", "project complete",
        "project ready", "nothing left to do", "no further work",
        "implementation complete", "done with all tasks",
        "everything is implemented", "ready for production",
    ]

    STATUS_RE = re.compile(r"RALPH_STATUS\s*:?\s*\n?(?P<body>(?:[ \t]*[A-Z_]+\s*:.*\n?){1,6})", re.IGNORECASE)
    EXIT_RE = re.compile(r"EXIT_SIGNAL\s*:\s*(true|false|yes|no|1|0)", re.IGNORECASE)
    STATUS_KEY_RE = re.compile(r"^\s*STATUS\s*:\s*(\S+)", re.IGNORECASE | re.MULTILINE)
    PROGRESS_RE = re.compile(r"PROGRESS_NOTE\s*:\s*(.+?)$", re.IGNORECASE | re.MULTILINE)

    @classmethod
    def analyze(cls, text: str) -> dict:
        text = text or ""
        lower = text.lower()
        completion_indicators = sum(1 for p in cls.COMPLETION_PHRASES if p in lower)
        # Treat checkbox completion as an indicator too
        if re.search(r"\b(\d+\s*/\s*\d+|\d+\s+of\s+\d+)\s+tasks?\s+(complete|done)\b", lower):
            completion_indicators += 1

        status_match = cls.STATUS_RE.search(text)
        exit_signal: bool | None = None
        status: str = ""
        progress_note: str = ""
        if status_match:
            body = status_match.group("body")
            em = cls.EXIT_RE.search(body)
            if em:
                exit_signal = em.group(1).lower() in {"true", "yes", "1"}
            sm = cls.STATUS_KEY_RE.search(body)
            if sm:
                status = sm.group(1).upper()
            pm = cls.PROGRESS_RE.search(body)
            if pm:
                progress_note = pm.group(1).strip()

        signature = re.sub(r"\s+", " ", text).strip()[:120]
        return {
            "completion_indicators": completion_indicators,
            "exit_signal": exit_signal,
            "status": status,
            "progress_note": progress_note,
            "signature": signature,
        }


# ===========================================================================
#  Ring-buffer log for live streaming to the UI
# ===========================================================================


class RingLog:
    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = capacity
        self._buf: deque[dict] = deque(maxlen=capacity)
        self._next_id = 1
        self._lock = threading.Lock()

    def append(self, level: str, source: str, message: str) -> None:
        with self._lock:
            entry = {
                "id": self._next_id,
                "ts": utc_now_iso(),
                "level": level,
                "source": source,
                "message": message,
            }
            self._next_id += 1
            self._buf.append(entry)

    def since(self, last_id: int, limit: int = 500) -> list[dict]:
        with self._lock:
            out = [e for e in self._buf if e["id"] > last_id]
            return out[-limit:]

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


# ===========================================================================
#  Ralph project files helper
# ===========================================================================


class RalphFiles:
    def __init__(self, project_root: Path) -> None:
        self.root = project_root
        self.dir = project_root / RALPH_DIR_NAME
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        (self.dir / "specs").mkdir(exist_ok=True)
        self._ensure_default("PROMPT.md", DEFAULT_PROMPT_MD)
        self._ensure_default("fix_plan.md", DEFAULT_FIX_PLAN_MD)
        self._ensure_default("AGENT.md", DEFAULT_AGENT_MD)
        self._ensure_default(".ralphrc", DEFAULT_RALPHRC)

    def _ensure_default(self, name: str, default: str) -> None:
        p = self.dir / name
        if not p.exists():
            p.write_text(default, encoding="utf-8")

    def path(self, name: str) -> Path:
        if name not in RALPH_EDITABLE_FILES:
            raise ValueError(f"not an editable Ralph file: {name}")
        return self.dir / name

    def read(self, name: str) -> str:
        return self.path(name).read_text(encoding="utf-8")

    def write(self, name: str, content: str) -> None:
        self.path(name).write_text(content, encoding="utf-8")

    def write_status(self, status: dict) -> None:
        try:
            (self.dir / RALPH_STATUS_FILE).write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def append_log(self, line: str) -> None:
        try:
            with (self.dir / "logs" / RALPH_LOG_FILE).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def fix_plan_unchecked_count(self) -> int:
        try:
            text = self.read("fix_plan.md")
        except Exception:
            return 0
        return len(re.findall(r"^\s*-\s*\[\s*\]\s+", text, re.MULTILINE))


# ===========================================================================
#  RALPH LOOP — the foundation. Wake / Sleep / Run / Stop
# ===========================================================================


class RalphLoop:
    """Autonomous loop. States: STOPPED -> RUNNING -> SLEEPING -> RUNNING -> STOPPED."""

    STOPPED, RUNNING, SLEEPING = "STOPPED", "RUNNING", "SLEEPING"

    def __init__(self, config: AppConfig, client: LMStudioClient,
                 files: RalphFiles, log: RingLog) -> None:
        self.config = config
        self.client = client
        self.files = files
        self.log = log
        self.rate_limiter = RateLimiter(config.max_calls_per_hour)
        self.breaker = CircuitBreaker(
            config.cb_no_progress_threshold,
            config.cb_same_error_threshold,
            config.cb_cooldown_minutes,
        )
        self.session_id = ""
        self.session_started_at: float | None = None
        self.history: list[dict[str, str]] = []
        self.state = self.STOPPED
        self.loop_count = 0
        self.idle_loops = 0
        self.last_response = ""
        self.last_progress = ""
        self.exit_reason = ""
        self.completion_indicators_streak = 0
        self.model = ""
        self.temperature = config.default_temperature
        self.max_tokens = config.default_max_tokens
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ---- session ----------------------------------------------------------
    def _new_session(self) -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.session_started_at = time.time()
        self.history = []
        self.completion_indicators_streak = 0
        self.exit_reason = ""
        self._info(f"New session {self.session_id}")

    def _session_expired(self) -> bool:
        if not self.session_started_at:
            return False
        age_h = (time.time() - self.session_started_at) / 3600.0
        return age_h >= self.config.session_expiry_hours

    def reset_session(self) -> None:
        with self._lock:
            self._new_session()

    # ---- public controls --------------------------------------------------
    def start(self, *, model: str, temperature: float, max_tokens: int,
              max_calls_per_hour: int) -> dict:
        with self._lock:
            if self.state != self.STOPPED:
                return {"ok": False, "error": f"loop already {self.state}"}
            if not model:
                return {"ok": False, "error": "model is required"}
            self.model = model
            self.temperature = float(temperature)
            self.max_tokens = int(max_tokens)
            self.rate_limiter.update_limit(int(max_calls_per_hour))
            self.breaker.reset()
            self._new_session()
            self.loop_count = 0
            self.idle_loops = 0
            self.state = self.RUNNING
            self._stop_event.clear()
            self._wake_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._info(f"Loop started • model={model} t={temperature} max_tok={max_tokens}")
            return {"ok": True}

    def stop(self) -> dict:
        with self._lock:
            if self.state == self.STOPPED:
                return {"ok": True}
            self._stop_event.set()
            self._wake_event.set()
            self._info("Stop requested")
            return {"ok": True}

    def sleep_now(self) -> dict:
        with self._lock:
            if self.state != self.RUNNING:
                return {"ok": False, "error": f"cannot sleep from {self.state}"}
            self.state = self.SLEEPING
            self._wake_event.clear()
            self._info("Sleeping (manual)")
            return {"ok": True}

    def wake_now(self) -> dict:
        with self._lock:
            if self.state == self.STOPPED:
                return {"ok": False, "error": "loop is stopped — start it first"}
            self.state = self.RUNNING
            self.idle_loops = 0
            self._wake_event.set()
            self._info("Wake requested")
            return {"ok": True}

    def reset_circuit(self) -> dict:
        self.breaker.reset()
        self._info("Circuit breaker reset")
        return {"ok": True}

    # ---- snapshot for UI --------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "model": self.model,
                "loop_count": self.loop_count,
                "idle_loops": self.idle_loops,
                "session_id": self.session_id,
                "session_age_seconds": (
                    int(time.time() - self.session_started_at)
                    if self.session_started_at else 0
                ),
                "last_progress": self.last_progress,
                "last_response": self.last_response[-4000:],
                "exit_reason": self.exit_reason,
                "completion_indicators_streak": self.completion_indicators_streak,
                "rate_limit": self.rate_limiter.snapshot(),
                "circuit": self.breaker.snapshot(),
                "fix_plan_open": self.files.fix_plan_unchecked_count(),
                "wake_interval_seconds": self.config.wake_interval_seconds,
                "sleep_after_idle_loops": self.config.sleep_after_idle_loops,
            }

    # ---- internals --------------------------------------------------------
    def _info(self, msg: str) -> None:
        self.log.append("info", "ralph", msg)
        self.files.append_log(f"[{utc_now_iso()}] [INFO] {msg}")

    def _warn(self, msg: str) -> None:
        self.log.append("warn", "ralph", msg)
        self.files.append_log(f"[{utc_now_iso()}] [WARN] {msg}")

    def _error(self, msg: str) -> None:
        self.log.append("error", "ralph", msg)
        self.files.append_log(f"[{utc_now_iso()}] [ERROR] {msg}")

    def _build_messages(self) -> list[dict[str, str]]:
        try:
            prompt_md = self.files.read("PROMPT.md")
        except Exception:
            prompt_md = DEFAULT_PROMPT_MD
        try:
            fix_plan = self.files.read("fix_plan.md")
        except Exception:
            fix_plan = DEFAULT_FIX_PLAN_MD
        try:
            agent_md = self.files.read("AGENT.md")
        except Exception:
            agent_md = ""

        system = (
            f"You are CatRalph, an autonomous coding agent running on LM Studio.\n\n"
            f"--- PROMPT.md ---\n{prompt_md}\n\n"
            f"--- AGENT.md ---\n{agent_md}\n"
        )
        user_now = (
            f"Loop #{self.loop_count + 1}. Current fix_plan.md:\n\n{fix_plan}\n\n"
            f"Pick the highest-priority unchecked task and do ONE step. "
            f"End with the RALPH_STATUS block as specified."
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        # session continuity — include trimmed history
        if self.config.session_continuity:
            budget = self.config.history_char_budget
            trimmed: list[dict[str, str]] = []
            spent = 0
            for msg in reversed(self.history):
                cost = len(msg.get("content", "")) + 8
                if spent + cost > budget:
                    break
                trimmed.append(msg)
                spent += cost
            messages.extend(reversed(trimmed))
        messages.append({"role": "user", "content": user_now})
        return messages

    def _maybe_exit(self, analysis: dict) -> str | None:
        """Return an exit reason string if the dual-condition gate fires."""
        ind = int(analysis.get("completion_indicators", 0) or 0)
        exit_signal = analysis.get("exit_signal")
        # Only accumulate streak when EXIT_SIGNAL is true (matches v0.11.1 fix)
        if exit_signal is True and ind >= self.config.completion_indicators_required:
            self.completion_indicators_streak += 1
        else:
            self.completion_indicators_streak = 0
        if (exit_signal is True
                and ind >= self.config.completion_indicators_required
                and self.files.fix_plan_unchecked_count() == 0):
            return "project_complete"
        if self.completion_indicators_streak >= 5:
            return "safety_circuit_5_consecutive_completions"
        return None

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                # SLEEP gate
                if self.state == self.SLEEPING:
                    woke = self._wake_event.wait(timeout=self.config.wake_interval_seconds)
                    if self._stop_event.is_set():
                        break
                    self._wake_event.clear()
                    with self._lock:
                        self.state = self.RUNNING
                        self.idle_loops = 0
                    self._info("Woke up" if woke else "Periodic wake")
                    # When waking from periodic timer, only continue if there's work
                    if not woke and self.files.fix_plan_unchecked_count() == 0:
                        with self._lock:
                            self.state = self.SLEEPING
                        self._info("No tasks queued — back to sleep")
                        continue

                # session expiration
                if self._session_expired():
                    self._warn("Session expired — rotating")
                    self._new_session()

                # rate limit
                ok, wait = self.rate_limiter.acquire()
                if not ok:
                    self._warn(f"Rate limited — waiting {wait}s")
                    waited = 0
                    while waited < wait and not self._stop_event.is_set():
                        time.sleep(1)
                        waited += 1
                    continue

                # circuit breaker
                if not self.breaker.can_call():
                    cb = self.breaker.snapshot()
                    self._warn(f"Circuit OPEN — cooldown {cb['cooldown_seconds_left']}s")
                    time.sleep(min(30, max(1, cb["cooldown_seconds_left"])))
                    continue

                # call the model
                self.loop_count += 1
                self._info(f"Loop {self.loop_count} → calling model")
                messages = self._build_messages()
                try:
                    result = self.client.chat(
                        model=self.model, messages=messages,
                        temperature=self.temperature, max_tokens=self.max_tokens,
                    )
                    content = result["content"]
                except Exception as exc:
                    self._error(f"Model call failed: {exc}")
                    self.breaker.record_error(str(exc))
                    time.sleep(self.config.inter_loop_seconds)
                    continue

                # remember in session history
                if self.config.session_continuity:
                    self.history.append({"role": "user", "content": "(loop tick)"})
                    self.history.append({"role": "assistant", "content": content})

                # analyze
                analysis = ResponseAnalyzer.analyze(content)
                self.last_response = content
                self.last_progress = analysis.get("progress_note") or analysis.get("status") or ""
                self.breaker.record_success(analysis["signature"])

                self._info(
                    f"Loop {self.loop_count} ← "
                    f"status={analysis['status'] or '?'} "
                    f"exit={analysis['exit_signal']} "
                    f"ci={analysis['completion_indicators']}"
                )

                # progress detection — count idle when nothing changed
                fp_open = self.files.fix_plan_unchecked_count()
                if analysis["progress_note"] or analysis["status"] in {"COMPLETE", "IN_PROGRESS"}:
                    self.idle_loops = 0
                else:
                    self.idle_loops += 1

                # auto-sleep when idle
                if self.idle_loops >= self.config.sleep_after_idle_loops and fp_open > 0:
                    with self._lock:
                        self.state = self.SLEEPING
                        self._wake_event.clear()
                    self._info(
                        f"Idle for {self.idle_loops} loops — sleeping "
                        f"(periodic wake every {self.config.wake_interval_seconds}s)"
                    )

                # auto-sleep when nothing left to do
                if fp_open == 0 and not analysis["exit_signal"]:
                    with self._lock:
                        self.state = self.SLEEPING
                        self._wake_event.clear()
                    self._info("fix_plan.md fully checked — sleeping until tasks are added")

                # exit gate
                reason = self._maybe_exit(analysis)
                if reason:
                    self.exit_reason = reason
                    self._info(f"Exit gate fired: {reason}")
                    break

                # push status
                self.files.write_status(self.snapshot())
                time.sleep(self.config.inter_loop_seconds)
        finally:
            with self._lock:
                self.state = self.STOPPED
            self._info("Loop stopped")
            self.files.write_status(self.snapshot())


# ===========================================================================
#  AutoGPT 0.4.7 — think -> plan -> act -> observe -> reflect
# ===========================================================================


AUTOGPT_SYSTEM = """You are an autonomous AutoGPT-style agent running on LM Studio.

You operate one step at a time. On every step, respond with ONE JSON object
matching this exact schema (no extra prose, no code fences):

{
  "thoughts": {
    "text": "...",
    "reasoning": "...",
    "plan": ["next step 1", "next step 2", "next step 3"],
    "criticism": "...",
    "speak": "what you'd say out loud"
  },
  "command": {
    "name": "<one of: write_file, read_file, list_files, delete_file, run_python, run_shell, web_fetch, remember, recall, finish>",
    "args": { ... }
  }
}

Available commands and their args are listed in TOOLS. Use `finish` ONLY when
the GOAL is fully achieved. Keep arguments small and specific. Do not add any
text outside the JSON object.
"""


class AutoGPTAgent:
    """Single-instance AutoGPT loop with wake / sleep mirroring Ralph."""

    STOPPED, RUNNING, SLEEPING = "STOPPED", "RUNNING", "SLEEPING"

    def __init__(self, config: AppConfig, client: LMStudioClient,
                 workspace: Path, sandbox: Sandbox, log: RingLog) -> None:
        self.config = config
        self.client = client
        self.workspace = workspace
        self.tools = ToolRegistry(workspace, sandbox)
        self.log = log
        self.state = self.STOPPED
        self.goal = ""
        self.model = ""
        self.temperature = 0.5
        self.max_tokens = 1024
        self.max_steps = 30
        self.step = 0
        self.last_thought = ""
        self.last_plan: list[str] = []
        self.last_command = ""
        self.last_observation = ""
        self.finish_summary = ""
        self.history: list[dict[str, str]] = []
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state, "goal": self.goal, "model": self.model,
                "step": self.step, "max_steps": self.max_steps,
                "last_thought": self.last_thought, "last_plan": list(self.last_plan),
                "last_command": self.last_command,
                "last_observation": self.last_observation[-4000:],
                "finish_summary": self.finish_summary,
                "notes": list(self.tools.notes),
            }

    def start(self, *, goal: str, model: str, temperature: float,
              max_tokens: int, max_steps: int) -> dict:
        with self._lock:
            if self.state != self.STOPPED:
                return {"ok": False, "error": f"agent is {self.state}"}
            if not goal.strip():
                return {"ok": False, "error": "goal is required"}
            if not model:
                return {"ok": False, "error": "model is required"}
            self.goal = goal.strip()
            self.model = model
            self.temperature = float(temperature)
            self.max_tokens = int(max_tokens)
            self.max_steps = max(1, int(max_steps))
            self.step = 0
            self.last_thought = self.last_command = self.last_observation = ""
            self.last_plan = []
            self.finish_summary = ""
            self.history = []
            self.tools.notes = []
            self.state = self.RUNNING
            self._stop.clear(); self._wake.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._log("info", f"AutoGPT started • goal={goal!r}")
            return {"ok": True}

    def stop(self) -> dict:
        self._stop.set(); self._wake.set()
        self._log("info", "AutoGPT stop requested")
        return {"ok": True}

    def sleep_now(self) -> dict:
        with self._lock:
            if self.state == self.RUNNING:
                self.state = self.SLEEPING
                self._wake.clear()
                self._log("info", "AutoGPT sleeping")
                return {"ok": True}
            return {"ok": False, "error": f"cannot sleep from {self.state}"}

    def wake_now(self) -> dict:
        with self._lock:
            if self.state == self.STOPPED:
                return {"ok": False, "error": "agent stopped"}
            self.state = self.RUNNING
            self._wake.set()
            self._log("info", "AutoGPT woke up")
            return {"ok": True}

    def _log(self, level: str, msg: str) -> None:
        self.log.append(level, "autogpt", msg)

    def _build_prompt(self, last_observation: str) -> list[dict[str, str]]:
        tool_spec = json.dumps(self.tools.spec(), indent=2)
        system = AUTOGPT_SYSTEM + "\n\nTOOLS:\n" + tool_spec
        user = (
            f"GOAL: {self.goal}\n\n"
            f"STEP: {self.step + 1} / {self.max_steps}\n"
            f"NOTES SO FAR ({len(self.tools.notes)}):\n"
            + ("\n".join(f"- {n}" for n in self.tools.notes[-10:]) or "(none)")
            + f"\n\nLAST OBSERVATION:\n{last_observation or '(none — first step)'}\n\n"
            "Respond with the JSON object only."
        )
        msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
        # short rolling history
        budget = 8000
        spent = 0
        trimmed: list[dict[str, str]] = []
        for m in reversed(self.history[-12:]):
            c = len(m.get("content", "")) + 8
            if spent + c > budget:
                break
            trimmed.append(m); spent += c
        msgs.extend(reversed(trimmed))
        msgs.append({"role": "user", "content": user})
        return msgs

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        # Strip code fences
        text = re.sub(r"```(?:json)?\s*|\s*```", "", text.strip())
        # Find the first {...} that parses
        start = text.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[start:i + 1]
                        try:
                            return json.loads(chunk)
                        except json.JSONDecodeError:
                            break
            start = text.find("{", start + 1)
        return None

    def _run(self) -> None:
        observation = ""
        try:
            while not self._stop.is_set() and self.step < self.max_steps:
                if self.state == self.SLEEPING:
                    self._wake.wait(timeout=60)
                    if self._stop.is_set():
                        break
                    self._wake.clear()
                    with self._lock:
                        self.state = self.RUNNING

                msgs = self._build_prompt(observation)
                try:
                    result = self.client.chat(
                        model=self.model, messages=msgs,
                        temperature=self.temperature, max_tokens=self.max_tokens,
                    )
                    raw = result["content"]
                except Exception as exc:
                    self._log("error", f"model call failed: {exc}")
                    time.sleep(2)
                    continue

                self.history.append({"role": "assistant", "content": raw})
                obj = self._extract_json(raw)
                if obj is None:
                    observation = "ERROR: response was not valid JSON. Try again with a single JSON object."
                    self._log("warn", "non-JSON response; retrying")
                    self.step += 1
                    continue

                thoughts = obj.get("thoughts") or {}
                command = obj.get("command") or {}
                cmd_name = str(command.get("name", "")).strip()
                cmd_args = command.get("args") or {}
                with self._lock:
                    self.step += 1
                    self.last_thought = str(thoughts.get("text", ""))[:1000]
                    plan_val = thoughts.get("plan") or []
                    if isinstance(plan_val, str):
                        plan_val = [plan_val]
                    self.last_plan = [str(x) for x in plan_val][:8]
                    self.last_command = f"{cmd_name}({json.dumps(cmd_args)[:200]})"

                self._log("info", f"step {self.step}: {self.last_command}")

                observation = self.tools.call(cmd_name, cmd_args if isinstance(cmd_args, dict) else {})
                with self._lock:
                    self.last_observation = observation

                if observation.startswith("FINISH:"):
                    with self._lock:
                        self.finish_summary = observation[len("FINISH:"):].strip()
                    self._log("info", f"AutoGPT finished: {self.finish_summary}")
                    break

                self.history.append({"role": "user", "content": f"OBSERVATION:\n{observation[:4000]}"})
                time.sleep(self.config.inter_loop_seconds)
        finally:
            with self._lock:
                self.state = self.STOPPED


# ===========================================================================
#  GPT-Engineer style multi-file project generator
# ===========================================================================


ENGINEER_SYSTEM = """You are gpt-engineer running on LM Studio. Given a project
description, produce a complete multi-file codebase.

Respond ONLY with one JSON object matching this schema:

{
  "files": [
    {"path": "relative/path.ext", "content": "FULL FILE CONTENTS"},
    ...
  ],
  "run": "command to run the project (one line)",
  "notes": "short summary of what was generated"
}

Rules:
  * Always include a README.md that explains how to run.
  * Pick reasonable defaults; do not ask clarifying questions.
  * Use only standard libraries unless the user explicitly asked otherwise.
  * Keep total output under ~30KB.
  * Do not wrap the JSON in code fences.
"""


class ProjectEngineer:
    """One-shot codebase generator (gpt-engineer flavour)."""

    def __init__(self, client: LMStudioClient, workspace: Path, log: RingLog) -> None:
        self.client = client
        self.workspace = workspace
        self.log = log
        self.last_run = ""
        self.last_notes = ""
        self.last_files: list[dict] = []
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            files = []
            for p in sorted(self.workspace.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(self.workspace).as_posix()
                    try:
                        size = p.stat().st_size
                    except OSError:
                        size = 0
                    files.append({"name": rel, "size": size})
            return {"run": self.last_run, "notes": self.last_notes, "files": files}

    def generate(self, *, description: str, model: str,
                 temperature: float, max_tokens: int) -> dict:
        if not description.strip():
            return {"ok": False, "error": "description is required"}
        if not model:
            return {"ok": False, "error": "model is required"}

        self.log.append("info", "engineer", f"generating project • model={model}")
        messages = [
            {"role": "system", "content": ENGINEER_SYSTEM},
            {"role": "user", "content": f"PROJECT DESCRIPTION:\n{description.strip()}"},
        ]
        try:
            result = self.client.chat(
                model=model, messages=messages,
                temperature=float(temperature), max_tokens=int(max_tokens),
            )
            raw = result["content"]
        except Exception as exc:
            self.log.append("error", "engineer", f"model call failed: {exc}")
            return {"ok": False, "error": str(exc)}

        obj = AutoGPTAgent._extract_json(raw)
        if obj is None or not isinstance(obj.get("files"), list):
            self.log.append("warn", "engineer", "model returned no file list")
            return {"ok": False, "error": "model did not return a valid {files: [...]} JSON"}

        written: list[dict] = []
        for entry in obj["files"]:
            if not isinstance(entry, dict):
                continue
            rel = str(entry.get("path", "")).strip()
            content = entry.get("content", "")
            if not rel or not isinstance(content, str):
                continue
            try:
                target = safe_join(self.workspace, rel)
            except ValueError as exc:
                self.log.append("warn", "engineer", str(exc))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append({"name": rel, "size": len(content)})

        with self._lock:
            self.last_run = str(obj.get("run", "")).strip()
            self.last_notes = str(obj.get("notes", "")).strip()
            self.last_files = written

        self.log.append("info", "engineer", f"wrote {len(written)} files")
        return {"ok": True, "files": written, "run": self.last_run, "notes": self.last_notes}


# ===========================================================================
#  Manus — long-horizon planning agent
# ===========================================================================


MANUS_PLANNER_SYSTEM = """You are Manus, a long-horizon task agent.

Given a GOAL, produce a numbered plan of 3-10 concrete steps that, when
executed in order, will produce the deliverables.

Respond ONLY with one JSON object:
{
  "plan": ["step 1...", "step 2...", "..."],
  "deliverables": ["expected file 1", "expected file 2"]
}
"""


MANUS_EXECUTOR_SYSTEM = """You are Manus executing one step of a plan.

Available TOOLS (JSON object only, no prose):
{
  "thought": "brief reasoning",
  "command": {
    "name": "<one of: write_file, read_file, list_files, delete_file, run_python, run_shell, web_fetch, remember, recall, finish>",
    "args": { ... }
  }
}

Use `finish` only after the current step is complete.
"""


class ManusAgent:
    STOPPED, PLANNING, RUNNING, SLEEPING = "STOPPED", "PLANNING", "RUNNING", "SLEEPING"

    def __init__(self, config: AppConfig, client: LMStudioClient,
                 workspace: Path, sandbox: Sandbox, log: RingLog) -> None:
        self.config = config
        self.client = client
        self.workspace = workspace
        self.tools = ToolRegistry(workspace, sandbox)
        self.log = log
        self.state = self.STOPPED
        self.goal = ""
        self.model = ""
        self.temperature = 0.5
        self.max_tokens = 1024
        self.max_steps_per_task = 6
        self.plan: list[dict] = []          # [{ "task": str, "status": "pending|done|skipped" }]
        self.deliverables: list[str] = []
        self.current_index = 0
        self.last_thought = ""
        self.last_command = ""
        self.last_observation = ""
        self.summary = ""
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state, "goal": self.goal, "model": self.model,
                "plan": list(self.plan),
                "deliverables": list(self.deliverables),
                "current_index": self.current_index,
                "last_thought": self.last_thought,
                "last_command": self.last_command,
                "last_observation": self.last_observation[-4000:],
                "summary": self.summary,
                "files": [
                    {"name": p.relative_to(self.workspace).as_posix(),
                     "size": p.stat().st_size}
                    for p in sorted(self.workspace.rglob("*")) if p.is_file()
                ],
            }

    def start(self, *, goal: str, model: str, temperature: float,
              max_tokens: int, max_steps_per_task: int) -> dict:
        with self._lock:
            if self.state != self.STOPPED:
                return {"ok": False, "error": f"agent is {self.state}"}
            if not goal.strip() or not model:
                return {"ok": False, "error": "goal and model are required"}
            self.goal = goal.strip()
            self.model = model
            self.temperature = float(temperature)
            self.max_tokens = int(max_tokens)
            self.max_steps_per_task = max(1, int(max_steps_per_task))
            self.plan = []
            self.deliverables = []
            self.current_index = 0
            self.last_thought = self.last_command = self.last_observation = ""
            self.summary = ""
            self.tools.notes = []
            self.state = self.PLANNING
            self._stop.clear(); self._wake.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self.log.append("info", "manus", f"Manus started • goal={goal!r}")
            return {"ok": True}

    def stop(self) -> dict:
        self._stop.set(); self._wake.set()
        self.log.append("info", "manus", "Manus stop requested")
        return {"ok": True}

    def sleep_now(self) -> dict:
        with self._lock:
            if self.state == self.RUNNING:
                self.state = self.SLEEPING
                self._wake.clear()
                return {"ok": True}
            return {"ok": False, "error": f"cannot sleep from {self.state}"}

    def wake_now(self) -> dict:
        with self._lock:
            if self.state == self.STOPPED:
                return {"ok": False, "error": "agent stopped"}
            self.state = self.RUNNING
            self._wake.set()
            return {"ok": True}

    def _make_plan(self) -> bool:
        msgs = [
            {"role": "system", "content": MANUS_PLANNER_SYSTEM},
            {"role": "user", "content": f"GOAL: {self.goal}"},
        ]
        try:
            result = self.client.chat(
                model=self.model, messages=msgs,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
        except Exception as exc:
            self.log.append("error", "manus", f"planner call failed: {exc}")
            return False
        obj = AutoGPTAgent._extract_json(result["content"])
        if not obj or not isinstance(obj.get("plan"), list):
            self.log.append("warn", "manus", "planner returned no plan")
            return False
        with self._lock:
            self.plan = [{"task": str(t), "status": "pending"} for t in obj["plan"] if t]
            self.deliverables = [str(d) for d in (obj.get("deliverables") or [])]
        self.log.append("info", "manus", f"plan with {len(self.plan)} step(s)")
        return True

    def _execute_step(self, idx: int) -> None:
        task = self.plan[idx]["task"]
        observation = ""
        for sub in range(self.max_steps_per_task):
            if self._stop.is_set():
                return
            tool_spec = json.dumps(self.tools.spec(), indent=2)
            system = MANUS_EXECUTOR_SYSTEM + "\n\nTOOLS:\n" + tool_spec
            user = (
                f"GOAL: {self.goal}\n\nCURRENT STEP ({idx+1}/{len(self.plan)}): {task}\n\n"
                f"PRIOR OBSERVATION:\n{observation or '(none)'}\n\n"
                "Respond with the JSON object only."
            )
            try:
                result = self.client.chat(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=self.temperature, max_tokens=self.max_tokens,
                )
            except Exception as exc:
                self.log.append("error", "manus", f"executor call failed: {exc}")
                return
            obj = AutoGPTAgent._extract_json(result["content"])
            if obj is None:
                observation = "ERROR: response was not valid JSON."
                continue
            command = obj.get("command") or {}
            cmd_name = str(command.get("name", "")).strip()
            cmd_args = command.get("args") or {}
            with self._lock:
                self.last_thought = str(obj.get("thought", ""))[:1000]
                self.last_command = f"{cmd_name}({json.dumps(cmd_args)[:200]})"
            self.log.append("info", "manus", f"step {idx+1}.{sub+1}: {self.last_command}")
            observation = self.tools.call(cmd_name, cmd_args if isinstance(cmd_args, dict) else {})
            with self._lock:
                self.last_observation = observation
            if observation.startswith("FINISH:") or cmd_name == "finish":
                self.plan[idx]["status"] = "done"
                return
        # ran out of sub-steps
        self.plan[idx]["status"] = "done"

    def _run(self) -> None:
        try:
            if not self._make_plan():
                return
            with self._lock:
                self.state = self.RUNNING
            for i in range(len(self.plan)):
                if self._stop.is_set():
                    break
                if self.state == self.SLEEPING:
                    self._wake.wait(timeout=60)
                    if self._stop.is_set():
                        break
                    self._wake.clear()
                    with self._lock:
                        self.state = self.RUNNING
                with self._lock:
                    self.current_index = i
                self._execute_step(i)
                time.sleep(self.config.inter_loop_seconds)
            with self._lock:
                done = sum(1 for s in self.plan if s["status"] == "done")
                self.summary = f"Completed {done}/{len(self.plan)} steps."
            self.log.append("info", "manus", self.summary)
        finally:
            with self._lock:
                self.state = self.STOPPED


# ===========================================================================
#  HTML UI — large, replaced in pass 2
# ===========================================================================

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CatSDK • Local LM Studio Playground</title>
<style>
  :root {
    /* Blue palette — preserved from CatSDK */
    --bg-0:#06101f;            /* deepest */
    --bg-1:#0a1830;            /* base */
    --bg-2:#0f2244;            /* surface */
    --bg-3:#143055;            /* raised */
    --bg-4:#1a3d6b;            /* hover */
    --line:#1d3f72;
    --line-2:#27548f;
    --ink:#e6f0ff;
    --ink-dim:#a8c0e6;
    --ink-mute:#6f8ab2;
    --accent:#3aa0ff;          /* primary blue */
    --accent-2:#1f7ad9;
    --accent-3:#73c2ff;
    --good:#4ad6a4;
    --warn:#ffc15c;
    --bad:#ff6b8a;
    --shadow:0 18px 48px rgba(0,12,30,0.55);
    --radius:14px;
    --radius-sm:9px;
    --mono:'JetBrains Mono','Menlo','SF Mono',ui-monospace,monospace;
    --sans:-apple-system,BlinkMacSystemFont,'Inter','SF Pro Display',sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0;height:100%;font-family:var(--sans);
    background:radial-gradient(circle at 18% -10%,#15316e 0%,transparent 55%),
               radial-gradient(circle at 110% 110%,#0d2a5a 0%,transparent 55%),
               linear-gradient(180deg,var(--bg-0),var(--bg-1));
    color:var(--ink);overflow:hidden}
  a{color:var(--accent-3);text-decoration:none}
  a:hover{color:var(--accent)}
  button{font-family:inherit;font-size:13px;cursor:pointer;border:1px solid var(--line-2);
    background:var(--bg-3);color:var(--ink);padding:7px 14px;border-radius:var(--radius-sm);
    transition:all .12s}
  button:hover:not(:disabled){background:var(--bg-4);border-color:var(--accent-2)}
  button:disabled{opacity:.45;cursor:not-allowed}
  button.primary{background:linear-gradient(180deg,var(--accent),var(--accent-2));
    border-color:var(--accent);color:#fff;font-weight:600}
  button.primary:hover:not(:disabled){filter:brightness(1.08)}
  button.danger{background:#3a0e1c;border-color:#7a2740;color:#ffb4c4}
  button.danger:hover:not(:disabled){background:#5a1730}
  button.ghost{background:transparent}
  input,select,textarea{font-family:inherit;font-size:13px;background:var(--bg-1);
    color:var(--ink);border:1px solid var(--line);border-radius:var(--radius-sm);
    padding:7px 10px;outline:none}
  input:focus,select:focus,textarea:focus{border-color:var(--accent-2);
    box-shadow:0 0 0 3px rgba(58,160,255,0.18)}
  textarea{resize:vertical;font-family:var(--mono);line-height:1.45}
  ::selection{background:rgba(58,160,255,0.35)}

  /* Layout */
  .app{display:grid;grid-template-rows:auto auto 1fr;height:100%}
  .topbar{display:flex;align-items:center;justify-content:space-between;
    padding:10px 18px;border-bottom:1px solid var(--line);
    background:linear-gradient(180deg,rgba(9,22,46,0.92),rgba(7,18,38,0.7));
    backdrop-filter:blur(8px)}
  .brand{display:flex;align-items:center;gap:12px}
  .brand .glyph{width:34px;height:34px;border-radius:10px;
    background:linear-gradient(135deg,#1f7ad9,#73c2ff);display:grid;place-items:center;
    box-shadow:0 6px 18px rgba(31,122,217,0.4);font-size:18px}
  .brand .title{font-weight:700;letter-spacing:.3px}
  .brand .sub{font-size:11px;color:var(--ink-mute);margin-top:1px}
  .topbar-right{display:flex;align-items:center;gap:10px}
  .badge{font-size:11px;padding:3px 9px;border-radius:999px;border:1px solid var(--line-2);
    background:var(--bg-2);color:var(--ink-dim)}
  .badge.ok{color:var(--good);border-color:rgba(74,214,164,0.4);
    background:rgba(74,214,164,0.08)}
  .badge.warn{color:var(--warn);border-color:rgba(255,193,92,0.4);
    background:rgba(255,193,92,0.08)}
  .badge.bad{color:var(--bad);border-color:rgba(255,107,138,0.4);
    background:rgba(255,107,138,0.08)}

  .tabs{display:flex;gap:2px;padding:0 14px;border-bottom:1px solid var(--line);
    background:rgba(7,18,38,0.55)}
  .tab{padding:10px 16px;border:none;background:transparent;color:var(--ink-dim);
    border-bottom:2px solid transparent;border-radius:0;font-weight:500;font-size:13px}
  .tab:hover{color:var(--ink);background:transparent}
  .tab.active{color:var(--accent-3);border-bottom-color:var(--accent)}

  .main{overflow:hidden;position:relative}
  .panel{position:absolute;inset:0;display:none;overflow:hidden}
  .panel.active{display:block}

  /* Cards */
  .card{background:linear-gradient(180deg,rgba(20,48,85,0.55),rgba(10,24,48,0.55));
    border:1px solid var(--line);border-radius:var(--radius);
    box-shadow:var(--shadow)}
  .card-header{padding:11px 14px;border-bottom:1px solid var(--line);
    display:flex;align-items:center;justify-content:space-between;gap:10px}
  .card-header h3{margin:0;font-size:13px;letter-spacing:.4px;
    color:var(--accent-3);text-transform:uppercase;font-weight:600}
  .card-body{padding:12px 14px}

  /* CHAT TAB ============================================================ */
  #panel-chat{display:none}
  #panel-chat.active{display:grid}
  #panel-chat{grid-template-rows:1fr auto;gap:12px;padding:14px 18px}
  .chat-stream{overflow:auto;padding:8px;display:flex;flex-direction:column;gap:10px}
  .chat-msg{max-width:78%;padding:10px 14px;border-radius:14px;line-height:1.5;
    white-space:pre-wrap;word-wrap:break-word;font-size:14px}
  .chat-msg.user{align-self:flex-end;background:linear-gradient(135deg,#1f7ad9,#3aa0ff);color:#fff;
    border-bottom-right-radius:4px}
  .chat-msg.assistant{align-self:flex-start;background:var(--bg-2);
    border:1px solid var(--line);border-bottom-left-radius:4px}
  .chat-msg.system{align-self:center;background:transparent;color:var(--ink-mute);
    font-size:12px;font-style:italic}
  .chat-msg .role{font-size:10px;text-transform:uppercase;letter-spacing:.6px;
    opacity:.7;margin-bottom:3px}
  .composer{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:end}
  .composer textarea{min-height:60px;max-height:240px}
  .chat-controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;
    padding:10px 4px}
  .chat-controls label{font-size:11px;color:var(--ink-dim);
    display:flex;align-items:center;gap:6px}
  .chat-controls input[type=number]{width:80px}
  .chat-controls input[type=range]{width:140px}

  /* RALPH TAB =========================================================== */
  #panel-ralph{display:none}
  #panel-ralph.active{display:grid}
  #panel-ralph{grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);
    grid-template-rows:auto 1fr;gap:14px;padding:14px 18px}
  .ralph-controls{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:8px;align-items:center;
    padding:10px 14px;background:var(--bg-2);border:1px solid var(--line);
    border-radius:var(--radius)}
  .ralph-controls .sep{width:1px;height:22px;background:var(--line);margin:0 4px}
  .ralph-controls label{font-size:11px;color:var(--ink-dim);
    display:flex;align-items:center;gap:6px}
  .ralph-controls input[type=number]{width:70px}
  .ralph-files{display:flex;flex-direction:column;min-height:0}
  .ralph-files .file-tabs{display:flex;gap:2px;padding:0 8px;
    border-bottom:1px solid var(--line)}
  .ralph-files .file-tab{padding:8px 12px;background:transparent;border:none;
    color:var(--ink-dim);border-bottom:2px solid transparent;font-family:var(--mono);
    font-size:12px;border-radius:0}
  .ralph-files .file-tab.active{color:var(--accent-3);border-bottom-color:var(--accent)}
  .ralph-files textarea{flex:1;min-height:0;border:none;border-radius:0;
    background:var(--bg-1);font-size:12.5px}
  .ralph-files .file-actions{padding:8px 14px;border-top:1px solid var(--line);
    display:flex;justify-content:space-between;align-items:center;gap:10px}
  .ralph-status{display:flex;flex-direction:column;gap:14px;min-height:0;overflow:hidden}
  .status-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:12px 14px}
  .stat{padding:10px;background:var(--bg-1);border:1px solid var(--line);
    border-radius:var(--radius-sm)}
  .stat .k{font-size:10px;text-transform:uppercase;color:var(--ink-mute);
    letter-spacing:.6px}
  .stat .v{font-family:var(--mono);font-size:14px;color:var(--ink);margin-top:3px;
    word-break:break-word}
  .ralph-log,.shared-log{flex:1;min-height:0;overflow:auto;padding:10px 14px;
    font-family:var(--mono);font-size:12px;line-height:1.55;background:var(--bg-1);
    border-top:1px solid var(--line)}
  .log-line{display:grid;grid-template-columns:auto auto 1fr;gap:8px;
    padding:1px 0}
  .log-line .ts{color:var(--ink-mute)}
  .log-line .src{color:var(--accent-3)}
  .log-line .msg.warn{color:var(--warn)}
  .log-line .msg.error{color:var(--bad)}

  /* SANDBOX TAB ========================================================= */
  #panel-sandbox{display:none}
  #panel-sandbox.active{display:grid}
  #panel-sandbox{grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);
    grid-template-rows:auto 1fr;gap:14px;padding:14px 18px}
  .sandbox-controls{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:8px;
    align-items:center;padding:10px 14px;background:var(--bg-2);
    border:1px solid var(--line);border-radius:var(--radius)}
  .sandbox-editor{display:flex;flex-direction:column;min-height:0}
  .sandbox-editor textarea{flex:1;min-height:0;border:none;border-radius:0;
    background:var(--bg-1);font-size:13px}
  .sandbox-editor .editor-tabs{display:flex;gap:2px;padding:0 8px;
    border-bottom:1px solid var(--line)}
  .sandbox-editor .editor-tab{padding:8px 12px;background:transparent;border:none;
    color:var(--ink-dim);border-bottom:2px solid transparent;font-family:var(--mono);
    font-size:12px;border-radius:0}
  .sandbox-editor .editor-tab.active{color:var(--accent-3);border-bottom-color:var(--accent)}
  .sandbox-output{display:flex;flex-direction:column;min-height:0;overflow:hidden}
  .out-panel{flex:1;min-height:0;overflow:auto;padding:10px 14px;
    font-family:var(--mono);font-size:12px;background:var(--bg-1);white-space:pre-wrap}
  .out-panel.err{color:var(--bad)}
  .artifact-list{padding:8px 14px;border-top:1px solid var(--line);max-height:160px;
    overflow:auto}
  .artifact-list .artifact{display:flex;justify-content:space-between;align-items:center;
    padding:5px 8px;border-radius:6px;font-size:12px;font-family:var(--mono)}
  .artifact-list .artifact:hover{background:var(--bg-3)}
  .artifact-list .artifact a{color:var(--accent-3)}
  .artifact-list .meta{color:var(--ink-mute);font-size:11px}

  /* ENGINEER TAB ======================================================== */
  #panel-engineer{display:none}
  #panel-engineer.active{display:grid}
  #panel-engineer{grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);
    grid-template-rows:auto 1fr;gap:14px;padding:14px 18px}
  .eng-controls{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:8px;
    align-items:center;padding:10px 14px;background:var(--bg-2);
    border:1px solid var(--line);border-radius:var(--radius)}
  .eng-input{display:flex;flex-direction:column;min-height:0}
  .eng-input textarea{flex:1;min-height:200px;border:none;border-radius:0;
    background:var(--bg-1)}
  .eng-result{display:flex;flex-direction:column;min-height:0;overflow:hidden}
  .eng-meta{padding:10px 14px;font-size:12px;color:var(--ink-dim);
    border-bottom:1px solid var(--line)}
  .eng-meta .run{font-family:var(--mono);background:var(--bg-1);padding:6px 8px;
    border-radius:6px;display:inline-block;margin-top:6px;font-size:11.5px}
  .eng-files{flex:1;min-height:0;overflow:auto;padding:6px 8px}
  .eng-file{display:flex;justify-content:space-between;align-items:center;
    padding:6px 10px;border-radius:6px;font-family:var(--mono);font-size:12px}
  .eng-file:hover{background:var(--bg-3)}
  .eng-file a{color:var(--accent-3)}
  .eng-preview{height:38%;min-height:120px;border-top:1px solid var(--line);
    display:flex;flex-direction:column}
  .eng-preview-header{padding:6px 12px;font-size:11px;color:var(--ink-mute);
    background:var(--bg-2);text-transform:uppercase;letter-spacing:.5px;
    border-bottom:1px solid var(--line)}
  .eng-preview pre{margin:0;flex:1;overflow:auto;padding:10px 14px;
    font-family:var(--mono);font-size:12px;background:var(--bg-1);color:var(--ink)}

  /* AUTOGPT TAB ========================================================= */
  #panel-autogpt{display:none}
  #panel-autogpt.active{display:grid}
  #panel-autogpt{grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);
    grid-template-rows:auto 1fr;gap:14px;padding:14px 18px}
  .ag-controls{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:8px;
    align-items:center;padding:10px 14px;background:var(--bg-2);
    border:1px solid var(--line);border-radius:var(--radius)}
  .ag-controls input[type=text]{flex:1;min-width:200px}
  .ag-state{display:flex;flex-direction:column;min-height:0;overflow:hidden}
  .ag-state .step-info{padding:12px 14px;border-bottom:1px solid var(--line);
    display:grid;grid-template-columns:1fr auto;gap:6px;align-items:center}
  .ag-state .step-info .step-num{font-family:var(--mono);font-size:18px;
    color:var(--accent-3)}
  .ag-state .thought{padding:10px 14px;background:var(--bg-1);
    border-bottom:1px solid var(--line);font-size:13px;line-height:1.55;max-height:30%;
    overflow:auto}
  .ag-state .thought .label{font-size:10px;text-transform:uppercase;
    color:var(--ink-mute);letter-spacing:.6px;margin-bottom:4px}
  .ag-state .plan{padding:10px 14px;border-bottom:1px solid var(--line);max-height:30%;
    overflow:auto}
  .ag-state .plan .label{font-size:10px;text-transform:uppercase;
    color:var(--ink-mute);letter-spacing:.6px;margin-bottom:6px}
  .ag-state .plan ol{margin:0;padding-left:18px;font-size:13px;line-height:1.55}
  .ag-state .observation{flex:1;min-height:0;overflow:auto;padding:10px 14px;
    font-family:var(--mono);font-size:12px;background:var(--bg-1);white-space:pre-wrap}

  /* MANUS TAB =========================================================== */
  #panel-manus{display:none}
  #panel-manus.active{display:grid}
  #panel-manus{grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);
    grid-template-rows:auto 1fr;gap:14px;padding:14px 18px}
  .manus-controls{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:8px;
    align-items:center;padding:10px 14px;background:var(--bg-2);
    border:1px solid var(--line);border-radius:var(--radius)}
  .manus-controls input[type=text]{flex:1;min-width:200px}
  .manus-plan{display:flex;flex-direction:column;min-height:0;overflow:hidden}
  .plan-list{flex:1;min-height:0;overflow:auto;padding:8px}
  .plan-step{padding:10px 12px;border-radius:8px;margin-bottom:6px;
    background:var(--bg-1);border:1px solid var(--line);font-size:13px;
    display:grid;grid-template-columns:auto 1fr;gap:10px;align-items:start}
  .plan-step.current{border-color:var(--accent);
    box-shadow:0 0 0 1px rgba(58,160,255,0.25)}
  .plan-step.done{opacity:.55;text-decoration:line-through}
  .plan-step .num{font-family:var(--mono);color:var(--accent-3);font-size:12px}
  .plan-step .status{font-size:10px;text-transform:uppercase;color:var(--ink-mute)}
  .deliverables{padding:10px 14px;border-top:1px solid var(--line);max-height:35%;
    overflow:auto}
  .deliverables .d-label{font-size:10px;text-transform:uppercase;
    color:var(--ink-mute);letter-spacing:.6px;margin-bottom:6px}
  .deliverables ul{margin:0;padding-left:18px;font-size:12px;font-family:var(--mono)}

  /* Empty states */
  .empty{display:grid;place-items:center;height:100%;color:var(--ink-mute);
    font-size:13px;text-align:center;padding:30px}
  .empty .big{font-size:28px;margin-bottom:8px;color:var(--ink-dim)}

  /* Utility */
  .row{display:flex;gap:8px;align-items:center}
  .grow{flex:1}
  .right{margin-left:auto}
  .small{font-size:11px;color:var(--ink-mute)}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;
    background:var(--bg-3);color:var(--ink-dim);border:1px solid var(--line-2)}
  .pill.ok{background:rgba(74,214,164,0.12);color:var(--good);
    border-color:rgba(74,214,164,0.4)}
  .pill.warn{background:rgba(255,193,92,0.12);color:var(--warn);
    border-color:rgba(255,193,92,0.4)}
  .pill.bad{background:rgba(255,107,138,0.12);color:var(--bad);
    border-color:rgba(255,107,138,0.4)}
  .pill.idle{background:var(--bg-3);color:var(--ink-mute)}
  .pill.run{background:rgba(58,160,255,0.18);color:var(--accent-3);
    border-color:var(--accent-2)}
  .pill.sleep{background:rgba(115,194,255,0.10);color:var(--accent-3);
    border-color:var(--line-2)}
</style>
</head>
<body>
<div class="app">

  <!-- TOP BAR ============================================================ -->
  <div class="topbar">
    <div class="brand">
      <div class="glyph">🐾</div>
      <div>
        <div class="title">CatSDK</div>
        <div class="sub">Chat • Ralph • Sandbox • Engineer • AutoGPT • Manus — on LM Studio</div>
      </div>
    </div>
    <div class="topbar-right">
      <span id="lm-badge" class="badge">LM Studio: …</span>
      <select id="global-model" title="Loaded model"></select>
      <button id="refresh-models" class="ghost" title="Refresh models">↻</button>
    </div>
  </div>

  <!-- TAB BAR ============================================================ -->
  <div class="tabs" id="tabs">
    <button class="tab active" data-tab="chat">💬 Chat</button>
    <button class="tab" data-tab="ralph">🔁 Ralph</button>
    <button class="tab" data-tab="sandbox">🧪 Sandbox</button>
    <button class="tab" data-tab="engineer">🏗️ Engineer</button>
    <button class="tab" data-tab="autogpt">🤖 AutoGPT</button>
    <button class="tab" data-tab="manus">📋 Manus</button>
  </div>

  <!-- MAIN =============================================================== -->
  <div class="main">

    <!-- ============================ CHAT ============================ -->
    <div class="panel active" id="panel-chat">
      <div id="chat-stream" class="chat-stream card">
        <div class="empty"><div><div class="big">💬</div>
          Pick a model and say hi. CatSDK keeps chat history in this tab.
        </div></div>
      </div>
      <div>
        <div class="chat-controls">
          <label>Temp <input id="chat-temp" type="range" min="0" max="1.5" step="0.05" value="0.65">
            <span id="chat-temp-val">0.65</span></label>
          <label>Max tokens <input id="chat-max" type="number" min="32" max="8192" value="768"></label>
          <label><input id="chat-history" type="checkbox" checked> Send history</label>
          <span class="right small" id="chat-meta"></span>
          <button id="chat-clear" class="ghost">Clear</button>
        </div>
        <div class="composer">
          <textarea id="chat-input" placeholder="Message CatSDK… (Enter sends, Shift+Enter newline)"></textarea>
          <button id="chat-send" class="primary">Send</button>
        </div>
      </div>
    </div>

    <!-- ============================ RALPH ============================ -->
    <div class="panel" id="panel-ralph">
      <div class="ralph-controls">
        <span class="pill" id="ralph-state-pill">STOPPED</span>
        <span class="sep"></span>
        <label>Calls/hr <input id="ralph-cph" type="number" min="1" max="2000" value="100"></label>
        <label>Temp <input id="ralph-temp" type="number" min="0" max="1.5" step="0.05" value="0.6"></label>
        <label>Max tok <input id="ralph-max" type="number" min="64" max="8192" value="1024"></label>
        <span class="sep"></span>
        <button id="ralph-start" class="primary">▶ Start</button>
        <button id="ralph-stop" class="danger">■ Stop</button>
        <button id="ralph-sleep">😴 Sleep</button>
        <button id="ralph-wake">⏰ Wake</button>
        <span class="sep"></span>
        <button id="ralph-reset-session" class="ghost">Reset Session</button>
        <button id="ralph-reset-circuit" class="ghost">Reset Circuit</button>
      </div>

      <!-- file editor -->
      <div class="card ralph-files">
        <div class="card-header">
          <h3>.ralph/ files</h3>
          <span class="small" id="ralph-file-meta"></span>
        </div>
        <div class="file-tabs" id="ralph-file-tabs"></div>
        <textarea id="ralph-file-content" spellcheck="false"></textarea>
        <div class="file-actions">
          <span class="small" id="ralph-file-status"></span>
          <button id="ralph-file-save" class="primary">Save</button>
        </div>
      </div>

      <!-- status + log -->
      <div class="card ralph-status">
        <div class="card-header">
          <h3>Status</h3>
          <span class="small" id="ralph-exit-reason"></span>
        </div>
        <div class="status-grid" id="ralph-stats"></div>
        <div class="card-header">
          <h3>Log stream</h3>
          <button id="ralph-log-clear" class="ghost">Clear view</button>
        </div>
        <div class="ralph-log" id="ralph-log"></div>
      </div>
    </div>

    <!-- ============================ SANDBOX ============================ -->
    <div class="panel" id="panel-sandbox">
      <div class="sandbox-controls">
        <span class="small">Code interpreter — Python &amp; shell, isolated workspace</span>
        <span class="right"></span>
        <button id="sb-run-py" class="primary">▶ Run Python</button>
        <button id="sb-run-sh">▶ Run Shell</button>
        <button id="sb-reset" class="danger">Reset workspace</button>
      </div>

      <div class="card sandbox-editor">
        <div class="editor-tabs" id="sb-tabs">
          <button class="editor-tab active" data-mode="python">python</button>
          <button class="editor-tab" data-mode="shell">shell</button>
        </div>
        <textarea id="sb-code" spellcheck="false"
          placeholder="# Try:&#10;import math&#10;print('hello cat')&#10;print(math.pi)"></textarea>
      </div>

      <div class="card sandbox-output">
        <div class="card-header">
          <h3>Output</h3>
          <span class="small" id="sb-meta"></span>
        </div>
        <div class="out-panel" id="sb-stdout"></div>
        <div class="out-panel err" id="sb-stderr"></div>
        <div class="card-header">
          <h3>Workspace files</h3>
          <button id="sb-refresh-files" class="ghost">↻</button>
        </div>
        <div class="artifact-list" id="sb-files"></div>
      </div>
    </div>

    <!-- ============================ ENGINEER ============================ -->
    <div class="panel" id="panel-engineer">
      <div class="eng-controls">
        <span class="small">gpt-engineer style: describe a project → get a multi-file codebase</span>
        <span class="right"></span>
        <label>Temp <input id="eng-temp" type="number" min="0" max="1.5" step="0.05" value="0.4" style="width:70px"></label>
        <label>Max tok <input id="eng-max" type="number" min="512" max="16384" value="4096" style="width:90px"></label>
        <button id="eng-generate" class="primary">⚡ Generate codebase</button>
      </div>

      <div class="card eng-input">
        <div class="card-header"><h3>Project description</h3>
          <span class="small">Be specific: language, files, behaviour, deps</span></div>
        <textarea id="eng-desc" placeholder="A small Flask app with two routes: / returns hello, /time returns the current UTC time as JSON. Include a README and a tiny test."></textarea>
      </div>

      <div class="card eng-result">
        <div class="card-header">
          <h3>Generated codebase</h3>
          <button id="eng-refresh" class="ghost">↻</button>
        </div>
        <div class="eng-meta" id="eng-meta">
          <div class="small">No project yet. Describe what you want and click Generate.</div>
        </div>
        <div class="eng-files" id="eng-files"></div>
        <div class="eng-preview">
          <div class="eng-preview-header" id="eng-preview-name">preview</div>
          <pre id="eng-preview"></pre>
        </div>
      </div>
    </div>

    <!-- ============================ AUTOGPT ============================ -->
    <div class="panel" id="panel-autogpt">
      <div class="ag-controls">
        <span class="pill" id="ag-state-pill">STOPPED</span>
        <input id="ag-goal" type="text" placeholder="Goal for the agent — e.g. 'Build a CSV → JSON converter and verify it on a sample.'">
        <label>Steps <input id="ag-steps" type="number" min="1" max="200" value="20" style="width:70px"></label>
        <label>Temp <input id="ag-temp" type="number" min="0" max="1.5" step="0.05" value="0.5" style="width:70px"></label>
        <label>Max tok <input id="ag-max" type="number" min="128" max="8192" value="1024" style="width:90px"></label>
        <button id="ag-start" class="primary">▶ Start</button>
        <button id="ag-stop" class="danger">■ Stop</button>
        <button id="ag-sleep">😴</button>
        <button id="ag-wake">⏰</button>
      </div>

      <div class="card ag-state">
        <div class="card-header"><h3>Agent state</h3>
          <span class="small" id="ag-finish"></span></div>
        <div class="step-info">
          <div class="small">Step</div>
          <div class="step-num" id="ag-step">0/0</div>
        </div>
        <div class="thought">
          <div class="label">Latest thought</div>
          <div id="ag-thought">—</div>
        </div>
        <div class="plan">
          <div class="label">Plan</div>
          <ol id="ag-plan"></ol>
        </div>
        <div class="card-header"><h3>Last command</h3></div>
        <div class="thought">
          <div id="ag-command" style="font-family:var(--mono)">—</div>
        </div>
      </div>

      <div class="card ag-state">
        <div class="card-header"><h3>Last observation</h3>
          <span class="small" id="ag-notes-count"></span></div>
        <div class="observation" id="ag-observation"></div>
        <div class="card-header"><h3>Workspace</h3>
          <button id="ag-refresh-files" class="ghost">↻</button></div>
        <div class="artifact-list" id="ag-files"></div>
      </div>
    </div>

    <!-- ============================ MANUS ============================ -->
    <div class="panel" id="panel-manus">
      <div class="manus-controls">
        <span class="pill" id="mn-state-pill">STOPPED</span>
        <input id="mn-goal" type="text" placeholder="Long-horizon goal — e.g. 'Research, plan and produce a 1-page brief on the history of ROM hacking.'">
        <label>Steps/task <input id="mn-substeps" type="number" min="1" max="20" value="6" style="width:70px"></label>
        <label>Temp <input id="mn-temp" type="number" min="0" max="1.5" step="0.05" value="0.5" style="width:70px"></label>
        <label>Max tok <input id="mn-max" type="number" min="128" max="8192" value="1024" style="width:90px"></label>
        <button id="mn-start" class="primary">▶ Start</button>
        <button id="mn-stop" class="danger">■ Stop</button>
        <button id="mn-sleep">😴</button>
        <button id="mn-wake">⏰</button>
      </div>

      <div class="card manus-plan">
        <div class="card-header"><h3>Plan</h3>
          <span class="small" id="mn-summary"></span></div>
        <div class="plan-list" id="mn-plan"></div>
        <div class="deliverables">
          <div class="d-label">Deliverables</div>
          <ul id="mn-deliverables"></ul>
        </div>
      </div>

      <div class="card ag-state">
        <div class="card-header"><h3>Latest action</h3></div>
        <div class="thought">
          <div class="label">Thought</div>
          <div id="mn-thought">—</div>
        </div>
        <div class="thought">
          <div class="label">Command</div>
          <div id="mn-command" style="font-family:var(--mono)">—</div>
        </div>
        <div class="card-header"><h3>Observation</h3></div>
        <div class="observation" id="mn-observation"></div>
        <div class="card-header"><h3>Workspace</h3></div>
        <div class="artifact-list" id="mn-files"></div>
      </div>
    </div>

  </div>
</div>

<script>
(() => {
  // ---- helpers --------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, txt) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt !== undefined) n.textContent = txt;
    return n;
  };
  const fmt = (n) => Number(n).toLocaleString();
  const fmtBytes = (n) => {
    if (n < 1024) return n + " B";
    if (n < 1024*1024) return (n/1024).toFixed(1) + " KB";
    return (n/1024/1024).toFixed(2) + " MB";
  };
  const pillClassFor = (state) => {
    if (state === "RUNNING") return "pill run";
    if (state === "SLEEPING") return "pill sleep";
    if (state === "PLANNING") return "pill warn";
    if (state === "STOPPED") return "pill idle";
    return "pill";
  };
  async function api(path, opts={}) {
    const init = Object.assign({headers:{"Content-Type":"application/json"}}, opts);
    if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
    const r = await fetch(path, init);
    const ct = r.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.statusText);
      return j;
    }
    if (!r.ok) throw new Error(r.statusText);
    return r;
  }

  // ---- tabs -----------------------------------------------------------
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".panel");
  tabs.forEach(t => t.addEventListener("click", () => {
    tabs.forEach(x => x.classList.remove("active"));
    panels.forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    $("panel-" + t.dataset.tab).classList.add("active");
  }));

  // ---- LM Studio status + model picker --------------------------------
  async function refreshLMStudio() {
    const badge = $("lm-badge");
    try {
      const ping = await api("/api/lmstudio");
      if (ping.ok) {
        badge.className = "badge ok";
        badge.textContent = `LM Studio: ${ping.models_loaded} model(s)`;
      } else {
        badge.className = "badge bad";
        badge.textContent = "LM Studio: offline";
      }
    } catch (e) {
      badge.className = "badge bad";
      badge.textContent = "LM Studio: offline";
    }
    try {
      const data = await api("/api/models");
      const sel = $("global-model");
      const prev = sel.value;
      sel.innerHTML = "";
      (data.models || []).forEach(m => {
        const o = document.createElement("option");
        o.value = m; o.textContent = m;
        sel.appendChild(o);
      });
      if (prev && data.models.includes(prev)) sel.value = prev;
    } catch (e) {/* ignore */}
  }
  $("refresh-models").addEventListener("click", refreshLMStudio);

  // =====================================================================
  // CHAT
  // =====================================================================
  const chatStream = $("chat-stream");
  const chatHistory = []; // [{role, content}]
  function renderChat() {
    chatStream.innerHTML = "";
    if (!chatHistory.length) {
      const e = el("div","empty");
      e.innerHTML = '<div><div class="big">💬</div>Pick a model and say hi.</div>';
      chatStream.appendChild(e);
      return;
    }
    chatHistory.forEach(m => {
      const d = el("div","chat-msg " + m.role);
      const r = el("div","role", m.role === "user" ? "you" : (m.role==="assistant"?"catsdk":"system"));
      const c = el("div","content", m.content);
      d.appendChild(r); d.appendChild(c);
      chatStream.appendChild(d);
    });
    chatStream.scrollTop = chatStream.scrollHeight;
  }
  $("chat-temp").addEventListener("input", e => $("chat-temp-val").textContent = e.target.value);
  $("chat-clear").addEventListener("click", () => { chatHistory.length=0; renderChat(); });
  async function sendChat() {
    const input = $("chat-input"); const text = input.value.trim();
    if (!text) return;
    const model = $("global-model").value;
    if (!model) { alert("Load a model in LM Studio first."); return; }
    chatHistory.push({role:"user", content:text}); renderChat();
    input.value = ""; $("chat-send").disabled = true;
    const sendHistory = $("chat-history").checked;
    try {
      const j = await api("/api/chat", {method:"POST", body:{
        model, message:text,
        history: sendHistory ? chatHistory.slice(0,-1) : [],
        temperature: parseFloat($("chat-temp").value),
        max_tokens: parseInt($("chat-max").value, 10),
      }});
      chatHistory.push({role:"assistant", content:j.reply});
      const u = j.usage || {};
      $("chat-meta").textContent = `model: ${j.model} • prompt ${u.prompt_tokens||0} / completion ${u.completion_tokens||0}`;
    } catch (e) {
      chatHistory.push({role:"system", content: "⚠ " + e.message});
    }
    renderChat();
    $("chat-send").disabled = false; input.focus();
  }
  $("chat-send").addEventListener("click", sendChat);
  $("chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });

  // =====================================================================
  // RALPH
  // =====================================================================
  const ralphFileTabs = $("ralph-file-tabs");
  const ralphContent = $("ralph-file-content");
  let ralphCurrentFile = "PROMPT.md";
  let ralphLogCursor = 0;
  let ralphFileDirty = false;

  async function loadRalphFiles() {
    const j = await api("/api/ralph/files");
    ralphFileTabs.innerHTML = "";
    j.files.forEach(name => {
      const b = el("button","file-tab" + (name===ralphCurrentFile?" active":""), name);
      b.addEventListener("click", () => switchRalphFile(name));
      ralphFileTabs.appendChild(b);
    });
    await switchRalphFile(ralphCurrentFile);
  }
  async function switchRalphFile(name) {
    if (ralphFileDirty && !confirm("Unsaved changes — discard?")) return;
    ralphCurrentFile = name;
    document.querySelectorAll("#ralph-file-tabs .file-tab").forEach(b =>
      b.classList.toggle("active", b.textContent === name));
    const j = await api("/api/ralph/file?name=" + encodeURIComponent(name));
    ralphContent.value = j.content;
    ralphFileDirty = false;
    $("ralph-file-status").textContent = "loaded " + name;
  }
  ralphContent.addEventListener("input", () => {
    ralphFileDirty = true;
    $("ralph-file-status").textContent = "● unsaved";
  });
  $("ralph-file-save").addEventListener("click", async () => {
    try {
      await api("/api/ralph/file", {method:"POST", body:{
        name: ralphCurrentFile, content: ralphContent.value
      }});
      ralphFileDirty = false;
      $("ralph-file-status").textContent = "saved " + ralphCurrentFile;
    } catch (e) { $("ralph-file-status").textContent = "⚠ " + e.message; }
  });

  function renderRalphStatus(s) {
    const pill = $("ralph-state-pill");
    pill.className = pillClassFor(s.state);
    pill.textContent = s.state;
    $("ralph-exit-reason").textContent = s.exit_reason ? "exit: " + s.exit_reason : "";
    const cb = s.circuit || {}; const rl = s.rate_limit || {};
    const sec = Math.max(0, s.session_age_seconds || 0);
    const sessionAge = sec >= 3600 ? `${(sec/3600).toFixed(1)}h` :
                       sec >= 60 ? `${Math.floor(sec/60)}m ${sec%60}s` : `${sec}s`;
    const stats = [
      ["Loop count", fmt(s.loop_count)],
      ["Idle loops", `${s.idle_loops} / ${s.sleep_after_idle_loops}`],
      ["Calls used", `${rl.calls_used||0} / ${rl.calls_limit||0}`],
      ["Rate reset", (rl.reset_seconds||0) + "s"],
      ["Circuit", cb.state || "—"],
      ["No-progress", `${cb.no_progress_count||0}`],
      ["Same errors", `${cb.same_error_count||0}`],
      ["Cooldown left", (cb.cooldown_seconds_left||0) + "s"],
      ["Session", s.session_id || "—"],
      ["Session age", sessionAge],
      ["Open tasks", String(s.fix_plan_open||0)],
      ["Last progress", s.last_progress || "—"],
    ];
    const grid = $("ralph-stats"); grid.innerHTML = "";
    stats.forEach(([k,v]) => {
      const c = el("div","stat");
      c.appendChild(el("div","k",k)); c.appendChild(el("div","v",v));
      grid.appendChild(c);
    });
  }

  async function pollRalph() {
    try {
      const s = await api("/api/ralph/status"); renderRalphStatus(s);
    } catch (e) { /* ignore */ }
    try {
      const j = await api("/api/logs?since=" + ralphLogCursor + "&source=ralph");
      const logBox = $("ralph-log");
      (j.entries||[]).forEach(en => {
        ralphLogCursor = Math.max(ralphLogCursor, en.id);
        const line = el("div","log-line");
        line.appendChild(el("span","ts", en.ts.slice(11,19)));
        line.appendChild(el("span","src", en.source));
        line.appendChild(el("span","msg " + en.level, en.message));
        logBox.appendChild(line);
      });
      if (j.entries && j.entries.length) logBox.scrollTop = logBox.scrollHeight;
    } catch (e) { /* ignore */ }
  }
  $("ralph-log-clear").addEventListener("click", () => { $("ralph-log").innerHTML = ""; });

  $("ralph-start").addEventListener("click", async () => {
    const model = $("global-model").value;
    if (!model) { alert("Pick a model"); return; }
    try {
      await api("/api/ralph/start", {method:"POST", body:{
        model,
        temperature: parseFloat($("ralph-temp").value),
        max_tokens: parseInt($("ralph-max").value, 10),
        max_calls_per_hour: parseInt($("ralph-cph").value, 10),
      }});
    } catch (e) { alert(e.message); }
  });
  $("ralph-stop").addEventListener("click", () => api("/api/ralph/stop", {method:"POST"}));
  $("ralph-sleep").addEventListener("click", () => api("/api/ralph/sleep", {method:"POST"}));
  $("ralph-wake").addEventListener("click", () => api("/api/ralph/wake", {method:"POST"}));
  $("ralph-reset-session").addEventListener("click", () => api("/api/ralph/session/reset", {method:"POST"}));
  $("ralph-reset-circuit").addEventListener("click", () => api("/api/ralph/circuit/reset", {method:"POST"}));

  // =====================================================================
  // SANDBOX
  // =====================================================================
  let sbMode = "python";
  document.querySelectorAll("#sb-tabs .editor-tab").forEach(b => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#sb-tabs .editor-tab").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      sbMode = b.dataset.mode;
      $("sb-code").placeholder = sbMode === "python"
        ? "# Try:\nimport math\nprint(math.pi)"
        : "# Try:\nls -la\nuname -a";
    });
  });
  async function runSandbox(kind) {
    const code = $("sb-code").value;
    if (!code.trim()) return;
    const url = kind === "python" ? "/api/sandbox/python" : "/api/sandbox/shell";
    const body = kind === "python" ? {code} : {command: code};
    $("sb-stdout").textContent = "Running…"; $("sb-stderr").textContent = "";
    try {
      const j = await api(url, {method:"POST", body});
      $("sb-stdout").textContent = j.stdout || "(no stdout)";
      $("sb-stderr").textContent = j.stderr || "";
      $("sb-meta").textContent = `exit ${j.exit_code} • ${j.duration_ms} ms` +
        (j.artifacts && j.artifacts.length ? ` • ${j.artifacts.length} new file(s)` : "");
      refreshSandboxFiles();
    } catch (e) {
      $("sb-stderr").textContent = "⚠ " + e.message;
    }
  }
  $("sb-run-py").addEventListener("click", () => runSandbox("python"));
  $("sb-run-sh").addEventListener("click", () => runSandbox("shell"));
  $("sb-reset").addEventListener("click", async () => {
    if (!confirm("Wipe sandbox workspace?")) return;
    await api("/api/sandbox/reset", {method:"POST"});
    refreshSandboxFiles();
  });
  async function refreshSandboxFiles() {
    try {
      const j = await api("/api/sandbox/files");
      const box = $("sb-files"); box.innerHTML = "";
      if (!j.files.length) { box.innerHTML = '<div class="small" style="padding:8px">(no files)</div>'; return; }
      j.files.forEach(f => {
        const row = el("div","artifact");
        const a = el("a","", f.name); a.href = "/api/sandbox/file?name=" + encodeURIComponent(f.name);
        a.target = "_blank";
        row.appendChild(a);
        row.appendChild(el("span","meta", fmtBytes(f.size)));
        box.appendChild(row);
      });
    } catch (e) {/* ignore */}
  }
  $("sb-refresh-files").addEventListener("click", refreshSandboxFiles);

  // =====================================================================
  // ENGINEER
  // =====================================================================
  $("eng-generate").addEventListener("click", async () => {
    const model = $("global-model").value;
    if (!model) { alert("Pick a model"); return; }
    const desc = $("eng-desc").value;
    if (!desc.trim()) { alert("Describe the project"); return; }
    $("eng-meta").innerHTML = '<div class="small">Generating… this can take a while on large models.</div>';
    $("eng-generate").disabled = true;
    try {
      const j = await api("/api/engineer/generate", {method:"POST", body:{
        description: desc, model,
        temperature: parseFloat($("eng-temp").value),
        max_tokens: parseInt($("eng-max").value, 10),
      }});
      if (!j.ok) throw new Error(j.error || "generation failed");
      await refreshEngineer();
    } catch (e) {
      $("eng-meta").innerHTML = '<div class="small" style="color:var(--bad)">⚠ ' + e.message + '</div>';
    }
    $("eng-generate").disabled = false;
  });
  async function refreshEngineer() {
    const j = await api("/api/engineer/status");
    const meta = $("eng-meta");
    if (!j.files.length) {
      meta.innerHTML = '<div class="small">No project yet. Describe what you want and click Generate.</div>';
    } else {
      meta.innerHTML = '<div>' + (j.notes || '') + '</div>' +
        (j.run ? '<div class="run">$ ' + j.run + '</div>' : '');
    }
    const box = $("eng-files"); box.innerHTML = "";
    j.files.forEach(f => {
      const row = el("div","eng-file");
      const a = el("a","", f.name); a.href = "#";
      a.addEventListener("click", async (ev) => {
        ev.preventDefault();
        const r = await fetch("/api/engineer/file?name=" + encodeURIComponent(f.name));
        const text = await r.text();
        $("eng-preview-name").textContent = f.name;
        $("eng-preview").textContent = text;
      });
      row.appendChild(a);
      row.appendChild(el("span","small", fmtBytes(f.size)));
      box.appendChild(row);
    });
  }
  $("eng-refresh").addEventListener("click", refreshEngineer);

  // =====================================================================
  // AUTOGPT
  // =====================================================================
  function renderAutoGPT(s) {
    const pill = $("ag-state-pill");
    pill.className = pillClassFor(s.state); pill.textContent = s.state;
    $("ag-step").textContent = `${s.step}/${s.max_steps}`;
    $("ag-thought").textContent = s.last_thought || "—";
    $("ag-command").textContent = s.last_command || "—";
    $("ag-observation").textContent = s.last_observation || "(no observation yet)";
    $("ag-finish").textContent = s.finish_summary ? ("✓ " + s.finish_summary) : "";
    $("ag-notes-count").textContent = (s.notes||[]).length + " note(s)";
    const planBox = $("ag-plan"); planBox.innerHTML = "";
    (s.last_plan||[]).forEach(p => planBox.appendChild(el("li","",p)));
  }
  async function pollAutoGPT() {
    try { renderAutoGPT(await api("/api/autogpt/status")); } catch (e) {}
  }
  async function refreshAutoGPTFiles() {
    try {
      const j = await api("/api/autogpt/files");
      const box = $("ag-files"); box.innerHTML = "";
      if (!j.files.length) { box.innerHTML='<div class="small" style="padding:8px">(no files)</div>'; return; }
      j.files.forEach(f => {
        const row = el("div","artifact");
        const a = el("a","",f.name); a.href = "/api/autogpt/file?name=" + encodeURIComponent(f.name);
        a.target = "_blank";
        row.appendChild(a); row.appendChild(el("span","meta", fmtBytes(f.size)));
        box.appendChild(row);
      });
    } catch (e) {}
  }
  $("ag-refresh-files").addEventListener("click", refreshAutoGPTFiles);
  $("ag-start").addEventListener("click", async () => {
    const model = $("global-model").value;
    if (!model) { alert("Pick a model"); return; }
    try {
      await api("/api/autogpt/start", {method:"POST", body:{
        goal: $("ag-goal").value, model,
        temperature: parseFloat($("ag-temp").value),
        max_tokens: parseInt($("ag-max").value, 10),
        max_steps: parseInt($("ag-steps").value, 10),
      }});
    } catch (e) { alert(e.message); }
  });
  $("ag-stop").addEventListener("click", () => api("/api/autogpt/stop", {method:"POST"}));
  $("ag-sleep").addEventListener("click", () => api("/api/autogpt/sleep", {method:"POST"}));
  $("ag-wake").addEventListener("click", () => api("/api/autogpt/wake", {method:"POST"}));

  // =====================================================================
  // MANUS
  // =====================================================================
  function renderManus(s) {
    const pill = $("mn-state-pill");
    pill.className = pillClassFor(s.state); pill.textContent = s.state;
    $("mn-summary").textContent = s.summary || "";
    const planBox = $("mn-plan"); planBox.innerHTML = "";
    (s.plan||[]).forEach((step, i) => {
      const cls = "plan-step" + (i === s.current_index && s.state !== "STOPPED" ? " current" : "")
        + (step.status === "done" ? " done" : "");
      const row = el("div", cls);
      row.appendChild(el("div","num", "#" + (i+1)));
      const right = el("div");
      right.appendChild(el("div","",step.task));
      right.appendChild(el("div","status", step.status));
      row.appendChild(right);
      planBox.appendChild(row);
    });
    const dlv = $("mn-deliverables"); dlv.innerHTML = "";
    (s.deliverables||[]).forEach(d => dlv.appendChild(el("li","",d)));
    $("mn-thought").textContent = s.last_thought || "—";
    $("mn-command").textContent = s.last_command || "—";
    $("mn-observation").textContent = s.last_observation || "(no observation yet)";
    const files = $("mn-files"); files.innerHTML = "";
    (s.files||[]).forEach(f => {
      const row = el("div","artifact");
      const a = el("a","",f.name); a.href = "/api/manus/file?name=" + encodeURIComponent(f.name);
      a.target = "_blank";
      row.appendChild(a); row.appendChild(el("span","meta", fmtBytes(f.size)));
      files.appendChild(row);
    });
  }
  async function pollManus() {
    try { renderManus(await api("/api/manus/status")); } catch (e) {}
  }
  $("mn-start").addEventListener("click", async () => {
    const model = $("global-model").value;
    if (!model) { alert("Pick a model"); return; }
    try {
      await api("/api/manus/start", {method:"POST", body:{
        goal: $("mn-goal").value, model,
        temperature: parseFloat($("mn-temp").value),
        max_tokens: parseInt($("mn-max").value, 10),
        max_steps_per_task: parseInt($("mn-substeps").value, 10),
      }});
    } catch (e) { alert(e.message); }
  });
  $("mn-stop").addEventListener("click", () => api("/api/manus/stop", {method:"POST"}));
  $("mn-sleep").addEventListener("click", () => api("/api/manus/sleep", {method:"POST"}));
  $("mn-wake").addEventListener("click", () => api("/api/manus/wake", {method:"POST"}));

  // ---- boot -----------------------------------------------------------
  refreshLMStudio(); loadRalphFiles(); refreshSandboxFiles();
  refreshEngineer().catch(()=>{}); refreshAutoGPTFiles();

  setInterval(refreshLMStudio, 15000);
  setInterval(pollRalph, 1500);
  setInterval(pollAutoGPT, 1500);
  setInterval(pollManus, 1500);
})();
</script>
</body>
</html>"""


# ===========================================================================
#  HTTP handler
# ===========================================================================


def create_handler(config: AppConfig):
    client = LMStudioClient(
        base_url=config.lm_studio_base_url,
        timeout=config.request_timeout_seconds,
        api_key=config.lm_studio_api_key,
    )
    log = RingLog(2000)

    # Ralph foundation
    ralph_files = RalphFiles(config.project_root)
    ralph_loop = RalphLoop(config, client, ralph_files, log)

    # Sandbox + agents (each gets its own workspace)
    sandbox = Sandbox(feature_dir(config, "sandbox"))
    autogpt_workspace = feature_dir(config, "autogpt")
    autogpt_sandbox = Sandbox(autogpt_workspace)
    autogpt = AutoGPTAgent(config, client, autogpt_workspace, autogpt_sandbox, log)

    engineer_workspace = feature_dir(config, "engineer")
    engineer = ProjectEngineer(client, engineer_workspace, log)

    manus_workspace = feature_dir(config, "manus")
    manus_sandbox = Sandbox(manus_workspace)
    manus = ManusAgent(config, client, manus_workspace, manus_sandbox, log)

    class CatSDKHandler(BaseHTTPRequestHandler):
        server_version = f"{APP_NAME}/{APP_VERSION}"

        # ---- low-level helpers -------------------------------------------
        def _send_json(self, status: int, payload: dict | list) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_html(self, html: str) -> None:
            raw = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_bytes(self, data: bytes, mime: str,
                        filename: str | None = None, inline: bool = True) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            disp = "inline" if inline else "attachment"
            if filename:
                self.send_header("Content-Disposition", f"{disp}; filename=\"{filename}\"")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                parsed = json.loads(body.decode("utf-8"))
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}

        def _query(self) -> dict[str, str]:
            parsed = urllib_parse.urlparse(self.path)
            return {k: v[0] for k, v in urllib_parse.parse_qs(parsed.query).items()}

        def _route(self) -> str:
            return urllib_parse.urlparse(self.path).path

        def log_message(self, fmt: str, *args) -> None:
            return

        # ---- GET ----------------------------------------------------------
        def do_GET(self) -> None:
            route = self._route()

            if route in {"/", "/index.html"}:
                self._send_html(HTML)
                return

            if route == "/health":
                self._send_json(HTTPStatus.OK, {"status": "ok", "app": APP_NAME, "version": APP_VERSION})
                return

            if route == "/api/models":
                try:
                    models = client.list_models()
                    self._send_json(HTTPStatus.OK, {"models": models, "apple_silicon": is_apple_silicon()})
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
                return

            if route == "/api/lmstudio":
                self._send_json(HTTPStatus.OK, client.ping())
                return

            if route == "/api/logs":
                q = self._query()
                try:
                    since = int(q.get("since", "0"))
                except ValueError:
                    since = 0
                source = q.get("source", "")
                entries = log.since(since)
                if source:
                    entries = [e for e in entries if e["source"] == source]
                self._send_json(HTTPStatus.OK, {"entries": entries})
                return

            # Ralph
            if route == "/api/ralph/status":
                self._send_json(HTTPStatus.OK, ralph_loop.snapshot())
                return
            if route == "/api/ralph/files":
                self._send_json(HTTPStatus.OK, {"files": list(RALPH_EDITABLE_FILES)})
                return
            if route == "/api/ralph/file":
                name = self._query().get("name", "")
                if name not in RALPH_EDITABLE_FILES:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "unknown file"})
                    return
                self._send_json(HTTPStatus.OK, {"name": name, "content": ralph_files.read(name)})
                return

            # Sandbox
            if route == "/api/sandbox/files":
                self._send_json(HTTPStatus.OK, {"files": sandbox.list_artifacts()})
                return
            if route == "/api/sandbox/file":
                name = self._query().get("name", "")
                try:
                    data, mime = sandbox.read_artifact(name)
                except FileNotFoundError:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                inline = self._query().get("inline", "1") == "1"
                self._send_bytes(data, mime, filename=Path(name).name, inline=inline)
                return

            # Engineer
            if route == "/api/engineer/status":
                self._send_json(HTTPStatus.OK, engineer.snapshot())
                return
            if route == "/api/engineer/file":
                name = self._query().get("name", "")
                try:
                    target = safe_join(engineer_workspace, name)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                if not target.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                self._send_bytes(target.read_bytes(),
                                 "text/plain; charset=utf-8",
                                 filename=Path(name).name, inline=True)
                return

            # AutoGPT
            if route == "/api/autogpt/status":
                self._send_json(HTTPStatus.OK, autogpt.snapshot())
                return
            if route == "/api/autogpt/files":
                files = []
                for p in sorted(autogpt_workspace.rglob("*")):
                    if p.is_file():
                        files.append({
                            "name": p.relative_to(autogpt_workspace).as_posix(),
                            "size": p.stat().st_size,
                        })
                self._send_json(HTTPStatus.OK, {"files": files})
                return
            if route == "/api/autogpt/file":
                name = self._query().get("name", "")
                try:
                    target = safe_join(autogpt_workspace, name)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                if not target.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                self._send_bytes(target.read_bytes(),
                                 "text/plain; charset=utf-8",
                                 filename=Path(name).name, inline=True)
                return

            # Manus
            if route == "/api/manus/status":
                self._send_json(HTTPStatus.OK, manus.snapshot())
                return
            if route == "/api/manus/file":
                name = self._query().get("name", "")
                try:
                    target = safe_join(manus_workspace, name)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                if not target.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                self._send_bytes(target.read_bytes(),
                                 "text/plain; charset=utf-8",
                                 filename=Path(name).name, inline=True)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        # ---- POST ---------------------------------------------------------
        def do_POST(self) -> None:
            route = self._route()
            payload = self._read_json()

            # ---- Chat (preserved verbatim) -------------------------------
            if route == "/api/chat":
                model = str(payload.get("model", "")).strip()
                message = str(payload.get("message", "")).strip()
                if not model:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Model is required."})
                    return
                if not message:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Message cannot be empty."})
                    return
                history = payload.get("history", [])
                if not isinstance(history, list):
                    history = []
                try:
                    temperature = float(payload.get("temperature", config.default_temperature))
                except (TypeError, ValueError):
                    temperature = config.default_temperature
                try:
                    max_tokens = int(payload.get("max_tokens", config.default_max_tokens))
                except (TypeError, ValueError):
                    max_tokens = config.default_max_tokens
                try:
                    response = client.chat(
                        model=model,
                        messages=build_chat_messages(history, message),
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    self._send_json(HTTPStatus.OK, {
                        "reply": response["content"],
                        "model": response["model"],
                        "usage": response["usage"],
                    })
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
                return

            # ---- Ralph ---------------------------------------------------
            if route == "/api/ralph/file":
                name = str(payload.get("name", ""))
                content = payload.get("content", "")
                if name not in RALPH_EDITABLE_FILES:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "unknown file"})
                    return
                if not isinstance(content, str):
                    content = str(content)
                ralph_files.write(name, content)
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            if route == "/api/ralph/start":
                model = str(payload.get("model", "")).strip()
                try:
                    temperature = float(payload.get("temperature", config.default_temperature))
                except (TypeError, ValueError):
                    temperature = config.default_temperature
                try:
                    max_tokens = int(payload.get("max_tokens", config.default_max_tokens))
                except (TypeError, ValueError):
                    max_tokens = config.default_max_tokens
                try:
                    cph = int(payload.get("max_calls_per_hour", config.max_calls_per_hour))
                except (TypeError, ValueError):
                    cph = config.max_calls_per_hour
                self._send_json(HTTPStatus.OK, ralph_loop.start(
                    model=model, temperature=temperature,
                    max_tokens=max_tokens, max_calls_per_hour=cph,
                ))
                return
            if route == "/api/ralph/stop":
                self._send_json(HTTPStatus.OK, ralph_loop.stop()); return
            if route == "/api/ralph/sleep":
                self._send_json(HTTPStatus.OK, ralph_loop.sleep_now()); return
            if route == "/api/ralph/wake":
                self._send_json(HTTPStatus.OK, ralph_loop.wake_now()); return
            if route == "/api/ralph/session/reset":
                ralph_loop.reset_session()
                self._send_json(HTTPStatus.OK, {"ok": True}); return
            if route == "/api/ralph/circuit/reset":
                self._send_json(HTTPStatus.OK, ralph_loop.reset_circuit()); return

            # ---- Sandbox -------------------------------------------------
            if route == "/api/sandbox/python":
                code = str(payload.get("code", ""))
                if not code:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "code is required"}); return
                r = sandbox.run_python(code)
                self._send_json(HTTPStatus.OK, {
                    "ok": r.ok, "exit_code": r.exit_code,
                    "stdout": r.stdout, "stderr": r.stderr,
                    "duration_ms": r.duration_ms, "artifacts": r.artifacts,
                })
                return
            if route == "/api/sandbox/shell":
                cmd = str(payload.get("command", ""))
                if not cmd:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "command is required"}); return
                r = sandbox.run_shell(cmd)
                self._send_json(HTTPStatus.OK, {
                    "ok": r.ok, "exit_code": r.exit_code,
                    "stdout": r.stdout, "stderr": r.stderr,
                    "duration_ms": r.duration_ms, "artifacts": r.artifacts,
                })
                return
            if route == "/api/sandbox/reset":
                sandbox.reset()
                self._send_json(HTTPStatus.OK, {"ok": True}); return

            # ---- Engineer ------------------------------------------------
            if route == "/api/engineer/generate":
                desc = str(payload.get("description", ""))
                model = str(payload.get("model", "")).strip()
                try:
                    temperature = float(payload.get("temperature", 0.4))
                except (TypeError, ValueError):
                    temperature = 0.4
                try:
                    max_tokens = int(payload.get("max_tokens", 4096))
                except (TypeError, ValueError):
                    max_tokens = 4096
                self._send_json(HTTPStatus.OK, engineer.generate(
                    description=desc, model=model,
                    temperature=temperature, max_tokens=max_tokens,
                ))
                return

            # ---- AutoGPT -------------------------------------------------
            if route == "/api/autogpt/start":
                self._send_json(HTTPStatus.OK, autogpt.start(
                    goal=str(payload.get("goal", "")),
                    model=str(payload.get("model", "")).strip(),
                    temperature=float(payload.get("temperature", 0.5) or 0.5),
                    max_tokens=int(payload.get("max_tokens", 1024) or 1024),
                    max_steps=int(payload.get("max_steps", 30) or 30),
                ))
                return
            if route == "/api/autogpt/stop":
                self._send_json(HTTPStatus.OK, autogpt.stop()); return
            if route == "/api/autogpt/sleep":
                self._send_json(HTTPStatus.OK, autogpt.sleep_now()); return
            if route == "/api/autogpt/wake":
                self._send_json(HTTPStatus.OK, autogpt.wake_now()); return

            # ---- Manus ---------------------------------------------------
            if route == "/api/manus/start":
                self._send_json(HTTPStatus.OK, manus.start(
                    goal=str(payload.get("goal", "")),
                    model=str(payload.get("model", "")).strip(),
                    temperature=float(payload.get("temperature", 0.5) or 0.5),
                    max_tokens=int(payload.get("max_tokens", 1024) or 1024),
                    max_steps_per_task=int(payload.get("max_steps_per_task", 6) or 6),
                ))
                return
            if route == "/api/manus/stop":
                self._send_json(HTTPStatus.OK, manus.stop()); return
            if route == "/api/manus/sleep":
                self._send_json(HTTPStatus.OK, manus.sleep_now()); return
            if route == "/api/manus/wake":
                self._send_json(HTTPStatus.OK, manus.wake_now()); return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    return CatSDKHandler


# ===========================================================================
#  CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=APP_NAME.lower(),
        description=APP_TAGLINE,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--lm-studio-url",
        default=os.environ.get("LM_STUDIO_BASE_URL", DEFAULT_LM_STUDIO_BASE_URL),
        help="LM Studio OpenAI-compatible base URL")
    parser.add_argument("--lm-studio-key",
        default=os.environ.get("LM_STUDIO_API_KEY", "lm-studio"),
        help="Bearer token (LM Studio accepts any value).")
    parser.add_argument("--project", default=os.getcwd(),
        help="Project root that holds .ralph/ and .catsdk/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_apple_silicon_optimizations(AppConfig(
        host=args.host, port=args.port,
        lm_studio_base_url=args.lm_studio_url,
        lm_studio_api_key=args.lm_studio_key,
        project_root=Path(args.project).expanduser().resolve(),
    ))

    probe = LMStudioClient(
        base_url=config.lm_studio_base_url,
        timeout=8.0, api_key=config.lm_studio_api_key,
    ).ping()

    print(f"{APP_NAME} v{APP_VERSION}: http://{config.host}:{config.port}")
    print(f"LM Studio endpoint: {config.lm_studio_base_url}")
    print(f"Project root      : {config.project_root}")
    if probe.get("ok"):
        print(f"LM Studio: connected • {probe['models_loaded']} model(s) loaded")
    else:
        print(f"LM Studio: NOT reachable — {probe.get('error', 'unknown error')}")
        print("  Tip: open LM Studio → 'Developer' / 'Local Server' → Start Server.")
    if is_apple_silicon():
        print("Apple Silicon optimizations: enabled")

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
