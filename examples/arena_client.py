"""Arena HTTP client + introspection + credential helpers.

This module is the "plumbing" you usually don't need to read or edit.
It wraps the Arena REST API with:

  - httpx client with 429/5xx retry + Retry-After honored
  - typed ArenaError for surfaced 4xx
  - introspection fetch + required-endpoint assertion (fail loud, not 404 mid-hand)
  - terminal phase/status resolution from the live schema
  - idempotent credential cache (re-verifies cached key via /agent/me)
  - small JSON state cache for cross-run continuity

Builders normally only touch `examples/agent.py` — this file is shared by
`agent.py`, `llm_agent.py`, and `mock.py`.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import httpx


DEFAULT_BASE = "https://arena.dev.fun/api/arena"
MOCK_BASE = "http://mock.local/api/arena"  # --dry-run rebinds to this
CREDS_PATH = Path(".arena-credentials")
CREDS_BACKUP_PATH = Path(".arena-credentials.rejected")
STATE_PATH = Path(".arena-poker-state")
RETRY_MAX = 3


def _move_creds_aside() -> bool:
    if not CREDS_PATH.exists():
        return False
    try:
        os.replace(str(CREDS_PATH), str(CREDS_BACKUP_PATH))
        return True
    except OSError as e:
        print(f"[arena-pokerkit] failed to back up creds aside: {e}", file=sys.stderr)
        return False


def _restore_creds_backup() -> bool:
    if not CREDS_BACKUP_PATH.exists():
        return False
    if CREDS_PATH.exists():
        return False
    try:
        os.replace(str(CREDS_BACKUP_PATH), str(CREDS_PATH))
        print("[arena-pokerkit] restored previous .arena-credentials after registration failure",
              file=sys.stderr)
        return True
    except OSError as e:
        print(f"[arena-pokerkit] failed to restore creds backup: {e}", file=sys.stderr)
        return False


def _discard_creds_backup() -> None:
    if CREDS_BACKUP_PATH.exists():
        try:
            CREDS_BACKUP_PATH.unlink()
        except OSError:
            pass


REQUIRED_ENDPOINTS = (
    ("POST", "/api/arena/auth/register"),
    ("GET",  "/api/arena/agent/me"),
    ("POST", "/api/arena/texas/benchmark/start"),
    ("GET",  "/api/arena/texas/benchmark/status"),
    ("GET",  "/api/arena/texas/pending-actions"),
    ("POST", "/api/arena/texas/action"),
)

FALLBACK_TERMINAL_PHASES = ("completed", "cancelled", "failed")
FALLBACK_TERMINAL_STATUSES = ("Completed", "Cancelled", "Failed")


class ArenaError(Exception):
    def __init__(self, status: int, body: Any, where: str = ""):
        super().__init__(f"{where} status={status} body={body}")
        self.status = status
        self.body = body
        self.where = where


class ArenaClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 15.0):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout, trust_env=False)

    def close(self) -> None:
        self._client.close()

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["x-arena-api-key"] = self.api_key
        return h

    def _req(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base}{path}"
        backoff = 0.5
        last_exc: Optional[Exception] = None
        for attempt in range(RETRY_MAX):
            try:
                r = self._client.request(method, url, headers=self._headers(), **kwargs)
            except httpx.HTTPError as e:
                last_exc = e
                time.sleep(backoff)
                backoff *= 2
                continue
            try:
                body = r.json()
            except Exception:
                body = r.text
            if r.status_code == 429 and attempt < RETRY_MAX - 1:
                ra = r.headers.get("Retry-After")
                try:
                    wait = float(ra) if ra else backoff
                except ValueError:
                    wait = backoff
                time.sleep(max(wait, 0.0))
                backoff *= 2
                continue
            if r.status_code >= 500 and attempt < RETRY_MAX - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            if not r.is_success:
                raise ArenaError(r.status_code, body, where=f"{method} {path}")
            return body
        raise ArenaError(0, str(last_exc), where=f"{method} {path}")

    def get(self, path: str, **kwargs) -> Any:
        return self._req("GET", path, **kwargs)

    def post(self, path: str, json_body: Optional[dict] = None) -> Any:
        return self._req("POST", path, json=json_body)


def fetch_introspection(client: ArenaClient) -> dict:
    try:
        schema = client.get("/__introspection")
    except ArenaError as e:
        raise SystemExit(
            f"[arena-pokerkit] introspection unreachable ({e.where} -> {e.status}). "
            "The live API may be down; cannot continue safely."
        )
    if not isinstance(schema, dict):
        raise SystemExit("[arena-pokerkit] introspection returned non-object — refusing to continue.")
    return schema


def assert_endpoints(schema: dict,
                     required: tuple[tuple[str, str], ...] = REQUIRED_ENDPOINTS) -> None:
    endpoints = schema.get("endpoints") or []
    present = {(e.get("method"), e.get("path")) for e in endpoints if isinstance(e, dict)}
    missing = [pair for pair in required if pair not in present]
    if missing:
        raise SystemExit(
            "[arena-pokerkit] live API schema missing endpoint(s): "
            + ", ".join(f"{m} {p}" for m, p in missing)
        )


def resolve_terminal_phases(schema: dict) -> tuple[set[str], set[str]]:
    phase_enum: list[str] = []
    status_enum: list[str] = []
    for ep in (schema.get("endpoints") or []):
        if not isinstance(ep, dict):
            continue
        if ep.get("path") != "/api/arena/texas/benchmark/start":
            continue
        out = ep.get("output") or {}
        match = ((out.get("properties") or {}).get("match")) or {}
        candidates = match.get("anyOf") or [match]
        for cand in candidates:
            props = (cand.get("properties") or {})
            ph = props.get("phase") or {}
            st = props.get("status") or {}
            if ph.get("enum"):
                phase_enum = ph["enum"]
            if st.get("enum"):
                status_enum = st["enum"]
            if phase_enum and status_enum:
                break
        break

    if not phase_enum:
        phase_enum = list(FALLBACK_TERMINAL_PHASES) + ["queued", "panel_acting", "waiting_user"]
    if not status_enum:
        status_enum = list(FALLBACK_TERMINAL_STATUSES) + ["Running"]

    live_phases = {"queued", "panel_acting", "waiting_user"}
    terminal_phases = {p for p in phase_enum if p.lower() not in live_phases and p != "Running"}
    terminal_statuses = {s for s in status_enum if s != "Running"}
    if not terminal_phases:
        terminal_phases = set(FALLBACK_TERMINAL_PHASES)
    if not terminal_statuses:
        terminal_statuses = set(FALLBACK_TERMINAL_STATUSES)
    return terminal_phases, terminal_statuses


def load_or_register(client: ArenaClient, handle: str, name: str, quote: str) -> dict:
    if CREDS_PATH.exists():
        try:
            creds = json.loads(CREDS_PATH.read_text())
        except Exception:
            creds = {}
        key = creds.get("apiKey") or ""
        agent_id_str = str(creds.get("agentId") or creds.get("id") or "")
        if agent_id_str == "agent_dry" or key.startswith("dry_") or key.startswith("mock_"):
            print(f"[arena-pokerkit] detected stale mock creds; re-registering", file=sys.stderr)
            _move_creds_aside()
            creds = {}
            key = None
        if key:
            client.api_key = key
            try:
                me = client.get("/agent/me")
                if isinstance(me, dict) and (me.get("id") or me.get("agentId") or me.get("handle")):
                    return creds
            except ArenaError as e:
                if e.status in (401, 403):
                    print(f"[arena-pokerkit] cached key rejected ({e.status}); re-registering",
                          file=sys.stderr)
                    client.api_key = None
                    _move_creds_aside()
                else:
                    raise

    attempt_handle = handle
    body = None
    try:
        for attempt in range(3):
            try:
                body = client.post("/auth/register", {
                    "handle": attempt_handle, "name": name, "quote": quote,
                    "description": "",
                })
                break
            except ArenaError as e:
                if e.status == 409 and _is_handle_taken(e.body) and attempt < 2:
                    suffix = secrets.token_hex(3)
                    attempt_handle = f"{handle}-{suffix}"
                    print(f"[arena-pokerkit] handle taken; retrying as {attempt_handle!r}",
                          file=sys.stderr)
                    continue
                raise
        if isinstance(body, dict) and "apiKey" in body:
            client.api_key = body["apiKey"]
        _atomic_write(CREDS_PATH, json.dumps(body, indent=2))
        _discard_creds_backup()
        return body if isinstance(body, dict) else {}
    except Exception:
        _restore_creds_backup()
        raise


def _is_handle_taken(body: Any) -> bool:
    if isinstance(body, dict):
        text = " ".join(
            str(v) for v in (body.get("error"), body.get("message"), body.get("detail")) if v
        )
    else:
        text = str(body or "")
    return "already taken" in text.lower() or "handle" in text.lower()


def _default_state() -> dict:
    return {
        "hands_played": 0, "bankroll": 0, "last_action": None,
        "timeout_count": 0, "rejection_count": 0, "stale_count": 0, "iterations": [],
    }


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
            if isinstance(state, dict):
                if "iterations" not in state or not isinstance(state.get("iterations"), list):
                    state["iterations"] = []
                for k, v in _default_state().items():
                    state.setdefault(k, v)
                return state
        except Exception:
            pass
    return _default_state()


def save_state(state: dict) -> None:
    _atomic_write(STATE_PATH, json.dumps(state, indent=2))


def append_iteration(entry: dict) -> dict:
    state = load_state()
    iters = state.get("iterations") or []
    if not isinstance(iters, list):
        iters = []
    record = dict(entry)
    record.setdefault("iter", len(iters))
    if "ts" not in record:
        import datetime as _dt
        record["ts"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    iters.append(record)
    state["iterations"] = iters
    save_state(state)
    return state


def _atomic_write(path: Path, contents: str) -> None:
    parent = path.parent if str(path.parent) else Path(".")
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp",
        dir=str(parent) if str(parent) else None,
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(contents)
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
