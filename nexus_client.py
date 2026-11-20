"""
Nexus API client for SmartPHC.

Talks to any OpenAI-compatible ``/chat/completions`` endpoint (this is the
format the Nexus / Nexusify API documents). Everything is configured through
environment variables (see .env.example) - nothing is hard-coded and the API
key is only ever read on the server, never sent to the browser.

Token-saving features
---------------------
* Responses are cached on disk (instance/nexus_cache.db): asking the same
  question over the same data costs zero tokens the second time.
* Every request has a hard max_tokens cap.
* A local daily call limit (NEXUS_DAILY_LIMIT) stops runaway usage.
* Usage (calls, tokens, cache hits) is recorded per day for the demo/status.

If the key is missing, the quota is used up or the API fails, a NexusError
with a human-readable message is raised. Nothing is ever faked.
"""
import hashlib
import json
import os
import sqlite3
import threading
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests

DB_PATH = Path(__file__).parent / "instance" / "nexus_cache.db"
_lock = threading.Lock()

DEFAULT_BASE_URL = "https://nexusapi.navigatelabs.ai/v1"
DEFAULT_MODEL = "gemini-2.5-flash"


class NexusError(Exception):
    """Base error. ``message`` is safe to show to the user."""
    status = 502

    def __init__(self, message, status=None):
        super().__init__(message)
        self.message = message
        self.excerpts = None  # optional: document excerpts shown with the error
        if status:
            self.status = status


class NexusNotConfigured(NexusError):
    status = 503


class NexusQuotaError(NexusError):
    status = 429


class NexusAPIError(NexusError):
    status = 502


def _env(name, default=""):
    return (os.environ.get(name) or default).strip()


def settings():
    return {
        "api_key": _env("NEXUS_API_KEY"),
        "base_url": _env("NEXUS_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        "model": _env("NEXUS_MODEL", DEFAULT_MODEL),
        "max_tokens": int(_env("NEXUS_MAX_TOKENS", "350") or 350),
        "timeout": int(_env("NEXUS_TIMEOUT", "45") or 45),
        "daily_limit": int(_env("NEXUS_DAILY_LIMIT", "150") or 150),
        "cache_ttl": int(_env("NEXUS_CACHE_TTL_HOURS", "24") or 24) * 3600,
    }


def is_configured():
    key = settings()["api_key"]
    return bool(key) and key.lower() not in ("your_api_key_here", "changeme")


# ------------------------------------------------------------------ storage
def _connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=5)
    conn.execute("CREATE TABLE IF NOT EXISTS cache "
                 "(key TEXT PRIMARY KEY, response TEXT, ts REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS usage "
                 "(day TEXT PRIMARY KEY, calls INTEGER DEFAULT 0, "
                 "prompt_tokens INTEGER DEFAULT 0, "
                 "completion_tokens INTEGER DEFAULT 0, "
                 "cache_hits INTEGER DEFAULT 0)")
    return conn


def _cache_get(key, ttl):
    with _lock, _connect() as conn:
        row = conn.execute("SELECT response, ts FROM cache WHERE key=?",
                           (key,)).fetchone()
    if row and (time.time() - row[1]) < ttl:
        return row[0]
    return None


def _cache_put(key, text):
    with _lock, _connect() as conn:
        conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)",
                     (key, text, time.time()))
        # keep the cache small
        conn.execute("DELETE FROM cache WHERE key NOT IN "
                     "(SELECT key FROM cache ORDER BY ts DESC LIMIT 500)")


def _bump(calls=0, prompt=0, completion=0, hits=0):
    today = date.today().isoformat()
    with _lock, _connect() as conn:
        conn.execute("INSERT OR IGNORE INTO usage (day) VALUES (?)", (today,))
        conn.execute("UPDATE usage SET calls=calls+?, prompt_tokens=prompt_tokens+?, "
                     "completion_tokens=completion_tokens+?, cache_hits=cache_hits+? "
                     "WHERE day=?", (calls, prompt, completion, hits, today))


def usage_today():
    today = date.today().isoformat()
    with _lock, _connect() as conn:
        row = conn.execute("SELECT calls, prompt_tokens, completion_tokens, cache_hits "
                           "FROM usage WHERE day=?", (today,)).fetchone()
    calls, p, c, hits = row if row else (0, 0, 0, 0)
    return {"calls": calls, "prompt_tokens": p, "completion_tokens": c,
            "cache_hits": hits, "daily_limit": settings()["daily_limit"]}


# --------------------------------------------------------------------- chat
def _api_error_text(resp):
    try:
        body = resp.json()
        err = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(err, dict):
            err = err.get("message") or json.dumps(err)
        return str(err)[:200]
    except ValueError:
        return (resp.text or "")[:200]


def chat(messages, max_tokens=None, temperature=0.2, use_cache=True):
    """Send one chat completion. Returns {"text", "cached", "usage"}."""
    cfg = settings()
    if not is_configured():
        raise NexusNotConfigured(
            "The Nexus API key is not configured. Add NEXUS_API_KEY to the .env "
            "file and restart the server.")

    max_tokens = int(max_tokens or cfg["max_tokens"])
    key = hashlib.sha256(json.dumps(
        [cfg["model"], messages, max_tokens, temperature],
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    if use_cache:
        hit = _cache_get(key, cfg["cache_ttl"])
        if hit is not None:
            _bump(hits=1)
            return {"text": hit, "cached": True, "usage": {}}

    if usage_today()["calls"] >= cfg["daily_limit"]:
        raise NexusQuotaError(
            f"Daily AI limit of {cfg['daily_limit']} requests reached (set by "
            "NEXUS_DAILY_LIMIT to protect your free quota). Try again tomorrow "
            "or raise the limit in .env.")

    payload = {"model": cfg["model"], "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Authorization": f"Bearer {cfg['api_key']}",
               "Content-Type": "application/json"}
    url = cfg["base_url"] + "/chat/completions"
    host = urlparse(cfg["base_url"]).netloc

    resp = None
    for attempt in (1, 2):
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=cfg["timeout"])
        except requests.Timeout:
            if attempt == 1:
                time.sleep(1)
                continue
            raise NexusAPIError("The Nexus API timed out. Please try again.")
        except requests.ConnectionError:
            raise NexusAPIError(f"Could not reach the Nexus API at {host}. "
                                "Check your internet connection and NEXUS_BASE_URL.")
        if resp.status_code >= 500 and attempt == 1:
            time.sleep(1)
            continue
        break

    code = resp.status_code
    if code in (401, 403):
        raise NexusAPIError("Nexus rejected the API key (invalid or expired). "
                            "Check NEXUS_API_KEY in .env.", status=502)
    if code in (402, 429):
        raise NexusQuotaError("Nexus rate limit or free credits reached. Wait a "
                              "minute, or check your credits on the Nexus dashboard.")
    if code != 200:
        raise NexusAPIError(f"Nexus API error {code}: {_api_error_text(resp)}")

    try:
        data = resp.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
    except (ValueError, KeyError, IndexError, TypeError):
        raise NexusAPIError("Nexus returned an unexpected response format.")
    if not text:
        raise NexusAPIError("Nexus returned an empty answer. Please try again.")

    usage = data.get("usage") or {}
    _bump(calls=1, prompt=int(usage.get("prompt_tokens") or 0),
          completion=int(usage.get("completion_tokens") or 0))
    if use_cache:
        _cache_put(key, text)
    return {"text": text, "cached": False, "usage": usage}
