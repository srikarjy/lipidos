"""Authenticated, rate-limited gateway in front of a vLLM OpenAI-compatible server.

The model behind it is a causal-LM domain adaptation (not instruction-tuned), so only
/v1/completions is exposed. Everything the caller can influence is validated here; vLLM
itself is never reachable from outside the container.

Security properties (see README.md for the threat model):
  * API keys are stored only as SHA-256 hashes and compared in constant time.
  * Per-key requests-per-minute and daily token quotas.
  * Strict request schema (unknown fields rejected), bounded prompt/length/body size.
  * No streaming, n=1, no vLLM-specific extras (guided decoding, logit bias, ...).
  * Audit log stores a prompt hash and sizes, never prompt text.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("gateway")

MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "lipidos-phi3-domain-adapt-merged")
UPSTREAM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
MAX_BODY_BYTES = 64 * 1024
MAX_PROMPT_CHARS = 8_000
MAX_NEW_TOKENS = 512
CHARS_PER_TOKEN_ESTIMATE = 3  # deliberately pessimistic for quota pre-checks


@dataclass
class KeyRecord:
    key_id: str
    secret_hash: str
    label: str = ""
    rpm: int = 20
    daily_tokens: int = 50_000


@dataclass
class Limiter:
    """In-memory limits. Exact only when the deployment runs a single replica."""

    window: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    spent: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    def check_rate(self, key: KeyRecord, now: float) -> bool:
        q = self.window[key.key_id]
        while q and now - q[0] >= 60:
            q.popleft()
        if len(q) >= key.rpm:
            return False
        q.append(now)
        return True

    @staticmethod
    def _day() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def remaining(self, key: KeyRecord) -> int:
        return key.daily_tokens - self.spent[(key.key_id, self._day())]

    def charge(self, key: KeyRecord, tokens: int) -> None:
        self.spent[(key.key_id, self._day())] += tokens


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def load_keys(raw: str | None) -> dict[str, KeyRecord]:
    """Parse API_KEYS_JSON: {"<key_id>": {"hash": "...", "label": "...", "rpm": 20, "daily_tokens": 50000}}."""
    if not raw:
        return {}
    keys: dict[str, KeyRecord] = {}
    for key_id, spec in json.loads(raw).items():
        keys[key_id] = KeyRecord(
            key_id=key_id,
            secret_hash=spec["hash"],
            label=spec.get("label", ""),
            rpm=int(spec.get("rpm", 20)),
            daily_tokens=int(spec.get("daily_tokens", 50_000)),
        )
    return keys


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = MODEL_NAME
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    max_tokens: int = Field(default=128, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stop: list[str] | None = Field(default=None, max_length=4)
    seed: int | None = None


def create_app(
    keys: dict[str, KeyRecord] | None = None,
    upstream_url: str = UPSTREAM_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    app = FastAPI(title="LipidOS inference gateway", docs_url=None, redoc_url=None, openapi_url=None)
    key_table = keys if keys is not None else load_keys(os.environ.get("API_KEYS_JSON"))
    limiter = Limiter()
    client = httpx.AsyncClient(base_url=upstream_url, timeout=120.0, transport=transport)

    def authenticate(request: Request) -> KeyRecord:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"})
        # Token format: <key_id>.<secret>. Always hash, even for unknown ids, to keep timing flat.
        key_id, _, secret = token.partition(".")
        record = key_table.get(key_id)
        candidate = hash_secret(secret)
        expected = record.secret_hash if record else hash_secret("unknown")
        if not hmac.compare_digest(candidate, expected) or record is None:
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})
        return record

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Model-Use"] = "research-only; not for clinical decisions"
        return response

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_NAME}

    @app.get("/v1/models")
    async def models(key: KeyRecord = Depends(authenticate)):
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, key: KeyRecord = Depends(authenticate)):
        started = time.monotonic()

        if body.model != MODEL_NAME:
            raise HTTPException(400, f"Unknown model; use '{MODEL_NAME}'")
        if not limiter.check_rate(key, time.time()):
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})

        worst_case = len(body.prompt) // CHARS_PER_TOKEN_ESTIMATE + body.max_tokens
        if worst_case > limiter.remaining(key):
            raise HTTPException(429, "Daily token quota exhausted")

        payload = body.model_dump(exclude_none=True)
        payload["n"] = 1
        payload["stream"] = False
        status, tokens = 502, 0
        try:
            upstream = await client.post("/v1/completions", json=payload)
            status = upstream.status_code
            if status != 200:
                raise HTTPException(502, "Model server error")
            data = upstream.json()
            tokens = int(data.get("usage", {}).get("total_tokens", worst_case))
            limiter.charge(key, tokens)
            return JSONResponse(data)
        except httpx.HTTPError as exc:
            status = 504
            raise HTTPException(504, "Model server unavailable") from exc
        finally:
            log.info(json.dumps({
                "event": "completion",
                "key_id": key.key_id,
                "prompt_sha256": hashlib.sha256(body.prompt.encode()).hexdigest(),
                "prompt_chars": len(body.prompt),
                "max_tokens": body.max_tokens,
                "total_tokens": tokens,
                "status": status,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }))

    return app
