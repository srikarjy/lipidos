import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gateway
from manage_keys import create


def make_client(rpm=3, daily_tokens=1000, upstream=None, calls=None):
    key, entry = create("test", rpm, daily_tokens)
    keys = gateway.load_keys(json.dumps(entry))

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(json.loads(request.content))
        if upstream is not None:
            return upstream(request)
        return httpx.Response(200, json={"choices": [{"text": "ok"}], "usage": {"total_tokens": 20}})

    app = gateway.create_app(keys, transport=httpx.MockTransport(handler))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
    return client, key


def auth(key):
    return {"authorization": f"Bearer {key}"}


BODY = {"prompt": "Raman bands of cholesterol", "max_tokens": 16}


@pytest.mark.anyio
async def test_requires_valid_key():
    client, key = make_client()
    assert (await client.post("/v1/completions", json=BODY)).status_code == 401
    assert (await client.post("/v1/completions", json=BODY, headers=auth("bad.key"))).status_code == 401
    wrong = key.split(".")[0] + ".wrongsecret"
    assert (await client.post("/v1/completions", json=BODY, headers=auth(wrong))).status_code == 401
    assert (await client.post("/v1/completions", json=BODY, headers=auth(key))).status_code == 200


@pytest.mark.anyio
async def test_auth_checked_before_body_validation():
    client, _ = make_client()
    response = await client.post("/v1/completions", json={"nope": 1})
    assert response.status_code == 401


@pytest.mark.anyio
async def test_rejects_unknown_fields_and_oversize():
    client, key = make_client()
    h = auth(key)
    assert (await client.post("/v1/completions", json={**BODY, "guided_json": {}}, headers=h)).status_code == 422
    assert (await client.post("/v1/completions", json={**BODY, "max_tokens": 5000}, headers=h)).status_code == 422
    assert (await client.post("/v1/completions", json={**BODY, "prompt": "x" * 9000}, headers=h)).status_code == 422
    assert (await client.post("/v1/completions", json={**BODY, "model": "other"}, headers=h)).status_code == 400


@pytest.mark.anyio
async def test_forces_safe_upstream_params():
    calls = []
    client, key = make_client(calls=calls)
    await client.post("/v1/completions", json={**BODY, "stop": ["\n"]}, headers=auth(key))
    assert calls[0]["n"] == 1 and calls[0]["stream"] is False


@pytest.mark.anyio
async def test_rate_limit():
    client, key = make_client(rpm=2)
    h = auth(key)
    assert (await client.post("/v1/completions", json=BODY, headers=h)).status_code == 200
    assert (await client.post("/v1/completions", json=BODY, headers=h)).status_code == 200
    third = await client.post("/v1/completions", json=BODY, headers=h)
    assert third.status_code == 429 and third.headers["retry-after"] == "60"


@pytest.mark.anyio
async def test_daily_token_quota():
    client, key = make_client(rpm=50, daily_tokens=100)
    h = auth(key)
    assert (await client.post("/v1/completions", json={**BODY, "max_tokens": 40}, headers=h)).status_code == 200
    # 20 tokens charged; next worst case (prompt/3 + 90) exceeds the 80 remaining
    assert (await client.post("/v1/completions", json={**BODY, "max_tokens": 90}, headers=h)).status_code == 429


@pytest.mark.anyio
async def test_upstream_errors_do_not_leak_or_charge():
    def boom(request):
        return httpx.Response(500, text="Traceback: secret internals")

    client, key = make_client(upstream=boom)
    response = await client.post("/v1/completions", json=BODY, headers=auth(key))
    assert response.status_code == 502 and "Traceback" not in response.text


@pytest.mark.anyio
async def test_audit_log_has_hash_not_prompt(caplog):
    client, key = make_client()
    with caplog.at_level("INFO", logger="gateway"):
        await client.post("/v1/completions", json=BODY, headers=auth(key))
    text = "\n".join(caplog.messages)
    assert "prompt_sha256" in text and BODY["prompt"] not in text and key.split(".")[1] not in text


@pytest.mark.anyio
async def test_oversized_body_rejected():
    client, key = make_client()
    response = await client.post("/v1/completions", content=b"x" * 70_000, headers={**auth(key), "content-type": "application/json"})
    assert response.status_code == 413


@pytest.mark.anyio
async def test_docs_not_exposed_and_health_open():
    client, _ = make_client()
    assert (await client.get("/docs")).status_code == 404
    assert (await client.get("/openapi.json")).status_code == 404
    assert (await client.get("/health")).status_code == 200
