# Serving the LipidOS model

vLLM on a single GPU, behind an authenticated gateway, so other people can run inference on
[`srikarjy025/lipidos-phi3-domain-adapt-merged`](https://huggingface.co/srikarjy025/lipidos-phi3-domain-adapt-merged)
for their own experiments.

**Status: code and gateway tests are done; nothing is deployed.** The gateway has 10 passing
tests. The Modal and Docker files have not been run against a real GPU.

## What the model is (and is not)

A causal-LM domain adaptation of Phi-3.5 Mini on lipid and Raman-spectroscopy abstracts. It is
not instruction-tuned, so the API exposes **text completion** (`/v1/completions`), not chat.
Give it the start of a passage and it continues it. Its measured gain is held-out perplexity
(4.911 to 3.955); that is language fit, not answer accuracy. Pair it with retrieval and the
citation check from the LipidOS pipeline for anything that matters.

## API

```bash
curl -X POST https://<your-deployment>/v1/completions \
  -H "authorization: Bearer <key_id>.<secret>" \
  -H "content-type: application/json" \
  -d '{"prompt":"The Raman band near 1440 cm-1 in lipids is assigned to","max_tokens":64}'
```

Limits: prompt up to 8,000 characters, `max_tokens` up to 512, one completion, no streaming,
64 KB request body. Unknown fields are rejected, so callers cannot reach vLLM-specific options.

## Security model

| Threat | Control |
|---|---|
| Anonymous or stolen-guess access | Bearer keys, stored only as SHA-256 hashes, constant-time compare, flat timing for unknown ids |
| One user exhausting the GPU or your bill | Per-key requests per minute and daily token quota; scale-to-zero; single replica cap |
| Abusing vLLM parameters | Strict request schema; `n`, `stream` forced server-side |
| Oversized or malformed input | Body size, prompt length, and `max_tokens` bounds; auth checked before body validation |
| Leaking internals | Upstream errors become generic 502/504; `/docs` and OpenAPI disabled |
| Sensitive prompts in logs | Audit log stores key id, prompt hash, sizes, tokens, latency, status, never prompt text |
| Reaching vLLM directly | vLLM binds to loopback only; the gateway is the sole public port |

Known limits: rate limits and quotas live in memory, so they reset on restart and are exact
only with one replica (the Modal file sets `max_containers=1`). Keys are rotated by editing the
secret. There is no per-user billing, abuse detection, or content filtering. Do not send
patient data: the model and this service are research tools, not clinical or HIPAA-grade systems.

## Deploy on Modal (scale to zero)

```bash
pip install modal && modal setup
python manage_keys.py create --label "alice@lab"      # prints the key once and a JSON fragment
modal secret create lipidos-api-keys API_KEYS_JSON='{"<key_id>": {...}}'
modal deploy modal_app.py
```

You pay for GPU time only while a container runs. The first request after idle waits for the
container and model to load, which can take minutes. Check Modal's current GPU pricing before
sharing the endpoint; the daily token quota is your cost ceiling per key.

Smoke test after the first deploy: `/health` returns 200, a request with no key returns 401,
and a valid request returns text. Pin the vLLM version that works for you.

## Other GPU hosts

`Dockerfile` and `start.sh` run the same stack anywhere with an NVIDIA GPU:

```bash
docker build -t lipidos-serve . 
docker run --gpus all -p 8000:8000 -e API_KEYS_JSON="$(cat keys.json)" lipidos-serve
```

Put it behind TLS (a reverse proxy or the host's load balancer) before exposing it.

## Tests

```bash
pip install fastapi httpx pydantic pytest anyio
python -m pytest tests -q
```

## Serving another model

Change `MODEL_REPO` and `SERVED_NAME`, adjust `--max-model-len`, and keep the gateway. If the
new model is instruction-tuned, add a `/v1/chat/completions` route with the same checks.
Healthcare models need de-identified public training data, a documented evaluation, and a
safety review before anyone outside your lab uses them.
