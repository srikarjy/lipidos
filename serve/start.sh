#!/usr/bin/env bash
set -euo pipefail
# vLLM listens on loopback only; the gateway is the sole public surface.
vllm serve "${MODEL_REPO:-srikarjy025/lipidos-phi3-domain-adapt-merged}" \
  --served-model-name "$SERVED_MODEL_NAME" --host 127.0.0.1 --port 8001 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 &
exec uvicorn gateway:create_app --factory --host 0.0.0.0 --port 8000
