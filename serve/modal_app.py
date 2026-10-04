"""Deploy the LipidOS model on Modal: vLLM on one GPU behind the authenticated gateway.

UNTESTED against a real GPU: written from Modal's and vLLM's documented interfaces. Run the
smoke test in README.md after the first deploy and pin versions that work for you.

    modal secret create lipidos-api-keys API_KEYS_JSON='{...}'
    modal deploy modal_app.py
"""

import subprocess
import time

import modal

MODEL_REPO = "srikarjy025/lipidos-phi3-domain-adapt-merged"
SERVED_NAME = "lipidos-phi3-domain-adapt-merged"
VLLM_PORT = 8001

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "fastapi", "httpx", "pydantic")
    .env({"HF_HOME": "/models/hf", "SERVED_MODEL_NAME": SERVED_NAME, "VLLM_URL": f"http://127.0.0.1:{VLLM_PORT}"})
    .add_local_python_source("gateway")
)

app = modal.App("lipidos-inference", image=image)
weights = modal.Volume.from_name("lipidos-weights", create_if_missing=True)


@app.cls(
    gpu="L4",
    volumes={"/models": weights},
    secrets=[modal.Secret.from_name("lipidos-api-keys")],
    scaledown_window=300,  # scale to zero after 5 idle minutes: you pay only while it runs
    max_containers=1,      # one replica keeps the in-memory rate limits exact
    timeout=600,
)
@modal.concurrent(max_inputs=16)
class Inference:
    @modal.enter()
    def start_vllm(self) -> None:
        self.proc = subprocess.Popen([
            "vllm", "serve", MODEL_REPO,
            "--served-model-name", SERVED_NAME,
            "--host", "127.0.0.1", "--port", str(VLLM_PORT),
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
        ])
        import httpx

        deadline = time.time() + 540
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("vLLM exited during startup")
            try:
                if httpx.get(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(3)
        raise RuntimeError("vLLM did not become healthy in time")

    @modal.exit()
    def stop_vllm(self) -> None:
        self.proc.terminate()

    @modal.asgi_app()
    def web(self):
        from gateway import create_app

        return create_app()
