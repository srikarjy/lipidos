# Portable alternative to Modal: any host with an NVIDIA GPU (RunPod, Lambda, your own server).
FROM vllm/vllm-openai:latest
RUN pip install --no-cache-dir fastapi httpx pydantic uvicorn
WORKDIR /app
COPY gateway.py start.sh ./
ENV SERVED_MODEL_NAME=lipidos-phi3-domain-adapt-merged VLLM_URL=http://127.0.0.1:8001
EXPOSE 8000
ENTRYPOINT ["/bin/bash", "/app/start.sh"]
