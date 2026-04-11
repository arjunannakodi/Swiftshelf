FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# No HEALTHCHECK — let HF Spaces manage health
# Use longer timeout so space doesn't die during cold start

CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--timeout-keep-alive", "120"]