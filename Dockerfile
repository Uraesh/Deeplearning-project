FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/latest/model.pt
ENV METRICS_PATH=/app/models/latest/metrics.json

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY data.csv ./data.csv
COPY models ./models

EXPOSE 8000

CMD ["sh", "-c", "python -m breast_cancer_ai.api --host 0.0.0.0 --port ${PORT:-8000}"]
