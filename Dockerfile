FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY data.csv ./data.csv

EXPOSE 8000

CMD ["sh", "-c", "if [ ! -f /app/models/latest/model.pt ]; then python -m breast_cancer_ai.train --config configs/train_config.yaml; fi && python -m breast_cancer_ai.api --host 0.0.0.0 --port ${PORT:-8000}"]
