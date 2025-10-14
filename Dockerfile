FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY wands_search/models ./wands_search/models

EXPOSE 8000
EXPOSE 80

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80", "--log-level", "info"]
