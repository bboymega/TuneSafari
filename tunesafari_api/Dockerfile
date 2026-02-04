FROM python:3.12-slim

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

COPY . /app/

EXPOSE 8000

ENV FLASK_APP=api.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["sh", "-c", "WORKERS=$(( $(nproc) * 2 + 1 )) && echo Starting Gunicorn with $WORKERS workers && exec gunicorn --bind 0.0.0.0:8000 --workers $WORKERS api:app --timeout 600"]
