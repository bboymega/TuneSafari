FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/

EXPOSE 80

ENV FLASK_APP=api.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["sh", "-c", "WORKERS=$(( $(nproc) * 2 + 1 )) && echo Starting Gunicorn with $WORKERS workers && exec gunicorn --bind 0.0.0.0:80 --workers $WORKERS api:app --timeout 600"]
