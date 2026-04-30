# Demo container: runs main.py end-to-end. NOT for training.
# Mount checkpoints + SMPL-X body models at runtime (see Makefile `docker-demo`).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libosmesa6 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.lock.txt ./
RUN pip install --upgrade pip && pip install -r requirements.lock.txt
RUN python -m spacy download en_core_web_sm

COPY . .

ENTRYPOINT ["python", "main.py"]
CMD ["a person walks forward"]
