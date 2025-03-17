FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN pip install --no-cache-dir poetry

ENV PYTHONPATH="/app"

COPY README.md pyproject.toml poetry.lock /app/

COPY . /app

RUN poetry install --with dev

RUN poetry run pip install -e .

CMD ["poetry", "run", "python", "experiments/hitsbert_example.py"]