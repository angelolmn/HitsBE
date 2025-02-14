FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* README.md /app/

COPY hitsbe/ /app/hitsbe/

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-root

# No se si es necesario
COPY . /app

CMD ["python", "hitsbe/main.py"]
