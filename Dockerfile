FROM nvidia/cuda:12.0.1-base-ubuntu22.04 

WORKDIR /app

RUN apt-get update && apt-get install -y python3.11 python3-pip 

RUN pip install poetry

COPY . /app

RUN poetry install

CMD ["poetry", "run", "python", "experiments/hitsbert_example.py"]