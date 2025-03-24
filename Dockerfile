FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

RUN apt update && apt install -y python3 python3-pip

RUN pip install poetry

# docker run -v /path_to/HitsBE:/hitsbe --gpus device=0 -ti hitsbe:latest
