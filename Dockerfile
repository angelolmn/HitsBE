FROM nvidia/cuda:12.8

RUN apt update && apt install -y python3 python3-pip 

RUN pip install poetry

# docker run -v /home/aolmedo/HitsBE:/hitsbe --gpus device=0 -ti hitsbe:latest
