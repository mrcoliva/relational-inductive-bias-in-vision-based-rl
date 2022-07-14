FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 gcc g++ git python-opengl xvfb

RUN conda install pytorch-scatter -c rusty1s \
 && conda install pytorch-sparse -c rusty1s \
 && pip install torch-geometric

ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt

WORKDIR /code
