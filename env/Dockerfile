# FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

# Add User & Group
ARG UID
ARG USER
ARG PASSWORD

# Add User & Group
ARG UID
ARG USER
ARG PASSWORD
RUN echo $UID, $USER, $PASSWORD


RUN groupadd -g 1000 ${USER}_group
RUN useradd -m --uid=${UID} --gid=${USER}_group --groups=sudo ${USER}
RUN echo ${USER}:${PASSWORD} | chpasswd
RUN echo 'root:root' | chpasswd

COPY requirements.txt /
# RUN pip install -r /requirements.txt
RUN pip install -r /requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113


# for Timezone setting
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tz$
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata
# timezone setting
ENV TZ=Asia/Tokyo
RUN apt-get update && apt-get upgrade -y && apt-get install -y git



# Install Python library
RUN pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
# COPY requirements.txt /
# RUN pip install -r /requirements.txt


# Install fish
RUN apt-get update && apt-get install -y --no-install-recommends fish nano git sudo curl


# for OpenCV
#RUN apt-get update && apt-get install -y --no-install-recommends \
RUN apt-get update && apt-get install -y libopencv-dev

ENV PYTHONPATH ${PYTONPATH}:"/workspace/"

RUN chmod 777 "/root"
ENV HOME "/root"



