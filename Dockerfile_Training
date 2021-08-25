FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN pip install scipy
RUN pip install pycocotools
RUN pip install einops
RUN pip install tensorboard

ARG UID
ARG USER
ARG GID
ARG PW

RUN groupadd --gid $GID $USER
RUN useradd -m -u ${UID} -g $USER $USER && echo $USER:$PW | chpasswd
USER ${UID}:${GID}

ENV TORCH_HOME=/workdir/.torch/