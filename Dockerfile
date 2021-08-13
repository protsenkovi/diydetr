FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN pip install scipy
RUN pip install pycocotools
RUN pip install einops
RUN pip install tensorboard