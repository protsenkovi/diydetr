#!/bin/bash
export UID=$(id -u)
export GID=$(id -g)

docker build \
  --build-arg USER=$USER \
  --build-arg UID=$UID \
  --build-arg GID=$GID \
  --build-arg PW=123 \
  -t diydetr \
  -f Dockerfile_Training .