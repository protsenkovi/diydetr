docker run -it --rm \
    --name diydetr \
    --gpus all \
    -v $(pwd):/workdir \
    -w /workdir \
    -p 6006:6006 \
    diydetr \
    tensorboard --logdir ./runs --bind_all