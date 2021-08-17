docker run -it --rm \
    --name diydetr_tensorboard \
    --gpus all \
    -v $(pwd):/workdir \
    -w /workdir \
    -p 6006:6006 \
    diydetr \
    tensorboard --logdir ./runs --host 0.0.0.0