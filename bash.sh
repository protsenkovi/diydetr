docker run -it --rm \
    --name diydetr \
    --gpus all \
    -v $(pwd):/workdir \
    -w /workdir \
    diydetr \
    bash