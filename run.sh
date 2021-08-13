docker run -it --rm \
    --name diydetr \
    --gpus all \
    -v $(pwd):/workdir \
    diydetr \
    bash -l -c "cd /workdir; python main.py"
