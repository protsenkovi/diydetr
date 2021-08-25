rm -rf runs/

docker run -it --rm \
    --name diydetr \
    --gpus all \
    --shm-size 1g \
    -v $(pwd):/workdir \
    diydetr \
    bash -l -c "cd /workdir; python main.py"
