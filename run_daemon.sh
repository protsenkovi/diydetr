docker stop diydetr
docker rm diydetr
docker run -d \
    --name diydetr \
    --gpus all \
    -v $(pwd):/workdir \
    --shm-size 1g \
    diydetr \
    bash -l -c "cd /workdir; python main.py"
