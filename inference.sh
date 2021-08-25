docker run -it --rm \
    --name diydetr_inference \
    --gpus all \
    -v $(pwd):/workdir \
    diydetr_inference \
    bash -l -c "cd /workdir; python inference.py"
