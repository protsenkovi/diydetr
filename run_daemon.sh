docker stop avaz_shmixer
docker rm avaz_shmixer
docker run -d \
    --name avaz_shmixer \
    --gpus all \
    -v $(pwd):/workdir \
    avaz_shmixer \
    bash -l -c "cd /workdir; python main.py"
