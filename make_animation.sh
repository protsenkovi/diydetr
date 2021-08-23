docker run --rm -it --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    --name diydetr_make_animation \
    -v $(pwd):/workdir \
    -w /workdir \
    romansavrulin/ffmpeg-cuda \
  -framerate 15 \
  -i runs/result_%06d.png \
  -c:v hevc_nvenc \
  -b:v 4M -maxrate:v 5M -bufsize:v 8M -profile:v main \
  diydetr_train_visualization.mp4  
