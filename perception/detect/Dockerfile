# Object-detection image for the G1 dashboard.
#
# Self-contained: built directly from the public Ultralytics Jetson base (NOT from
# g1-pose -- detection uses nothing pose-specific). It adds only YOLO-World's CLIP
# text-encoder PACKAGE. The 354 MB CLIP weight is NOT baked here -- it lives in the
# project at perception/detect/models/clip/ViT-B-32.pt and is mounted read-only at
# runtime (see run_detect.sh), so the model runs offline and nothing is hidden in
# the image.
#
# Build:  sudo docker build -t g1-detect:latest /home/unitree/projects/g1/perception/detect
FROM ultralytics/ultralytics:latest-jetson-jetpack5

# Trigger Ultralytics' own CLIP setup (installs the exact clip build it needs),
# then drop the downloaded weight in the SAME layer -- the runtime mount supplies
# it from the project folder, keeping the image lean.
RUN python3 -c "from ultralytics import YOLOWorld; m = YOLOWorld('yolov8s-worldv2.pt'); m.set_classes(['door']); print('CLIP package installed OK')" \
 && rm -f /ultralytics/weights/clip/ViT-B-32.pt
