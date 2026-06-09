# Tiny derived image: the official jetson Ultralytics stack + the ByteTrack dep
# baked in, so the pose service starts fast and works offline (no per-start pip).
#
# Build:  sudo docker build -t g1-pose:latest /home/unitree/perception/pose
FROM ultralytics/ultralytics:latest-jetson-jetpack5
RUN pip install --no-cache-dir "lap>=0.5.12"
