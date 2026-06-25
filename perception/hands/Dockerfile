# Hand-landmark service image: the community MediaPipe jetson build, plus the
# LD_PRELOAD that headless MediaPipe needs on this L4T. Without it the very first
# `import mediapipe` dies with:
#   ImportError: libGLdispatch.so.0: cannot allocate memory in static TLS block
# Preloading that lib first reserves its TLS slot before the rest loads. No extra
# pip is needed -- opencv 4.8 + numpy + mediapipe 0.10.7 are baked into the base.
#
# Build:  sudo docker build -t g1-hands:latest /home/unitree/projects/g1/perception/hands
FROM ghcr.io/lanzani/mediapipe:l4t35.4.1-py3.8.10-ocv4.8.0-mp0.10.7
ENV LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0
