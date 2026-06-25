# Hands — finger landmark detection (Phase 1)

Adds **21 finger landmarks per hand** to the dashboard. The landmarks are drawn
on the same camera canvas as the body skeleton and ride the **Skeleton** toggle:
turn Skeleton on, and any visible hands get landmarked too. This is the
foundation layer only — **no gesture recognition / no command logic** (that is a
later phase).

## How it fits the existing perception architecture

Same demand-gated, shm-JSON, canvas-overlay pattern as `../pose` and `../detect`:

```
camera_service.py ──► /dev/shm/g1_camera.jpg ──► hands_service.py (g1-hands container)
                                                      │  MediaPipe Hand Landmarker (GPU)
                                                      ▼
                                 /dev/shm/g1_hands_tracks.json  {w,h,items:[{hand,score,landmarks:[[x,y,z]x21]}]}
                                                      ▲ heartbeat /dev/shm/g1_hands_demand
   browser: pose.js (Skeleton toggle) ──► hands.js polls /camera/hands/tracks ──► cam-overlay.js drawHands()
```

- We **never open the camera** — `/dev/video0` is single-consumer; we only re-read
  the JPEG `camera_service.py` already publishes.
- **Demand-gated:** the GPU only runs while the Skeleton overlay is on (the
  browser poll heartbeats `g1_hands_demand`; stale after 3 s → the service idles).
- Landmark x,y are in **source-frame pixels**, so they map through the exact same
  canvas projector the skeleton uses → fingers line up pixel-perfect with the body.

## Install notes (detected environment + what worked)

| Item | Value |
|---|---|
| L4T / JetPack | **R35.3.1 / JetPack 5.1.1** |
| Host Python | 3.8.10 |
| CUDA | 11.4 |
| Install option used | **A — prebuilt Docker image** (`ghcr.io/lanzani/mediapipe:l4t35.4.1-py3.8.10-ocv4.8.0-mp0.10.7`) |
| MediaPipe | 0.10.7 (Tasks API `HandLandmarker`) |
| GPU vs CPU | **GPU delegate** — confirmed working (`Created TensorFlow Lite delegate for GPU`) |

Three non-obvious things that were required to make it run (all baked into the
Dockerfile / run script):

1. **`LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0`** — without it the very
   first `import mediapipe` dies with *"cannot allocate memory in static TLS
   block"*. Set as `ENV` in the Dockerfile.
2. **`--runtime nvidia` is required even though inference is "just" landmarks** —
   the HandLandmarker graph needs an EGL/GPU context to build (`kGpuService`);
   headless without the nvidia runtime it fails with `eglGetDisplay` errors.
3. **GPU delegate, not CPU** — on this build the **CPU delegate loads but detects
   ZERO hands** (the float16 model misbehaves on the XNNPACK CPU path here). The
   GPU delegate detects reliably (clear hands at 0.94–0.99 confidence, ~34 ms).

The legacy `mp.solutions.hands` API is **broken** in this image ("Unable to find
the type for stream 'image'"); the modern Tasks API is what works, which is why
`models/hand_landmarker.task` is shipped here.

## Files

| File | Role |
|---|---|
| `hand_detector.py` | Importable `HandDetector` (no display / no shm). Phase 2 imports this. |
| `hands_service.py` | Demand-gated producer loop (reads camera shm, writes hands JSON). |
| `still_test.py` | Dev visualiser — runs the detector on still images, saves annotated JPEGs. |
| `Dockerfile` | `g1-hands:latest` = lanzani mediapipe image + the `LD_PRELOAD` env. |
| `run_hands.sh` | Launches the container detached, demand-gated, `--restart unless-stopped`. |
| `models/hand_landmarker.task` | Pretrained MediaPipe hand model (float16). |

## Run

```bash
# build once
sudo docker build -t g1-hands:latest /home/unitree/projects/g1/perception/hands
# launch (idempotent; survives reboot; idles until Skeleton is on)
/home/unitree/projects/g1/perception/hands/run_hands.sh
```

The dashboard routes live in `scripts/robot_web_controller.py`
(`/camera/hands/{status,tracks}`); after editing the controller, restart it:
`sudo systemctl restart g1-web.service`.

## Config knobs (env, see `run_hands.sh`)

| Env | Default | Meaning |
|---|---|---|
| `INFER_HZ` | 10 | Max inference rate (caps GPU load shared with pose/detect). |
| `MAX_HANDS` | 2 | Most hands to landmark per frame. |
| `MIN_DET_CONF` | 0.6 | Min hand-detection confidence (raise to cut phantom hands). |
| `MIN_TRK_CONF` | 0.6 | Min tracking confidence (raise to drop an uncertain hand sooner). |
| `ALWAYS_ON` | 0 | `1` ignores demand-gating (manual testing). |

## Test still images

```bash
sudo docker run --rm --runtime nvidia \
  -e LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0 \
  -v /home/unitree/projects/g1/perception/hands:/app \
  g1-hands:latest python3 /app/still_test.py /app/test_images/woman_hands.jpg
# -> writes /app/test_images/woman_hands_hands.jpg with landmarks drawn
```
