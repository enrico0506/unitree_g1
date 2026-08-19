# G1 Dashboard

Web-based teleop + perception dashboard for a Unitree G1, running on the robot's
onboard NVIDIA Jetson Orin NX. Serves a control UI (drive, modes, LiDAR map) and a
live head-camera feed, with optional **people-skeleton** (YOLO11-pose) and
**object-detection** (NanoOWL+NanoSAM, open-vocabulary) overlays.

Open it over WLAN at `http://<robot-ip>:8080`.

---

## Quick start

The dashboard is a FastAPI app, installed as the `g1-web` systemd service (auto-starts
on boot; `sudo systemctl restart g1-web` / `sudo systemctl status g1-web`). To run it
manually in a terminal instead (e.g. for debugging — the interactive safety prompt is
skipped automatically when not attached to a TTY, so it works either way):

```bash
cd /home/unitree/projects/g1
export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds
export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH
python3 scripts/robot_web_controller.py     # read the warnings, then press Enter
```

It prints `Web controller live at http://<robot-ip>:8080` and spawns the camera
grabber as a separate process.

To also get the **skeleton overlay**, start the pose engine (see
[People-pose feature](#people-pose-feature-skeletons--labels)):

```bash
perception/pose/run_pose.sh               # or: sudo docker start g1-pose
```

---

## Architecture (how it fits together)

```
                    robot (videohub RPC, locomotion) ── DDS ──┐
                                                              │
  camera_service.py ──reads videohub──> /dev/shm/g1_camera.jpg
        (own process, single camera consumer)         │
                                                       │ reads
  robot_web_controller.py (FastAPI :8080) ─────────────┤ serves MJPEG /camera/stream
        ├─ /ws            control + telemetry          │
        ├─ /ws/lidar      3D point cloud               │
        └─ /camera/pose/* skeleton lane ───────────────┘
                                                       ▲ writes
  pose_service.py (in Docker) ──reads g1_camera.jpg──> /dev/shm/g1_pose.jpg
        YOLO11-pose + ByteTrack, skeletons + names
```

- **One camera consumer.** Only `camera_service.py` talks to the robot's single-
  consumer videohub; everything else reads the JPEG it writes to shared memory.
- **Processes are isolated** so a blocking locomotion RPC can't stall the camera.
- The robot's balance/walk control runs on the robot's own control boards, **not**
  this Jetson — the Jetson runs the dashboard, LiDAR, and (optionally) pose.

Modes (UI): `zero_torque` (FSM 0), `damp` (1), `stand` (4), `walk` (802). The robot
only moves in **walk**; it boots in **damp** and won't move until you command it.

---

## People-pose feature (skeletons + labels)

Draws 17-point stick-figure skeletons on people in the camera feed, tracks each as a
stable ID (ByteTrack), and lets you **name each ID** from the dashboard. It is a
**toggle** on the camera panel — the raw feed is untouched when it's off.

**Use it:**
1. Start the pose engine: `perception/pose/run_pose.sh` (or `sudo docker start g1-pose`).
2. In the dashboard, click **Skeleton** (top-right of the camera window).
3. People in view get a skeleton + an ID. Type a name in the **People** bar to label
   that ID; the name follows the skeleton until the track is lost.
4. Click **Skeleton** again to return to the raw feed.

**GPU is demand-gated:** the engine only runs while someone is watching the skeleton
view. Toggle off / close the tab → it idles at ~0% GPU, protecting the robot.

The code for this feature lives in **[perception/pose/](perception/pose/)**
(its own [README](perception/pose/README.md)) plus the `/camera/pose/*` routes
in `scripts/robot_web_controller.py` and `web/perception/pose.js`.

---

## Docker (the pose GPU environment)

The pose model needs GPU PyTorch + TensorRT, which is painful to install natively on
Jetson. Instead we run it inside NVIDIA/Ultralytics' **prebuilt** image — a sealed,
known-good GPU environment. **The robot's own Python is never touched.**

**What is prebuilt vs. ours:**
- *Prebuilt (downloaded as-is):* `ultralytics/ultralytics:latest-jetson-jetpack5`
  — PyTorch (CUDA 11.4), TensorRT, OpenCV, the YOLO library. ~13.7 GB.
- *Our only Docker addition:* a 2-line [Dockerfile](perception/pose/Dockerfile)
  that pre-installs one helper lib (`lap`, the tracker) → image `g1-pose:latest`.
- *Our actual logic:* all in local files under `perception/pose/` — **not** baked
  into the image. They are *mounted* into the container at runtime.

**Where things live (nothing important is "inside" the container):**

| Thing | On disk | In container |
|---|---|---|
| Sealed GPU environment (image) | Docker storage `/var/lib/docker` | — |
| Our code (`pose_service.py`) | `perception/pose/` ← **edit here** | `/app` |
| Model files (`.pt` / `.engine`) | `perception/pose/models/` | `/models` |
| Frames in/out | `/dev/shm/g1_*.jpg` | `/dev/shm` |

The container runs **detached with `--restart unless-stopped`**, so it survives a
crash *and* a full power cycle: the Docker daemon is enabled on boot and re-launches
it automatically (the GPU stays demand-gated, so an always-present container still
idles at ~0% GPU until someone watches). `sudo docker stop g1-pose` keeps it stopped
until you start it again. Your code, models, and the image all persist on disk
regardless.

### Accessing / managing the container

```bash
sudo docker ps                      # is it running?
sudo docker logs -f g1-pose         # watch its output live (Ctrl-C to stop watching)
sudo docker exec -it g1-pose bash   # shell INSIDE it; `ls /app` shows your local files
sudo docker stop g1-pose            # stop (skeleton view then shows "no signal")
perception/pose/run_pose.sh       # start again
sudo docker images                  # list images
```

### Changing pose behavior

1. Edit `perception/pose/pose_service.py`, or tune env vars in `run_pose.sh`
   (`INFER_HZ`, `CONF`, `IMGSZ`, `MODEL`).
2. `sudo docker stop g1-pose && perception/pose/run_pose.sh` — the new code is
   re-read from your mounted folder (no rebuild needed for `.py` changes).

Docker access requires root; a sudoers rule grants passwordless `sudo docker` for the
`unitree` user (`/etc/sudoers.d/unitree-docker`).

---

## Key files

| Path | Purpose |
|---|---|
| `scripts/robot_web_controller.py` | FastAPI server: control WS, camera/pose/lidar routes |
| `scripts/camera_service.py` | Separate camera grabber → `/dev/shm/g1_camera.jpg` |
| `scripts/camera_source.py` | Reads the camera/pose JPEG from shared memory |
| `scripts/lidar_source.py`, `map_builder.py` | LiDAR cloud + mapping |
| `config/robot.yaml` | Tuning: port, speed caps, stream rates |
| `web/` | Frontend (`index.html`, `controller.js`, `camera.js`, `pose.js`, `lidar.js`, `style.css`) |
| `perception/pose/` | People-pose service, Docker setup, models |
| `perception/detect/` | Object-detection service (NanoOWL+NanoSAM, open-vocab), Docker setup, models |

---

## Known issues

- **Only one camera consumer allowed.** Don't start a second `camera_service.py` or a
  second dashboard instance — both would fight the single-consumer videohub. Kill stray
  `camera_service.py` processes before starting a fresh dashboard.
