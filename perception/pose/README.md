# G1 people-pose (skeletons + manual labeling)

Draws YOLO11-pose stick-figure skeletons on people in the head-camera feed, with
stable per-person ByteTrack IDs the operator can name from the dashboard. Surfaced
as a **toggle** on the existing camera panel.

## How it fits in (mirrors the camera pipeline)

```
camera_service.py (host) --> /dev/shm/g1_camera.jpg --read--> pose_service.py (docker)
                                                               YOLO11-pose .track(persist=True)
                                                               result.plot() + name labels
   /dev/shm/g1_pose.jpg        <-- annotated JPEG  (served at /camera/pose/stream)
   /dev/shm/g1_pose_tracks.json<-- [{id,name,cx,cy}] (served at /camera/pose/tracks)
   /dev/shm/g1_pose_labels.json--> {"<id>":"name"}  (written by POST /camera/pose/label)
   /dev/shm/g1_pose_demand     --> heartbeat; pose only infers while someone is watching
```

The pose process runs **inside** `ultralytics/ultralytics:latest-jetson-jetpack5`
(JetPack 5.1.1 / CUDA 11.4). The host web server gains no new Python deps.

## Files
- `pose_service.py` — the service (runs in the container).
- `smoke_test.py` — Phase-1 GPU/model check on one frame.
- `run_pose.sh` — `docker run` wrapper (env-configurable).
- `g1-pose.service` — systemd unit (staged; install after validation).
- `models/` — cached `yolo11n-pose.pt` and exported `.engine`.

## Run / stop (manual)
```bash
# manual run (always-on, ignores demand-gating, for testing):
ALWAYS_ON=1 /home/unitree/perception/pose/run_pose.sh
# stop:
sudo docker stop g1-pose
```

## Tuning (env vars for run_pose.sh)
- `MODEL` — `yolo11n-pose.pt` (default) → `yolo11n-pose.engine` after TensorRT export.
- `INFER_HZ` — inference cap (default 12); lower if it competes with control.
- `CONF` — confidence threshold (default 0.5).
- `IMGSZ` — inference size (default 640); 480 is faster.

## Dashboard
Toggle button on the camera panel swaps the `<img>` between `/camera/stream` (raw)
and `/camera/pose/stream` (skeletons). The "People" panel lists current IDs; typing
a name labels that skeleton until the track is lost.
