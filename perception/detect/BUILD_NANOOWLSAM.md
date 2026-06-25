# NanoOWL + NanoSAM detector (open-vocab labels + segmentation masks)

Adds a third object detector behind the existing `detector.py` seam:
`DETECTOR_IMPL=nanoowlsam`. It runs **NanoOWL** (TensorRT OWL-ViT) for
open-vocabulary labeled boxes, then **NanoSAM** (TensorRT ResNet18 encoder +
MobileSAM decoder) to segment **each box into a mask** — so you get *both* labels
*and* full per-object coverage. The dashboard draws the masks as translucent
polygons under the boxes (no UI toggle change — it's the same "Object Detection"
overlay). YOLO-World stays available as the default/fallback.

- Pipeline: `frame → OWL boxes+labels → SAM mask per box → {cls,conf,box,mask}`.
- The image is encoded **once per frame**; the cheap decoder runs per box, so masks
  cost little beyond OWL itself. OWL dominates the latency budget.
- Optional `SEG_EVERYTHING=1`: a class-agnostic point-grid pass that adds `"object"`
  masks for full coverage. **Costly (seconds/frame)** — a debug/exploration toggle.

Host target: Jetson, **JetPack 5 / L4T r35.3.1, Python 3.8, CUDA 11.4, TensorRT 8.5**.

---

## Why this shape (the load-bearing decisions)

- **Base image = `dustynv/nanosam:r35.3.1`**, NanoOWL layered on top
  (`Dockerfile.nanoowlsam`). That base already ships a known-good
  torch/torch2trt/TensorRT-8.5 stack built for *this exact L4T*, plus the NanoSAM
  package and **prebuilt SAM engines**. Both predictors then live in one process /
  one CUDA context (matters for memory on smaller Orins).
- **Reuse the prebuilt SAM engines** instead of rebuilding the mask decoder. The
  MobileSAM decoder ONNX→engine conversion is documented to **fail on some TRT
  versions** (`IIOneHotLayer cannot be used to compute a shape tensor`). The baked
  r35.3.1 engine avoids that.
- **Engines are mounted, not baked** (they're device/TRT-specific), exactly like the
  YOLO-World CLIP weight is mounted today: from `perception/detect/models/data`.
- **`OWL_THRESHOLD` is separate from `CONF`.** OWL scores run lower (~0.1) than
  YOLO-World's 0.35; reusing `CONF=0.35` would suppress every detection.

---

## Build & run (on the Jetson)

```bash
cd /home/unitree/projects/g1/perception/detect      # the checkout that holds this branch

# 1) Build the image (~7 GB; pulls the dustynv base + layers NanoOWL).
sudo docker build -f Dockerfile.nanoowlsam -t g1-detect-nanoowlsam:latest .

# 2) Populate the three TensorRT engines into models/data (one-time).
#    Extracts the prebuilt SAM engines from the base; builds the OWL engine on-device.
chmod +x prepare_nanoowlsam_engines.sh run_detect_nanoowlsam.sh
./prepare_nanoowlsam_engines.sh

# 3) Run it (REPLACES the YOLO-World g1-detect container — same name/contract).
./run_detect_nanoowlsam.sh
```

Watch the logs for:
```
[detect_service] impl=nanoowlsam model=google/owlvit-base-patch32 ... seg_everything=False ...
[detect_service] warmup done
```
`warmup done` is the key signal: all three engines deserialized and ran on this host.

---

## Verify it works

1. **Service warmup** — `warmup done` in the logs (above). A crash here usually means
   an engine didn't deserialize on this host's TRT (see Troubleshooting).
2. **Geometry has masks** — open the Object Detection overlay in the dashboard, then:
   ```bash
   curl -s localhost:8080/camera/detect/objects | python3 -m json.tool | head -40
   ```
   (adjust port to your web server). Items should look like
   `{"cls":"person","conf":0.34,"box":[x1,y1,x2,y2],"mask":[[x0,y0,...]]}`.
3. **Dashboard render** — toggle **Object Detection**: boxes as before, now with
   translucent orange mask polygons filling each detected object. Hard-refresh the
   browser once (the cache-buster bumped to `cam-overlay.js?v=5`).

---

## Tuning

| Env | Default | Notes |
|-----|---------|-------|
| `DETECT_PROMPTS` | person . door . chair . … | Open-vocab classes; period/comma separated. Order is internal-only. |
| `OWL_THRESHOLD` | `0.1` | Raise to cut false positives, lower for recall. |
| `INFER_HZ` | `6` | OWL-bound. ~8–10 on AGX Orin, ~3–5 on Orin Nano. |
| `SEG_EVERYTHING` | `0` | `1` adds the costly class-agnostic mask pass; drop `INFER_HZ` to ~1–2 when on. |
| `MODEL` | google/owlvit-base-patch32 | OWL-ViT HF id (text encoder + processor). |

Engine paths can be overridden with `OWL_ENGINE`, `SAM_ENCODER`, `SAM_DECODER`.

---

## Known risks / troubleshooting (verify on-device)

- **An engine fails to deserialize** (error on startup / no `warmup done`): the
  prebuilt engine doesn't match this host's TRT patch. Rebuild that engine on-device:
  - SAM encoder: `trtexec --onnx=resnet18_image_encoder.onnx --saveEngine=resnet18_image_encoder.engine --fp16`
  - SAM decoder (FP32, may hit the IIOneHotLayer bug):
    `trtexec --onnx=mobile_sam_mask_decoder.onnx --saveEngine=mobile_sam_mask_decoder.engine --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:1x1x2,point_labels:1x1 --maxShapes=point_coords:1x10x2,point_labels:1x10`
  - OWL encoder: `python3 -m nanoowl.build_image_encoder_engine owl_image_encoder_patch32.engine --onnx_opset=16`
  (ONNX sources: NanoSAM README's gdown links; OWL downloads from HF automatically.)
- **Nothing detected** even with objects in view → `OWL_THRESHOLD` too high, or the
  text encoder weights weren't cached. Lower `OWL_THRESHOLD` to ~0.05 to confirm recall.
- **`transformers` import / OWL-ViT load error** on Py3.8 → the pin in
  `Dockerfile.nanoowlsam` (`transformers==4.41.2`) may need adjusting. 4.42+ dropped
  Python 3.8.
- **OWL engine build hangs / OOM** (8 GB Orin Nano, nanoowl issue #33) → add 8–16 GB
  swap and disable zram before `prepare_nanoowlsam_engines.sh`, or extract a baked
  engine from `dustynv/nanoowl:r35.3.1` instead.
- **`set_image` channel order** → the detector converts BGR→RGB before NanoSAM/NanoOWL;
  a missing swap silently degrades masks. Don't remove the `cv2.cvtColor` call.
- **Disk** → the image is ~7 GB; confirm headroom (`df -h /`).

---

## Revert to YOLO-World

```bash
cd /home/unitree/projects/g1/perception/detect
./run_detect.sh        # rebuilds the g1-detect container with the YOLO-World detector
```
No dashboard change needed — YOLO-World items just omit `mask`, and the canvas skips
the fill pass for them.
