# Motion Record → Recreate — ON-DEVICE plan (all on the G1's Jetson Orin NX 16GB)

**Goal:** person stands in front of the G1 → **Record** → person moves → **Pause** → **Recreate** →
the robot reproduces the motion. Own dashboard **Motion** tab with a **clean camera feed**
(same `/camera/stream` as the main view, NO object-detection overlay). **Sim-first** (MuJoCo), real robot later.

**Pipeline (revised for on-device):**
`clip.mp4 → ROMP (body SMPL) → GMR (retarget to 29-DOF G1 joints) → SONIC (whole-body tracking) → MuJoCo G1`
— **everything runs on the robot's Orin NX. No workstation.**

---

## ⚠️ Read first — on-device reality (verified 2026-07)

- **Runs entirely on the Orin NX 16GB.** GMR + SONIC are Jetson-ready — SONIC is *literally* NVIDIA's official G1-onboard deploy target (JetPack 6 C++ TensorRT stack builds the engine on-device from a ~91 MB ONNX). GMR is CPU Python with native aarch64 wheels.
- **GVHMR is replaced by ROMP** (a lighter body-SMPL estimator). GVHMR can't run on ARM (x86-pinned, 8–12 GB, zero Jetson evidence). **Tradeoff:** ROMP is camera-relative, so it loses GVHMR's world-grounded global trajectory → **spins / turns / root translation are rougher.** Mitigate with (a) the **static camera** (robot stands still → GVHMR's camera-motion compensation isn't needed) + (b) **temporal smoothing of root orientation**; IMU/odometry fusion is the stretch option if spins are too jittery. Fine for the stated bar: *overall move with rotation, not finger/face detail.*
- **Body-only SMPL is enough** — no fingers/face. GMR → SONIC is body-only 29-DOF anyway.
- **29-DOF, sim-first.** GMR/SONIC ship only 29-DOF; the real robot is 23-DOF → real-robot "Recreate" is deferred (needs a custom 23-DOF model+policy). Deliverable = the sim loop.
- **Memory is the tight constraint** (16 GB unified, shared with your live camera+pose+hands+detect). **During Recreate, STOP the live perception containers** to free RAM, then restart them after. (Ignore NVIDIA's "10–12 GB savings" marketing number — it was refuted.)
- **Setup landmine:** SONIC needs **TensorRT 10.7**, but stock JetPack 6.2 ships **10.3** → you must install 10.7. A version mismatch produces *wrong* motion (dangerous on the real robot; sim-first protects you).

## License note (fine for internal research)
ROMP = Apache-2.0/BSD-ish (check repo) · SMPL body model = MPI non-commercial (register/download) · GMR = MIT · SONIC = Apache-2.0 code + NVIDIA Open Model License weights. No AGPL YOLO (that was a GVHMR dependency — gone).

---

## Build order (each phase is a standalone checkpoint — don't proceed until it passes)
**Phase 1 (record/tab/stub, no ML — buildable now) → Phase 0b (JetPack6/TensorRT10.7) → spike SONIC's own examples → Phase 2 pose → Phase 3 GMR → Phase 4 SONIC → Phase 5 wire it up.**

### Phase 0 — repo + on-device base
- `motion/` folder (this repo). Add third-party code as submodules under `third_party/`: `Arthur151/ROMP`, `YanjieZe/GMR`, `NVlabs/GR00T-WholeBodyControl`.
- **Base (do early, has friction):** confirm **JetPack 6** on the Orin; install **TensorRT 10.7** on top of stock 10.3 (SONIC needs the exact version). Use NVIDIA's L4T PyTorch wheel for anything torch-based.
- Fill `config/pipeline.yaml` (pose estimator=romp, control_hz=50, robot=unitree_g1, dof=29, static_cam=true, "pause these containers during recreate" list, model paths).

### Phase 1 — App / tab / record layer (Jetson, buildable NOW, no ML, ~1–2 days)
Full Record→Pause→Recreate UX against a **stub** replay provider, before any ML. Unchanged by the pose choice.
- `app/recorder.py` — sample `/dev/shm/g1_camera.jpg` → `data/clips/<job>/clip.mp4`. **Measure true fps over wall-clock and stamp it into `clip.json`** (the pose/retarget stages assume a known fps). Detect dropped/dup frames.
- `app/jobs.py` — on-disk job store + state machine (`IDLE→RECORDING→RECORDED→PROCESSING→READY→ERROR`), `status.json`.
- `app/replay.py` — `ReplayProvider` interface + `StubProvider` (Recreate copies clip→`replay.mp4`, marks READY after a fake staged delay). The real pipeline slots in behind this.
- `app/routes.py` — FastAPI `APIRouter`: `POST /motion/record/{start,stop}`, `GET /motion/jobs`, `POST /motion/jobs/<id>/recreate`, `GET /motion/jobs/<id>/status`, `GET /motion/jobs/<id>/replay.mp4`. Mount in `scripts/robot_web_controller.py` next to `/camera/*`; gate Record/Recreate behind the existing `/ws` single-controller lock.
- **Clean feed + tab:** add `<button class="tab" data-tab="motion">Motion</button>` + `<div id="tab-motion" class="tab-panel">` in `web/index.html` (existing `controller.js` tab-switch picks it up free). Feed is `<img src="/camera/stream">` with **no `<canvas>`, no pose/detect toggles** → identical framing, zero overlay. Add `web/motion.js` + `web/motion.css` (reuse `style.css` tokens). One primary button that morphs by state (● Record → ❚❚ Pause → Recreate), a "takes" list, a processing stepper (Pose→GMR→SONIC), a result view (original clip vs replay side by side).
- `sudo systemctl restart g1-web`.
- **✅ Verify:** record yourself, Pause → `clip.mp4` exists, `ffprobe` fps within ~1 of true rate; Recreate walks IDLE→PROCESSING→READY, stub replay plays; Motion feed clean while Drive tab still shows boxes. `pytest tests/test_recorder.py`.

### Phase 2 — Pose: ROMP (Orin) → body SMPL (replaces GVHMR)
- `third_party/ROMP` submodule; install `simple_romp` (needs L4T torch). Download SMPL body model → `models/smpl` (MPI registration).
- `pipeline/stage_pose/run_pose.sh` — run ROMP on `clip.mp4` → per-frame SMPL (pose 24-joint, betas, camera). **Free RAM first:** stop `g1-pose`/`g1-hands`/`g1-detect` for the duration.
- `pipeline/glue/pose_to_smpl.py` — pack ROMP's per-frame output into an AMASS-style SMPL sequence GMR accepts (`global_orient`, `body_pose` 21-joint/63-dim, `betas`, `transl`), at `clip.json.true_fps`. **Apply root-orientation temporal smoothing** here (SLERP low-pass on `global_orient`) to keep spins from flipping.
- **✅ Verify (no GMR yet):** ROMP's own mesh overlay tracks the person; the SMPL sequence loads and length == clip frames; eyeball that a slow turn produces a smoothly rotating root (not per-frame flips).
- **⚠️ Honest note:** ROMP's monocular **root translation** (walking across the floor) is weak/approximate, and fast spins/back-turned frames will be the roughest part. If unacceptable, that's when you add IMU/odom fusion for global yaw.

### Phase 3 — GMR (Orin, CPU): SMPL → 29-DOF G1 joints
- `third_party/GMR` submodule; `pip install -e` (mujoco, mink, smplx — all aarch64 wheels).
- `pipeline/stage_gmr/run_gmr.sh` — `scripts/smplx_to_robot.py` fed the Phase-2 SMPL sequence, `--robot unitree_g1 --save_path gmr.pkl`. Output `.pkl`: `root_pos (T,3)` m, `root_rot (T,4)` **XYZW**, `dof_pos (T,29)` radians (MJCF order), fps.
- Pass the true fps through (GMR defaults to 30 — set `--tgt_fps`/patch the loader). GMR applies a Y-up→Z-up fix — **sanity-check the person isn't lying sideways.**
- **✅ Verify:** GMR's `robot_motion_viewer.py` plays `gmr.pkl` on the 29-DOF G1 in MuJoCo — recognizable, upright, right speed. This validates pose→retarget before the hard SONIC seam.

### Phase 4 — SONIC (Orin): track the reference in sim (hardest)
**Derisk first, before writing the converter:**
- `third_party/GR00T-WholeBodyControl`; get `model_encoder.onnx` + `model_decoder.onnx` + `observation_config.yaml`. Build `gear_sonic_deploy` (needs the TensorRT 10.7 from Phase 0).
- **① Spike SONIC's OWN example motions** through the on-Orin sim2sim loop (`deploy.sh --motion-data <example_dir> sim`). If this doesn't run on your Orin, stop and fix the TensorRT/CUDA setup — nothing downstream matters until it does.

Then the make-or-break glue:
- `pipeline/glue/gmr_to_sonic_csv.py` (+ `joint_maps.py`) — GMR `.pkl` → SONIC's **6-CSV bundle** (all 50 Hz, header rows, equal row counts): `joint_pos/joint_vel.csv [T,29]` radians in **IsaacLab order** (NOT MJCF), `body_pos.csv` m, `body_quat.csv` **WXYZ** (docs: "critical"), `metadata.txt`. Steps: reindex MJCF→IsaacLab (pinned 29-name table — issue #78 is a silent-corruption trap); quat **XYZW→WXYZ**; resample→exactly 50 Hz (SLERP/linear); finite-diff velocities at dt=0.02 s; FK for body arrays (start root-only, verify empirically).
- `pipeline/stage_sonic/run_sonic.sh` — `deploy.sh --motion-data <csv_dir> sim`.
- **✅ Verify:** SONIC tracks your CSVs and the sim G1 reproduces the motion; `pytest tests/test_gmr_to_sonic_csv.py` (row parity, 50 Hz spacing, vel finite-diff, headers, metadata).

### Phase 5 — wire the real replay + view (all local — no network)
- `sim/sim_to_feed.py` — render MuJoCo (offscreen/EGL on the Orin) → `replay.mp4` OR live frames into `/dev/shm/g1_motion_sim.jpg`, served like `/camera/stream` via a new `/motion/sim/stream` route.
- Swap `StubProvider` → `SonicProvider` = `pipeline/run_pipeline.sh` (pause perception → ROMP → GMR → SONIC → render), updating `status.json` per stage, restart perception at the end.
- **Fallback:** if SONIC can't cleanly track the ROMP-derived reference, `SonicProvider` renders **GMR's kinematic playback** as the Recreate output — the feature still ships.
- **✅ Verify:** press Recreate → clean feed pauses/records → pipeline runs on-device → `replay.mp4` plays in the Motion tab beside the original.

### Phase 6 — real robot (LATER, blocked on 23-DOF work)
Not attempted until sim is solid AND a 23-DOF G1 model+policy exists. Open space, spotter, e-stop, short clip first. Double-check the TensorRT version (a mismatch = wrong motion on real hardware).

---

## Gotchas checklist
- **RAM:** free the 16 GB during Recreate — stop `g1-pose`/`g1-hands`/`g1-detect`, restart after.
- **TensorRT:** SONIC needs 10.7; stock JetPack 6.2 = 10.3. Install 10.7 or you get wrong motion.
- **Rotation:** ROMP is camera-relative → smooth root orientation; expect rougher spins than GVHMR.
- **Root translation:** monocular = approximate; don't expect precise walking-across-the-floor.
- **fps:** measure true fps at record → carry through pose+GMR (both default to 30) → resample to 50 Hz for SONIC.
- **Joint order:** GMR MJCF ≠ SONIC IsaacLab (issue #78, silent). Pin the table, validate visually.
- **Quaternions:** GMR `.pkl` = XYZW; SONIC = WXYZ. Convert once.
- **Up-axis:** ROMP/GMR → assert the person is upright, not sideways.
- **DOF:** everything 29-DOF for sim; 23-DOF hardware is a separate deferred project.
```
