# G1 Mid-360 + D435i Fusion Roadmap — Fine Small-Obstacle Detection

Goal: detect fine, small obstacles everywhere — especially on the ground and inside the Mid-360 near-front blind wedge (~1 m) — while staying real-time (~10 Hz) on the Jetson Orin NX (shared with the YOLO pose service; pure-numpy CPU pipeline).

## Guiding architecture decision

Keep the existing **polar log-odds occupancy ring (`occupancy.py`) as the shared fusion substrate** and do **late fusion**: two per-sensor inverse sensor models summed in log-odds. Reasons: `occupancy.py` already implements log-odds + tau-decay + VFH+ hysteresis and its `update()` already accepts a `covered_mask`, so per-source/blind-wedge gating is a natural drop-in; adding log-odds IS the correct recursive-Bayes update for independent sensors. **Reject early point-merge** (it biases the single robust plane fit — the exact floor-drag failure `_fit_floor` guards against) and **defer GPU substrates** (elevation_mapping_cupy / nvblox / STVL) — they are a framework rewrite (Isaac ROS/Humble/Jazzy or Nav2 costmap this stack lacks) and contend for the single Orin GPU already held by pose inference. A small **cartesian near-front elevation patch** is the only non-polar piece worth piloting later.

The `/dev/video0` single-open constraint is the central plumbing fact: the dashboard `LidarSource` (domain 0) is the sole camera owner; `obstacle_node` is domain 99. The plan therefore lands the near-front win **guard-side first** (no cross-domain work), then moves to true perception-side cloud fusion via a one-way camera-cloud bridge.

---

## Phase 0 — Calibration & frame prerequisites (gate everything; do FIRST)

Nothing downstream is trustworthy until the depth frame is correct — a wrong pitch/height false-stops or silently misses (why `depth_fusion.enabled=false` today).

- **P0.1 Fix the pitch at the source (`scripts/lidar_source.py::_depth_to_cloud`, :177).** Today it bakes only `MOUNT_HEIGHT=1.3` into z and omits the ~47.6° down mount pitch; every consumer (viz, mapper, `realsense_cloud_pub`) sees a pitch-uncorrected cloud and only `DepthNearField._compute` undoes it in-place. Apply the pitch rotation once here so all consumers get a floor-referenced cloud, and delete the derotation from `depth_nearfield.py`. Single ground model, single source of truth.
- **P0.2 Measure & validate camera→base extrinsic.** Confirm `pitch_deg` (URDF `d435_joint` 0.83078 rad ≈ 47.6°) and `camera_height_m` (yaml 1.30, note says measure ~1.27) against the real robot. This camera→base transform is the ONLY calibration the near-front blind-wedge fill needs, because the camera is the *sole* source there — do not block on lidar↔camera overlap.
- **P0.3 Lidar↔camera extrinsic (only for the overlap band).** Use `koide3/direct_visual_lidar_calibration` (ROS2-friendly), initialized from the URDF TF prior. Budget a target/checkerboard fallback: the rig has *marginal* FOV overlap (Mid-360 looks up, D435i looks down; they co-observe only mid-range vertical edges), so targetless edge self-calib may converge poorly. This gates only Phase 2 overlap-band fusion quality, not Phase 1.
- **P0.4 Fix `DEPTH_SCALE` coupling note.** If P4 depth-unit tuning (100 µm) is ever applied, `DEPTH_SCALE=0.001` in `lidar_source.py` must become `0.0001` or depths go 10× wrong. Flag now to avoid a later silent break.

---

## Phase 1 — Quick wins (pure software, modules already exist & unit-tested)

### QW1. Wire per-cell elevation ground segmentation into the node (`ground.py` → `obstacle_node.py`)
**This is the single dominant cap on small ground obstacles.** The node's inline `_fit_floor`/`_ground_mask` fits ONE global tilted plane with a ±0.12 m inlier band — it absorbs residual floor structure up to ~12 cm and cannot follow undulation/ramps/curbs, so the effective 8 cm clearance varies across the scene. `ground.py::segment_obstacles_elevation` (cell=0.20, ground_clearance=0.08, min_cell_pts=3) already localizes the floor **per 20 cm cell** and is purpose-built for small ground objects — but is unwired.
- Replace the inline `_ground_mask`/`_fit_floor` call in `_on_cloud` (~:384/:686) with `ground.segment_obstacles`/`segment_obstacles_elevation`, feeding the gravity vector already available from deskew/IMU.
- Once per-cell floor is local, `ground_clearance_m` can be lowered toward the sensor noise floor (see Phase 4 physics limit) without floor-drag false positives.
- Effort: low-medium. Impact: high. No new deps.

### QW2. Replace the flat inline ring with the unused range-scaled filters (`filters.py` → `obstacle_node.py`)
The wired ring uses a flat `ring_min_points=3` with no range scaling and no outlier cleaning, so a small/thin object giving 1–2 pts/5°-sector is dropped unless within the 1.6 m tripwire. `filters.py` already implements (and unit-tests) the fix, unused:
- Pre-clean with `dror_filter` (range-scaled radius-outlier removal) to strip lone fliers so thresholds can be lowered safely.
- Swap the inline per-sector logic in `_ring_distances` (~:456/:560) for `sector_kth_nearest` + `sector_tripwire` with **range-scaled required-points** (keep sparse far/thin hazards, reject noise), merged via the min-merge helper.
- Effort: low. Impact: medium-high (small/thin object retention, fewer conservative drops).

### QW3. Range-graded decimation instead of blind stride (`filters.py::custommsg_to_numpy` path)
The CustomMsg path does `decimate = point_num//6000` — blind stride that thins an already-sparse small object further, below the ring/tripwire/cluster thresholds. There is no voxel step, so this is a cheap fix: **keep-all near / thin far under a total point budget** (density-graded stride, or a coarse near-voxel), so near-ground small-object density is preserved. Must be paired with QW1 (raw near points amplify floor noise without the per-cell ground fit).
- Effort: low. Impact: medium.

### QW4. Turn depth ON and upgrade `DepthNearField` to a per-sector ring merged into the guard ring (guard-side, Option B)
Fastest path to fill the near-front blind wedge across **all** forward directions with **zero cross-domain plumbing** (camera stays in domain 0). Today only a single forward scalar (`kth=4` inside a ±0.35 m corridor) is min()-merged on the forward axis only (`guard.apply` :495–514).
- After P0.1/P0.2, set `depth_fusion.enabled=true` (validated).
- Upgrade `scripts/depth_nearfield.py::_compute` to emit a **per-sector ring** in the same `{n, bin_deg, start_deg, dist[]}` schema the guard already understands, over the forward-facing sectors the camera covers (± corridor widened to the D435i horizontal FOV).
- Min-merge it into the guard ring path (`_ring_cone_min` :705 / `apply()` ring block :520–564) instead of only `_stop_check("fwd")`.
- Add DBSCAN/small-min-cluster + register on the cropped, ground-removed depth band (upgrade from kth-nearest) so a small object surfaces as an entity, not a corridor scalar.
- Effort: low-medium. Impact: high (directly closes the stated primary goal). Two ring builders temporarily coexist — acceptable bridge to Phase 2.
- **Safety rule (ship with it):** a depth *hole* (invalid/missing return on dark/specular/low-texture floor) must NEVER emit "clear/free." Treat invalid depth as UNKNOWN and reject implausible far spikes below the expected-floor depth (cheap given known extrinsic). This is the most safety-relevant single item.

---

## Phase 2 — Perception-side fusion (the "right place"): one shared cloud/ground/ring/occupancy

Promote fusion from the guard forward-min into `obstacle_node._on_cloud` so both sensors flow through the SAME ground seg, ring, occupancy, wedges, and shm contract — giving fine near-ground detail across the whole ring, not just forward.

- **P2.1 Camera-cloud bridge (respects single-open).** Keep the dashboard as sole `/dev/video0` owner; have `LidarSource` publish its **pitch-corrected** decoded cloud (P0.1) to a one-way transport into domain 99 — a `/dev/shm/g1_camera_cloud` buffer (mtime-poll, mirroring the existing `g1_obstacle.json` contract) or a DDS topic. Avoids the mutually-exclusive `realsense_cloud_pub` second-open. Attach a timestamp for sync.
- **P2.2 Per-source downsample before merge.** The occupancy tuning (`occ_l_high≈2 returns/sector`) assumes sparse lidar; a dense depth carpet would flood sectors. Voxel-downsample the D435i cloud keyed to detection resolution before concatenation into the `(N,3)` array (right after de-skew + finite filter, ~:337–402).
- **P2.3 Shared ground handling.** Route the fused near-ground points through the QW1 elevation grid (or a camera-specific ground cut for the 0.5–2 m band). One ground model, not two.
- **P2.4 Per-sensor inverse sensor models + late log-odds fusion (`occupancy.py::update`).** D435i triangulation error grows ~z² (textbook): down-weight its log-odds beyond ~2 m where the up-tilted lidar takes over, up-weight it in the 0.5–2 m near band the lidar cannot see; Mid-360 stays sharp/far. Sum log-odds with **disjoint-sector / blind-wedge gating via `covered_mask`** to avoid double-counting in the overlap wedge (independence only holds on disjoint coverage). Optional per-region lower occupied threshold for the near-front. Supereight2-style additive `w=min(w+1,wmax)` clamp bounds confidence cheaply.
- Effort: medium-high. Impact: high. Requires P0.3 extrinsic validated.

---

## Phase 3 — Pilots (new capabilities; gated on Phase 1–2 landing)

### P3.1 Negative-obstacle detection (drop-offs / descending stairs / curb edges) — genuinely missing
The pipeline has ZERO negative-obstacle detection; a descending stair or curb drop-off is a real fall hazard the positive stop-ring can never see, and the D435i at ~47° down is near-ideal geometry (reference used 40°). Add a geometric gap + point-spacing detector on the depth band (numpy, no GPU). Risks: IR dropouts on dark/specular floor read as phantom holes → false stop, so require multi-frame confirmation and careful rim/shadow thresholds; indoor/humanoid-head transfer is unproven. Pilot after P0 frame is validated. Effort: medium. Impact: high (new safety category).

### P3.2 Motion-compensated temporal accumulation + lower `min_cluster_points`
`obstacle_node` already has rolling accumulation (`viz_accum_s=0.6`) + `deskew.py`. The new part is consuming a real SE(3) pose (`rt/odommodestate` leg-odom, or FAST-LIO on demand) to add **translation** compensation (currently rotation-only; `deskew_full` unused), enabling a longer window and a lower `min_cluster_points` with per-point timestamps — and, crucially, **carrying an object seen at 0.5–1 m forward into the front blind wedge under the robot** (the fixed robot-frame grid only fades, it does not transport). Risks: leg-odom is noisy in z/roll/pitch on a bouncing torso; needs a ghost/occlusion filter or walking people accumulate into phantom walls → false stops; the yaml already notes 1.0 s over-accumulated sway and "boxed the robot in," so the useful window is near its ceiling. Pilot with pose-reuse + a cheap newest-point ghost filter. Effort: medium. Impact: medium.

### P3.3 Cartesian near-front elevation patch + height-residual persistence
Only if Phase 2's polar grid proves too coarse for the tiny near-front wedge: a small 2–3 cm cartesian per-cell max-height-above-ground patch (reusing the ground substrate) with log-odds persistence, instead of `min_cluster_points`. Order-dependent (needs the fused grid + a reliable per-cell ground ref) and low value standalone. Full per-ray visibility cleanup is expensive on CPU numpy — keep the region tiny. Pilot. Effort: medium. Impact: medium.

---

## Phase 4 — Config/tuning pilots & deferred heavy items

- **P4.1 Patchwork++ ground backend.** `ground.py` already scaffolds `pypatchworkpp` as an auto-backend with elevation-grid fallback — adoption is `pip install pypatchworkpp` (aarch64 build) + validate on Mid-360 scans (~18 ms/scan CPU, fits budget) and re-tune uprightness/sensor-height for the 1.2 m up-tilted close-range Livox (automotive-tuned defaults won't transfer). Improves ground/non-ground separation only; does NOT fill the blind wedge. Effort: low. Impact: medium.
- **P4.2 D435i librealsense boot-JSON tuning.** High Accuracy preset (collision robots), modest disparity shift (50–100, for down-pointing cameras), 100 µm depth units for the near band — applied once at boot via SDK/JSON (persists in HW until power-cycle), then V4L2 takes over. Requires librealsense installed + a boot step (NOT pure config) and the P0.4 `DEPTH_SCALE` fix. Validate High Accuracy's sparser return doesn't starve small-object clusters. Effort: medium. Impact: medium. Pilot.
- **P4.3 Livox driver / DDS receive-buffer tuning.** CustomMsg 5 Hz cap is a real driver bug; kernel/CycloneDDS receive-buffer tuning unlocks 20–50 Hz — but only worth it once P3.2 accumulation actually consumes the extra sweeps. Low impact enabler. Effort: low.
- **Deferred (confirm GPU budget first):** `elevation_mapping_cupy` and `nvblox` ESDF. Both are SOTA for legged robots but are framework rewrites off the numpy CPU stack (ROS2 community port / Isaac ROS on Foxy→Humble migration) and contend for the single Orin NX GPU already running pose inference. Revisit only if the stack moves to a full GPU perception subsystem.

---

## Explicitly dropped (with reason)

- **Early point-level fusion** — biases the single robust plane fit (floor-drag); use late log-odds fusion instead.
- **STVL / OpenVDB** — a 3D voxel *costmap for Nav2*; this stack has no Nav2/costmap. Same blind-wedge win comes free from late fusion on the existing ring.
- **nvblox GPU TSDF/ESDF** — Isaac ROS + Humble/Jazzy vs current Foxy = version+framework migration; GPU contention; 1 cm figure is AGX-optimistic and still misses sub-cm.
- **GroundGrid** — redundant with `ground.py`'s per-cell elevation grid; C++/ROS2-Jazzy foreign node; authors flag it weak on sparse clouds + small obstacles (the exact target).
- **C-ARC** — no Euclidean clustering stage exists to replace (pipeline is kth-nearest-per-sector); new single-threaded C++17 dep across the numpy boundary for marginal reactive-guard gain.
- **U/V-depth histogram** — weak geometric fit at 47° down (floor fills/compresses the frame); duplicates the plane-fit residual with no compute saving (occupancy already <1 ms).
- **LV-DOT** — dynamic *people* tracking (off-goal: goal is static ground clutter + blind wedge); ROS1 Noetic + YOLOv11; duplicates the existing pose detector.
- **RFNet learned semantic segmentation** — GPU model on a contended Jetson; needs G1-viewpoint labels/fine-tune; pitched at sub-10 cm yet D435i noise is exactly what bounds sub-10 cm. Defer to a later semantic layer.
- **Multi-res ROG-Map grid** — a performance optimization, premature before a basic uniform fusion grid exists; near-front wedge is tiny so uniform 2–3 cm is already cheap.
- **Asymmetric log-odds clamp / hit-count persistence tuning** — already shipped in `occupancy.py` (clamps [-2.0,+3.5]; `occ_l_high=1.5` already = ~2-return persistence). Not a new win.

---

## Physics ceiling (set expectations)

D435i active-stereo floor RMS at 0.5–2 m on a 47° tilt is ~1–3 cm, so **sub-~2 cm debris is physically inseparable from floor noise** regardless of algorithm — set `ground_clearance` against MEASURED floor RMS, not aspiration. Mid-360 datasheet: 0.1–1 m thin/low-reflectivity "detection cannot be guaranteed," 0.1–0.2 m "reference only" — the strongest independent argument for the D435i carrying the near band. Realistic reliable small-object floor is ~sub-10 cm with the fused geometric stack; smaller needs temporal averaging and is best-effort.

## Expected wins by phase

- **Phase 0+1:** near-front blind wedge filled across all forward directions; small ground objects down to ~8 cm (and lower where floor is flat) retained instead of absorbed into a global plane; sparse thin/far hazards kept; fewer conservative drops. This delivers most of the stated goal.
- **Phase 2:** fine near-ground detail across the whole 360 ring (not just forward), with per-sensor noise-aware confidence and persistence — the true "fine small obstacles everywhere."
- **Phase 3:** drop-off/stair fall-hazard coverage (new category) + blind-wedge memory under motion.

## Risks / cross-cutting

- Wrong depth frame (pitch/height) is the top risk — false-stops or silent misses; P0 gates everything.
- Marginal lidar↔camera FOV overlap may defeat targetless self-calibration — budget a target-based fallback; but Phase 1 depends only on camera→base, not overlap.
- Dense depth flooding sparse-tuned occupancy — per-source downsample + ISM weighting mandatory before any merge.
- Cross-domain (0↔99) transport + no timestamp alignment/de-skew on the depth cloud — the shm/DDS bridge must carry timestamps.
- Depth holes on dark/specular floor = catastrophic "hole==free" — the UNKNOWN rule (QW4) is non-negotiable.
- GPU contention with the pose service — the reason CPU polar late fusion stays the backbone and GPU substrates are deferred.

## Sources

- Patchwork++ (KAIST, IROS'22): https://github.com/url-kaist/patchwork-plusplus
- Targetless LiDAR–camera extrinsic calibration (koide3, ROS2): https://github.com/koide3/direct_visual_lidar_calibration
- Livox–camera calibration (HKU-MARS): https://github.com/hku-mars/livox_camera_calib
- elevation_mapping_cupy (ETH RSL): https://github.com/leggedrobotics/elevation_mapping_cupy
- nvblox (NVIDIA Isaac ROS): https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox
- STVL (spatio-temporal voxel layer): https://github.com/SteveMacenski/spatio_temporal_voxel_layer
- Intel RealSense depth quality / presets / disparity shift: https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance and https://dev.intelrealsense.com/docs/depth-post-processing
- Livox Mid-360 spec (near-range/low-reflectivity limits): https://www.livoxtech.com/mid-360
- livox_ros_driver2 CustomMsg 5 Hz cap: https://github.com/Livox-SDK/livox_ros_driver2/issues/187


---

## Top changes to implement first

| # | Change | Files | Effort | Impact |
|---|--------|-------|--------|--------|
| 1 | Wire per-cell elevation ground segmentation into the node | obstacle/obstacle_node.py, obstacle/ground.py | medium | high |
| 2 | Validate depth frame, fix pitch at source, and merge a per-sector depth ring into the guard | scripts/lidar_source.py, scripts/depth_nearfield.py, obstacle/guard.py, obstacle/obstacle.yaml | medium | high |
| 3 | Replace the flat inline ring with the unused range-scaled filters | obstacle/obstacle_node.py, obstacle/filters.py | low | high |
| 4 | Range-graded decimation instead of blind stride | obstacle/filters.py, obstacle/obstacle_node.py | low | medium |
| 5 | Establish and validate the camera->base and lidar<->camera extrinsics | obstacle/obstacle.yaml, scripts/lidar_source.py | medium | high |

### Details

**1. Wire per-cell elevation ground segmentation into the node** (effort medium, impact high)

- Files: `obstacle/obstacle_node.py`, `obstacle/ground.py`
- Change: Replace the inline single-global-plane _ground_mask/_fit_floor call in _on_cloud (~:384/:686) with ground.segment_obstacles / segment_obstacles_elevation (cell=0.20, ground_clearance=0.08, min_cell_pts=3), passing the gravity vector already produced by deskew/IMU. The per-cell 20 cm floor localizes the ground so small objects are no longer absorbed into a tilted global plane with a 12 cm inlier band.
- Why: This is the dominant cap on small ground obstacles: the global plane + 0.12 m band absorbs residual floor up to ~12 cm and cannot follow undulation/ramps. The better per-cell grid already exists, is unit-tested, and is simply unwired. Also lets ground_clearance drop toward the sensor noise floor safely.

**2. Validate depth frame, fix pitch at source, and merge a per-sector depth ring into the guard** (effort medium, impact high)

- Files: `scripts/lidar_source.py`, `scripts/depth_nearfield.py`, `obstacle/guard.py`, `obstacle/obstacle.yaml`
- Change: In lidar_source.py::_depth_to_cloud (:177) apply the ~47.6 deg mount pitch (currently omitted; only MOUNT_HEIGHT baked) so every consumer gets a floor-referenced cloud; remove the derotation from depth_nearfield. Measure/confirm pitch_deg and camera_height_m, set depth_fusion.enabled=true. Upgrade DepthNearField._compute from a single forward scalar to a per-sector ring in the guard's {n,bin_deg,start_deg,dist[]} schema over the camera FOV, with DBSCAN small-min-cluster, and min-merge it into the guard ring path (_ring_cone_min :705 / apply ring block :520-564) instead of only _stop_check('fwd'). Add the invalid-depth=UNKNOWN rule so a depth hole never emits clear.
- Why: Fastest path to fill the ~1 m near-front blind wedge across all forward directions with zero cross-domain plumbing (camera stays domain 0). Fixing pitch at the source gives one correct cloud for all consumers. The UNKNOWN rule prevents the catastrophic hole==free false-clear on specular/dark floor.

**3. Replace the flat inline ring with the unused range-scaled filters** (effort low, impact high)

- Files: `obstacle/obstacle_node.py`, `obstacle/filters.py`
- Change: Pre-clean the cloud with filters.dror_filter, then swap the inline _ring_distances logic (~:456/:560) for filters.sector_kth_nearest + sector_tripwire using range-scaled required-points, min-merged. Replaces the flat ring_min_points=3 with a density-aware count that keeps sparse far/thin hazards while rejecting lone fliers.
- Why: A small/thin object giving 1-2 pts per 5 deg sector is currently dropped unless within the 1.6 m tripwire. These functions are already implemented and unit-tested but unused; wiring them retains fine objects and removes noise so thresholds can be lowered.

**4. Range-graded decimation instead of blind stride** (effort low, impact medium)

- Files: `obstacle/filters.py`, `obstacle/obstacle_node.py`
- Change: Replace the blind decimate = point_num//6000 stride in the CustomMsg->numpy path with density-graded decimation (keep-all near, thin far) or a coarse near-voxel under a fixed point budget, so near-ground small-object density is preserved. Pair with the elevation-grid ground fix so raw near points don't amplify floor noise.
- Why: Blind stride thins an already-sparse small object below the ring/tripwire/cluster thresholds precisely in the near band that matters most; density-grading preserves the small-object point budget while staying within the ~10 Hz Jetson limit.

**5. Establish and validate the camera->base and lidar<->camera extrinsics** (effort medium, impact high)

- Files: `obstacle/obstacle.yaml`, `scripts/lidar_source.py`
- Change: Measure camera pitch/height on the real robot (gates all depth use). Run koide3 direct_visual_lidar_calibration initialized from the URDF d435_joint (0.83078 rad) TF prior for the lidar<->camera extrinsic; budget a checkerboard fallback because the up/down mounts have only marginal mid-range FOV overlap. Record the DEPTH_SCALE coupling (1mm->100um would need 0.001->0.0001) before any depth-unit tuning.
- Why: A wrong extrinsic makes fused output worse than lidar-alone (false-stops or silent misses) - it is the mandatory prerequisite for every fusion step. Camera->base alone unblocks Phase 1 (camera is the sole near-front source); lidar<->camera gates only Phase 2 overlap-band quality.


---
_Generated by 15-agent research workflow (map→research→verify→synthesize), 2026-07-02._
