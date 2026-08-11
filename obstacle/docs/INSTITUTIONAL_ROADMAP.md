# G1 Obstacle Perception — Institutional-Grade Accuracy Roadmap

_15-agent research workflow (map→research→verify→synthesize), 2026-07-02. Pass 2 (builds on FUSION_ROADMAP.md)._

# Roadmap: G1 Obstacle Pipeline → Institutional-Grade Fine Detection

**Thesis.** "Institutional accuracy" is a *measured* claim. Today every accuracy assertion (recall, floor RMS, latency, frame time) is qualitative or datasheet-cited. Therefore the first deliverable is a measurement substrate that emits confusion matrices, PR/ROC curves, distance-error, detection-latency and per-stage timing — and gates every later change as a regression bench. Only after that do we spend effort on accuracy upgrades, and we spend it CPU-first (the Orin GPU is owned by pose + NanoOWL + NanoSAM; no CUDA MPS, so any GPU tenant time-multiplexes and can starve the safety-critical pose service).

**Platform budget rule (applies to every item).** obstacle_node (domain 99) → `/dev/shm/g1_obstacle.json` → guard (domain 0) is pure-numpy CPU at ~10 Hz. Keep it that way. Every ADOPT item below is CPU/numpy, zero GPU, cannot contend with pose. Every learned PILOT is gated behind DLA/INT8 + demand-scheduling into the autonomous-walk window (NanoOWL/NanoSAM/pose are demand-gated and idle while walking).

---

## Phase 0 — Measurement Foundation (BLOCKS everything; do first)

All offline, in `scripts/sim_perception.py` + `scripts/perception_harness.py` (which already load the REAL node math via stubbed rclpy) + a new `scripts/perception_bench.py`. Runs on a workstation or the Orin off-shift — never in the reactive hot path, never touches the GPU.

| # | Item | Modules | Effort | Impact |
|---|------|---------|--------|--------|
| 0.1 | **Per-sector ground-truth ring** `_true_ring()` beside existing `_true_clearance` (scalar only today, line 308) + rasterize sim obstacles into the 72-sector polar grid | sim_perception.py | M | High |
| 0.2 | **Stratified confusion matrix + PR curves**, binned by size × range × azimuth × material; COCO AP_S / Waymo distance-bin protocol on the ring | new perception_bench.py | M | High |
| 0.3 | **Detection-latency / reaction-margin** = TTC-at-first-detection − brake time; add per-obstacle first-detection frame (today `run()` keeps only a `detected_any` bool OR, line 252) — deterministic via existing `seed=` param | sim_perception.py | M | **High** |
| 0.4 | **False-stop vs miss ROC**, operating-point selection by sweeping `occ_l_high` + guard `stop_at`; report **per-scenario** (NOT per-hour — 8–24 s adversarial courses are not a duty cycle) | perception_bench.py | S | High |
| 0.5 | **Hoss TP/FP/FN checklist + FOV/area-of-vision coverage mask** in obstacle.yaml, scored before metrics; MUST ship with an explicit coverage-gap report (fraction of hazard space unmonitored: sides/rear-low, at-feet <0.45 m, negatives) so masking cannot launder blind spots | obstacle.yaml, perception_bench.py | S | Med |
| 0.6 | **Effort-based deceleration-criticality weighting**: weight each error by the required-deceleration / TTC-reduction it induces on `guard.apply()` (GT-ring cmd vs perceived-ring cmd, both already computable per frame). Right-sized for a velocity governor. | perception_bench.py, guard.py | M | Med |
| 0.7 | **Regression gate**: freeze recall / FP-rate / mean-stop-distance / P50–P99 frame-time as tracked KPIs per commit; wire an autoresearch-style bench loop so every tuning change (l_high, thresholds, accum window) has a quantitative guardrail | perception_bench.py, CI | M | High |

**PILOT (foundation, higher cost):**
- **Real-log GT** via deterministic MCAP rosbag replay through the real node, hand/mocap-labeled on a *small* staged corpus first. This is the sim-to-real anchor and the ceiling for "institutional," but ROS2 deterministic replay is documented-hard and mocap is capital/space. Defer full auto-labeling (SurroundOcc-style multi-frame aggregation) until Phase 0.1–0.6 deliver value; scoring against imperfect pseudo-GT is circular unless the pseudo-GT is itself validated on a hand-labeled subset.
- **Small/negative-obstacle metrics** (SMIYC FPR95 / sIoU_gt / PPV / mean-F1, Lost-and-Found protocol) — the *scoring* is a free reuse of 0.2, but the *data* is the bottleneck: the ray-cast sim models the floor as z=0 with no hole geometry, so negatives can't be rendered until the sim is extended (Phase 4) or real staged logs exist. FPR95 is degenerate while negative recall is 0.0 — pair with component-F1 and the coverage mask.

*Expected measured gain:* none directly — this is the yardstick. Everything below reports its gain against this bench.

---

## Phase 1 — Calibration & Frame Correctness (safety gate; prerequisite for trusting ANY real-data metric)

`depth_fusion.enabled=true` is **currently running on an unvalidated frame** — this is the top open safety item.

| # | Item | Modules | Effort | Impact |
|---|------|---------|--------|--------|
| 1.1 | **Fix D435i pitch at the SOURCE** (roadmap P0.1). `lidar_source._depth_to_cloud` bakes only MOUNT_HEIGHT into z and omits the ~47.6° mount pitch; the pitch is then undone in-place in `depth_nearfield._compute` with hardcoded `pitch_deg=47.6 / camera_height=1.30`. **Fix pitch+height at source, delete the in-place derotation**, record a measured camera→base residual before trusting depth. A wrong pitch/height is the stated false-stop / silent-miss risk. | scripts/lidar_source.py, scripts/depth_nearfield.py | S | **Critical** |
| 1.2 | **Target-based Mid-360↔D435i extrinsic** (ACSC / lvt2calib, non-repetitive-Livox-aware) in the thin ~0.5–2 m forward-low overlap band + D435i on-chip depth self-cal + RGB intrinsic refit. Replaces `depth_nearfield`'s hand-set 2-DoF (pitch+height, assumes zero roll/yaw/lateral). A 1–2° unmodeled roll/yaw shifts a min-merged depth sector a full 5° at 1–2 m → false brake or wrong-heading assignment. Offline, dockerized ROS1, zero runtime cost. | depth_nearfield.py, config | M | High |
| 1.3 | **Hardware time-sync**: Mid-360 PTPv2 slave to host master + D435i `RS2_OPTION_GLOBAL_TIME_ENABLED`, then ApproximateTime for association only. Config-only, no new HW, no GPU. (Gain is *medium* not high here: intra-lidar sync is already solved in deskew via per-point offset_time; camera↔lidar stays coarse ~0.3 s. First confirm the Mid-360 is actually publishing/fused at runtime — `lidar_source.py` suggests it may not be.) | config | S | Med |
| 1.4 | **Online calibration health watchdog**: numpy running residual of lidar-vs-depth range in overlap sectors → raise "calibration degraded" flag → bias guard toward its existing depth-only-lowers fail-safe. Turns silent extrinsic drift (foot-impact/vibration/thermal) into a reportable metric. Threshold needs on-robot tuning. | obstacle_node.py, guard.py | S | High |
| 1.5 | **Feed measured gravity into ground segmentation.** Node calls `segment_obstacles_elevation` on raw sensor-frame z (obstacle_node.py:815) with **no gravity vector**, bypassing ground.py's IMU-accel levelling; uprightness rests on gyro-deskew + assumed z-up. Pass the deskew IMU gravity estimate so the elevation grid is levelled against measured gravity, and feed the floor-normal/height estimate into the health checks. | obstacle_node.py, ground.py | S | Med |

**PILOT:** LI-Init targetless lidar↔IMU (cheap offline sanity, but the lidar+IMU are the same Mid-360 with a factory extrinsic — marginal); `direct_visual_lidar_calibration` (Koide, ROS2-native, named as P0.3) as an *independent second estimate* to cross-check 1.2 — agreement across methods IS the calibration certificate.

**DROP:** GRIL-Calib (redundant — ground.py's per-frame seed+refit floor fit already absorbs the static pitch/z error it targets); iKalibr (SOTA but offline-desktop-heavy, academic overkill vs a checkerboard for a 0.6 m guard).

*Expected measured gain:* removes a whole class of false-brake / wrong-heading errors from the depth ring; makes every subsequent real-data metric trustworthy. Set `ground_clearance` and stop thresholds against the MEASURED floor RMS this phase produces.

---

## Phase 2 — Highest-Leverage Accuracy Upgrades (CPU-only, gated by Phase 0 bench)

| # | Item | Modules | Effort | Impact |
|---|------|---------|--------|--------|
| 2.1 | **SE(2) ego-motion compensation** of the ring + accumulation before temporal fusion — *do this before any tracker*. One SE(2) transform per ~0.1 s frame using Unitree body odom (`OdomReader` → `rt/odommodestate` x/y/yaw, already consumed by step_pacer). Fixes DOGMa "ghost velocity," lets the 0.6 s window transport a memorized obstacle into the under-body front blind wedge instead of only fading it, and lifts the accumulation ceiling (yaml notes 1.0 s "boxed the robot in" precisely because points went geometrically stale). **Caveat:** odom lives in the domain-0 SDK process, not domain-99 node → needs a shm bridge or a 2nd SDK subscriber; legged dead-reckoning drifts → fuse with the already-buffered deskew IMU and keep the window to ~1 frame. | obstacle_node.py, occupancy.py, new odom shm bridge | M | **High** |
| 2.2 | **Heteroscedastic inverse sensor model + per-cell variance channel**. `occupancy.update()` (line 149) takes only `sector_range` and adds FIXED log-odds; `filters.sector_kth_nearest` (obstacle_node.py:655) already returns `_counts` and *discards them*. Extend `update()` to accept per-sector (range, count, incidence) + a precomputed weight LUT (D435i down-weighted ~z² beyond 2 m, up-weighted in the 0.5–2 m near band; Mid-360 sharp/far). <0.1 ms vectorized. Re-tunes the `occ_l_high=1.5` latch. | occupancy.py, obstacle_node.py | M | High |
| 2.3 | **Conformal safety-margin on stop distance**. Convert the hand-tuned 0.6 / 0.30 / 0.35 m magic numbers into a distribution-free quantile margin from logged clearance residuals (`_true_clearance` GT exists). One quantile add in `guard.apply()`. Pilot in sim; **re-calibrate on logged field residuals before claiming any coverage %** (sim guarantee ≠ field guarantee; use Adaptive/Learnable CP for time-correlated residuals; clamp the margin so heavy tails don't balloon into nuisance stops). | guard.py | M | High |
| 2.4 | **Per-sensor fault / degradation / OOD health monitor** gating sensor trust — the most safety-relevant uncertainty item: kills the scariest failure, a blinded Livox / sun-saturated D435i whose *absence of returns* reads as free space. Cheap numpy aggregates → existing shm JSON → guard's existing FAULT state. The **largest-contiguous-depth-hole** metric doubles as a negative-obstacle prior (a floor-region hole where geometry says floor should be). **Livox-aware baselining is mandatory** (non-repetitive scan → point count varies hugely; naive threshold false-trips); D435i fill-rate legitimately collapses on reflective/sunlit floor when healthy. Fold the reframed lidar-vs-depth raw-distance disagreement flag in here. | obstacle_node.py, guard.py | M | High |
| 2.5 | **Post-hoc probability calibration (temperature/Platt) + ECE / reliability diagrams**. Value is the *measurement backbone* (the ECE/PR numbers the goal requires) more than the scalar T. Needs per-cell occ/free GT → add a rasterizer projecting sim obstacles into the polar grid (real work, not "one scalar"); use per-range-band T (couples to 2.2). | occupancy.py, perception_bench.py | S | Med |
| 2.6 | **Short-window (K=3–5) Livox multi-frame accumulation in the odom frame** — cheap densifier (deskew already per-frame, odom from 2.1). Primarily boosts THIN/FAR positive recall and is the enabler that makes negative-obstacle reasoning (Phase 4) reliable (more ground returns to reason about their *absence*). Bound K≤5 @10 Hz (≤0.5 s) to avoid dynamic smear. | obstacle_node.py | S | Med |

*Expected measured gain (vs Phase 0 bench):* 2.1 lifts near-field/blind-wedge persistence recall and removes phantom-velocity FPs; 2.2+2.5 give calibrated, range-aware evidence (recall lift for far/thin, FP cut for grazing-incidence noise); 2.3+2.4 give a *bounded, risk-based* stop distance and eliminate silent-blind failures. These are the items whose deltas you headline as "recall X / FP Y at N m."

---

## Phase 3 — Advanced: Tracking, Learned Perception, Uncertainty (measured, gated, mostly CPU)

### Tracking (CPU, tens of objects, microseconds — zero GPU)
| # | Item | Modules | Effort | Impact |
|---|------|---------|--------|--------|
| 3.1 | **ByteTrack two-stage association + SORT track management** on clustered ring detections. No canonical point-cloud port → re-implement association as **angular + range gating** (standard substitution). Source = existing `PolarOccupancyGrid.nearest_occupied()` (OCCUPIED=strong, sub-l_high=weak). Enables obstacle-velocity, TTC, static/dynamic — and track-level FP suppression so walking people stop accumulating into phantom walls (which today caps the useful window). **Bounded max-age mandatory** (a coasting track can veto motion). Add track-level P/R scoring to the bench first. | new tracker module, obstacle_node.py, perception_harness.py | M | High |
| 3.2 | **IMM static-vs-dynamic** — start a **2-model CP/CV bank** (not the full CP+CV+CT/CA), + a simple speed gate. Hard-coupled to 2.1: on a swaying biped without ego-motion comp, Constant-Position never wins for real walls and IMM mislabels every static obstacle as dynamic. Only after 2.1. | tracker module | M | Med |
| 3.3 | **Occlusion / blind-wedge track HOLD** with bounded coasting + position-gated re-acquisition — matches the "depth only lowers, never clears" philosophy; contingent on 3.1. Re-introduces the stale-phantom-freeze bug the team already fought (needed `predict_timeout_s`) unless hold time is bounded. | tracker module, guard.py | M | Med |

**DROP (tracking):** DONEX grid-level dynamic-cell (a *fork* producing the same static/dynamic labels as 3.1+3.2 + same ego-motion dep — redundant with the chosen object-level spine); DP-TBD track-before-detect (adds N-frame confirmation LATENCY to the most safety-critical returns — drop-offs/thin-far — fatal for a reactive guard; tau-decay ring already does crude accumulation); GM-PHD/RFS (academic overkill until clutter is *measured* to break SORT/ByteTrack — solving an unproven problem).

### Learned perception (GPU — DLA/INT8 + demand-gated into walk window ONLY; build the labeled P/R eval first)
- **PILOT — Range-view LiDAR semantic seg (SalsaNext slim, 2-class INT8)**: the *only* learned option that uniquely attacks the headline blind spot — 360° semantics for low SIDE/REAR hazards geometry and forward-only depth cannot see. Feasibility is the hard gate: SalsaNext is the only range-view net that hits real-time on Jetson and is the *least accurate* evaluated (and that was on AGX Orin, stronger than this NX); FRNet won't hit real-time here. CPU-side range projection is 35–83% of runtime and must be pipelined. Viable only as a slimmed 2-class INT8 head run during autonomous walk when detect/pose are demand-gated off. No labeled eval yet proves it beats the geometric ring — build that first.
- **PILOT — Deep freespace/terrain-safety seg for NEGATIVE obstacles from forward RGB**: targets the only ZERO-detection mode (0→nonzero recall = largest single safety gain). DLA/INT8-friendly, runs where depth returns nothing. No off-the-shelf weights → site-specific data + fine-tune on THIS robot's floors/stair edges; monocular is not confident enough to be sole authority (a false-negative at a ledge is catastrophic) → must be fail-safe-fused and AND/OR-paired with the geometric drop-off cue (Phase 4). Forward-only.
- **PILOT — RFNet RGB-D "unexpected small obstacle" class**: MobileNet+INT8 is DLA-friendly (could co-exist), but heavily overlaps NanoOWL/NanoSAM which already produce forward labeled boxes+masks on this exact cone. Only unique delta is the Lost-and-Found sub-threshold-debris class — a narrow gain that must be shown real against current NanoSAM masks first. Forward-only, does nothing for the sides/rear priority.

**DROP (learned):** GndNet ground seg (redundant — Patchwork++ already deployed in ground.py; can't detect holes it never sampled; PointPillars maps poorly to DLA → lands on saturated GPU); EviLOG learned evidential ISM (another PointPillars net on the no-MPS GPU + bespoke-simulator training path = largest lift, marginal guard-level benefit); self-supervised traversability (WVN/STERLING/RoadRunner) — outputs a costmap with no consumer (this is a scalar governor, not Nav2), biggest integration lift for a benefit the guard can't act on; monocular reflective-ground net (2nd learned model starving pose — worse safety failure than the blind spot it patches).

### Uncertainty
- **PILOT — Dempster-Shafer evidential ring** (free/occ/unknown masses + conflict K): two float32 planes on the CPU ring is trivial, but its headline 2-sensor-conflict payoff rests on an architecture misread — **there is no depth-ISM** (guard.py:562–566 min-merges depth as a RAW distance, outside occupancy.py). Real conflict needs routing depth through its own ISM into the grid (rewrite of node fusion + shm JSON contract + guard consumer). Available now: single-sensor *temporal* conflict (free-cell→occupied), a smaller win. For a scalar-consuming governor the ignorance channel largely duplicates the existing `_observed` flag + never-free-on-NaN policy. Pilot only if 2.2/2.4 prove insufficient.

---

## Phase 4 — Blind-Spot & Negative-Obstacle Closure (CPU-only; the missing safety categories)

| # | Item | Modules | Effort | Impact |
|---|------|---------|--------|--------|
| 4.1 | **Ray-cast missing-ground-return negative-obstacle detector** on the existing 20 cm elevation grid (drop-offs / descending stairs / curb edges — currently ZERO detection). Reuses the 47.6° down-tilt geometry, pure numpy, folds into `guard._merge_depth_ring` fail-safe. **Must** implement the occlusion ray-test + N-frame persistence or it over-triggers at corners / behind positive obstacles. Horizon bounded to D435i ~0.5–2 m band = ~4 s margin at ≤0.5 m/s (adequate). Provenance note: the real method is CSIRO Virtual Surfaces (2010.16018), not STEP. | new negative module, ground.py, obstacle_node.py, guard.py | M | **High** |
| 4.2 | **Down-tilt geometric negative scan-line vote on D435i** (ground-height-drop + rear-wall height/density; point-spacing-jump as secondary) — an *independent* geometric vote AND/OR'd with 4.1 buys PRECISION on negatives (only fire when both agree = institutional bar). On a stereo D435i lean on height-drop/rear-wall; point-spacing is noisier than the 92.7% orchard-LiDAR figure. | new negative module, depth_nearfield.py | M | High |
| 4.3 | **Odometry-anchored rolling temporal grid** carrying seen hazards into the side/rear blind wedge (Mid-360 near-floor-blind, depth forward-only). Uses the same odom + buffer as 2.1/2.6. **Frustum-gated clearing is load-bearing** — never blind-clear the un-sensed rear just because no points arrived (STVL rule). Bound cell age a few seconds (legged odom drift smears world-frame cells). CPU/numpy — NOT elevation_mapping_cupy (GPU contends with pose). | obstacle_node.py, occupancy.py | M | High |
| 4.4 | **Conservative invalid-depth semantics + bounded never-free hole-fill**. The depth RING already encodes NaN=no-override; harden the scalar `front_near_m` path to the same rule and add a bounded hole-fill so d=0 is never read as free space. Near-free numpy, highest-value cheap safety fix. **Split off and DROP the Depth-Anything-V2 infill half** — ~10 FPS ViT-S consumes the whole GPU, no place in the reactive hot path (gate to invalid-heavy frames at low rate only, or drop). | depth_nearfield.py, guard.py | S | High |
| 4.5 | **Coverage/visibility-aware caution policy**: today the grid marks sectors UNKNOWN but the guard fails-OPEN on NaN. Treat a sustained near-field UNKNOWN arc along the heading as a soft hazard — cap speed by observed free-space rather than assuming unobserved==clear. Directly addresses side/rear + at-feet + large depth holes. | guard.py | S | Med |
| 4.6 | **Sim extension: negative-obstacle geometry** (drop-off edge, hole, step-down) + neg-GT metric — required to *measure* 4.1/4.2 recall (0.0 today) since the current ray-cast sim is a z=0 plane. Feeds the Phase-0 SMIYC/Lost-and-Found pilot. | sim_perception.py | M | High |

**PILOT:** Patchwork++ RNR / R-VPF ground upgrade — `ground.py` already imports `pypatchworkpp` as an auto-selected backend, so this is "install + enable + tune + A/B on the harness," not new work (verify it's installed on the Jetson). Off-dimension for negatives (RNR/R-VPF target specular reflection + low *positive* structure, not drop-offs); incremental win is specular-floor RNR. Don't assume it beats the current elevation-grid fallback — A/B it.

**DROP:** monocular freespace net (see Phase 3 — GPU starvation).

*Expected measured gain:* 4.1+4.2 move negative-obstacle recall from **0.0 → nonzero** (the single largest available safety gain), with the AND/OR vote controlling FP; 4.3+4.5 close the side/rear/at-feet unknown-arc into a *speed-capped* policy instead of fail-open; 4.4 removes the dark/specular-floor "hole read as free" failure. Report all against the Phase-4.6 sim scenarios + staged real logs.

---

## Residual physics floor (out of scope — state honestly in the coverage-gap report)
Sub-~2 cm debris (point-spacing floor), at-feet inside ~0.45 m without a memorized transport, and any hazard a nose-up Livox never samples inside a drop-off (elevation there is extrapolation over empty cells, not detection). These belong in the Hoss coverage-gap report (0.5), not hidden by masking.

---

## Sources
- **Eval:** Hoss et al. (TP/FP/FN reproducibility checklist + area-of-vision masking); COCO AP_S; Waymo distance-bin protocol; SegmentMeIfYouCan (FPR95, sIoU_gt, PPV, mean-F1) + Lost-and-Found; Philion et al., *Planning-centric Metrics* (PKL), CVPR 2020 (nv-tlabs/planning-centric-metrics).
- **Calibration/sync:** Livox Mid-360 IEEE-1588v2 PTP (PointXYZRTLT); Intel D435i `RS2_OPTION_GLOBAL_TIME_ENABLED`; ACSC (non-repetitive Livox↔camera); lvt2calib; koide3 `direct_visual_lidar_calibration` (P0.3); LI-Init (hku-mars, arXiv 2202.11006); FUSION_ROADMAP.md P0.
- **Tracking:** ByteTrack; SORT; IMM (radar/aerospace estimation); DOGMa/radar-grid ego-motion compensation.
- **Uncertainty:** occupancy inverse-sensor-model (OctoMap/STVL heritage); temperature/Platt scaling + ECE; conformal prediction / Adaptive-CP for robotics safety margins; Nuss et al. RFS/DST occupancy grids.
- **Negative/blind-spot:** CSIRO Virtual Surfaces (arXiv 2010.16018); "Detecting negative obstacles" patent; orchard-LiDAR negative detection (PMC11679008); Patchwork++ (IROS 2022, RNR/R-VPF); STVL frustum-gated clearing; Depth-Anything-V2 (Orin NX ~10 FPS ViT-S — noted as GPU-prohibitive here).
- **Learned/feasibility:** SalsaNext / FRNet / CENet range-view; RFNet + Lost-and-Found unexpected-obstacle class; Jetson no-MPS time-multiplexing + DLA INT8/TensorRT co-location limits (documented 3-model concurrent-inference hang on Orin).

---

# Measurement Framework (Phase 0 — gates every later change)

**Why first:** "institutional accuracy" is unfalsifiable today. sim_perception.py emits only booleans (`detected_obstacle`, `collided`, `robot_fault_collision`, `reached_goal`) + gait stats (`max_accel_vx`, `max_jerk_vx`, `accel_violations`) and a single scalar `min_surface_dist_m` from `_true_clearance` (line 308). No per-sector GT, no confusion matrix, no PR/ROC, no distance-error, no detection latency, no per-stage timing. Build the yardstick before spending effort on accuracy.

## 1. Ground-truth capture
- **Synthetic (now):** add `_true_ring()` beside `_true_clearance` in sim_perception.py — per-sector true surface distance over the 72-sector polar grid (trivial with existing `_ray_cylinder`/`_ray_panel`/`_nearest_dir` helpers). Add a rasterizer projecting sim obstacles into the polar grid for per-cell occ/free GT (needed by calibration/ECE). Extend the sim with negative-obstacle geometry (drop-off/hole/step-down) — the current z=0-plane sim literally cannot render negatives (Phase 4.6).
- **Real (staged, pilot):** deterministic MCAP rosbag replay through the REAL node (perception_harness already loads node math without ROS). Hand-label a small staged corpus (physical boxes at known range/azimuth/size, a step-down edge, a specular-floor patch); mocap/total-station only if available. Validate any auto/pseudo-GT against the hand-labeled subset before trusting it — scoring against unvalidated pseudo-GT is circular.

## 2. Metrics (primary KPIs)
- **Stratified confusion matrix → Precision / Recall / F1 / PR curves**, binned by **size × range × azimuth × material** (COCO AP_S / Waymo distance-bin protocol on the ring). Target statements like "recall 0.95 / FP 0.02 for a 6 cm object at 2 m."
- **Distance accuracy:** bias + RMSE of per-sector ring distance vs `_true_ring()`. Set `ground_clearance`/thresholds against MEASURED floor RMS (real logs), not datasheet cites.
- **Detection latency / reaction margin:** TTC-at-first-detection − brake time; per-obstacle first-detection frame (must be ADDED — `run()` keeps only a `detected_any` OR today). Highest value-per-effort: PR curves are structurally blind to LATE detection, the dominant reactive-guard failure at closing speed, and this catches "a filter change added one frame of lag."
- **False-stop vs miss ROC + operating-point selection** by sweeping `occ_l_high` + guard `stop_at` (rebuild node per threshold — cheap). Report **false-stops-per-scenario**, NOT per-hour (the 13 scenarios are 8–24 s adversarial courses, not a duty cycle — a per-hour headline is fictitious until real nominal-operation logs exist).
- **Small/negative-obstacle metrics:** SMIYC FPR95 (pair with component-F1 — FPR95 is degenerate while negative recall is 0.0), sIoU_gt, PPV, mean-F1; Lost-and-Found protocol.
- **Criticality-weighted error:** effort-based required-deceleration / TTC-reduction — weight each error by its impact on `guard.apply()` (GT-ring cmd vs perceived-ring cmd, both computable per frame). Right-sized for a velocity governor; full PKL is optional/mismatched (built for sampling planners).
- **Calibration quality:** ECE + reliability diagrams (temperature/Platt) on occupancy probabilities (needs the per-cell GT rasterizer).
- **Per-stage compute:** end-to-end frame-time histogram (P50/P95/P99), per-stage timing (deskew/filters/ground/occupancy/guard), CPU% + thermal + pose-starvation logging measured **on the actual Orin under concurrent YOLO-pose load** — replace the asserted "<1 ms occupancy / ~10 Hz / 3–4 ms deskew" with profiled worst-case.

## 3. Coverage / honesty guard
- **Hoss TP/FP/FN checklist** + explicit **area-of-vision / FOV coverage mask** (a coverage config in obstacle.yaml) applied before scoring. **MUST** ship with a coverage-gap report stating the fraction of hazard space unmonitored (sides/rear-low, at-feet <0.45 m, negatives, sub-2 cm debris) — otherwise masking launders the known blind spots and flatters the miss rate.

## 4. Regression / operating point
- Freeze recall, FP-rate, mean-stop-distance, detection-latency, P99 frame-time as tracked KPIs **per commit** (autoresearch-style bench loop). Today tests are pass/fail asserts with no metric tracked across commits, so tuning changes (l_high, thresholds, accum window) have no quantitative guardrail and silent perception regressions are invisible.
- Deterministic + seeded (the `seed=` param exists) so gating is stable.
- Operating point (thresholds) chosen from the ROC/partial-AUC, then frozen and re-validated on real logs before any coverage % is claimed.

## 5. Compute discipline
Entirely OFFLINE (workstation or Orin off-shift) via the pure-numpy sim/harness — never touches the reactive hot path, never contends for the GPU, cannot starve pose. Honest caveat baked into every report: sim numbers are necessary-but-optimistic (idealized ray-cast omits floor-specular holes, real deskew noise, gait sway) — the sim-to-real gap means real-log validation (pilot) is the ceiling for "institutional," and sim/field ECE + conformal coverage guarantees are SIM guarantees until re-calibrated on field residuals.

---

## Top changes (ranked, do #1 first)

| # | Change | Files | Effort | Impact |
|---|--------|-------|--------|--------|
| 1 | Build the measurement foundation (confusion-matrix + PR/ROC + latency + timing bench) | scripts/sim_perception.py, scripts/perception_harness.py, scripts/perception_bench.py, obstacle/obstacle.yaml | high | high |
| 2 | Fix D435i depth frame at the source + delete the in-place derotation (safety-critical P0.1) | scripts/lidar_source.py, scripts/depth_nearfield.py | low | high |
| 3 | SE(2) ego-motion compensation of the ring + accumulation | obstacle/obstacle_node.py, obstacle/occupancy.py | medium | high |
| 4 | Negative-obstacle detection: ray-cast missing-ground-return + down-tilt geometric vote | obstacle/ground.py, obstacle/obstacle_node.py, scripts/depth_nearfield.py, scripts/sim_perception.py | medium | high |
| 5 | Per-sensor health/degradation monitor + conservative never-free invalid-depth | obstacle/obstacle_node.py, obstacle/guard.py, scripts/depth_nearfield.py | medium | high |
| 6 | Heteroscedastic inverse sensor model + conformal stop-distance margin | obstacle/occupancy.py, obstacle/guard.py | medium | high |

### Details

**1. Build the measurement foundation (confusion-matrix + PR/ROC + latency + timing bench)** (effort high, impact high)

- Files: `scripts/sim_perception.py`, `scripts/perception_harness.py`, `scripts/perception_bench.py`, `obstacle/obstacle.yaml`
- Change: Add _true_ring() per-sector GT beside the scalar _true_clearance; a polar-grid rasterizer for per-cell occ/free GT; a stratified confusion matrix (size x range x azimuth x material) -> P/R/F1/PR curves; per-obstacle first-detection frame -> detection-latency/reaction-margin; false-stop-vs-miss ROC by sweeping occ_l_high + guard stop_at; Hoss TP/FP/FN + FOV coverage mask with a coverage-gap report; per-stage frame-time histogram profiled on the Orin under pose load; freeze all as per-commit regression KPIs.
- Why: Institutional accuracy is unprovable without this. The sim emits only booleans (detected/collided/reached) + a single clearance scalar today. Every later change must report its measured gain against, and be gated by, this bench. Offline pure-numpy, zero GPU, cannot starve pose.

**2. Fix D435i depth frame at the source + delete the in-place derotation (safety-critical P0.1)** (effort low, impact high)

- Files: `scripts/lidar_source.py`, `scripts/depth_nearfield.py`
- Change: lidar_source._depth_to_cloud bakes only MOUNT_HEIGHT and omits the ~47.6deg mount pitch; depth_nearfield._compute then undoes pitch in-place with hardcoded pitch_deg=47.6/camera_height=1.30. Correct pitch+height at source, delete the in-place derotation, record a measured camera->base residual, and feed the deskew IMU gravity vector into segment_obstacles_elevation (obstacle_node.py:815 currently passes none).
- Why: depth_fusion.enabled=true is running on exactly the unvalidated frame P0 was meant to gate. A wrong pitch/height is the stated false-stop / silent-miss risk and shifts a min-merged depth sector a full 5deg at 1-2m. Makes all real-data metrics trustworthy.

**3. SE(2) ego-motion compensation of the ring + accumulation** (effort medium, impact high)

- Files: `obstacle/obstacle_node.py`, `obstacle/occupancy.py`
- Change: Bridge Unitree body odom (rt/odommodestate x/y/yaw, already read by step_pacer) into the domain-99 node via shm or a 2nd SDK subscriber; apply one SE(2) transform per ~0.1s frame to the ring + accumulation before temporal fusion; fuse with the already-buffered deskew IMU and keep the window ~1 frame to bound legged-odom drift.
- Why: Highest ROI-per-line accuracy fix. Removes DOGMa ghost-velocity FPs, lets the 0.6s window TRANSPORT a memorized obstacle into the under-body front blind wedge (today it only fades), and lifts the accumulation ceiling that boxed the robot in. Prerequisite for any tracker. CPU-only.

**4. Negative-obstacle detection: ray-cast missing-ground-return + down-tilt geometric vote** (effort medium, impact high)

- Files: `obstacle/ground.py`, `obstacle/obstacle_node.py`, `scripts/depth_nearfield.py`, `scripts/sim_perception.py`
- Change: New pure-numpy negative module: ray-cast no-return-gap on the 20cm elevation grid (with occlusion ray-test + N-frame persistence to stop corner over-trigger) AND/OR'd with a D435i down-tilt scan-line vote (ground-height-drop + rear-wall height/density). Fold into guard._merge_depth_ring fail-safe. Extend the z=0 sim with drop-off/hole geometry + neg-GT to actually measure recall.
- Why: Drop-offs/descending stairs/curb edges are currently ZERO detection - a whole missing safety category. Moving recall 0.0 -> nonzero is the single largest available safety gain; the two-cue AND/OR buys precision to the institutional bar. CPU-only, reuses existing 47.6deg geometry.

**5. Per-sensor health/degradation monitor + conservative never-free invalid-depth** (effort medium, impact high)

- Files: `obstacle/obstacle_node.py`, `obstacle/guard.py`, `scripts/depth_nearfield.py`
- Change: Cheap numpy aggregates (Livox-baselined point-count collapse, D435i fill-rate, largest-contiguous-depth-hole, lidar-vs-depth raw-distance disagreement) -> existing shm JSON -> guard's existing FAULT state, with graduated fallback. Harden the scalar front_near_m path so d=0 is never read as free; bounded never-free hole-fill. Drop the Depth-Anything-V2 infill (whole-GPU).
- Why: Kills the scariest failure - a blinded Livox / sun-saturated D435i whose absence of returns reads as free space. The depth-hole metric doubles as a negative-obstacle prior. Livox-aware baselining is mandatory (non-repetitive scan false-trips naive thresholds). CPU-only.

**6. Heteroscedastic inverse sensor model + conformal stop-distance margin** (effort medium, impact high)

- Files: `obstacle/occupancy.py`, `obstacle/guard.py`
- Change: Extend occupancy.update() (line 149, fixed log-odds today) to accept per-sector (range, count, incidence) via a weight LUT - the counts already exist in filters.sector_kth_nearest (obstacle_node.py:655) and are discarded; add a per-cell variance channel. Replace the hand-tuned 0.6/0.30/0.35m guard magic numbers with a distribution-free conformal quantile margin from logged clearance residuals; re-tune occ_l_high; re-calibrate on field residuals before claiming coverage %.
- Why: Turns fixed, range-blind evidence and magic-number thresholds into calibrated, range-aware, risk-based bounds - the hallmark of an institutional safety layer. <0.1ms vectorized, CPU-only. Sequence AFTER the bench exists so the LUT/margin are measured, not re-named magic numbers.
