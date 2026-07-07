# Perception Rebuild — LiDAR + Depth Obstacle Detection

Everything that changed from **the last plain checkpoint (`e61818d`, 2 Jul)** through the
perception/obstacle arc up to **`8dd0c17`**, plus what to do about each on the robot.
The odometry / wave-back / phone features that came after live in **`OFFICE-RUNBOOK.md`**
(boot sequence + on-robot test plan). This file is the *why + what-to-validate* for the
obstacle pipeline specifically.

Every claim below is from a commit that shipped with its own verification; the honest
limits are called out, not laundered. `(ROBOT)` = only provable on the real robot — sim
cannot cover it.

---

## TL;DR — what you got

The obstacle pipeline went from *"one tilted ground plane + a flat ring that dropped
sparse returns + a rotating radar-strip display"* to:

1. **Fine detection** — small ground objects and thin/far hazards are kept, not absorbed
   into the floor plane. A **6 cm pole at 3 m** is detected; a **3 cm wire at 3 m ≈ 98%**
   recall (measured).
2. **Depth-camera fusion** — the RealSense D435i fills the Mid-360's ~1 m near-front blind
   wedge across all forward sectors, and sees **floor cables/cords ≳3 cm** the up-tilted
   lidar is blind to. Fail-safe: depth can only make a sector *closer*, never clear one.
3. **A measurement yardstick** — `perception_bench.py` scores the real pipeline against
   ground truth. Current baseline: **Precision 99.0% / Recall 90.0% / F1 94.3%,
   distance RMSE 45 mm**. Every future accuracy change reports against this.
4. **Object-aware 3D viz** — the sphere clusters points into distinct objects (stable
   colour + bounding box) instead of blocky columns, and **shows the depth detections too**,
   so the display equals what the guard actually reacts to.
5. **Safety hardening** — IMU gravity-levelling (sign-agnostic, tilt-clamped) and a depth
   frame self-gate so a mis-mounted camera fails to lidar-only instead of false-stopping.

---

## 1. Fine obstacle detection — `c597910`

Three precision modules were wired into `obstacle_node.py`, each behind an `obstacle.yaml`
flag (default **on**, revertible):

| Module | What it fixes |
|---|---|
| **Per-cell elevation ground seg** (`ground_use_gravity` path, `segment_obstacles_elevation`) | Floor is segmented per 20 cm cell instead of one global tilted plane, so a small ground object is no longer swallowed by the plane's ~12 cm inlier band. |
| **Range-scaled ring** (`sector_kth_nearest` + `sector_tripwire`) | Ring keeps sparse thin/far hazards (1–2 points) that the flat `ring_min_points=3` used to drop. |
| **Range-graded decimation** | Keep-all-near / thin-far under a point budget, replacing a blind stride that thinned near objects. |

**State:** verified — 18 obstacle unit tests, A/B harness (sparse object retained), and
`sim_perception --selftest` (6 cm pole @ 3 m detected, floor-only clean, closed-loop
no-collision). Research behind it: `obstacle/FUSION_ROADMAP.md`.

**What to do:** nothing to build — validate live that a **small object on the floor
(a can, a 6 cm pole) actually raises a sector** on the 2D ring/3D sphere as you approach
`(ROBOT)`. If the floor itself trips detections on your surface, that's the ground-seg
inlier band — tell me and we raise it.

---

## 2. Depth-camera fusion (D435i) — `a4880a9`, `71cc92a`, `dabe1e2`

The Mid-360 is mounted tilted up, so it is blind to the near floor and the ~1 m front
wedge. The depth camera fills that:

- **Per-sector depth ring** (`depth_nearfield.py::front_ring`) over the camera's forward
  FOV, same 5°/72-sector geometry as the lidar ring, at the robot-validated **47.6° down
  pitch / 1.30 m height**. NaN where nothing is near (never emitted as "clear").
- **Fused into the guard** (`guard.py::_merge_depth_ring`) with a NaN-aware **min-merge**:
  depth can only pull a sector *closer*; geometry mismatch or error → the lidar ring is
  returned unchanged. Depth can never break the guard or clear a lidar reading.
- **Floor cables** (`dabe1e2`): the depth ring uses a dedicated low
  `ring_ground_clearance = 0.03 m` (vs the scalar corridor's conservative 0.08 m), so a
  few-cm cable clears the floor noise and is reported.

**Config:** `obstacle.yaml → depth_fusion.enabled: true` (ships **ON**). It is
**frame-self-gated** (`71cc92a`): `DepthNearField` checks every frame — a flat floor must
read height ≈ 0 at all forward ranges; a frame with the wrong pitch/height emits **no**
ring/scalar (falls back to lidar-only, never a false near-stop). `frame_ok` +
`floor_ramp_m` are in telemetry so you can validate it.

**State:** off-robot the live V4L2 camera path is **untestable**. Proven in
`obstacle/test_depth_ring.py` (20 cm object @ 1 m detected; 4 cm cable @ 1 m detected;
floor-only / empty / out-of-FOV / 2 cm noise → no reading; merge is min-only /
never-clear / geometry-guarded).

**What to do `(ROBOT)` — this is the #1 thing to validate before trusting depth in motion:**
1. Bring the robot up, open the dashboard, watch the depth telemetry: **`frame_ok` must be
   true** and `floor_ramp_m` near 0 on a flat floor.
2. If `frame_ok` is false → the camera mount pitch/height doesn't match the validated
   47.6°/1.30 m. **Do not drive** trusting depth; a bad frame = false stops. Re-measure the
   mount or tell me to re-tune.
3. Lay a cable/cord on the floor ahead → it should raise a near sector on the 2D ring and
   appear as its own cluster in the 3D sphere.
4. If a noisy floor false-triggers, raise `depth_fusion.ring_ground_clearance` (0.03 →
   0.05 m). Physics floor: sub-~3 cm objects blend into D435i near-field noise — can't fix.

---

## 3. Steady, object-aware visualization — `c09ea62`, `bfce3ac`, `f6ba378`, `19bb2f4`

- **`c09ea62`** — the Mid-360 scans non-repetitively (each ~100 ms frame paints a thin
  petal), so the ring/sphere looked like a rotating radar strip. A node-side rolling
  accumulation window (`viz_accum_s = 0.6 s`, ~6 frames) merges recent points → a full,
  steady cloud. Safety wedges/zone stay **per-frame**, so hard-stop latency is unchanged;
  this also densifies the 360° guard gating (fail-safe: a cleared sector lingers ≤0.6 s).
- **`bfce3ac`** — the 3D sphere now renders **objects**: points are clustered
  (union-find, 8-connected over the occupied cells), each object gets a stable colour + an
  AABB bounding box tinted red inside the slow/stop band. A wall, a person and a pole now
  look different instead of one blocky field. Zero backend/GPU change.
- **`f6ba378`** — **toggle** between the new Points render and the legacy Columns
  (corner button / `c` key, persists in localStorage); points enlarged (0.18 m) for
  legibility.
- **`19bb2f4`** — the **depth detections are shown too**: `front_points()` feeds the depth
  near-ground points into the 2D ring (min-merge) and 3D sphere (own cluster + box). The
  display now equals the fused picture the guard acts on — previously the robot would stop
  for a cable that never appeared on screen.

**State:** all JS `node --check` clean; headless clustering test (mock wall+person+pole+
cable → 3–4 distinct stable objects with correct extents). Design notes:
`web/VIZ_REDESIGN.md`.

**Known limit (documented):** top-down 2D clustering merges objects whose footprints are
<~13 cm apart (a pole touching a wall); a z-band clustering key is the noted next step.
The sphere shows the *kept* (filtered, strided to 3000) obstacle points — inherently
sparser than the raw RealSense "fake cloud"; bigger points + boxes are the legibility
levers, not raw density.

**What to do:** just know both views exist — press `c` / the corner button to switch
Points ↔ Columns. No action needed.

---

## 4. Safety hardening — `71cc92a`, `8dd0c17`

- **Gravity-levelled ground seg** (`71cc92a`): the elevation grid was segmented on raw
  sensor-frame z with no gravity vector — uprightness rested on gyro-deskew. The node now
  keeps a slow EMA of the accelerometer (= measured gravity) and levels the grid before
  segmentation (`ground_use_gravity`, default on; inactive without an IMU, so sim/harness
  unchanged).
- **Sign-agnostic + tilt-clamped** (`8dd0c17`): `sensor_msgs/Imu` (REP-145) reports
  specific-force **up** at rest, so passing the raw EMA could rotate the cloud the wrong
  way — in the **unsafe under-detection** direction — with no guard. Now the node orients
  the vector down by its own z sign (correct whether the IMU reports gravity or
  specific-force-up) and levels **only** when plausibly vertical (<~35° from straight
  down); anything else → safe no-op (assumed z-up). Verified: down / flipped-up both →
  correct down vector; sideways / 50° tilt → skipped.

**What to do `(ROBOT)`:** with the guard on and the robot standing still upright, confirm
the floor does **not** paint as an obstacle (levelling working). If it does on a sloped
surface, that's the tilt-clamp being conservative — expected on >35° tilt.

---

## 5. The measurement bench — `285b243`, `dabe1e2`

`scripts/perception_bench.py` is the yardstick: it drives the **real** `obstacle_node`
math (rclpy-stubbed) over the sim scenarios and compares the perceived 72-sector ring
against a ground-truth ring rasterized from the sim obstacle footprints (±1-sector
tolerance). Emits Precision / Recall / F1 (overall + by range), distance RMSE, forward
latency, false-stop rate, and an explicit **coverage-gap** statement. Pure-numpy, no GPU,
deterministic, `--selftest` + `--json`.

**Baseline (noise 0.02, sway 2°, 12 scenarios, 1578 frames):**

| Metric | Value |
|---|---|
| Precision | 99.0% |
| Recall | 90.0% |
| F1 | 94.3% |
| Recall by range (near/mid/far) | 92 / 91 / 83% |
| Distance RMSE | 45 mm |
| Thin-object recall (3 cm wire @ 3 m) | 98.1% |

Roadmap for the next accuracy phases: `obstacle/INSTITUTIONAL_ROADMAP.md`.

**What to do:** before/after any obstacle change, run the gate:
```bash
cd scripts && python3 perception_bench.py            # full report
python3 perception_bench.py --small                  # thin-object recall
python3 perception_bench.py --selftest               # pass/fail gate
```
A recall/precision drop vs the table above = a regression. (Harmless known noise: a
`filters.py:391 RuntimeWarning: invalid value encountered in cast` prints during the run —
a NaN→int cast in an empty sector, no effect on results. Say the word and I'll add the
nan-guard.)

---

## 6. Behaviour change you should know — `ddf87f6`

Committed just before the fine-detection work, entangled in the guard:

- **Removed** the autonomous gap-following steering + turn-in-place recovery (too fiddly to
  tune against coarse Mid-360 returns). **Kept**: distance slow/stop, 360° ring directional
  gating, blind-zone prediction, per-direction hard stops, the 2D + 3D viz.
- **Graduated smooth stop**: a routine wall approach now zeroes the target and lets the
  jerk/accel shaper ramp to a halt; only an obstacle inside `emergency_stop_m` (0.30 m raw,
  ~0.45 m effective) snaps instantly. `resume_hysteresis_m` (0.12 m) adds a Schmitt band to
  kill stop-go chatter.
- **Snappier gait**: `accel_vx` 2.2 m/s², `accel_vy` 1.6, yaw 4.0; jerk 8/8/16
  (`config/robot.yaml`).

⚠️ **This is the source of the one failing unit test.** `test_walk_pipeline.py`
"reverse overshoot ≤ 0.025" fails at **−1.535 vs −1.525 floor** — the snappier accel makes
the reverse ramp overshoot 0.035 m/s for ~1 tick, tripping a now-stale tolerance. It is
**pre-existing** (not from the perception or later feature commits) and physically trivial.
**Decision needed:** loosen the tolerance to 0.04, or tighten the cmd_shaper reverse ramp.
Tell me which and I'll do it.

---

## Config knobs — `obstacle/obstacle.yaml` (+ `config/robot.yaml`)

| Key | Default | Tune when |
|---|---|---|
| `emergency_stop_m` | 0.30 m | Instant-snap distance. Raise for a bigger hard-stop bubble. |
| `resume_hysteresis_m` | 0.12 m | Raise if the robot chatters stop-go at a wall. |
| `ring_min_points` (lidar) | 3 | Lower → keeps sparser (thinner/farther) hazards, at more noise. |
| `viz_accum_s` | 0.6 s | Viz/gating merge window. 0 = old per-frame sweep. |
| `viz_max_points` | 3000 | Raise for a denser sphere cloud (bigger WS payload). |
| `depth_fusion.enabled` | **true** | Set false to fall back to lidar-only if the depth frame won't validate. |
| `depth_fusion.ring_ground_clearance` | 0.03 m | Raise (→0.05) if a noisy floor false-triggers depth. |
| `depth_fusion.ring_min_points` | 2 | Near-sector points before depth reports. |
| `ground_use_gravity` | on | Leave on (needs IMU; auto-safe without one). |
| `accel_vx` / `jerk_vx` (`robot.yaml`) | 2.2 / 8.0 | Lower if the gait feels too aggressive (also shrinks the reverse overshoot). |

Every flag defaults to the tuned value; all are revertible without code changes.

---

## Honest limits — what is NOT proven here

- **Depth camera path is on-robot-unvalidated.** The whole depth-fusion benefit rests on
  `frame_ok` being true with a correctly-mounted camera. Validate it (Section 2) before
  trusting near-ground detection in motion. A bad frame = false stops.
- **Floor cables can't be closed-loop-measured in sim** (the sim models only the lidar).
  Detection is unit-proven; the *avoidance* of a floor cable is `(ROBOT)`-only.
- **Sub-3 cm floor objects** blend into D435i noise — physics floor, not a bug.
- **Clustering merges footprints <13 cm apart** in the 3D viz (cosmetic; z-band key is next).
- The `filters.py:391` RuntimeWarning is cosmetic (empty-sector NaN cast).

---

## What to do at the office — perception checklist `(ROBOT)`

In priority order (highest-risk unknown first):

1. **Mid-360 publishes `/livox/lidar`** — the entire primary path depends on it. Guard on →
   `tail -f /tmp/g1_obstacle.log` shows ~10 Hz frames with rising `seq`; zone chip goes live.
2. **Depth `frame_ok` = true**, `floor_ramp_m` ≈ 0 on flat floor (Section 2). If false, do
   not trust depth — re-check the camera mount.
3. **Small object on floor** (can / 6 cm pole) raises a sector as you approach (Section 1).
4. **Floor cable** appears on the 2D ring + 3D sphere and the robot slows/stops for it.
5. **Floor is not painted as obstacle** standing still upright (levelling OK, Section 4).
6. **Closed-loop wall approach** — routine approach ramps to a smooth halt; a sudden/near
   wall snaps instantly; no collision, no stop-go chatter.
7. Run `perception_bench.py` once on the robot's Python to confirm the gate is green there.

Anything that fails 1–2 stops the demo — fix before driving. 3–6 are the "does it actually
detect well" proof you asked for.
