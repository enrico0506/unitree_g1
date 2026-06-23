# g1_nav — Nav2 Migration Status

Replacing the hand-rolled reactive obstacle node (`/home/unitree/projects/g1/obstacle`)
with a standard Nav2 stack for thin / low-obstacle detection + blind-spot memory.

```
Mid-360 (Livox CustomMsg) --livox_to_pc2--> /livox/points
  --Patchwork++--> /patchwork/nonground (+ground)
  (+ optional /camera/points from RealSense) --STVL--> Nav2 local costmap
  --DWB (holonomic)--> /cmd_vel --g1_cmd_vel_bridge (d99->d0)--> Unitree LocoClient.SetVelocity
```

Map-free, rolling local costmap in the FAST-LIO `odom` frame. ROS 2 **Foxy**, `ROS_DOMAIN_ID=99`
(CycloneDDS), sourced via `/home/unitree/g1_mapping_ws/env.sh`. Jetson Orin (aarch64).

---

## WHERE IT STANDS

| Component | State | Notes |
|---|---|---|
| `livox_to_pc2.py` | **Built, verified good** | Consumes existing `/livox/lidar` CustomMsg (xfer_format=1), republishes `sensor_msgs/PointCloud2` on `/livox/points`. Runs NO driver of its own (single-consumer safe). Correct field layout/frames/QoS. |
| `g1_cmd_vel_bridge.py` | **Built, verified good, OFF by default** | d99 sub `/cmd_vel` -> LocoClient d0/eth0; clamps max_vx 1.5 / max_vy 0.7 / max_vyaw 2.0; 0.5 s watchdog; e-stop on exit. Now gated behind `use_bridge:=false`. |
| `realsense_cloud_pub.py` | **Built, OFF (Phase 3)** | V4L2 Z16 -> `/camera/points`, reuses `scripts/lidar_source.py`. Single-open conflict with dashboard. Gated behind `use_realsense:=false`. |
| `nav2_params.yaml` | **Verified good (Foxy-correct)** | DWB holonomic (vy_samples>0); NavFn; recoveries_server; STVL block + commented VoxelLayer fallback; `min_obstacle_height: 0.03` (DO NOT RAISE). Physical values flagged `<ASK ENRICO>`. |
| `bringup.launch.py` | **Structurally sound** | Brings Nav2 servers up directly (no nav2_bringup). Lifecycle list correct. Bridge now gated OFF. Patchwork commented out (graceful). |
| `tf.launch.py` | **Built, needs measured values** | base_link->livox_frame + base_link->camera_link static TFs, all 6 values `0.0` placeholders. Correctly does NOT publish odom->base_link (FAST-LIO owns it). |
| `setup.py` / `package.xml` | **Fixed this pass** | Entry-point module names + manifest deps corrected (see FIXES APPLIED). |
| **nav2_bringup** | **MISSING** (apt-trivial) | Not strictly needed; launch works around it. |
| **STVL** (spatio_temporal_voxel_layer) | **MISSING** | Only `nav2_voxel_grid` present. See Blocker (b). |
| **Patchwork++** | **MISSING** (source-only) | No apt package. See Blocker (c). |
| Live topics `/livox/lidar`, `/Odometry` | **Not present** | Livox driver + FAST-LIO not running at check time (expected — bring them up per smoke-test). |

**Headline: the package now builds and launches safely.** The degraded smoke-test below
needs **zero new builds** — it runs on the already-installed costmap binaries. A first
costmap-with-a-pole-in-RViz is roughly one bring-up sequence away (Livox driver + FAST-LIO
+ a one-block params edit). STVL and Patchwork++ are drop-in upgrades after that.

---

## BLOCKERS & EXACT NEXT STEPS (in order)

### (a) Degraded smoke-test — first costmap + bridge validation (NO STVL, NO Patchwork++)
**Difficulty: trivial. No new builds.** Uses the costmap's own `min_obstacle_height` z-gate
as a throwaway ground filter, and the already-installed `nav2_voxel_grid`/costmap binaries.

1. Build the package (after this pass's fixes):
   ```
   source /home/unitree/g1_mapping_ws/env.sh
   cd /home/unitree/g1_mapping_ws        # or wherever g1_nav is overlaid
   colcon build --packages-select g1_nav
   source install/setup.bash
   ```
2. In `config/nav2_params.yaml`, switch to the FALLBACK VoxelLayer (smoke-test only):
   - set `plugins: ["obstacle_layer", "inflation_layer"]`
   - uncomment the FALLBACK VoxelLayer block (~L152-180)
   - in it, point the source `topic` at the converter output `/livox/points` (NOT
     `/patchwork/nonground`, which is empty without Patchwork++)
   - set `min_obstacle_height: 0.08` **as the throwaway floor cut** — this is a
     SMOKE-TEST value only. **DO NOT COMMIT 0.08 over the 0.03 "DO NOT RAISE" line.**
     Restore `0.03` the moment Patchwork++ supplies a true nonground cloud.
3. Bring up, in order (single Livox driver only — see SAFETY):
   ```
   # 1) the mapping ws Livox driver (CustomMsg / xfer_format=1)
   # 2) FAST-LIO  (supplies odom frame + odom->base_link TF)
   ros2 launch /home/unitree/projects/g1/g1_nav/launch/bringup.launch.py
   #   (tf static + livox_to_pc2 + Nav2 servers; bridge stays OFF)
   ```
4. RViz over **WLAN, not the VS Code forwarded port** (per camera-lag memory): add
   Costmap/Map on `local_costmap/costmap`. Stand a pole in front; confirm marked cells.
5. **Bridge validation — robot OFF the ground or e-stop in hand:**
   - first `ros2 topic echo /cmd_vel` (domain 99) to confirm DWB emits velocity toward a goal
   - only then re-launch with `use_bridge:=true` to confirm d99->d0->LocoClient
     (watchdog stops on no-cmd; e-stop on exit).

### (b) Source-build STVL (+ OpenVDB) — **likely apt-trivial on this robot, NOT the aarch64 nightmare**
The original premise was "OpenVDB/STVL on aarch64 is the showstopper." Build-feasibility
review found `ros-foxy-spatio-temporal-voxel-layer` (`2.1.4-1focal arm64`) **and its full
OpenVDB-6.2/PCL-1.10 closure are apt binaries in the configured mirror** — no source build,
no OpenVDB source compile.
- **Try apt first (do NOT install without Enrico's go-ahead):**
  ```
  sudo apt-get install -y ros-foxy-spatio-temporal-voxel-layer
  # pulls libopenvdb6.2 + libopenvdb-dev, PCL 1.10, IlmBase/OpenEXR transitively
  ```
- **Then** restore STVL in `nav2_params.yaml` (it's the default block already), repoint the
  source to `/patchwork/nonground`, restore `min_obstacle_height: 0.03`.
- **Risk (low):** only real failure is the apt plugin not *loading* into the
  already-installed `nav2_costmap_2d` at activation (ABI). Both come from the same Foxy
  mirror, so unlikely.
- **Fallback only if the plugin won't load (moderate, NOT the classic nightmare):**
  ```
  cd /home/unitree/g1_mapping_ws/src
  git clone https://github.com/SteveMacenski/spatio_temporal_voxel_layer -b foxy-devel
  cd /home/unitree/g1_mapping_ws && colcon build --packages-select spatio_temporal_voxel_layer
  # apt-installed libopenvdb-dev satisfies the hard dep — still NO OpenVDB source build
  ```

### (c) Source-build Patchwork++ — the only genuine source build (moderate)
No apt package exists. Eigen is already installed; Open3D is viz-only and absent — build with
Open3D OFF.
```
source /home/unitree/g1_mapping_ws/env.sh
cd /home/unitree/g1_mapping_ws/src
git clone https://github.com/url-kaist/patchwork-plusplus-ros.git
# confirm the package name in its package.xml (patchworkpp / patchwork_pp_ros vary by fork)
cd /home/unitree/g1_mapping_ws
colcon build --packages-select <pkg_name> --cmake-args -DCMAKE_BUILD_TYPE=Release
```
- **Risks:** (a) CMake `FetchContent` pulls the core `patchworkpp` lib mid-build — **needs
  internet on the robot**; (b) some forks default Open3D viz ON and fail (pass the fork's OFF
  flag, e.g. `-DPATCHWORK_BUILD_VISUALIZER=OFF`); (c) wire its input to `/livox/points` and
  remap output to `/patchwork/nonground` (+ `/patchwork/ground`) to match the pinned param
  topics. Budget 30-60 min.
- Then uncomment the `patchwork` Node in `bringup.launch.py` (block ~L143-163) AND add it to
  the returned LaunchDescription list; set its `sensor_height` (see ASK ENRICO).

### (d) nav2_bringup — apt-trivial, optional (not on the critical path)
```
sudo apt-get install -y ros-foxy-nav2-bringup
```
The launch already brings servers up directly; this only adds canned launch files + a
known-good RViz config to crib from. Cheap insurance, but not required to run.

---

## ASK ENRICO INVENTORY (physical values the stack needs)

- [ ] **base_link -> livox_frame** static extrinsic: x y z + yaw pitch roll (`tf.launch.py`, all `0.0` now)
- [ ] **base_link -> camera_link** static extrinsic: x y z + yaw pitch roll (`tf.launch.py`, all `0.0` now)
- [ ] **LiDAR height above the floor** (m) — for Patchwork++ `sensor_height` (= base_link->livox_frame z + base height off ground); reuse the obstacle.yaml `sensor_height_m` if it's the same mount
- [ ] **RealSense depth vertical FOV** (rad) — `nav2_params.yaml` realsense source (D435 ~1.01 rad; confirm exact module)
- [ ] **RealSense depth horizontal FOV** (rad) — `nav2_params.yaml` realsense source (D435 ~1.52 rad; confirm exact module)
- [ ] **G1 footprint polygon** — local_costmap footprint (currently placeholder); the real foot/torso outline
- [ ] **inflation_radius** (m) — robot inscribed radius + safety margin (`nav2_params.yaml`, `0.45` placeholder)
- [ ] **Safe nav velocities** vx / vy / vyaw + **accelerations** for DWB — the robot caps are 1.5 / 0.7 / 2.0 but the *safe nav* limits are likely lower; confirm DWB `max_vel_*` and `acc_lim_*`
- [ ] **Recovery speeds** — spin/backup velocities for recoveries_server
- [ ] **E-stop / spotting plan** — who holds the e-stop, robot on stand vs floor for each phase
- [ ] **Single-process dual-domain confirm** — verify the bridge's d99 rclpy + d0 LocoClient participants coexist in one process on this robot without DDS interference

---

## HUMAN GATES (RViz / walking validations — cannot be skipped)

1. **TF / extrinsic alignment** — with measured static TFs, the `/livox/points` cloud must
   land on real-world surfaces in RViz (floor at z~0, walls vertical). Misaligned extrinsics
   = phantom or missing obstacles.
2. **Patchwork++ keeps a low object + a pole** — confirm a curb/cable-ramp-height object AND a
   pole both survive ground segmentation into `/patchwork/nonground` (the whole point of the
   migration). Tune at the source, not by raising `min_obstacle_height`.
3. **RealSense covers the near-floor blind cone** (Phase 3) — confirm `/camera/points` fills
   the under-foot region the Mid-360 misses, without fighting the dashboard for `/dev/video0`.
4. **Costmap remembers a pole after it leaves view** — with STVL, walk so a pole exits the FOV
   and confirm the cell stays marked for `voxel_decay` (4 s) — the blind-spot-memory promise.
5. **Lifecycle active** — `ros2 lifecycle get /controller_server` etc. all `active`; the STVL
   plugin actually loaded (controller_server doesn't fail at activation).
6. **/cmd_vel drives + watchdog stops** — with bridge ON and e-stop in hand: a goal produces
   motion toward it; stopping `/cmd_vel` halts the robot within the 0.5 s watchdog; killing the
   bridge e-stops.

---

## SAFETY

- **Bridge OFF by default.** `g1_cmd_vel_bridge` is gated behind `use_bridge:=false` in
  `bringup.launch.py` — it will NOT drive the robot unless you explicitly pass
  `use_bridge:=true`. It keeps its 0.5 s watchdog + e-stop-on-exit.
- **Single-consumer Mid-360.** The Nav2 Livox driver, the mapping driver, and the obstacle/
  driver are **mutually exclusive** (single UDP port). Run exactly ONE. `livox_to_pc2` runs no
  driver of its own — it consumes an existing `/livox/lidar`. The broken PointCloud2 path
  (xfer_format=0) stalls after one frame; always use CustomMsg (xfer_format=1).
- **RealSense V4L2 single-open.** `/dev/video0` (Z16) cannot be read by `realsense_cloud_pub`
  while the dashboard holds the camera. RealSense source is Phase-3 / OFF until then.
- **Keep the old `obstacle/` node as fallback** until the new Nav2 stack passes end-to-end on
  the floor. Do not delete it.

---

## FIXES APPLIED (this pass)

- **`setup.py`** — corrected all three `console_scripts` module names to the real files:
  `livox_to_pointcloud2`->`livox_to_pc2`, `realsense_cloud_publisher`->`realsense_cloud_pub`,
  `cmd_vel_bridge`->`g1_cmd_vel_bridge`. (Before: every `ros2 run`/launch Node died with
  `ModuleNotFoundError` — no executables were built.)
- **`launch/bringup.launch.py`** — gated `g1_cmd_vel_bridge` OFF by default: added
  `use_bridge` LaunchArgument (default `false`) + `condition=IfCondition(use_bridge)` on the
  Node, mirroring the `use_realsense` pattern. (Before: the bridge launched unconditionally
  and drove the robot on every bring-up — the single most dangerous item flagged.)
- **`package.xml`** — fixed manifest so it actually parses + matches the launch:
  - removed illegal `--` (double-hyphen) and a literal `<exec_depend>` tag from inside an XML
    comment that made the manifest **fail catkin_pkg/colcon parsing** (a real, pre-existing
    build blocker not caught by the prose reviews).
  - dropped `nav2_bringup` exec_depend (not installed -> `rosdep install` would fail; launch
    doesn't use it).
  - added `nav2_recoveries`, `nav2_bt_navigator` (launched directly by bringup) and
    `livox_ros_driver2` (`livox_to_pc2` imports its `CustomMsg`).

### Deliberately NOT changed (risky / uncertain / hardware-dependent — recorded only)
- **`realsense` left in STVL `observation_sources`** while the node is OFF (perpetual "no
  observations" warnings). The param file itself documents leaving it listed; removing it
  touches wiring/tuning intent. Decide with Enrico at Phase 3.
- **Patchwork++ input QoS vs `/livox/points` best-effort pub** — a reliable-sub/best-effort-pub
  mismatch would silently drop all data. Cannot verify until Patchwork++ is built. Verify then.
- **`livox_to_pc2` per-point Python comprehension** on the full ~20k-pt frame — likely under
  the 10 Hz budget but measure on-target; decimate only if it stalls.
- **RealSense publish-time stamp + per-frame random decimation** — by design (V4L2 gives no
  ROS stamp); ensure costmap transform tolerance / `expected_update_rate` is lenient at Phase 3.
- All physical `<ASK ENRICO>` values (extrinsics, FOV, footprint, velocities) — left as
  placeholders by design; see ASK ENRICO INVENTORY.

---

## SMOKE-TEST RUNBOOK

A **DEGRADED, STATIONARY** end-to-end check of the plumbing
`Mid-360 -> livox_to_pc2 -> Nav2 local costmap (VoxelLayer) -> DWB -> /cmd_vel`,
the goal being to **see a pole in the local costmap in RViz**. Uses ONLY the
already-installed `nav2` + the now-built `g1_nav`. **NO** STVL, Patchwork++, FAST-LIO,
RealSense, or bridge. The robot **stands still** — a STATIC identity `odom->base_link`
TF replaces FAST-LIO, so the costmap does **NOT** track real motion (stationary test only).

Throwaway pieces (do not commit over the real pipeline):
- `config/nav2_params_smoketest.yaml` — copy of `nav2_params.yaml` with the obstacle source
  swapped to a `VoxelLayer` reading the **raw** `/livox/points`, with `min_obstacle_height: 0.10`
  as a throwaway floor cut. The REAL pipeline uses Patchwork nonground at `0.03` (**DO NOT RAISE**).
- `launch/smoketest.launch.py` (+ `smoketest.sh` wrapper) — starts the Livox driver, the static
  `odom->base_link` TF, then `bringup.launch.py` with the smoke-test params, `use_bridge:=false`,
  `use_realsense:=false`.
- `tf.launch.py` `base_link->livox_frame` z is set to `1.10` (LiDAR height above floor; obstacle.yaml
  `sensor_height_m` default — **<ASK ENRICO: MEASURE the real height>**). A floor return lands at
  `odom z ~= 0`, a pole at `odom z > 0`, so `min_obstacle_height 0.10` drops the floor, keeps the pole.

### Pre-flight (single-consumer Mid-360!)
1. **Stop EVERYTHING that touches the Mid-360** — the mapping driver, the old `obstacle/` node,
   any other Livox driver. The Mid-360 is single-consumer (one UDP port); run exactly ONE driver.
   The dashboard (camera/perception) is fine to leave running — it does not use the LiDAR.
   Verify nothing holds it:
   ```
   source /home/unitree/g1_mapping_ws/env.sh
   ros2 topic list | grep livox     # expect NOTHING (no driver running yet)
   ```
2. **No bridge, no motion.** The bridge stays OFF (`use_bridge:=false`); the robot will NOT move.
   E-stop in hand anyway is good practice.

### Run
```
bash /home/unitree/projects/g1/g1_nav/smoketest.sh
```
This starts, in order: the Livox driver (CustomMsg / xfer_format=1) -> `livox_to_pc2`
(`/livox/lidar` -> `/livox/points`) -> the static `odom->base_link` and `base_link->livox_frame`
TFs -> the Nav2 servers (controller/planner/recoveries/bt_navigator) + lifecycle, with the
smoke-test params. The bridge stays OFF.

### Check topics (a SECOND shell, env.sh-sourced)
```
source /home/unitree/g1_mapping_ws/env.sh
ros2 topic hz /livox/points          # expect ~10 Hz, steady (NOT one frame then stall)
ros2 topic list | grep costmap       # expect local_costmap/costmap + global_costmap/costmap
ros2 lifecycle get /controller_server   # expect: active
```
If `/livox/points` stalls after one frame, the driver is in the broken PointCloud2 path
(`xfer_format=0`); confirm it is CustomMsg (`xfer_format=1`). See the Mid-360 memory note.

### RViz
Open over **WLAN, NOT the VS Code forwarded port** (multi-second lag through the tunnel —
see the camera-lag memory):
```
rviz2 -d /home/unitree/projects/g1/g1_nav/rviz/smoketest.rviz
```
Fixed Frame is `odom`. You should see: the TF tree (`odom -> base_link -> livox_frame`), the
`/livox/points` cloud (floor + walls), and the `local_costmap/costmap` + `global_costmap/costmap`
overlays (costmap color scheme).
- **Stand a pole / box ~1-2 m ahead** -> expect **marked** cells (lethal) **+ inflated** halo in
  the local costmap where the pole is. Move it -> the marks follow (cells clear behind it).
- **Lower a low box onto the floor** -> expect it **marked** too: `min_obstacle_height 0.10`
  keeps returns taller than ~10 cm while dropping the floor.
- If the cloud does NOT land on real surfaces (floor not at `z~0`, walls not vertical), the
  `base_link->livox_frame` extrinsic is wrong — **MEASURE the real LiDAR height** and fix
  `tf.launch.py` before trusting the costmap.

### /cmd_vel validation (NO motion — bridge OFF)
```
# shell A: watch the controller output
ros2 topic echo /cmd_vel
# shell B: send a short local goal ~2 m ahead (or use the "Nav2 Goal" tool in RViz)
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: odom}, pose: {position: {x: 2.0, y: 0.0, z: 0.0}, \
   orientation: {w: 1.0}}}}"
```
DWB should emit a non-zero `geometry_msgs/Twist` on `/cmd_vel` toward the goal. **Nothing
drives** — the bridge is OFF, so `/cmd_vel` goes nowhere. (Identity `odom->base_link` means the
robot never appears to make progress, so the goal won't "complete" — that is expected for a
stationary test; we only care that DWB produces a sane Twist.)

### Teardown
`Ctrl-C` the `smoketest.sh` launch. Confirm the Livox driver exited (frees the Mid-360 for the
mapping / obstacle stack):
```
ros2 topic list | grep livox     # expect NOTHING again
```

> **Reminder:** identity `odom->base_link` means the costmap does NOT track real robot motion.
> This validates plumbing + a pole-in-the-costmap only — it is a STATIONARY test, not navigation.
