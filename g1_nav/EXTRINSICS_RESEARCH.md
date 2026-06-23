# G1 (EDU U6) Extrinsics & Geometry Research — for Nav2 `<ASK ENRICO>` placeholders

Researched 2026-06-23 from the web, to replace hand-measurement of the Nav2 physical values.

## Gold source used

**Unitree official URDF:** `unitreerobotics/unitree_ros` → `robots/g1_description/g1_23dof_rev_1_0.urdf`
(raw: <https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/g1_description/g1_23dof_rev_1_0.urdf>)

This URDF **does contain the sensor mounts** (contrary to the README, which doesn't mention them).
The relevant fixed joints, quoted exactly from the URDF:

```xml
<!-- base frame of the URDF is `pelvis` -->
<joint name="waist_yaw_joint" type="...">       <!-- pelvis -> torso_link -->
  <origin xyz="-0.0039635 0 0.044" rpy="0 0 0"/>
  <parent link="pelvis"/>  <child link="torso_link"/>
</joint>

<joint name="mid360_joint" type="fixed">         <!-- torso_link -> mid360_link (Livox Mid-360) -->
  <origin xyz="0.0002835 0.00003 0.428434" rpy="3.141592653589793 0.05112069379091391 0"/>
  <parent link="torso_link"/>  <child link="mid360_link"/>
</joint>

<joint name="d435_joint" type="fixed">           <!-- torso_link -> d435_link (RealSense D435i) -->
  <origin xyz="0.0576235 0.01753 0.42987" rpy="0 0.8307767239493009 0"/>
  <parent link="torso_link"/>  <child link="d435_link"/>
</joint>
```

The sensors are children of **`torso_link`**, not `pelvis`. The Nav2 stack uses `base_link`.
Because the `pelvis -> torso_link` joint has **zero rotation**, composing `pelvis -> sensor` is a
simple translation add (torso xyz + sensor xyz) with the sensor's own rpy carried through unchanged.

> **IMPORTANT — base frame mapping:** the URDF root is `pelvis`. Our Nav2/TF uses `base_link`.
> These transforms are **pelvis -> sensor**. They are correct as `base_link -> sensor` **only if
> `base_link` is defined coincident with `pelvis`** (same origin, same orientation). Confirm what
> your `base_link` actually is on the robot before publishing these as static TFs. If `base_link`
> is on the ground (foot-level) or elsewhere, add that offset.
>
> **IMPLEMENTATION DECISIONS taken in `tf.launch.py` (differ from the raw URDF — read this):**
> 1. **`base_link` = the FLOOR projection of the robot, not pelvis.** So the published
>    `base_link -> livox_frame` z is the **full floor->LiDAR height ~1.27 m** (URDF 0.472 above
>    pelvis + ~0.8 m pelvis stance), NOT the URDF 0.472. The `ground_seg`/Patchwork++
>    `sensor_height` is the SAME 1.27 m. (x, y stay as the URDF -0.00368 / 0.00003.)
> 2. **LiDAR roll is set to 0, NOT pi.** The Livox driver already applies the 180-deg (roll=pi)
>    mount de-roll to the points it publishes, so `livox_frame` comes out UPRIGHT. The URDF mount
>    roll=pi and the driver de-roll CANCEL, leaving only the mount pitch (0.05112). Publishing
>    roll=pi on top would double-flip (floor on the ceiling). **Verify in RViz** that floor points
>    read below the sensor. The camera (`d435_link`) keeps its URDF pitch 0.83078 and z 0.47387
>    (camera is a Phase-3/optional source, not yet de-rolled by a driver).

Note: `g1_dual_arm.urdf` has the LiDAR/camera z **0.01 m lower** than `g1_23dof_rev_1_0.urdf`
(known discrepancy, unitree_rl_gym issue #72). I used the `_rev_1_0` 23-DOF file.

---

## Findings table

| # | Value | Number | Source | Confidence | Verify on robot? |
|---|-------|--------|--------|-----------|------------------|
| 1 | **base_link(pelvis) -> Mid-360 LiDAR** | xyz = **(-0.00368, 0.00003, 0.47243)** m ; rpy = **(π, 0.05112, 0)** = **(3.14159, 0.05112, 0.0)** rad | unitree_ros g1_23dof_rev_1_0.urdf (`mid360_joint` ∘ `waist_yaw_joint`) | **High** (URDF), conditional on base_link≡pelvis | Yes — confirm base_link definition; the **roll = π** means the Mid-360 frame is mounted "upside-down" relative to pelvis — keep it, don't zero it |
| 2 | **base_link(pelvis) -> RealSense depth cam** | xyz = **(0.05366, 0.01753, 0.47387)** m ; rpy = **(0, 0.83078, 0)** rad | unitree_ros g1_23dof_rev_1_0.urdf (`d435_joint` ∘ `waist_yaw_joint`) | **High** (URDF), conditional on base_link≡pelvis | Yes — note pitch ≈ **0.831 rad ≈ 47.6° downward-ish tilt**; this is the *optical mount* frame, the RealSense optical frame adds the usual -90/0/-90 rotation if you publish `camera_link` vs `camera_depth_optical_frame` |
| 3 | **Mid-360 height above floor** (Patchwork++ `sensor_height`) | **≈ 1.27 m** (best estimate). = lidar z above pelvis (0.4724 m) + pelvis stance height (~0.80 m sim / ~0.74–0.79 m real) | URDF + unitree_rl_gym `g1_config.py` base init `pos=[0,0,0.8]` | **Medium** (posture-dependent) | **YES — MEASURE.** With pelvis at 0.80 m → 1.265 m; at 0.74 m → 1.21 m. The old obstacle.yaml default of 1.10 m is **too low** — closer to ~1.2–1.27 m. Patchwork++ is sensitive to this; tune on robot |
| 4a | **Head depth camera model** | **Intel RealSense D435i** | Weston Robot G1 dev guide; URDF link name `d435_link` | **High** | No |
| 4b | **D435(i) DEPTH FOV (H × V)** | **87° × 58°** = **1.5184 rad × 1.0123 rad** (datasheet tolerance 87°±3° × 58°±1°) | Intel RealSense D435i product spec / D400 datasheet 337029-005 | **High** | No |
| 5a | **G1 body footprint (top-down)** | overall **width ≈ 0.45 m** (shoulder), **depth ≈ 0.20 m** (front-back), height 1.32 m | Unitree G1 datasheet / RoboStore EDU specs (1320×450×200 mm) | **High** (static dims); **Medium** as a *moving* footprint | Yes — feet swing wider than torso; pad it (see note below) |
| 5b | **Proposed footprint polygon** (base_link m, x fwd / y left, CCW) | half-width y = 0.25, half-depth x = 0.20 → **`[[0.20,0.25],[0.20,-0.25],[-0.20,-0.25],[-0.20,0.25]]`** | Derived from 5a + ~0.10 m walking/arm margin on each side | **Medium** (inference) | Yes — start here, widen if the gait clips obstacles. A 0.30 m **radius circle** footprint is a safer first cut for a swaying biped |
| 6a | **Max forward speed** | **2.0 m/s** rated (spec); **~1.5 m/s** practical/comfortable | Unitree spec; SVRC review; matches dashboard `max_vx: 1.5` | **High** (rated), **High** (practical from your own config) | Cap Nav2 `max_vel_x` at **1.0–1.5 m/s**, not 2.0 |
| 6b | **Max lateral (strafe) speed** | **≈ 0.5 m/s** (conservative) — not officially published; biped strafe is much slower than forward | Inference (Unitree publishes only the 2 m/s forward figure) | **Low** | YES — set from the dashboard's real `max_vy`; treat 0.5 m/s as a safe ceiling |
| 6c | **Max yaw rate** | **≈ 0.6 rad/s** conservative (older Unitree SDK `rotateSpeed` range ±0.6 rad/s; not a published G1 number) | Inference from Unitree legged SDK `comm.h` | **Low** | YES — set from the dashboard's real turn cap |
| 7 | **Inscribed radius** (Nav2 `inflation_radius` basis) | footprint inscribed radius ≈ **0.20 m** (half the 0.40 m depth) → recommend `inflation_radius` ≈ **0.35–0.45 m** (inscribed + margin) | Derived from footprint #5 | **Medium** (inference) | Yes — the existing 0.45 m placeholder is reasonable; keep ≥ 0.35 m |

---

## Direct values for the config / launch files

### `g1_nav/launch/tf.launch.py` — static transforms

`static_transform_publisher` arg order is **x y z yaw pitch roll** (Foxy), so:

```
# base_link -> livox_frame  (= base_link -> mid360_link)
#   x        y         z         yaw   pitch     roll
  -0.00368   0.00003   0.47243   0.0   0.05112   3.14159

# base_link -> camera_link  (= base_link -> d435_link, mount frame)
#   x        y         z         yaw   pitch     roll
   0.05366   0.01753   0.47387   0.0   0.83078   0.0
```
(rpy from URDF: lidar roll=π pitch=0.05112 yaw=0 ; cam roll=0 pitch=0.83078 yaw=0.
Re-ordered to yaw pitch roll for the CLI.)

### Patchwork++ `sensor_height`
```
sensor_height: 1.27   # MEASURE on robot; 0.4724 (URDF) + ~0.80 (stance). Was 1.10 (too low).
```

### Nav2 `nav2_params.yaml`
```yaml
footprint: "[[0.20,0.25],[0.20,-0.25],[-0.20,-0.25],[-0.20,0.25]]"   # x fwd, y left, CCW
inflation_radius: 0.40
# RealSense depth FOV (radians):
horizontal_fov_angle: 1.5184   # 87 deg
vertical_fov_angle:   1.0123   # 58 deg
# velocity caps (start conservative, raise after testing):
max_vel_x:        1.0     # rated 2.0, practical 1.5 — start lower for Nav2
max_vel_y:        0.5     # LOW confidence — set from dashboard max_vy
max_vel_theta:    0.6     # LOW confidence — set from dashboard turn cap
```

---

## Honesty notes on confidence

- **Transforms (#1, #2): HIGH** — straight out of Unitree's official URDF. The only caveat is the
  `base_link` vs `pelvis` identity, which must be confirmed on the robot. The numbers themselves are exact.
- The Mid-360 `roll = π` is real (sensor mounted inverted); the D435 `pitch ≈ 0.83 rad` (downward tilt) is real.
  Do **not** "clean these to zero."
- **sensor_height (#3): MEDIUM** — the 0.4724 m lidar-above-pelvis is exact; the pelvis-above-ground
  (~0.74–0.80 m) is posture-dependent and the dominant uncertainty. Measure with a tape on the standing robot.
- **Footprint / radius / lateral & yaw speeds (#5b, #6b, #6c, #7): LOW–MEDIUM (inference).** Unitree only
  publishes the 2 m/s forward figure and static body dimensions. Lateral/yaw caps and the *swept* footprint
  are not published — derive the real caps from your own dashboard limits (the project memory records
  `max_vx: 1.5`).
- D435i model and 87°×58° depth FOV: **HIGH** (Intel datasheet + Unitree dev guide naming).

## Sources
- Unitree URDF (gold): https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_23dof_rev_1_0.urdf
- Sensor pos discrepancy issue: https://github.com/unitreerobotics/unitree_rl_gym/issues/72
- Base init height: https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py
- G1 sensors (D435i + Mid-360): https://docs.westonrobot.com/tutorial/unitree/g1_dev_guide/
- Intel RealSense D435i spec (depth FOV 87×58): https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html
- Intel D400 datasheet 337029-005: https://cdrdv2-public.intel.com/841984/Intel-RealSense-D400-Series-Datasheet.pdf
- G1 dimensions/speed: https://robostore.com/blogs/news/unitree-g1-edu-ultimate-technical-specifications
- G1 datasheet (ROS Components): https://www.roscomponents.com/wp-content/uploads/2024/11/G1-UNITREE-DATASHEET.pdf
- G1 speed (review): https://www.roboticscenter.ai/blog/unitree-g1-review
