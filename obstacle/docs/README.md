# Obstacle avoidance — LiDAR proximity guard

A self-contained slow-down / stop / 360° directional-gating layer for the G1, driven
by the head **Livox Mid-360**. It is **opt-in** (off by default) and never replaces
the operator: it only *trims* the velocity the operator already commands.

## The one-number idea

Everything starts from a single number: **the nearest obstacle distance in the
wedge straight ahead of the robot.** From that one number we decide a `zone` and a
forward-speed multiplier:

```
front distance ≥ slow_at_m (2.0)  → CLEAR  → full speed
        between slow_at and stop_at → SLOW   → speed scaled linearly toward 0
front distance ≤ stop_at_m (0.7)  → STOP   → forward velocity = 0
```

```
speed_scale = clamp((front_filt_m - stop_at_m) / (slow_at_m - stop_at_m), 0, 1)
            (front_filt_m == null → 1.0, i.e. nothing seen → no limit)
```

Reverse, yaw and strafe are **never** blocked by the slow/stop logic — you can
always back away. Side wedges (left / right) raise indicator flags and spoken
warnings but do not brake. A full-circle **360° ring** of per-sector distances lets
the guard brake in the actual **travel direction** (not just the four wedges); the
robot only ever *slows or stops* — it never autonomously steers.

## Architecture — two halves and a shared-memory bridge

The feature is split across two ROS/SDK domains that cannot share a process, so
they talk through one atomically-written JSON file in `/dev/shm`:

```
        ROS2 Foxy, domain 99                 dashboard process, domain 0
   ┌──────────────────────────┐         ┌────────────────────────────────────┐
   │  obstacle_node.py         │         │  ObstacleManager (manager.py)      │
   │  (rclpy + livox msgs)     │  spawns │   starts/stops the node subprocess  │
   │                           │◀────────┤   (Popen of run_obstacle.sh)        │
   │  /livox/lidar  ─▶ filter  │         │                                     │
   │   ground-plane removal +  │         │  ObstacleGuard (guard.py)          │
   │   footprint mask, robust  │         │   reads shm ~20 Hz, caches it,      │
   │   nearest, per-dir dist,  │  write  │   guard.apply(vx,vy,vyaw) trims     │
   │   360° ring distances     │ ──────▶ │   the command at 30 Hz before it    │
   │                           │  shm    │   reaches the Unitree SDK.          │
   └──────────────────────────┘    │    └────────────────────────────────────┘
                                    ▼
                       /dev/shm/g1_obstacle.json
            (node WRITES atomically — tmp + os.replace; guard READS by mtime)
```

* **`obstacle_node.py`** — the perception half. Needs the sourced ROS2
  environment (rclpy + Livox messages, `ROS_DOMAIN_ID=99`). Runs as its own
  process; it does all the geometry and is the **source of truth** for the JSON
  contract and every sign convention below.
* **`guard.py` (`ObstacleGuard`)** — runs *inside* the dashboard (domain 0,
  Unitree SDK, **no** rclpy). It only reads the shm file and scales / hard-stops the
  commanded velocity. It also runs the fail-safe state machine and the spoken /
  LED warnings.
* **`manager.py` (`ObstacleManager`)** — starts/stops `obstacle_node.py` as a
  subprocess, mirroring `scripts/map_builder.py`'s lifecycle exactly
  (`start_new_session=True`, `os.killpg(..., SIGINT)`, `wait`, `SIGKILL`
  fallback).
* **`web/obstacle.js`** — browser panel: the toggles, the readouts, the zone
  chip / speed bar, and the FAULT / auto-disabled banners.

## Staged build (Stages 0–8)

The feature was built in checkpoints, each independently testable:

| Stage | Checkpoint |
|-------|-----------|
| **0** | Package skeleton, `obstacle.yaml` tuning file, `run_obstacle.sh` bring-up. |
| **1** | Node subscribes to `/livox/lidar`, prints frame rate / point counts. |
| **2** | Filter: tilt-aware ground-plane removal + robot footprint self-mask (see below), range crop. (Legacy height band — `min/max_height_m`, `blind_radius_m` — kept for `ground_removal: false`.) |
| **3** | Robust nearest distance in the front wedge (`min_cluster_points`-th point), EMA-smoothed (`filter_alpha`); writes `front_m` / `front_filt_m` to shm. |
| **4** | Per-direction distances (`left/right/back_m`), `zone`, `speed_scale`, side flags. Guard applies the slow/stop + hard-stop floor. |
| **5** | Spoken announcements + LED warnings (cooldown-gated, off-thread). |
| **6** | Fail-safe state machine: STARTING → ACTIVE → FAULT → auto-disable. |
| **7** | Dashboard wiring: WebSocket protocol, `ObstacleManager`, mutual-exclusion with mapping, the `web/obstacle.js` UI. |
| **8** | Full-circle **360° ring** (`ring_bin_deg` sectors) with a near-field tripwire. The guard gates vx/vy by the ring in the **travel direction** (`ring_gating`), a strict superset of the four wedges. Drives the dashboard 2D radar + 3D sphere. |

## How to run

Obstacle avoidance and mapping **cannot run at the same time** (see caveat below);
the dashboard enforces this. For a standalone bring-up of the perception node:

```bash
# 1) sourced ROS2 env (rclpy + Livox msgs, ROS_DOMAIN_ID=99)
source ~/g1_mapping_ws/env.sh

# 2) start the perception node (writes /dev/shm/g1_obstacle.json)
obstacle/run_obstacle.sh
#    ...log goes to /tmp/g1_obstacle.log
```

In normal operation you do **not** run the node by hand — start it from the
dashboard:

1. Open the dashboard, go to the **Drive** tab → **Obstacle Avoidance** panel.
2. Toggle **Obstacle Guard: On**. The dashboard launches `obstacle_node.py`
   (via `ObstacleManager`) and the guard begins trimming velocity. Mapping is
   forced off while the guard is on.
3. Optionally enable **Depth fusion (D435i)** to add the near-ground blind-zone
   fill (independent of the guard; toggle it to validate the reading).

The panel shows live front distance, a coloured zone chip (CLEAR/SLOW/STOP), a
speed-scale bar, L/R side indicators, and a prominent banner on FAULT or
auto-disable.

## Ground-plane removal + footprint self-mask (the point filter)

The first stage of every frame decides **which LiDAR points are real obstacles**.
A blunt sensor-frame height band cannot do this: the *flat floor*, the *robot's own
legs/feet*, and a *real low obstacle* (a harness pole, a gantry foot on the floor)
all sit **low and close**. Raise the band and it goes blind to low obstacles and
drives into them; lower it and it false-freezes on the floor and its own legs. A
height cut alone cannot separate them — so we use **spatial + ground-relative**
discrimination instead (`ground_removal: true`, the default).

**1. Tilt-aware ground-plane removal.** The floor is **not** assumed level: a
walking humanoid sways a few degrees, so a fixed `z` threshold is unreliable.
Instead we fit the local floor as a tilted plane `z = a·x + b·y + c`:

* *Candidates* = in-range points that are clearly low: `z < −sensor_height_m +
  ground_band_m`.
* If there are at least `ground_min_points` candidates, fit `a, b, c` by least
  squares (`np.linalg.lstsq` on `[x, y, 1] → z`), with **one** light
  robustification refit that drops candidates whose residual > 0.15 m.
* Otherwise (too few candidates) fall back to a **level** floor: `a = b = 0`,
  `c = −sensor_height_m`.
* The fit is **sanity-bounded**: if `|a|` or `|b|` > 0.4 (≈ 22° tilt — implausible)
  or `|c + sensor_height_m|` > 0.5 m, it is discarded for the level fallback, so a
  bad fit can never invent a floor that clips real obstacles or hide them.

Each point's height **above the fitted floor** is `above = z − (a·x + b·y + c)`. We
keep points with `ground_clearance_m < above < obstacle_max_above_m`: the floor
itself drops out, anything sticking up more than ~8 cm is kept (even when it is low
and close), and the ceiling / overhead above ~1.9 m is cut.

**2. Robot footprint self-mask.** The robot's own legs/feet are removed by
**position, not height** — a box around the base:

```
in_self = (x > −self_back_m) & (x < self_front_m) & (|y| < self_half_width_m)
```

Points inside that box are dropped; everything outside it is judged purely by the
ground-plane test. So a forward leg swing inside the box is ignored, but a harness
pole just outside it that sticks up from the floor is **kept**, even when it is low
and close — exactly the case the old height band could not handle.

The final keep mask is `in_range & keep_ground & ~in_self`. Everything downstream
(the wedge robust-nearest, EMA, scales, `tight_factor`, the 360° ring, the shm
write) is **unchanged** — only the set of kept points differs.

> ⚠ **Measure the footprint for the real robot.** `self_front_m`, `self_back_m` and
> `self_half_width_m` default to a box that should cover the G1's stance and a
> normal leg swing, but they are **not** measured on your machine — tune them.
> **Residual caveat:** a leg swung *beyond* `self_front_m` lands outside the box
> and **can still be seen** as an obstacle (a transient false freeze). If that
> happens, widen the footprint — but only as far as needed, since a too-large box
> blinds the robot to genuine obstacles right in front of its feet.

**Reverting.** Set `ground_removal: false` to fall back to the exact old behaviour
— the `[min_height_m, max_height_m]` band plus `blind_radius_m`. Those three keys
are kept in config and code solely for that fallback path.

The first-frame sanity log now also prints the fitted floor tilt
(`floor fit a=… b=… c=…`) so the plane fit is observable while tuning, and the
per-frame status line reports the kept-point count (`pts <raw>-><kept>`).

## Tuning (`obstacle/obstacle.yaml`)

`obstacle.yaml` is the **single source of tuning**, read by both halves. Every key
also has a baked-in default in code, so a missing key never crashes a consumer.

| Key | Default | Meaning |
|-----|---------|---------|
| `topic` | `/livox/lidar` | Mid-360 point-cloud topic (already x-fwd/y-left/z-up). |
| `sensor_height_m` | `1.10` | Mid-360 height above floor — **measure on your robot**. Floor sits at z ≈ −this. |
| `obstacle_range_m` | `2.0` | Farthest horizontal point kept. |
| `slow_at_m` | `2.5` | Start slowing at this distance — earlier = more gradual onset. |
| `stop_at_m` | `0.7` | Hard-stop at/under this distance (enforced per direction). |
| `ground_removal` | `true` | Use ground-plane removal + footprint self-mask. `false` = legacy height band. |
| `ground_clearance_m` | `0.08` | Keep points sticking up MORE than this above the fitted floor (floor itself drops). |
| `ground_band_m` | `0.5` | Floor-candidate points are below `sensor_height + this` (used only to FIT the plane). |
| `ground_min_points` | `80` | Min candidates needed to fit a plane (else assume a level floor). |
| `obstacle_max_above_m` | `1.9` | Cut points higher than this above the floor (ceiling / overhead). |
| `self_front_m` | `0.35` | Footprint self-mask: forward extent (covers a leg swing) — **measure/tune**. |
| `self_back_m` | `0.35` | Footprint self-mask: rearward extent — **measure/tune**. |
| `self_half_width_m` | `0.30` | Footprint self-mask: half-width (covers the leg stance) — **measure/tune**. |
| `min_height_m` | `-0.4` | **Legacy** (`ground_removal: false` only): height-band lower edge (cuts the floor). |
| `max_height_m` | `0.50` | **Legacy** (`ground_removal: false` only): height-band upper edge (cuts ceiling / overhead). |
| `blind_radius_m` | `0.45` | **Legacy** (`ground_removal: false` only): drop self-returns closer than this. |
| `min_cluster_points` | `10` | k-th-closest point = robust nearest (rejects single-point noise). |
| `filter_alpha` | `0.25` | Per-direction distance EMA; lower = smoother. |
| `ease_scale` | `true` | Node: smoothstep S-curve for the distance→scale ramp (`false` = linear). |
| `tight_open_m` | `1.6` | Node: a direction starts counting as "closing in" below this. |
| `tight_min` | `0.45` | Node: floor of the global tight-space multiplier (slowest overall factor). |
| `scale_slew_per_s` | `1.8` | Guard: max change in the applied speed-scale per second (smoothness). |
| `announce_cooldown_s` | `3.0` | Min seconds between repeats of the same announcement. |
| `speaker_id` | `0` | `AudioClient.TtsMaker` speaker id. |
| `led_warnings` | `true` | Tint LED orange (slow) / red (stop). |
| `stale_timeout_s` | `1.0` | No fresh frame for this long while enabled → FAULT. |
| `startup_grace_s` | `8.0` | After enabling, wait this long for the first frame before faulting. |
| `fault_stop_s` | `3.0` | On FAULT: stop forward for this long, announce, then auto-disable. |
| `ring_bin_deg` | `5.0` | Ring angular resolution → 360/this sectors. |
| `ring_min_points` | `3` | Points before a ring sector reads a distance. |
| `tripwire_range_m` | `1.6` | Near-field only: fire the tripwire for returns within this range. |
| `tripwire_min_points` | `2` | Min returns inside `tripwire_range_m` to report a sector's nearest point. |
| `ring_gating` | `true` | Guard gates vx/vy by the ring in the travel direction (`false` → legacy 4-wedge). |
| `direction_cone_deg` | `25.0` | Min half-angle of the travel-direction gating cone (floor). |
| `direction_cone_max_deg` | `70.0` | Cap on the auto-widened cone half-angle. |
| `robot_half_width_m` | `0.25` | Half the G1 body width — **measure**; sizes the swept corridor. |
| `clearance_margin_m` | `0.10` | Extra room demanded on each side of the travel corridor. |
| `enabled_default` | `false` | Guard off until the operator turns it on. |
| `publish_hz` | `10` | Node processing / shm-write rate. |

## Fail-safe behaviour

The guard fails **safe and loud**, then gets out of the way:

```
DISABLED ──operator enables──▶ STARTING ──first fresh frame──▶ ACTIVE
   ▲                              │ no frame within startup_grace_s          │ frames go stale
   │                              ▼                                          ▼ (age > stale_timeout_s)
   └──── auto-disable ◀──── FAULT ◀──────────────────────────────────────────┘
        (enabled=False,      for fault_stop_s (3.0 s):
         auto_disabled=True)   · zero FORWARD vx only (reverse/yaw/strafe stay free)
                               · speak "obstacle detection failed, inactive now" ONCE
                               · after fault_stop_s → auto-disable, then pass-through
```

* **STARTING** does *not* brake — it waits up to `startup_grace_s` for the first
  frame.
* In **FAULT** only forward motion is cut (`vx = min(vx, 0)`); you can always back
  out or turn. The announcement is spoken **once**, off-thread (TTS is a blocking
  RPC — never called inline).
* After `fault_stop_s` the guard **auto-disables itself** (`enabled=False`,
  `auto_disabled=True`) so the UI shows the banner *"guard auto-disabled —
  re-enable when sensor is back"*. From then on `apply()` is a pure pass-through
  until the operator re-enables.

## ⚠ CRITICAL caveat — obstacle avoidance and mapping are mutually exclusive

Both the obstacle node and the FAST-LIO mapping pipeline drive the **same head
Mid-360 over the same Livox UDP ports**. Only one Mid-360 driver can bind those
ports at a time, so **obstacle avoidance and mapping cannot run simultaneously.**
The dashboard enforces this mutual exclusion: enabling the guard stops mapping and
vice-versa. Do not start `run_obstacle.sh` by hand while mapping is running (or
the reverse) — the second driver will fail to bind.

## JSON contract — `/dev/shm/g1_obstacle.json`

`obstacle_node.py` is the **source of truth** for this contract. The node WRITES
the file atomically (write to `path + ".tmp"`, then `os.replace(tmp, path)`); the
guard READS it (polling by mtime). `null` means *clear / no obstacle / infinite
distance*.

```jsonc
{
  "seq": 0,                 // increments each processed frame
  "stamp": 0.0,             // node wall clock = time.time()
  "front_m": null,          // RAW robust nearest distance in the front wedge
  "front_filt_m": null,     // EMA-smoothed front distance (drives zone/speed_scale)
  "left_m": null, "right_m": null, "back_m": null,
  "back_filt_m": null,      // EMA-smoothed back distance
  "left_filt_m": null,      // EMA-smoothed left distance
  "right_filt_m": null,     // EMA-smoothed right distance
  "zone": "CLEAR",          // "CLEAR" | "SLOW" | "STOP" (driven by front_filt_m)
  "speed_scale": 1.0,       // 0.0..1.0 forward multiplier (== scale_front, back-compat)
  "scale_front": 1.0,       // 0..1 smoothstep scale from front_filt_m
  "scale_back": 1.0,        // 0..1 smoothstep scale from back_filt_m
  "scale_left": 1.0,        // 0..1 smoothstep scale from left_filt_m
  "scale_right": 1.0,       // 0..1 smoothstep scale from right_filt_m
  "tight_factor": 1.0,      // 0..1 global multiplier (1 = open, → tight_min when boxed in)
  "side": { "left": false, "right": false },   // real obstacle in the L/R wedge
  "ring": {
    "n": 72,                // sector count (360 / bin_deg)
    "bin_deg": 5.0,         // sector angular width
    "start_deg": -180.0,    // sector i center = start_deg + (i + 0.5) * bin_deg
    "dist": [null, null]    // per-sector robust-nearest distance (m), or null = clear
  },
  "points": [ /* x0,y0,z0, x1,y1,z1, ... */ ]  // decimated kept obstacle points (3D sphere viz)
}
```

### Frames & sign convention (source of truth: `obstacle_node.py`)

* **x = forward, y = LEFT, z = UP.** Origin at the sensor; floor at z ≈
  −`sensor_height_m`.
* `angle = atan2(y, x)`: **0° = straight ahead, +90° = left, −90° = right,
  ±180° = behind.**
* **`ring.dist[i]`** is the nearest obstacle in sector `i`, whose center angle is
  `start_deg + (i + 0.5) * bin_deg` in the same convention (+ = left).
* **`points`** is a flat `[x,y,z, …]` list in the viz frame (x fwd, y left, z =
  height above the fitted floor), decimated to `viz_max_points`.

The Livox driver already applies the 180° mount roll before publishing, so
`/livox/lidar` is **already** x-fwd / y-left / z-up — the node must **not**
re-rotate it.
