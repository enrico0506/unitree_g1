# Unitree G1 — Office Runbook (arriving with the real robot)

Branch `home`, pushed to `origin/main` + `origin/home` + `origin/office` (all aligned — at the office just `git checkout office && git pull`). At home there is no robot/camera/arm, so every on-robot item below is UNVERIFIED here — the `(VERIFY)` markers flag what to confirm live before trusting it.

> ## Update — changes after this runbook was first written
> - **"Greet" is now "🤝 Interaction" mode** (same toggle, same place). When ON the robot
>   responds to a person's gesture: **wave → waves back**. Wave detection now **fuses the
>   skeleton with the MediaPipe palm feed**: a clean shoulder-based wave fires on the skeleton
>   alone, but when the short robot can't see the head/shoulders on a close person, an **open
>   palm oscillating on the wrist** corroborates the (weaker) elbow-only skeleton wave so it
>   still fires — while a lone weak signal never does, keeping false waves down. Every
>   detected gesture is **labelled on the camera feed** (a "👋 wave" chip on that person) and
>   **written to the dashboard log** (`👋 #3 wave → robot wave`, or "held — not safe"). Still
>   OFF by default; only fires upright in Walk, not moving, not busy. (`e58f133`)
> - **Multi-client access model** (`24536e0`): multiple phones/sites can connect at once.
>   Driving (joysticks) + modes + Dance/Climb + hands_up/release_arm stay exclusive to ONE
>   lock owner; **arm greetings (wave/high-five/clap/shake/hug/heart/kiss) can be fired by
>   anyone**. Take-over is gated: you can only seize the drive lock when the robot is stopped
>   ("driving · stop to take"), never mid-walk. Server-enforced.
> - **Handshake ("gives hand")**: auto-detect is NOT shipped (unreliable from a single camera
>   — would false-fire on a casual reach). `shake` is available as a **tap** (any device) →
>   robot offers its hand. Decision to attempt a measured auto-detector is still open.

---

## Status

- **Branch/commit:** `home` @ `b12d341` — this cycle's 6 commits are the tip of `origin/main`.
- **Pushed:** `origin/main` and `origin/home` both carry `b12d341`.
- **Pull onto the OFFICE machine** (office branch fast-forwards to `origin/main`):

```bash
git fetch origin
git checkout office
git merge --ff-only origin/main     # fast-forwards office -> b12d341; fails loudly if office has diverged
git log --oneline -6                # confirm b12d341 ... 722030c are on top
```

If `--ff-only` refuses (office diverged), stop and reconcile — do NOT force. The archive tags (`archive/oldmain`, `archive/feat360`, `archive/nanoowl-detector`) exist for recovery.

---

## What's new this cycle

> The **LiDAR + depth obstacle-detection rebuild** that preceded these features (fine
> detection, depth fusion, the measurement bench, viz overhaul, safety hardening) is
> documented in full — with per-area *what-to-do* — in **`PERCEPTION-REBUILD.md`**. Read
> that for the obstacle pipeline; this section covers the features that came after it.

**Odometry**
- `722030c` — Fused SE2 odometry (leg + IMU + lidar): a pure-numpy complementary filter that fuses leg-odom, IMU yaw-rate and FAST-LIO into one smooth, non-drifting pose (sim: leg-only 4.33 m RMSE → fused 0.04 m). **It is a library, not yet wired in** — zero consumers, and the on-robot adapter's `start()` raises `NotImplementedError` (the FAST-LIO subscribe is a stub). Nothing changes on the robot until someone finishes the wiring.

**Wave-back demo**
- `fcc48ec` — Human-gesture → robot-response reactor (wave-back), proven offline.
- `a3f9ff2` — Dropped the both-arms-up gesture; high-five now fires only at head-level-or-above.
- `b12d341` — Auto wave-back wired end-to-end: `GreetingService` in the web controller + a dashboard **"🤖 Greet"** toggle (OFF by default). Person waves at the head camera → robot waves back.

**Phone**
- `5dafacd` — Phone "Actions" surface: arm gestures (wave/high-five/clap/shake/hug/heart/kiss/hands-up/release) + whole-body Dance (503) / Climb (812) via hold-to-fire, plus UI polish.
- `f2686d9` — Grab-on-first-tap control: any stick touch or tap on a connected phone takes the single-controller lock instantly (no "take control" button gate); handoff zeroes velocity so there's no lurch.

---

## Cold-start boot sequence

Run on the robot's Jetson. The robot must already be **powered and in damp**, e-stop in reach. The web controller is the one backbone process — it fans out to camera + obstacle viz by itself.

```bash
# 1) Environment (every session)
cd /home/unitree/projects/g1
export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds
export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH

# 2) THE BACKBONE — start the dashboard (DDS domain 0 / eth0, serves :8080).
#    Reads the two safety warnings, then press Enter (TTY-gated; systemd starts headless).
#    This ONE process: inits DDS, spawns camera_service.py, auto-starts the domain-99
#    obstacle node for always-on viz, starts DepthNearField + ObstacleGuard (disabled)
#    + GreetingService (OFF).
python3 scripts/robot_web_controller.py

# 3) OPTIONAL — pose container (only for the wave-back / skeleton demo).
#    pose_service.py writes /dev/shm/g1_pose_tracks.json. Demand-gated: the CONTAINER
#    must be running (this step starts it); turning Greet ON then touches POSE_DEMAND to
#    keep the GPU inferring even with the skeleton view closed.
#    NOTE: run_pose.sh ends by `exec docker logs -f` (tails forever) — run it in its own
#    terminal, or background it, then check `docker ps` from another shell.
perception/pose/run_pose.sh          # or, if the container already exists: sudo docker start g1-pose
sudo docker ps                       # confirm g1-pose is Up

# 3b) OPTIONAL but recommended for the wave-back — hands container (palm fusion). Lets a
#     wave still register when the short robot can't see a close person's head/shoulders.
#     Demand-gated too: Interaction mode now heartbeats g1_hands_demand. Without it, only
#     clean shoulder-based waves fire (graceful fallback, no crash). NB: pose + hands + camera
#     share the Jetson GPU — expect some video lag while both infer.
perception/hands/run_hands.sh        # or: sudo docker start g1-hands ; confirm g1-hands is Up

# 4) Open the UI
#    Laptop console: http://<robot-ip>:8080/?full=1
#    Phone teleop:   http://<robot-ip>:8080/phone   (phone-sized devices auto-redirect)

# 5) In the dashboard, bring the robot upright: damp -> stand -> walk (802).
#    (Stand needs the harness on this robot; Walk is harness-free.)

# 6) OPTIONAL — obstacle guard: Drive tab -> Obstacle Avoidance -> "Obstacle Guard: On".
#    Do NOT hand-start run_obstacle.sh; the controller already spawns the node,
#    and mapping is forced off while the guard runs (shared Mid-360 UDP ports).
```

Do NOT normally run by hand (debug/alt paths only):

```bash
# (VERIFY) obstacle node standalone — controller already starts this at boot.
#          run_obstacle.sh ALSO sources env.sh itself (redundant to prefix it) and has
#          `set -e`. Sources /home/unitree/g1_mapping_ws/env.sh (NOT in repo — confirm it
#          exists). Binds the Mid-360 UDP ports — mutually exclusive with mapping/nav.
#          Manager log -> /tmp/g1_obstacle.log; per-frame JSON -> /dev/shm/g1_obstacle.json.
obstacle/run_obstacle.sh

# (VERIFY) production autostart — g1-web.service is NOT in this repo (the repo ships only
#          deploy/unitree-g1web, a sudoers policy scoping `systemctl {start,stop,restart}
#          g1-web` for the unitree user). An out-of-repo copy of the unit lives at
#          /home/enrico/Dokumente/g1_bot/g1-web.service. Whether it's installed/enabled on
#          the robot is unconfirmed. README says run manually.
sudo systemctl start g1-web

# Nav2 / mapping — NOT part of normal boot (obstacle source is `lidar`, not nav2costmap).
#          Conflicts with the auto-started obstacle node over the single Mid-360.
g1_nav/check_prereqs.sh   # verify Nav2 pkgs + live /livox/lidar,/Odometry first
```

**Not runnable as shipped:** the fused-odometry on-robot adapter — `G1FusedOdometryAdapter.start()` raises `NotImplementedError` (`fused_odometry.py:527-529`). Someone must write the FAST-LIO `/Odometry` subscribe before any on-robot odometry test.

---

## Test plan — do this at the office

### Must test on the robot

Ordered so the highest-risk unknowns come first.

1. **Mid-360 actually publishes `/livox/lidar`.** *The entire primary obstacle path depends on it.* Toggle Obstacle Guard On; `tail -f /tmp/g1_obstacle.log` for a ~10 Hz frame rate and increasing `seq`; the dashboard zone chip goes live (CLEAR/SLOW/STOP).
   - **Fail looks like:** no frames within the 8 s startup grace (`guard.py:62 startup_grace_s`) → guard STARTING → FAULT → auto-disables (says "off", brake released). `lidar_source.py:3` warns the Mid-360 "isn't publishing on this robot" — resolve this **before** trusting the guard.

2. **`env.sh` present + single-Mid-360 isolation.** Confirm `/home/unitree/g1_mapping_ws/env.sh` exists (every obstacle/nav script sources it; not in repo). Do NOT also start mapping/nav.
   - **Fail looks like:** missing env.sh → obstacle node won't launch (`set -e`; dashboard still comes up, but 2D/3D viz is dead). Running mapping too → one consumer stalls after a single frame.

3. **Camera feed live.** Dashboard camera panel shows video; MJPEG at `/camera/stream`. Confirm no second `camera_service.py` (single-consumer).
   - **Fail looks like:** blank panel / stale frame.

4. **Wave-back end-to-end (the client path).** Pose container up, Walk FSM, Greet ON, person waves at the head camera ~1.5 s (wrist above shoulder, side-to-side). Caption: "watching for a wave…" → "saw wave -> waving back"; arm waves back once (arm-action 26 / `high_wave` — the native `LocoClient.WaveHand()` is a dead no-op on this G1), debounced.
   - **Fail looks like:** caption stuck on "watching for a wave…" forever = dead pose feed OR arm client failed to init (see run-of-show gates).
   - **Close-range / small-robot case (palm fusion):** stand CLOSE so the head + shoulders leave frame, then wave with an **open palm**. With `g1-hands` up it still fires (the dashboard gesture log shows `source: skeleton+palm`); without `g1-hands` it won't (a weak wave needs corroboration) — that's expected, start the hands container.
   - **Shoulder-level wave (must FIRE):** wave normally with the hand at shoulder height (forearm bent up, not raised overhead) — it now fires (`source: skeleton`). **Held / non-wave (must NOT fire):** raise the forearm and HOLD it still (palm down), or stir/point/clap — the robot must stay put. If a held hand still triggers a wave-back on-robot, the pose jitter is heavier than assumed → raise `wave_swing_min_px` (14.0) in `ReactorConfig`. Known residual: an animated talker doing a genuine open-palm side-to-side at shoulder height can still read as a subtle wave (indistinguishable) — the safety gate limits it to upright/idle.

5. **Startup line `G1 ArmActionClient ready (gestures).`** Watch controller stdout at boot (`robot_web_controller.py:1149`).
   - **Fail looks like:** `ArmActionClient init failed` (`:1152`) → the wave-back now runs as arm-action 26 (`high_wave`), so a dead arm client means it never fires (and the safety gate's `arm_client is not None` check blocks it). Catch this before the demo, not during.

6. **Manual High Five button (arm action 18).** The auto high-five (raise a hand → high-five) was removed; the manual button stays. Tap **High Five** on the dashboard/phone → high-five fires, auto-releases after ~4 s.
   - **Fail looks like:** nothing fires = arm-action service path down (the same `arm` service the wave-back now uses).

7. **Greeting safety gate physically holds arms.** Greet ON but robot in **damp** (not Walk), wave → caption "saw wave -- holding (not safe)", ZERO arm motion. Repeat while driving.
   - **Fail looks like:** any arm motion when not upright+idle+walk = gate broken; stop the demo.

8. **Depth near-field fusion (D435i).** Ships ON (`obstacle.yaml:194`). Watch `depth.front_near` telemetry / `frame_ok`. Place a low cable/box 0.5–2 m ahead (Mid-360 blind zone) → front stop fires.
   - **Fail looks like:** `[DEPTH] frame rejected ... CHECK D435i pitch/height` log → silently falls back to lidar-only. Verify `pitch_deg 47.6` (`obstacle.yaml:195`) / `camera_height_m 1.30` (`:196`) against the real mount.

9. **Footprint self-mask matches the real G1.** Watch per-frame `pts <raw>-><kept>`; a forward leg-swing must not cause a false freeze.
   - **Fail looks like:** too-small footprint = legs false-trigger STOP; too-large = blind to an obstacle at the feet. Tune `self_front/back/half_width` + `robot_half_width_m` minimally (all unmeasured defaults, `obstacle.yaml:79-81,111`).

10. **Floor-fit sanity.** Check first-frame `floor fit a=.. b=.. c=..`. Harness ~-1.5 m, ground ~-1.2 m.
    - **Fail looks like:** fitted plane clips real obstacles or invents a floor.

11. **End-to-end velocity trim while walking.** Drive toward a wall → smooth SLOW then STOP; reverse/yaw/strafe stay free during the stop; no stop-go chatter. Confirm guard-on stops mapping and vice-versa.

12. **Blind-zone predictor + emergency snap.** Drive straight at a wall until it vanishes into the ~1 m front blind cone → dead-reckoned stop still fires; an obstacle inside ~0.45 m effective snaps instantly (shaper bypass).

13. **Phone: grab-on-first-tap steals from laptop, no lurch.** Laptop drives; tap the phone Move stick once → server logs a handoff, laptop chip flips to "○ Phone has control · Take back", robot does NOT jerk.

14. **SAFETY — read-only phone STOP is a no-op.** Laptop drives; a *second* (read-only) phone taps STOP → **nothing happens**. The phone STOP button sends only `{type:"stop"}` with no take-control (`phone.js:686`); the server drops non-owner messages at the ownership gate (`robot_web_controller.py:1709`). Everyone must know: a bystander's phone STOP does not stop a laptop-driven robot — the physical e-stop is the only universal stop.

15. **Laptop take-back is manual.** Phone driving → click laptop chip "Take back" → lock moves, phone sticks go inert. Laptop must NOT auto-reclaim while the phone still holds it.

16. **STOP zeroes velocity, keeps gait hot.** In walk, drive, tap phone STOP → immediate halt, then drivable again WITHOUT re-entering walk (sends `Move(0,0,0)`, never `StopMove`). STOP does NOT lower raised arms or exit dance/climb.

17. **Owner watchdog.** Drive from phone in walk, then lock the phone / kill wifi → stop within ~2 s. Owner closes tab → lock frees, robot stops.

18. **Mode ladder swipe-to-confirm.** Tap Walk, drag knob ≥85% → real FSM transitions (0/1/4/802). A mis-tap without the swipe must NOT switch modes.

19. **Sticks feel + mapping.** In walk, Move up=forward / left=strafe-left, Turn left=yaw-left; gentle (0.5× robot max); dead outside walk; knob anchors to touch-down point.

20. **Dance/Climb hold-to-fire + exit.** Press-hold ~1.3 s to fire (dance 503 / climb 812). Exit via Stand or Walk. Do this on a gantry / clear area.

### Already proven in sim — do NOT spend robot time

All GREEN when re-run at home this cycle. Trust them.

- **7 selftest suites:** `test_cmd_shaper`, `test_fused_odometry`, `test_gesture_reactor`, `test_step_pacer`, `sim_gestures --selftest`, `sim_odometry --selftest`, `sim_perception`. Plus `sim_walk.py --all`.
- **Obstacle geometry** — ground-plane fit, footprint mask, tilt/deskew, occupancy: `obstacle/` 24-test suite passes (re-run green here: `test_deskew`/`test_filters`/`test_ground`/`test_occupancy`/`test_depth_ring`). Don't re-verify the math on the robot.
- **Full closed-loop avoidance** — floor-only no-detect, wall@2m, 6 cm pole@3m, no-collision + smooth (0 accel violations): `sim_perception.py --selftest` exercises the REAL node + REAL guard + REAL shaper. Pipeline logic is proven.
- **Depth ring low-object sensitivity** — 4 cm cable → 33 viz points, sub-clearance noise → 0: `test_depth_ring.py`. Only the real D435i *frame validity* needs the camera (robot item #8).
- **Fused-odometry filter math** — straight/rotation/yaw-wrap, no-lidar==leg-only exactly, drift correction, dropout coasting, outlier + kidnapped-robot re-acquire, NaN guards, drift magnitude (fused ≤0.5× leg): `test_fused_odometry.py` (18/18) + `sim_odometry.py --selftest`.
- **Gesture decision logic** — a wave fires once at ANY height (overhead OR **shoulder-level**, via the elbow-based raised test); a SUBTLE small wave fires only with open-palm corroboration; and a big **non-wave / false-wave catalog fires nothing**: idle, walking, static-reach, single sweep, clap (bimanual veto), circular stir (horizontal-dominance), fast tremor (rate cap), and — the key on-robot fix — a **held / jittery raised hand at distance** (hysteresis swing counting with an absolute px floor, so pose noise can't fake reversals). Proven: `gesture_reactor --selftest` + `sim_gestures --selftest` + `test_gesture_reactor` (53 tests).
- **Velocity ramp / step-pacing on handoff reset** — `cmd_shaper` + `step_pacer` green; handoff just calls `shaper.reset()`/`pacer.reset()`. Only verify the *integration* (no-lurch) via robot item #13.

### Skip / not ready

- **Fused odometry on the robot** — NOT ready. Adapter `start()` raises `NotImplementedError`; changes no runtime behavior until the FAST-LIO `/Odometry` subscribe (+ shared clock, gyro sign check, frame alignment) is built. It never touches `OdomReader` or the control loop, so nothing to test tomorrow unless you finish the wiring first.
- **nav2costmap obstacle backend** (`g1_nav/`) — obstacle source is `lidar` (`obstacle.yaml:9`), so the raw-LiDAR node is the shipped path. Nav2 is the alternative and conflicts over the Mid-360. Out of scope.
- **Phone greeting control** — the Greet toggle is laptop-dashboard-only; drive the demo from the laptop.

---

## Client wave-back demo — run of show

1. Robot powered, e-stop in reach.
2. Start the backbone: `python3 scripts/robot_web_controller.py`. **Watch stdout for `G1 ArmActionClient ready (gestures).`** — if it says init failed, stop and fix; the wave will silently no-op.
3. Start the pose container: `perception/pose/run_pose.sh`; confirm `sudo docker ps` shows **g1-pose Up**. For the close-range case also start the hands container: `perception/hands/run_hands.sh` (**g1-hands Up**) — enables palm fusion.
4. Open `http://<robot-ip>:8080/?full=1` on the laptop (lone laptop auto-holds control via take-if-free, so the toggle is allowed).
5. Bring the robot upright to **Walk (802)**: damp → stand → walk.
6. Flip **"🤖 Greet: off" → ON**. Caption shows "watching for a wave…".
7. Person stands in the **head-camera** FOV and waves ~1.5 s: wrist **above shoulder**, side-to-side.
8. Expect: caption "saw wave -> waving back", arm waves back within ~1.5–2 s, **once** (debounced).

**Silent-fail gates to check if it doesn't wave (debug in this order):**

- **Arm client didn't init** → the wave-back now runs as arm-action 26 (`high_wave`), so the arm client genuinely must be up; the greeting safety gate also requires `arm_client is not None`. No `G1 ArmActionClient ready` line = wave never fires. (Arm-client init + ready/failed print at `robot_web_controller.py:1145-1152`; the `_greeting_safe` gate that checks it at `:1287-1295`.)
- **Dead pose feed** → g1-pose stopped or camera not writing frames shows the SAME "watching for a wave…" as "no one is waving". `sudo docker ps` + check `/dev/shm/g1_pose_tracks.json` is fresh. Nothing distinguishes the two in the UI.
- **Not in Walk** → gate blocks unless upright + idle + Walk FSM. Caption "saw wave -- holding (not safe)" = you're in damp/stand or moving.
- **Robot moving** → `is_moving` blocks the auto-fire; stop driving before demoing.
- **Arms already raised** → `not state.arm_raised` (`:1291`) blocks the wave if a prior `hands_up`/gesture left the arms up. Send `release` (lower the arms) first; there is no distinct caption for this.
- **Just fired / queued** → the gate also blocks while a `pending_cmd` is queued or within the per-gesture debounce (`greeting_busy_until`, `:1292-1294`; wave ~3.5 s). Wait a few seconds between waves.
- **Greet still OFF** → it's OFF by default every boot.

---

## Known issues & decisions needed

- **DECISION — pre-existing walk test failure (NOT from these 6 commits; walk files untouched).** `test_walk_pipeline.py` "reverse overshoot negligible (≤0.025)" FAILS at `-1.53501` vs `-1.525` floor — 0.035 m/s reverse overshoot, physically trivial. Root: commit `ddf87f6` "snappier accel" tightened the ramp; the 0.025 tolerance is now stale.
  - **Option A:** loosen the tolerance to match the new ramp (accept the snappier accel).
  - **Option B:** tighten the `cmd_shaper` reverse ramp to get back under 0.025.
  - Owner picks; does not block anything tomorrow.
- **Depth fusion ships ON but config default disagrees with code.** `obstacle.yaml:194 enabled: true` vs code default OFF (`depth_nearfield.py:31`). On-robot-unvalidated. Self-gates on a frame-sanity check and fails safe to lidar-only, but a mis-measured D435i pitch/height causes silent per-frame rejections. Watch `frame_ok` before relying on it.
- **Mid-360 publishing is the key unknown.** `lidar_source.py:3` note says the Mid-360 "isn't publishing on this robot" (in the 3D-viz fallback context). If genuinely down, the whole obstacle guard FAULTs and auto-disables. Verify first (robot test #1).
- **Two clouds, don't confuse them when debugging.** The dashboard 3D-sphere viz + the D435i depth near-field both come from the **RealSense** (`lidar_source.py` reads `/dev/video0` via V4L2). The **Mid-360** feeds only the ROS2 obstacle_node ring/wedges.
- **Fused odometry is not wired in** — zero consumers; adapter `start()` raises `NotImplementedError`. No runtime effect until built.
- **README stale on two points:** (1) claims g1-web crash-loops on the `input()` prompt — now gated on `sys.stdin.isatty()` (`:1834`), so systemd starts headless; (2) the "run manually" guidance predates the isatty fix.
- **Unmeasured robot params** — footprint self-mask, `robot_half_width_m`, `sensor_height_m 1.20`, `camera_height_m 1.30` are all MEASURE/tune defaults. Tune on the real G1 (robot tests #8–10).
- **`(VERIFY)` unconfirmed-from-repo:** `g1-web.service` (not in repo; only the `deploy/unitree-g1web` sudoers policy is) and `/home/unitree/g1_mapping_ws/env.sh` (sourced by every obstacle/nav script, not in repo). If env.sh is missing on the robot, the obstacle node and all nav fail to launch.
- **Cosmetic:** `scripts/filters.py:391` emits a numpy `RuntimeWarning: invalid value encountered in cast` during the sim run; tests still pass (harmless NaN-in-cast).

---

## Safety

Non-negotiables — read before powering on near people.

- **The physical e-stop is the only universal STOP.** The dashboard/phone STOP only zeroes velocity **for the client that holds the lock**; a read-only phone's STOP does nothing (server drops non-owner messages, `robot_web_controller.py:1709`), and STOP never lowers raised arms or exits dance/climb. Keep the e-stop in hand.
- **Greeting mode is OFF by default and only fires upright in Walk.** The safety gate (`_greeting_safe`, `:1287-1295`) blocks the auto-wave unless the robot is upright + idle + in the Walk FSM (and arms not already raised). Verify it holds arms in damp (robot test #7) before demoing near people. Turn Greet OFF when not actively demoing.
- **Obstacle guard is opt-in and only trims forward speed** — it never steers; reverse/yaw/strafe are always free. It fails safe: no lidar frames → FAULT → auto-disable (forward brake released). Do not treat it as a collision-proof bumper; it is a speed trim, not a wall.
- **Grab-on-first-tap is a footgun.** Any touch on a connected phone steals control from the laptop instantly. Do NOT leave a connected phone unattended while driving from the laptop.
- **Keep the area clear** for Dance (503) / Climb (812) and any whole-body motion — run on a gantry / clear space. Stand needs the harness on this robot.
- **One consumer per sensor.** Don't run mapping/nav while the obstacle node is up (shared Mid-360 UDP ports), and don't start a second `camera_service.py` or a second dashboard (single-consumer head camera).
- The robot's balance/walk control runs on the robot's own boards, not the Jetson — killing the dashboard does not stop the gait; use STOP / mode ladder / e-stop.
