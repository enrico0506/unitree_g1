# motion/sim — run any motion in sim2sim

The easy button for "does this motion actually work": takes a clip from
`motion_library/` and runs it through HoloMotion's real MuJoCo sim2sim eval
against the shared general-purpose tracking policy, so you get a video +
trajectory dump without training anything per-motion.

```bash
./motion/sim/run_holomotion.sh wave_v2          # run it -- watch live in your browser (see below)
./motion/sim/run_holomotion.sh --list           # see what's in motion_library
./motion/sim/app.sh                             # the "app": pick a motion from a menu, repeat
```

Output lands in `motion/holomotion_ckpt/exported/mujoco_output_model_14000/`
as `<motion>_holomotion.mp4` (rendered rollout) and
`<motion>_holomotion_robot.npz` (actual robot trajectory).

**Heads up — shared 16GB Jetson RAM/GPU.** If the robot's perception
(g1-detect/g1-pose/g1-hands) is running, pause it first or the eval can get
OOM-killed mid-run:

```bash
docker stop g1-detect g1-pose g1-hands
./motion/sim/run_holomotion.sh <name>
docker start g1-detect g1-pose g1-hands
```

## Watching it live: your browser (recommended)

Every headless run streams its frames live. `run_holomotion.sh` prints a URL
the moment it starts:

```
Live view: http://192.168.123.164:8098/  (open this in your browser now)
```

Open that on your laptop (any browser, no X server or SSH juggling needed)
and leave the tab open — it shows whatever's currently playing, and just
sits idle on the last frame between runs. `app.sh` does this automatically
too, once, the first time you run it in a session.

This works over an ordinary SSH connection, VS Code Remote-SSH included —
no X11 forwarding involved. Frames come from the same EGL-rendered path the
headless mp4 output already uses, just also dropped live as a JPEG
(`motion/sim/live/frame.jpg`) that `live_view_server.py` serves as an MJPEG
stream. `run_holomotion.sh` starts that server automatically if it isn't
already running.

## `--gui`: the real interactive MuJoCo window (usually not worth it)

`./motion/sim/run_holomotion.sh <name> --gui` X11-forwards the actual
`mujoco.viewer.launch_passive` window instead. It's wired up and the plumbing
works (DISPLAY/XAUTHORITY forwarding, host networking so the container can
reach your session's X proxy) — but on this hardware it hits a real ceiling:
this image's NVIDIA driver has no GLX vendor registration and indirect GLX
(the classic X11-forwarded 3D protocol) is capped at an ancient OpenGL
version, so MuJoCo's renderer refuses the context (`OpenGL version 1.5 or
higher required`) even once the X11 plumbing itself is correct. Confirmed
2026-08-11 — not a config issue, a protocol-level ceiling.

If you want to chase this further, [VirtualGL](https://virtualgl.org/) is
the real fix (renders locally on the actual GPU, ships finished frames
instead of GL commands, sidesteps indirect GLX entirely) — more setup, and
Tegra/Jetson GPUs aren't its most common target. The browser view above
gets you "watch it happen live" today without any of that.

If you still want to try `--gui`: get an X server running on your laptop
(macOS: [XQuartz](https://www.xquartz.org/); Windows: [MobaXterm](https://mobaxterm.mobatek.net/)
portable edition works well and needs no admin rights, or
[VcXsrv](https://sourceforge.net/projects/vcxsrv/); Linux: already have one),
connect with `ssh -X unitree@192.168.123.164` (a plain terminal — **VS
Code's Remote-SSH tunnel doesn't carry X11**), sanity-check with `xclock`
first, then `./motion/sim/run_holomotion.sh <name> --gui`.

## How it's wired

`run_holomotion.sh` (host) starts/creates the `holomotion_sim2sim` docker
container and execs `sim2sim.py` (container-side) inside it. That container
bind-mounts `motion/holomotion`, `motion/holomotion_ckpt`, `motion/motion_library`,
and this `motion/sim` folder itself — so any motion you drop into
`motion_library/single/` or `combined/` is automatically pickable by name,
no extra wiring needed. It also runs on the host's network namespace
(`--network host`), needed for `--gui`'s X11 forwarding to reach your SSH
session's X proxy.

If the container doesn't exist (fresh machine, or it got removed),
`run_holomotion.sh` bootstraps it automatically via `setup_container.sh` —
no manual `docker run` needed. `setup_container.sh` reuses a local
`holomotion-sim2sim-provisioned` image if one exists (has the extra pip deps
the upstream deploy image doesn't ship — mujoco, hydra, ray, pandas,
tabulate, tqdm), or builds+saves one from scratch if not.

`live_view_server.py` runs on the host (not in the container) — it just
tails `motion/sim/live/frame.jpg`, which the container writes straight to
via the `motion/sim` bind mount, and re-serves it as MJPEG. The write hook
itself is a small, guarded addition to the vendored
`eval_mujoco_sim2sim.py` (search `motion/sim live-view hook` in that file) —
no-op unless `LIVE_STREAM_FRAME_PATH` is set, so it doesn't touch normal
headless/`--gui` behavior at all.
