# motion/sim — run any motion in sim2sim

The easy button for "does this motion actually work": takes a clip from
`motion_library/` and runs it through HoloMotion's real MuJoCo sim2sim eval
against the shared general-purpose tracking policy, so you get a video +
trajectory dump without training anything per-motion.

```bash
./motion/sim/run_holomotion.sh wave_v2          # run it
./motion/sim/run_holomotion.sh --list           # see what's in motion_library
./motion/sim/run_holomotion.sh cartwheel --gui  # interactive viewer instead of headless+video
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

## Watching it live: `--gui`

`./motion/sim/run_holomotion.sh <name> --gui` opens the real interactive
MuJoCo window (`mujoco.viewer.launch_passive`) instead of writing a video —
X11-forwarded from the container, through the Jetson, to your laptop.

This needs an actual X11-forwarded SSH connection, which **VS Code's
Remote-SSH tunnel does not provide** (it doesn't carry X11). Use a plain
terminal instead:

1. **Get an X server running on your laptop** (skip if Linux — you already
   have one):
   - macOS: install & launch [XQuartz](https://www.xquartz.org/)
   - Windows: install & launch [VcXsrv](https://sourceforge.net/projects/vcxsrv/) (or Xming/X410)
2. **Open a plain terminal** (not VS Code's) and connect with X11 forwarding:
   ```bash
   ssh -X unitree@192.168.123.164
   ```
3. **Sanity-check forwarding works before touching MuJoCo** — a window
   should pop up on your laptop within a couple seconds:
   ```bash
   xclock
   ```
   If nothing appears: the X server on your laptop isn't running, or `-X`
   didn't take (try `-Y` for trusted forwarding, some setups need it). This
   is entirely laptop-side — the Jetson's sshd already has `X11Forwarding
   yes` and `xauth` installed, so if `xclock` fails, the problem isn't here.
4. **Run it**, from that same terminal:
   ```bash
   cd ~/projects/g1
   ./motion/sim/run_holomotion.sh wave_v2 --gui
   ```

Expect it to feel laggy — GLX rendering over a forwarded X11 connection
sends every draw command over the SSH link, unlike the default headless
path (which renders locally via EGL and only ships out a finished video).
Fine for eyeballing whether a motion tracks or falls; not fine for judging
smoothness.

### The "app": `app.sh`

`./motion/sim/app.sh` is a picker that stays open — run it once, pick a
motion from the numbered menu, watch it live, land back on the menu for the
next one. No retyping `run_holomotion.sh <name> --gui` each time:

```bash
cd ~/projects/g1
./motion/sim/app.sh
```

Same X11 setup as above applies (this is just `run_holomotion.sh ... --gui`
under a menu loop).

## How it's wired

`run_holomotion.sh` (host) starts/creates the `holomotion_sim2sim` docker
container and execs `sim2sim.py` (container-side) inside it. That container
bind-mounts `motion/holomotion`, `motion/holomotion_ckpt`, `motion/motion_library`,
and this `motion/sim` folder itself — so any motion you drop into
`motion_library/single/` or `combined/` is automatically pickable by name,
no extra wiring needed.

If the container doesn't exist (fresh machine, or it got removed),
`run_holomotion.sh` bootstraps it automatically via `setup_container.sh` —
no manual `docker run` needed. `setup_container.sh` reuses a local
`holomotion-sim2sim-provisioned` image if one exists (has the extra pip deps
the upstream deploy image doesn't ship — mujoco, hydra, ray, pandas,
tabulate, tqdm), or builds+saves one from scratch if not.
