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
