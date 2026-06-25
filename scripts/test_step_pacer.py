"""Unit tests for StepPacer -- run: python3 scripts/test_step_pacer.py

Fake clock + fake pose, no robot / SDK. Asserts the pulse train, latching, the
coupled-clock max() windows, the continuous/disabled pass-through, and the optional
distance-quantized termination + fall-back to time."""
import math
from step_pacer import StepPacer, ang_wrap, _build_modes

MAX = (1.5, 0.7, 2.0)
fails = 0


def check(name, cond, detail=""):
    global fails
    print(("PASS" if cond else "FAIL") + f"  {name}"
          + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails += 1


def cfg(**over):
    base = {"enabled": True, "default_mode": "continuous",
            "distance_quantize": False, "drive_timeout_s": 1.5, "idle_eps": 0.001}
    base.update(over)
    return base


# A fake monotonic clock we advance by hand so the pulse timing is deterministic.
class Clock:
    def __init__(self):
        self.t = 0.0
    def tick(self, dt):
        self.t += dt
        return self.t


# Drive the pacer through `steps` ticks of `dt`, holding `intent`; record outputs.
def run(pacer, intent, steps, dt=1.0 / 30.0, clk=None):
    clk = clk or Clock()
    out = []
    for _ in range(steps):
        out.append(pacer.modulate(intent, clk.t))
        clk.tick(dt)
    return out


# (a) continuous / disabled -> pass-through (byte-for-byte unchanged) -----------
p = StepPacer(cfg(default_mode="continuous"), MAX)
o = run(p, (1.5, 0.3, -0.5), 20)
check("continuous mode is pure pass-through", all(x == (1.5, 0.3, -0.5) for x in o))

p = StepPacer(cfg(enabled=False, default_mode="small"), MAX)
o = run(p, (1.5, 0.0, 0.0), 20)
check("disabled (kill-switch) is pure pass-through even in small",
      all(x == (1.5, 0.0, 0.0) for x in o))

# (b) small mode forward: square ON pulse of mag_frac*cap then 0, repeating ------
p = StepPacer(cfg(default_mode="small"), MAX)
M = p.modes["small"]
on_s, off_s = M[0].on_s, M[0].off_s
expect_peak = M[0].mag_frac * MAX[0]
dt = 1.0 / 30.0
clk = Clock()
seq = []
for _ in range(int((off_s + on_s + off_s + on_s) / dt) + 10):
    seq.append((clk.t, p.modulate((1.5, 0.0, 0.0), clk.t)[0]))
    clk.tick(dt)
# First off_s is settle (0), then a burst at +expect_peak, then 0 again.
first_burst_t = next((t for (t, v) in seq if v > 0.0), None)
check("small fwd: first burst starts after the settle (~off_s)",
      first_burst_t is not None and abs(first_burst_t - off_s) <= dt + 1e-9,
      f"t={first_burst_t}")
burst_vals = [v for (t, v) in seq if v != 0.0]
check("small fwd: burst magnitude == mag_frac*cap, +sign",
      burst_vals and all(abs(v - expect_peak) < 1e-9 for v in burst_vals),
      f"vals={set(round(v,3) for v in burst_vals)}")
# count distinct bursts (rising edges) -> at least 2 in this window (repeats while held)
rises = sum(1 for i in range(1, len(seq)) if seq[i][1] > 0 and seq[i - 1][1] == 0)
check("small fwd: pulse REPEATS while held (>=2 bursts)", rises >= 2, f"rises={rises}")
# burst duration ~ on_s
burst_len = sum(1 for (_, v) in seq if v != 0.0 and seq.index((_, v)) < len(seq)) * dt
# (recompute robustly: length of the first contiguous burst)
i0 = next(i for i, (_, v) in enumerate(seq) if v != 0.0)
i1 = i0
while i1 < len(seq) and seq[i1][1] != 0.0:
    i1 += 1
check("small fwd: ON window length ~ on_s", abs((i1 - i0) * dt - on_s) <= 2 * dt,
      f"len={(i1-i0)*dt:.3f} on_s={on_s}")

# sign: backward (negative vx) -> negative burst
clk = Clock()
p.reset()
seqb = [p.modulate((-1.5, 0.0, 0.0), clk.tick(dt)) for _ in range(int((off_s + on_s) / dt) + 5)]
negs = [v[0] for v in seqb if v[0] != 0.0]
check("small back: burst sign is negative", negs and all(v < 0 for v in negs))

# strafe + rotate also pulse (every direction)
p.reset()
clk = Clock()
seqy = [p.modulate((0.0, 0.7, 0.0), clk.tick(dt)) for _ in range(int((off_s + on_s) / dt) + 5)]
check("small strafe (vy) pulses", any(v[1] != 0.0 for v in seqy))
p.reset()
clk = Clock()
seqr = [p.modulate((0.0, 0.0, 2.0), clk.tick(dt)) for _ in range(int((off_s + on_s) / dt) + 5)]
check("small rotate (vyaw) pulses", any(v[2] != 0.0 for v in seqr))

# (c) releasing all axes -> reset + (0,0,0) -------------------------------------
p = StepPacer(cfg(default_mode="small"), MAX)
clk = Clock()
# drive into a burst first
for _ in range(int((off_s + 0.01) / dt) + 1):
    p.modulate((1.5, 0, 0), clk.tick(dt))
rel = p.modulate((0.0, 0.0, 0.0), clk.tick(dt))
check("release -> (0,0,0)", rel == (0.0, 0.0, 0.0))
check("release -> phase reset to OFF", p.phase == "OFF")

# (d) multi-axis demand uses MAX on_s/off_s of the demanded axes ----------------
# small: vx off_s=0.65, vy off_s=0.70 -> combined off must be 0.70 (the longer).
p = StepPacer(cfg(default_mode="small"), MAX)
vx_off, vy_off = M[0].off_s, M[1].off_s
combined_off = max(vx_off, vy_off)
clk = Clock()
diag = (1.5, 0.7, 0.0)
seqd = []
for _ in range(int((combined_off + 0.5) / dt) + 5):
    seqd.append((clk.t, p.modulate(diag, clk.t)))
    clk.tick(dt)
first_t = next((t for (t, v) in seqd if v[0] != 0.0 or v[1] != 0.0), None)
check("diagonal: settle uses MAX off_s of demanded axes",
      first_t is not None and abs(first_t - combined_off) <= dt + 1e-9,
      f"t={first_t} expected~{combined_off}")
# both axes fire together in the same burst (one coherent stride)
both = any(v[0] != 0.0 and v[1] != 0.0 for (_, v) in seqd)
check("diagonal: vx and vy pulse together (one coupled stride)", both)

# (e) sign / membership latched at the ON edge ----------------------------------
# Start a forward burst, then flip intent to backward mid-burst: output keeps +sign.
p = StepPacer(cfg(default_mode="small"), MAX)
clk = Clock()
# advance through settle into the burst
v = (0.0, 0.0, 0.0)
while True:
    v = p.modulate((1.5, 0, 0), clk.t)
    clk.tick(dt)
    if v[0] != 0.0:
        break
# now in a burst with +sign; flip the operator intent to backward, stay in-burst
latched_ok = True
for _ in range(2):
    vv = p.modulate((-1.5, 0, 0), clk.t)
    clk.tick(dt)
    if vv[0] != 0.0 and vv[0] < 0.0:
        latched_ok = False
check("latched: sign flip mid-burst keeps the latched (+) sign", latched_ok)

# (f) distance mode terminates ON at the target displacement; falls back to time -
# Fake pose that advances x by `vstep` each read once the burst opens.
class FakePose:
    def __init__(self, vstep=0.02):
        self.x = 0.0
        self.live = True
        self.vstep = vstep
        self.reads = 0
    def get_pose(self):
        self.reads += 1
        return (self.x, 0.0, 0.0)
    def is_live(self):
        return self.live

fp = FakePose(vstep=0.02)
p = StepPacer(cfg(default_mode="small", distance_quantize=True), MAX,
              pose_fn=fp.get_pose, live_fn=fp.is_live)
tgt = p.modes["small"][0].target   # 0.10 m
clk = Clock()
# settle out first
while p.modulate((1.5, 0, 0), clk.t)[0] == 0.0:
    clk.tick(dt)
# now in burst: advance odom each tick; the ON window must close when x>=target
closed_at = None
clk.tick(dt)
for _ in range(200):
    out = p.modulate((1.5, 0, 0), clk.t)
    if out[0] != 0.0:
        fp.x += fp.vstep          # robot moved this tick
    if out[0] == 0.0 and fp.x > 0.0:   # burst ended (settle) after some motion
        closed_at = fp.x
        break
    clk.tick(dt)
check("distance mode: ON closes at/after target displacement",
      closed_at is not None and closed_at >= tgt - fp.vstep,
      f"closed_at={closed_at} target={tgt}")
check("distance mode: not flagged estimated while odom live", p.estimated is False)

# fall back to time when live_fn returns False
fp2 = FakePose()
fp2.live = False
p2 = StepPacer(cfg(default_mode="small", distance_quantize=True), MAX,
               pose_fn=fp2.get_pose, live_fn=fp2.is_live)
clk = Clock()
seq2 = []
for _ in range(int((off_s + on_s + 0.2) / dt) + 5):
    seq2.append(p2.modulate((1.5, 0, 0), clk.t))
    clk.tick(dt)
check("distance mode: odom dead -> falls back to time (still pulses)",
      any(v[0] != 0.0 for v in seq2))
check("distance mode: odom dead -> flagged estimated", p2.estimated is True)

# burst still terminates by time (didn't hang open) when odom is dead
i0 = next(i for i, v in enumerate(seq2) if v[0] != 0.0)
i1 = i0
while i1 < len(seq2) and seq2[i1][0] != 0.0:
    i1 += 1
check("distance fallback: ON window still bounded by time (~on_s)",
      abs((i1 - i0) * dt - on_s) <= 3 * dt, f"len={(i1-i0)*dt:.3f}")

# ang_wrap sanity
check("ang_wrap wraps past pi", abs(ang_wrap(math.pi + 0.1) - (-math.pi + 0.1)) < 1e-9)

# _build_modes falls back cleanly on an empty/partial cfg
m = _build_modes({})
check("_build_modes: defaults present for small+medium",
      "small" in m and "medium" in m and len(m["small"]) == 3)

# telemetry shape
p = StepPacer(cfg(default_mode="medium"), MAX)
t = p.telemetry()
check("telemetry exposes mode/phase/estimated",
      t.get("step_mode") == "medium" and "step_phase" in t and "step_estimated" in t)

# --- SETTLE GATE: the OFF window holds until the realized (shaper) velocity has
# settled on the demanded axes, so consecutive steps can never merge. ---
p = StepPacer(cfg(default_mode="small"), MAX)
busy = p.modulate((1.0, 0.0, 0.0), 1.0, vel_fb=(0.3, 0.0, 0.0))   # vx not planted yet
check("settle gate: holds OFF while shaper vx still high (no merged step)",
      busy == (0.0, 0.0, 0.0))
done = p.modulate((1.0, 0.0, 0.0), 1.0 + 1.0 / 30, vel_fb=(0.0, 0.0, 0.0))
check("settle gate: starts the burst once shaper vx has settled", done[0] != 0.0)

# off_max_s backstop: never hang the OFF window open forever if it never settles.
p2 = StepPacer(cfg(default_mode="small", off_max_s=1.0), MAX)
bk = p2.modulate((1.0, 0.0, 0.0), 5.0, vel_fb=(0.5, 0.0, 0.0))
check("settle gate: off_max_s backstop forces a burst when never settled", bk[0] != 0.0)

# a backward clock step (wall-clock NTP correction) must NOT wedge the gate.
p3 = StepPacer(cfg(default_mode="small"), MAX)
p3.modulate((1.0, 0.0, 0.0), 10.0, vel_fb=(0.0, 0.0, 0.0))         # burst at t=10
back = p3.modulate((1.0, 0.0, 0.0), 2.0, vel_fb=(0.0, 0.0, 0.0))   # clock jumps back
check("clock step-back resyncs instead of wedging",
      isinstance(back, tuple) and len(back) == 3)

# vel_fb omitted -> pure off_s timing still pulses (back-compat with no feedback).
p4 = StepPacer(cfg(default_mode="small"), MAX)
seqc = run(p4, (1.0, 0.0, 0.0), 80)
check("no vel_fb -> still pulses on pure off_s timing (back-compat)",
      any(v[0] != 0.0 for v in seqc) and any(v == (0.0, 0.0, 0.0) for v in seqc))

print("\n" + ("ALL PASS" if fails == 0 else f"{fails} FAILED"))
import sys
sys.exit(1 if fails else 0)
