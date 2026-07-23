"""End-to-end walking-pipeline test: replays the exact command_loop ordering
(operator intent -> clamp -> CommandShaper.shape -> ObstacleGuard.apply ->
shaper.sync) with the REAL shaper + REAL guard (both import without the SDK /
rclpy), and asserts the ACTUALLY-SENT velocity is smooth AND that safety stays
instant. Run: python3 scripts/test_walk_pipeline.py"""
import sys, os, time, importlib.util, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "obstacle"))
from cmd_shaper import CommandShaper

_g = importlib.util.spec_from_file_location("guard", os.path.join(ROOT, "obstacle", "guard.py"))
G = importlib.util.module_from_spec(_g); _g.loader.exec_module(G)

cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "robot.yaml")))
MAXES = (cfg["speeds"]["max_vx"], cfg["speeds"]["max_vy"], cfg["speeds"]["max_vyaw"])
DT = 1.0 / cfg["control"]["send_hz"]
A = cfg["motion"]["accel_vx"]
shaper = CommandShaper(cfg=cfg["motion"], max_speeds=MAXES)
guard = G.ObstacleGuard(os.path.join(ROOT, "obstacle", "obstacle.yaml"), audio=None, limits=MAXES)


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def tick(intent, guard_on=False, ring=None):
    """One command_loop iteration (OUTPUT-side shaping); returns the sent velocity.
    intent -> normalize -> guard.apply -> shaper.shape(bypass=hard-stop flags)."""
    target = tuple(clamp(intent[i], -MAXES[i], MAXES[i]) for i in range(3))
    target = shaper.normalize(target)
    if guard_on:
        guard.enabled = True; guard.mode = "ACTIVE"
        with guard._lock:
            guard._data = {"ring": ring, "tight_factor": 1.0} if ring else {"tight_factor": 1.0}
            guard._last_fresh = time.monotonic(); guard._live = True
        governed = guard.apply(target)
        hard = guard.hard_stop_flags()
    else:
        governed = target
        hard = (False, False, False)
    return shaper.shape(governed, DT, bypass=hard)


def mk_ring(pairs, n=72, bd=5.0, st=-180.0):
    d = [None] * n
    for a, v in pairs:
        d[int(((a - st) / bd)) % n] = v
    return {"n": n, "bin_deg": bd, "start_deg": st, "dist": d}


def racc(seq):
    return [(seq[i + 1][0] - seq[i][0]) / DT for i in range(len(seq) - 1)]


fails = 0
def ck(name, cond, detail=""):
    global fails
    print(("PASS" if cond else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        fails += 1


# 1) press W -> smooth ramp, exact max, no velocity jump
shaper.reset()
seq = [(0, 0, 0)] + [tick((1.5, 0, 0)) for _ in range(120)]
ck("press W reaches max exactly", abs(seq[-1][0] - 1.5) < 1e-6, f"{seq[-1][0]:.5f}")
ck("press W accel bounded (no jump)", max(abs(a) for a in racc(seq)) <= A + 1e-3)
ck("press W first move is a ramp not a jump", seq[1][0] < 0.1, f"{seq[1][0]:.4f}")

# 2) release -> smooth decel to exactly 0
seq2 = [tick((0, 0, 0)) for _ in range(120)]
ck("release decel to 0 exactly", abs(seq2[-1][0]) < 1e-6)
ck("release decel bounded", max(abs(a) for a in racc([seq[-1]] + seq2)) <= A + 1e-3)

# 3) diagonal normalized onto the speed ellipse
shaper.reset()
s3 = [tick((1.5, 0.7, 0)) for _ in range(120)][-1]
ck("diagonal stays on the speed ellipse", (s3[0] / 1.5) ** 2 + (s3[1] / 0.7) ** 2 <= 1.0 + 1e-3)

# 4) guard hard-stop is INSTANT, and no lurch when the obstacle clears
shaper.reset()
for _ in range(120):
    tick((1.5, 0, 0), guard_on=True, ring=mk_ring([]))
obst = mk_ring([(0, 0.4), (-2.5, 0.4), (2.5, 0.4)])
stop = tick((1.5, 0, 0), guard_on=True, ring=obst)
ck("guard hard-stop is INSTANT (1 tick)", stop[0] == 0.0, f"{stop[0]:.3f}")
for _ in range(5):
    tick((1.5, 0, 0), guard_on=True, ring=obst)
clr = tick((1.5, 0, 0), guard_on=True, ring=mk_ring([]))
ck("no lurch on clear: re-accel from rest", clr[0] < 0.12, f"{clr[0]:.3f}")
res = [clr] + [tick((1.5, 0, 0), guard_on=True, ring=mk_ring([])) for _ in range(120)]
ck("resume after clear is smooth", max(abs(a) for a in racc(res)) <= A + 1e-3)

# 5) reverse direction -> no overshoot past -max
shaper.reset()
for _ in range(120):
    tick((1.5, 0, 0))
rv = [tick((-1.5, 0, 0)) for _ in range(220)]
ck("reverse reaches -max exactly", abs(rv[-1][0] + 1.5) < 1e-6)
ck("reverse overshoot negligible (<=0.025)", min(s[0] for s in rv) >= -1.5 - 0.025, f"{min(s[0] for s in rv):.5f}")

# 6) guard SLOW zone: deceleration into a slow-zone obstacle must be SMOOTH
#    (bounded decel <= accel limit) -- the whole point of output-side shaping.
shaper.reset()
for _ in range(120):                                  # at full speed, clear path
    tick((1.5, 0, 0), guard_on=True, ring=mk_ring([]))
slowring = mk_ring([(0, 1.3), (-2.5, 1.3), (2.5, 1.3)])   # 1.3 m = SLOW zone (>stop_at)
slow = [tick((1.5, 0, 0), guard_on=True, ring=slowring) for _ in range(200)]
decel = max((slow[i][0] - slow[i + 1][0]) / DT for i in range(len(slow) - 1))   # worst decel
ck("guard slow-zone settles 0<v<max (scaled, not stopped)", 0.05 < slow[-1][0] < 1.5, f"v={slow[-1][0]:.3f}")
ck("guard slow-zone DECEL is smooth (<=accel limit, not a lurch)", decel <= A + 0.05, f"decel={decel:.3f} lim={A}")
ck("guard slow-zone steady at the end", abs(slow[-1][0] - slow[-2][0]) < 0.01, f"d={abs(slow[-1][0]-slow[-2][0]):.4f}")

# 7) smoothness WITHOUT the guard (guard off): pressing W from a clear stop is a
#    bounded ramp (regression guard for the guard-off path).
shaper.reset()
g7 = [(0, 0, 0)] + [tick((1.5, 0, 0), guard_on=False) for _ in range(120)]
acc7 = max(abs(a) for a in racc(g7))
ck("guard OFF: forward ramp is smooth (accel bounded)", acc7 <= A + 1e-3, f"acc={acc7:.3f}")
ck("guard OFF: reaches max exactly", abs(g7[-1][0] - 1.5) < 1e-6)

print("\n" + ("ALL PASS -- pipeline smooth + safe" if fails == 0 else f"{fails} FAILED"))
if __name__ == "__main__":
    sys.exit(1 if fails else 0)
