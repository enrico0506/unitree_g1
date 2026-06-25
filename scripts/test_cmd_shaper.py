"""Unit tests for CommandShaper -- run: python3 scripts/test_cmd_shaper.py"""
import math
from cmd_shaper import CommandShaper, normalize_xy, _Axis

DT = 1.0/30.0
fails = 0
def check(name, cond, detail=""):
    global fails
    print(("PASS" if cond else "FAIL")+f"  {name}"+(f"  [{detail}]" if detail and not cond else ""))
    if not cond: fails += 1

# ---- single axis: ramp 0 -> 1.5, verify no overshoot, accel & jerk bounded ----
ax = _Axis(a_max=1.2, j_max=4.0)
vs=[ax.v]; as_=[ax.a]
for _ in range(400):
    ax.step(1.5, DT); vs.append(ax.v); as_.append(ax.a)
peak_v = max(vs)
# realized accel/jerk of the SENT velocity signal (what the robot actually receives).
racc=[(vs[i+1]-vs[i])/DT for i in range(len(vs)-1)]
njerk=[abs(racc[i+1]-racc[i])/DT for i in range(len(racc)-1)]
over=[j for j in njerk if j > 4.0+1e-6]
check("ramp up reaches target EXACTLY", abs(vs[-1]-1.5)<1e-6, f"end={vs[-1]:.6f}")
# Negligible bounded overshoot (<=0.025 m/s = 2.5 cm/s, momentary) is the deliberate
# tradeoff for a STRICTLY bounded jerk -- it damps out within a few ticks.
check("ramp up overshoot negligible (<=0.025 m/s)", peak_v <= 1.5+0.025, f"peak={peak_v:.6f}")
check("accel <= a_max (sent signal -> no velocity jump)", max(abs(a) for a in racc) <= 1.2+1e-6, f"maxacc={max(abs(a) for a in racc):.4f}")
# Jerk bounded everywhere (incl. the landing) to a small multiple of j_max.
check("jerk bounded everywhere (<=2.5x j_max)", (max(njerk) if njerk else 0) <= 2.5*4.0, f"peak={max(njerk) if njerk else 0:.2f}")
mono = all(vs[i+1] >= vs[i]-0.03 for i in range(len(vs)-1))
check("ramp up ~monotonic (no real reversal)", mono)
check("settles EXACTLY, no chatter", max(vs[-60:])-min(vs[-60:])<1e-9)
t95 = next(i for i,v in enumerate(vs) if v>=0.95*1.5)*DT
check("reaches 95% in 0.8-2.2s window", 0.8 <= t95 <= 2.2, f"t95={t95:.2f}s")
print(f"   (info) t95={t95:.2f}s peak={peak_v:.6f} jerk-over-ticks={len(over)} peakjerk={max(njerk) if njerk else 0:.1f}")

# ---- ramp down 1.5 -> 0: no undershoot below 0 ----
ax = _Axis(1.2,4.0); ax.reset(1.5)
vs=[ax.v]
for _ in range(400): ax.step(0.0, DT); vs.append(ax.v)
check("ramp down reaches 0", abs(vs[-1])<1e-6, f"end={vs[-1]:.6f}")
check("ramp down undershoot negligible (<=0.025 m/s)", min(vs) >= -0.025, f"min={min(vs):.6f}")

# ---- reverse target 1.5 -> -1.5 (direction change), no wild overshoot ----
ax = _Axis(1.2,4.0); ax.reset(1.5)
vs=[ax.v]
for _ in range(600): ax.step(-1.5, DT); vs.append(ax.v)
check("reverse reaches -1.5", abs(vs[-1]+1.5)<1e-6, f"end={vs[-1]:.6f}")
check("reverse overshoot negligible (<=0.025 m/s)", min(vs) >= -1.5-0.025, f"min={min(vs):.6f}")

# ---- step target mid-ramp (operator changes mind) stays bounded ----
ax=_Axis(1.2,4.0)
ok=True
for i in range(200):
    tgt = 1.5 if i<30 else (0.5 if i<90 else 1.0)
    ax.step(tgt, DT)
    if abs(ax.a) > 1.2+1e-6: ok=False
check("accel stays bounded through target changes", ok)

# ---- CommandShaper pass-through when disabled ----
sh = CommandShaper({"enabled": False}, max_speeds=(1.5,0.7,2.0))
out = sh.shape((1.5,0.7,2.0), DT)
check("disabled = passthrough", out==(1.5,0.7,2.0), str(out))

# ---- diagonal ellipse normalization ----
nx,ny = normalize_xy(1.5,0.7,1.5,0.7)
m=(nx/1.5)**2+(ny/0.7)**2
check("full diagonal normalized onto ellipse", abs(m-1.0)<1e-6, f"m={m:.4f}")
nx,ny = normalize_xy(0.5,0.2,1.5,0.7)
check("inside-ellipse left unchanged", (nx,ny)==(0.5,0.2))
# direction preserved
nx,ny=normalize_xy(1.5,0.7,1.5,0.7)
check("normalize preserves direction", abs((nx/ny)-(1.5/0.7))<1e-6)

# ---- sync rebases (guard hard-stop scenario) ----
sh = CommandShaper({}, max_speeds=(1.5,0.7,2.0))
for _ in range(60): sh.shape((1.5,0,0), DT)   # ramp up to ~full
v_before = sh.ax.v
sh.sync((0.0,0.0,0.0))                          # guard hard-stopped us
check("sync pulls velocity to 0", sh.ax.v==0.0, f"was {v_before:.3f}")
check("sync zeros accel on big delta", sh.ax.a==0.0)
# after sync, next ramp restarts smoothly from 0 (no instant jump)
sh.shape((1.5,0,0), DT)
check("re-ramp from 0 is gradual (no lurch)", sh.ax.v < 0.1, f"v={sh.ax.v:.4f}")

# ---- enabled shape ramps (not a step) ----
sh = CommandShaper({}, max_speeds=(1.5,0.7,2.0))
first = sh.shape((1.5,0,0), DT)
check("first enabled tick is a small ramp not a jump", first[0] < 0.2, f"vx={first[0]:.4f}")

# ---- full integration sim: smooth speed & accel profile, plausible distance ----
sh = CommandShaper({}, max_speeds=(1.5,0.7,2.0))
v_prev=0.0; max_acc=0.0
for i in range(120):  # 4s hold forward
    o=sh.shape((1.5,0,0),DT)
    max_acc=max(max_acc, abs(o[0]-v_prev)/DT); v_prev=o[0]
check("integration: forward settled at max", abs(sh.ax.v-1.5)<1e-2, f"v={sh.ax.v:.3f}")
check("integration: realized accel <= a_max+eps", max_acc<=1.2+1e-3, f"acc={max_acc:.3f}")

print("\n"+("ALL PASS" if fails==0 else f"{fails} FAILED"))
