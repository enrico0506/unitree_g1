#!/usr/bin/env python3
"""sim_person_fusion.py -- proves the person bearing/distance fusion math in
isolation, NO ROBOT, NO camera, NO depth sensor.

Track A0 (delegated-wishing-stream.md, Track A5's person_fusion.py is the eventual
on-robot module; this harness proves the math it will run BEFORE it touches a
camera). Mirrors sim_odometry.py's culture: synthesize the two REAL data sources,
fuse them with the same approach the on-robot module will use, and MEASURE the
recovered bearing/distance against the known truth.

THE TWO SOURCES SYNTHESIZED
    1. A detector bounding box, mirroring perception/detect/detect_service.py's
       DETECT_TRACKS shm shape EXACTLY: {"w":, "h":, "items":[{"cls","conf","box"}]}
       with box = [x1, y1, x2, y2] in SOURCE-FRAME PIXELS (confirmed against
       perception/detect/detector.py's NanoOwlSamDetector.detect()); "cls":"person"
       is a free-text open-vocab string (DETECT_PROMPTS), not a numeric id.
    2. A depth patch over the box region, mirroring what depth_nearfield.py reads
       from the D435i: a per-pixel range in metres, sparsely NaN where the sensor
       has no return (occlusion / a person's silhouette edge / sensor noise floor).

THE FUSION
    bearing:  box-CENTER-x mapped through the camera's horizontal FOV, LINEARLY
              (the same simple mapping depth_nearfield.py's per-sector ring uses:
              see its ring_fov_half_deg -- re-imported here, not re-invented, so a
              real HFOV re-tune only has to happen in one place).
    distance: a ROBUST, NaN-guarded read over the box's depth patch -- np.isfinite
              screens the patch first (mirrors fused_odometry.py's _all_finite
              door-guard idiom: a non-finite reading must be DROPPED, never let
              through to poison a median/mean), then the MEDIAN of what remains
              (robust to a stray near/far flier at the silhouette edge). Too few
              finite pixels (heavy occlusion) -> None, i.e. "unknown", never a
              guess -- the same fail-safe posture guard.py and DepthNearField take
              on a corrupt/missing reading.

Run:  python3 scripts/sim_person_fusion.py                 # default scenario
      python3 scripts/sim_person_fusion.py --all --json
      python3 scripts/sim_person_fusion.py --list
      python3 scripts/sim_person_fusion.py --selftest       # ASSERTS tolerances + edge cases
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import depth_nearfield as _dnf  # noqa: E402  -- reuse its ESTABLISHED D435i HFOV constant

# depth_nearfield.py's _DEFAULTS["ring_fov_half_deg"] is the D435i's measured
# horizontal HALF-field-of-view (~43 deg -> ~87 deg HFOV, documented there). Reused
# verbatim rather than re-declared, so a future re-measurement only changes one file.
HFOV_HALF_DEG = float(_dnf._DEFAULTS["ring_fov_half_deg"])

# Robust-median depth read: need at least this many FINITE patch pixels to trust a
# reading at all (mirrors DepthNearField's min_points fail-safe floor -- a patch
# that is mostly occluded/NaN must report None, not a poisoned few-sample median).
MIN_VALID_DEPTH_PX = 6


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# --------------------------------------------------------- synthetic detector box
def make_detection(frame_w, frame_h, bearing_rad, box_w_px=80, box_h_px=180,
                    cls="person", conf=0.9):
    """One detection item, mirroring detect_service.py's DETECT_TRACKS item shape
    EXACTLY ({"cls","conf","box":[x1,y1,x2,y2]} pixel coords), whose box CENTER
    maps -- via bearing_from_box_center()'s inverse -- to exactly `bearing_rad`."""
    # angle convention (matches obstacle ring / depth_nearfield: 0=ahead, +=LEFT,
    # -=RIGHT); image x increases RIGHTWARD, so a LEFT (positive) bearing sits at a
    # SMALLER pixel x (left of center) -- hence the negated offset here.
    off = clamp(-math.degrees(bearing_rad) / HFOV_HALF_DEG, -1.0, 1.0)
    cx = 0.5 * frame_w * (1.0 + off)
    cy = 0.5 * frame_h
    x1, x2 = cx - box_w_px / 2.0, cx + box_w_px / 2.0
    y1, y2 = cy - box_h_px / 2.0, cy + box_h_px / 2.0
    x1 = clamp(x1, 0, frame_w - 1); x2 = clamp(x2, 0, frame_w - 1)
    y1 = clamp(y1, 0, frame_h - 1); y2 = clamp(y2, 0, frame_h - 1)
    return {"cls": cls, "conf": conf, "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]}


def bearing_from_box_center(box, frame_w):
    """Inverse of make_detection()'s offset: box-center-x -> bearing (rad), via the
    SAME linear HFOV mapping depth_nearfield.py's ring uses (no separate model)."""
    x1, _y1, x2, _y2 = box
    cx = 0.5 * (x1 + x2)
    off = clamp((cx - 0.5 * frame_w) / (0.5 * frame_w), -1.0, 1.0)
    return -math.radians(off * HFOV_HALF_DEG)


# ------------------------------------------------------------- synthetic depth patch
def make_depth_patch(distance_m, rows=12, cols=8, noise_m=0.0, nan_frac=0.0, rng=None):
    """A patch of per-pixel ranges over the box region: uniformly `distance_m`,
    with optional Gaussian sensor noise and a fraction of NaN dropouts (mimicking a
    person's own body / silhouette edges producing sparse near/occluded returns,
    the exact edge case depth_nearfield.py's frame-sanity gate exists to catch)."""
    arr = np.full((rows, cols), float(distance_m), dtype=np.float64)
    if noise_m > 0.0:
        r = rng if rng is not None else np.random.RandomState()
        arr = arr + r.normal(0.0, noise_m, arr.shape)
    if nan_frac > 0.0:
        r = rng if rng is not None else np.random.RandomState()
        mask = r.random_sample(arr.shape) < nan_frac
        arr[mask] = np.nan
    return arr


def robust_depth(patch):
    """NaN-GUARDED median-over-box-region depth read.

    Mirrors fused_odometry.py's _all_finite() door-guard idiom: screen for finite
    values FIRST (a NaN must never reach a comparison/aggregate -- e.g. np.median
    on an array containing NaN silently returns NaN, which would then poison
    everything downstream), then require a minimum finite-pixel count before
    trusting the result at all (a patch that is mostly occluded reports None --
    "unknown" -- rather than a median computed from a handful of stray survivors).
    """
    if patch is None:
        return None
    arr = np.asarray(patch, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size < MIN_VALID_DEPTH_PX:
        return None
    return float(np.median(finite))


# ------------------------------------------------------------------------- fusion
def fuse_person(box, frame_w, depth_patch, track_id=1, t=0.0):
    """Fuse one detection box + its depth patch into the
    /dev/shm/g1_person_track.json person-entry contract (Cross-workstream decisions
    #3): {bearing_rad, distance_m, box, track_id}. Returns None (not a person entry
    with a poisoned distance) if the depth read is unusable -- the fail-safe
    posture: absence of a trustworthy reading must never be silently reported as
    a confident (wrong) one."""
    bearing = bearing_from_box_center(box, frame_w)
    distance = robust_depth(depth_patch)
    if distance is None:
        return None
    return {
        "bearing_rad": round(bearing, 4),
        "distance_m": round(distance, 3),
        "box": [round(float(v), 1) for v in box],
        "track_id": track_id,
    }


def make_person_track_msg(persons, t=0.0):
    """The full shm payload shape: {t, persons:[...]}"""
    return {"t": t, "persons": persons}


# ------------------------------------------------------------------------ scenarios
def scenarios():
    return {
        "centered_near": dict(
            frame_w=640, frame_h=480, bearing_deg=0.0, distance_m=1.0,
            noise_m=0.0, nan_frac=0.0, seed=None,
            desc="person dead ahead, close: clean box + clean depth"),
        "left_far": dict(
            frame_w=640, frame_h=480, bearing_deg=30.0, distance_m=2.5,
            noise_m=0.0, nan_frac=0.0, seed=None,
            desc="person off to the LEFT (+bearing), farther away"),
        "right_close": dict(
            frame_w=640, frame_h=480, bearing_deg=-25.0, distance_m=0.6,
            noise_m=0.0, nan_frac=0.0, seed=None,
            desc="person off to the RIGHT (-bearing), close"),
        "sparse_occluded_edges": dict(
            frame_w=640, frame_h=480, bearing_deg=10.0, distance_m=1.2,
            noise_m=0.02, nan_frac=0.6, seed=3,
            desc="60% of the depth patch is NaN (silhouette/occlusion dropout) "
                 "but enough survives -- fusion must still recover bearing/distance"),
        "fully_occluded": dict(
            frame_w=640, frame_h=480, bearing_deg=15.0, distance_m=1.5,
            noise_m=0.0, nan_frac=1.0, seed=None,
            desc="the ENTIRE depth patch is NaN -- must report None (unknown), "
                 "never a poisoned guess"),
        "noisy_but_valid": dict(
            frame_w=640, frame_h=480, bearing_deg=-10.0, distance_m=1.8,
            noise_m=0.05, nan_frac=0.0, seed=5,
            desc="realistic Gaussian depth sensor noise, no dropout"),
        "at_hfov_edge": dict(
            frame_w=640, frame_h=480, bearing_deg=HFOV_HALF_DEG - 1.0, distance_m=1.4,
            noise_m=0.0, nan_frac=0.0, seed=None,
            desc="person near the very edge of the camera's HFOV"),
        "near_min_valid_count": dict(
            frame_w=640, frame_h=480, bearing_deg=5.0, distance_m=1.0,
            noise_m=0.01, nan_frac=0.0, seed=1, rows=3, cols=2,
            desc="depth patch just barely at/above MIN_VALID_DEPTH_PX -- edge of the "
                 "robust-count floor, still must recover a valid reading"),
    }


def run(scn):
    frame_w, frame_h = scn["frame_w"], scn["frame_h"]
    bearing_true = math.radians(scn["bearing_deg"])
    distance_true = scn["distance_m"]
    rng = np.random.RandomState(scn["seed"]) if scn.get("seed") is not None else None

    det = make_detection(frame_w, frame_h, bearing_true)
    patch = make_depth_patch(distance_true, rows=scn.get("rows", 12), cols=scn.get("cols", 8),
                              noise_m=scn.get("noise_m", 0.0), nan_frac=scn.get("nan_frac", 0.0), rng=rng)

    person = fuse_person(det["box"], frame_w, patch, track_id=7)
    msg = make_person_track_msg([person] if person is not None else [], t=1.0)

    bearing_err_deg = distance_err_m = None
    if person is not None:
        bearing_err_deg = math.degrees(person["bearing_rad"] - bearing_true)
        distance_err_m = person["distance_m"] - distance_true

    return {
        "scenario": scn.get("name", "?"),
        "desc": scn["desc"],
        "bearing_true_deg": scn["bearing_deg"],
        "distance_true_m": distance_true,
        "detection_box": det["box"],
        "person": person,
        "msg": msg,
        "bearing_err_deg": bearing_err_deg,
        "distance_err_m": distance_err_m,
        "recovered": person is not None,
    }


# -------------------------------------------------------------------------- main
BEARING_TOL_DEG = 4.0     # "a few degrees" per the spec
DISTANCE_TOL_M = 0.15     # "~10-15 cm" per the spec


def verdict(m, expect_recovered=True):
    ok = True
    notes = []
    if expect_recovered:
        if not m["recovered"]:
            ok = False; notes.append("fusion failed to recover a person (expected one)")
        else:
            if abs(m["bearing_err_deg"]) > BEARING_TOL_DEG:
                ok = False; notes.append(f"bearing error {m['bearing_err_deg']:.2f} deg > {BEARING_TOL_DEG}")
            if abs(m["distance_err_m"]) > DISTANCE_TOL_M:
                ok = False; notes.append(f"distance error {m['distance_err_m']:.3f} m > {DISTANCE_TOL_M}")
    else:
        if m["recovered"]:
            ok = False; notes.append("fusion reported a person when the depth was unusable (should be None)")
    return ok, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="centered_near")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    scns = scenarios()
    if args.list:
        for name, s in scns.items():
            print(f"{name:24s} {s['desc']}")
        return 0

    names = list(scns) if args.all else [args.scenario]
    results = []
    for name in names:
        if name not in scns:
            print(f"unknown scenario '{name}' (try --list)"); return 2
        scn = dict(scns[name]); scn["name"] = name
        expect = name != "fully_occluded"
        m = run(scn)
        ok, notes = verdict(m, expect_recovered=expect)
        results.append((m, ok, notes))

    if args.json:
        print(json.dumps([{**m, "ok": ok, "notes": notes} for (m, ok, notes) in results], indent=2))
    else:
        overall_ok = True
        for (m, ok, notes) in results:
            overall_ok = overall_ok and ok
            print(f"\n=== {m['scenario']} === {'PASS' if ok else 'FAIL'} {'; '.join(notes)}")
            print(f"  {m['desc']}")
            print(f"  truth: bearing={m['bearing_true_deg']:.1f} deg  distance={m['distance_true_m']:.2f} m")
            print(f"  detection box: {m['detection_box']}")
            print(f"  fused person: {m['person']}")
            if m["recovered"]:
                print(f"  bearing_err={m['bearing_err_deg']:.2f} deg  distance_err={m['distance_err_m']:.3f} m")
        return 0 if overall_ok else 1
    return 0


def selftest():
    """Runs every built-in scenario and ASSERTS the recovery tolerances + the
    occlusion edge case, deterministically."""
    ok = True

    def c(name, cond):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + "  " + name)
        ok = ok and bool(cond)

    for name, base in scenarios().items():
        scn = dict(base); scn["name"] = name
        m = run(scn)
        expect = name != "fully_occluded"
        v_ok, notes = verdict(m, expect_recovered=expect)
        print(f"\n--- {name} --- {m['desc']}")
        print(f"  truth bearing={m['bearing_true_deg']:.2f} deg distance={m['distance_true_m']:.2f} m "
              f"-> recovered={m['recovered']} bearing_err={m['bearing_err_deg']} distance_err={m['distance_err_m']}")
        c(f"[{name}] overall verdict ({'; '.join(notes) if notes else 'clean'})", v_ok)

        if name == "fully_occluded":
            c(f"[{name}] a fully-NaN depth patch reports NO person (never a poisoned guess)",
              not m["recovered"])
            c(f"[{name}] the detection box itself was still computed fine (bearing math is independent "
              f"of depth availability)", m["detection_box"] is not None)
        else:
            c(f"[{name}] bearing recovered within {BEARING_TOL_DEG} deg", abs(m["bearing_err_deg"]) <= BEARING_TOL_DEG)
            c(f"[{name}] distance recovered within {DISTANCE_TOL_M} m", abs(m["distance_err_m"]) <= DISTANCE_TOL_M)

        if name == "sparse_occluded_edges":
            c(f"[{name}] recovered DESPITE 60% NaN dropout (robust median over survivors)",
              m["recovered"])

        # determinism (seeded scenarios only -- unseeded ones use a fresh RandomState
        # and are not expected to reproduce bit-for-bit).
        if base.get("seed") is not None:
            m2 = run(scn)
            c(f"[{name}] deterministic (same seed -> identical fused reading)",
              m["person"] == m2["person"])

    # cross-cutting: a person exactly at bearing 0 with box math alone (no depth
    # involvement) must map back to bearing ~0 -- a basic sanity anchor independent
    # of the tolerance-based scenarios above.
    det0 = make_detection(640, 480, 0.0)
    b0 = bearing_from_box_center(det0["box"], 640)
    c("bearing-neutral box (dead ahead) maps back to ~0 rad", abs(b0) < 1e-6)

    # MIN_VALID_DEPTH_PX floor: one pixel short of the minimum must ALSO report None
    # (not just "mostly NaN" -- the raw insufficient-sample-count path).
    tiny_patch = make_depth_patch(1.0, rows=1, cols=MIN_VALID_DEPTH_PX - 1)
    c(f"[floor] a patch with fewer than MIN_VALID_DEPTH_PX ({MIN_VALID_DEPTH_PX}) finite pixels reports None",
      robust_depth(tiny_patch) is None)
    exact_patch = make_depth_patch(1.0, rows=1, cols=MIN_VALID_DEPTH_PX)
    c("[floor] a patch with EXACTLY MIN_VALID_DEPTH_PX finite pixels DOES report a value",
      robust_depth(exact_patch) is not None)

    print("\n" + ("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
