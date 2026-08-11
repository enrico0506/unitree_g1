# walk_to_point — plan v5 (stance-pivot injector confirmed as the required build)

> **STATUS: HISTORICAL DESIGN RECORD — not the spec.**
> This document is why the module is shaped the way it is. Its *reasoning* is
> still the only record of several decisions (why turns must be real motion,
> why assembly is greedy and measured, why the stop tail cuts at foot
> adjacency). Roughly a third of its *literal numbers*, however, were found
> unsatisfiable by the real source clip once the code was measured against it,
> and were superseded by measured-excess-over-natural bounds. Those are marked
> **(SUPERSEDED)** inline and listed with their replacements and code pointers
> in [RESULTS / DEVIATIONS](#results--deviations-measured-supersessions) at the
> end. **Do not "fix" the code back to a superseded literal** — each one has a
> measured bug behind it.

## Goal

Given an arbitrary target point `(dx, dy)` (any distance, any direction),
produce a HoloMotion reference motion (`ref_dof_pos` npz, G1 29-DOF) where
the robot walks there and stops with both feet roughly parallel. Priority:
**natural-looking gait > millimeter-exact arrival**. Milestone #1 is smooth
playback in MuJoCo, judged numerically before eyeballing (QA is a hard gate,
not optional).

## Critical finding (v4): the file's absolute direction is not trustworthy at deploy

Confirmed by reading the actual deployment source, not by guessing or
running a live experiment:
`deployment/unitree_g1_ros2_29dof/src/humanoid_policy/observation_evaluator.py`
has `_align_ref_quat_for_motion_entry()`, wrapping every reference-quaternion
read used for tracking — it re-anchors the reference clip's heading to
**the robot's actual real heading at the moment playback starts**. Same
pattern in the training-side code
(`holomotion/src/env/isaaclab_components/isaaclab_motion_tracking_command.py`:
`get_ref_motion_cur_heading_aligned_root_pos` — reference position/heading
computed relative to the robot's *current* pose, not as absolute world
coordinates).

**Consequence**: rotating the whole clip's saved numbers to "aim" it at a
target (the earlier v1-v3 plan's step 2) does nothing at real deployment —
the system re-orients the clip to match wherever the robot is *actually*
facing when playback begins, silently discarding whatever absolute
direction was baked into the file. The only thing that survives this
re-anchoring is **relative motion the robot actually performs on itself** —
i.e., a genuine turn, executed as real stepping/leg motion within the clip,
not a label on the data.

**Decision (confirmed, this session)**: build the turn-in-clip machinery
as required scope from the start, not deferred to a v2. A version without
it can only ever point "wherever the robot already happens to be facing" —
not usable for the actual "any direction" requirement.

## What already exists to build on

- `motion/motion_library/single/walk_straight/` — 180 frames @ 30fps (6s),
  net ~6.07m forward, ~0.34m natural lateral drift (kept, not corrected).
  `raw_kimodo.npz` alongside it has `foot_contacts[180,4]`.
- `motion/motion_library/single/stair_climbing/` frame 0 — a known-good
  both-feet-planted stance, used as a *fallback* stop reference (see Step 3).
- `exhibition_patrol/patrol_logic.py` — has `carrot_point()`/bearing-to-
  target math, useful as a **build-time algorithm to reuse** for the
  turn's greedy re-aiming (Step 1) — but verified this round to have **no
  actual recorded motion data** behind it (pure runtime control logic), so
  it is NOT a source of turning motion to splice in, only a math pattern
  worth reusing at generation time.
- `build_walk_climb_walk.py`'s `align_segment()` / `crossfade()` /
  `timewarp_decelerate()`, built on the shared FK pipeline
  (`motion/holomotion/scripts/pkl_to_offline_npz.py`).

## Why distance is easy but direction needed this extra round

Stair-size-parametric needed real IK (leg joint *shape* must change with
step height) — tried and abandoned this session (see repo history/prior
plan versions), needs real RL training later. Distance-parametric doesn't
have that problem — same gait, just repeated. **Direction turned out to
have its own version of this same lesson**: it looked like a free rotation
of saved numbers, but the deployment system doesn't honor absolute
file-level orientation — direction has to be *real, relative motion* the
same way distance is real, relative stepping. Both are clip-editing
problems, not IK problems — but direction specifically needs an actual turn
motion spliced in, not a label.

## Plan v4

### Step 0 — preprocess `walk_straight` once

**Heel-strike detection** (from `raw_kimodo.npz`'s `foot_contacts[180,4]`):
1. Collapse 4→2 via per-foot OR of its two contact points (verify which
   columns pair with which foot, don't assume).
2. Debounce: any run (contact or gap) shorter than 3 frames (~100ms) merges
   into its neighbors.
3. Heel-strike = rising edge of the debounced per-foot signal.
4. Validate with cheap asserts: ~5±1 strikes/foot, alternating L/R, duty
   factor 0.55-0.7 **(SUPERSEDED → 0.42-0.7)**, never both feet airborne
   >1 frame, cycle-length std < ~2 frames.
5. Cross-check independently via FK: during each contact interval, that
   foot's z should stay within ~1cm of its own minimum, horizontal speed
   under ~5cm/s **(SUPERSEDED → interval-core trimming, 2.5cm / 12cm/s)**.
   If the stored signal disagrees, regenerate contacts from FK directly
   (flat ground makes z+speed thresholding trivial) — FK wins.

**Segment**: start segment (frame 0 → first heel-strike — check whether
frame 0 is standing/idle or mid-stride, don't assume; ~~if mid-stride,
synthesize a start via the mirror of the stop-tail technique — ease-in
timewarp-accelerate into the first full stride, instead of popping into
motion at frame 0~~ **(SUPERSEDED — never built; the actual fix was to use
the clip's own uncut ramp-up `[a, ca)`)**) + one steady gait cycle template
(heel-strike → next same-side heel-strike).

**Measure once**: per-cycle net displacement, per-cycle net yaw drift `δ`.

### Step 1 — turning (NEW, required — replaces the old "aim rotation" step 2)

**Verified empirically this round, not assumed**: `exhibition_patrol` has
**zero recorded motion npz** (`patrol_logic.py`'s `carrot_point` is pure
runtime control logic — exactly the category of thing already rejected as
"the runtime controller" — there is no actual motion data behind it to
mine). `walk_circle`'s real curvature was also checked directly: **1.57°
total heading change over its whole 5.2m path** — negligible, essentially
straight, not usable as a turn source either. **Conclusion: no usable
recorded turning motion exists anywhere in this repo right now.** The
stance-pivot injector is therefore not a fallback — it's the only viable
path today.

**Build: the stance-pivot injector (required, build first).**

Compute the required turn angle `θ` = angle between the robot's current
facing and the direction to `(dx, dy)`, in the robot's **start body frame**
(pin this convention: `(dx,dy)` is defined relative to the robot's actual
starting pose, since that's the frame the deployment re-anchoring
preserves — clip frame-0 pose ≡ robot's actual real pose at that moment).

Spec:
- **Gate injection to single-support intervals only.** Pivoting during
  double support (~20% of the cycle) drags the second, already-planted
  foot — trips QA #2 at every single step it's applied during. Ramp
  `0→Δψ` with a smoothstep *inside* each single-support window, hold
  constant through the following double-support phase.
- **Implement as a rigid rotation of root pos AND root quat together**,
  about the vertical axis passing through the *measured stance-foot
  contact point* for that interval — not just adding yaw to the quat while
  leaving root xy untouched (that sweeps the planted foot sideways,
  instant skate, same class of bug as the rejected v1 root-rescale).
- **Cap ~10-15° per step**, distribute the needed turn evenly over
  `ceil(θ / cap)` steps.

**The turn displaces the robot too — re-aim after, don't pre-compute
before.** A turn isn't in-place; it covers ~0.3-0.8m along a curve while
turning. Greedy version (same "measure, don't compute on paper" principle
as Step 2's distance assembly): while assembling turn steps, compare the
current *measured* heading against the bearing-to-target from the current
*measured* position after each step; stop turning once inside a trim band;
the straight-cycle assembly (Step 2) then walks whatever the *measured*
residual distance actually is, not a distance computed before turning
started. This is the same underlying math as `patrol_logic.py`'s
`carrot_point()` — bearing/distance-to-target from current state — reused
at **build time**, applied to the clip's own measured state, not running
live on the robot. Legitimate reuse; stays on the generation side of the
line already drawn against a runtime controller.

**Optional future polish, not v1**: if/when an actual recorded
constant-curvature turning clip exists (would need generating one — e.g.
through whatever pipeline produced `raw_kimodo.npz` — not mining
`exhibition_patrol`, which has no motion data), splicing that in for large
turns could look more natural than many small pivots. Not pursued now
since the data doesn't exist and the pivot injector is required regardless
(needed for fine residual-heading trim even if a coarse arc handles the
bulk of a large turn — see the rejected math below).

**Why arc-splicing alone, even if data existed, wouldn't have been
sufficient on its own**: arc splicing quantizes heading at whatever
resolution its own heel-strikes provide — at plausible curvature that's
roughly 15-20°/step, leaving residual heading error up to ~±10°. Heading
error converts to lateral miss as `d·sin(ε)` — 10° over a 10m straight leg
is ~1.7m, far outside the ~0.3-0.4m tolerance. The injector (spread thin,
3-5°/step) would still be needed to trim that residual even with arc data
available — reinforcing that it's foundational, not replaceable by arc
splicing later.

### Step 2 — assemble straight-line distance (greedy, measured)

Unchanged from v3: assemble **greedily** (append start/turn, then one gait
cycle at a time via `align_segment()` + `crossfade()`), measuring actual
cumulative displacement after each append — not computed analytically —
because each crossfade seam consumes real travel (~0.03m/frame × window
frames) that an on-paper `k = round(target/cycle_distance)` calculation
would silently miss.

Stop appending full cycles once within one cycle's distance of the
(remaining, post-turn) target. The final cycle is NOT simply truncated at
the nearest heel-strike (see Step 3's fix for why) — see Step 3 for how the
tail is actually handled.

**Crossfade window**: 6-8 frames for phase-matched straight-cycle seams
(shrunk from `walk_climb_walk`'s 20 — with matched gait phase the
discontinuity being smoothed is already tiny; log per-seam dof
discontinuity, confirm < ~0.05 rad **(SUPERSEDED as an ABSOLUTE bound → the
source's own per-frame step is 0.3259 rad; bounded as excess over natural.
Also, `CROSSFADE_WINDOW_CAP = 7` is a CAP, not a fixed width — the used
width is adaptive per seam)**). Keep 0.3-0.5s only for heterogeneous
seams (turn↔straight, decel-tail↔stop-stance).

**Yaw drift correction**: if per-cycle `δ` exceeds ~1°, counter-rotate each
repetition by `−δ`, computed fresh off the measured end-of-sequence-so-far
every time (never pre-composed analytically) — structural guard against
drift accumulation, not a numerical one.

**Minimum distance / feasibility clamp**: not a naive `hypot(dx,dy)` check.
A target 180° behind the robot becomes a walking U-turn (12-18 pivot steps,
several meters of actual path) — perfectly fine, natural-looking. But a
target e.g. 1.5m *behind* the robot cannot be reached by any curve this
machinery produces at all (the turn itself covers more ground than that).
The check must account for the turn's own path length plus the
straight-leg and stop distance *along the actual assembled path*, not the
straight-line distance to the target — reject infeasible targets
explicitly rather than forcing a degenerate result.

### Step 3 — decelerate and stop (FIXED — v3's version guaranteed a QA fail)

**The bug in v3**: truncating at heel-strike (the moment of *maximum*
fore-aft foot split, ~0.6m) and crossfading straight into a parallel stance
forces the rear foot to close that ~0.5m gap during a 0.3-0.5s blend —
joint-space blending drags it along the ground at >1 m/s. That's a long,
visible foot-drag at the most-scrutinized moment of the whole clip, and
it's exactly what QA check #2 (stance-foot skate detector) is designed to
catch.

**Fix**: keep heel-strike as the distance *bookkeeping* granularity (Step 2
unchanged), but let the kept motion continue **about half a swing past**
the final heel-strike. `timewarp_decelerate()` through that final swing,
and cut at **foot adjacency**: swing-foot xy within ~10cm of
beside-the-stance-foot, low (~2-5cm **(SUPERSEDED → a 0.05..0.13 z ladder
plus a best-combined-score fallback)**) and slow (post-deceleration) — not at
the heel-strike instant itself. From there, the blend into the parallel
stance only has to lower a foot a few cm in place — invisible, not a drag.
Distance bookkeeping doesn't change (still measured greedily); the extra
half-swing just adds ~0.3m that folds into the same measured logic.

**Stop stance source, in preference order**:
1. **`walk_straight`'s own frame 0, if it's genuinely standing** (checked
   in Step 0 — **MEASURED OUTCOME: it is NOT.** Double-support but the root
   is already translating 0.151 m/s, so preference #2 is what actually gets
   used. The check stays live in `pick_stop_stance()`, not hardcoded) —
   preferred over `stair_climbing` frame 0, because: (a)
   same-clip arm posture/torso lean (stair frame 0 was authored heading
   into a climb, its arm position may not match a plain standing walk),
   and (b) end-pose = start-pose makes future chaining trivial (walk A→B,
   then B→C, using the same clip's own bookend stance both times).
2. **Fallback: `stair_climbing` frame 0`** if `walk_straight`'s frame 0
   turns out not to be a clean standing pose.

**Settled facing** for the stop stance: circular mean of root yaw over the
*last ~0.5s of the decelerated tail* (not the final frame alone — root yaw
sways ±2-4° per step, a single last-frame sample could land on a sway
extreme and plant the stance visibly twisted; not the chord-to-target
either — match the body's actual settled orientation).

**Position**: xy via `align_segment()`. Z via FK floor-snap (offset root z
so the stance's lowest contact point sits at ground level — its authored
height came from a different clip/context originally, even if source #1 is
used, since the settle point's exact z may differ slightly).

**One-time sanity check** (not per-generation), extended per this round's
feedback beyond stance width alone: compare the stop-stance's lateral
stance width, **pelvis height, and knee angles** against `walk_straight`'s
own double-support values. Width within ~3cm and knee/pelvis differences
small: ignore (invisible at near-zero blend speed). A slightly crouched
stance (knee/pelvis mismatch) causes a visible *sink* during the stop blend
that width alone wouldn't catch — check for it explicitly.

~~`crossfade()`~~ **(SUPERSEDED → the dedicated `_stance_blend()`)** (~0.3-0.5s)
from the near-static decelerated tail into the (yaw- and z-corrected)
stance — both sides near-stopped, so this blends near-static into static,
avoiding the "floaty" frozen-endpoint failure from this session's very first
crossfade attempt (that failure came from blending two clips at meaningfully
*different* speeds; not the case here). Hold the final stance ~0.5s.
This one seam does NOT use `crossfade()`: its velocity-eased root bridge
measured a real 23 cm/s stance-foot drag here, because the stance leg's own
joints interpolate at the same time and pelvis motion + leg motion don't
cancel. See `_stance_blend`'s docstring.

### Step 4 — output

Run the assembled `dof_pos`/`root_pos`/`root_quat` through the shared
FK/central-diff/legacy-npz pipeline. **Assert the final output is uniform
30fps dt** after all the timewarping — the npz format silently assumes
this, and nothing else in the pipeline checks it.

## QA — numbers before eyeballing (hard gate)

1. **Max per-frame dof delta** (L2 norm, all 29 columns) — flag > ~0.15 rad
   **(SUPERSEDED → excess over the source's own 0.3259 rad natural step
   must stay < 0.05 rad)**.
2. **Stance-foot horizontal speed while in ground contact — the skate
   detector**. IMPORTANT (new this round): contacts for the *assembled
   output* must be **FK-derived** (z + speed threshold against the
   assembled clip's own data) — the original `foot_contacts` array from
   `raw_kimodo.npz` does not survive assembly/blending and can't be reused
   for this check. Flag > 2-3cm/s during any contact interval **(SUPERSEDED
   for contact cores → excess over the source's own 13.36 cm/s baseline must
   stay < 1.5 cm/s. The near-static stop blend/hold IS still held to the
   absolute 3 cm/s.)**
3. **Root-speed continuity across every seam** **(SUPERSEDED → checked
   GLOBALLY against a modeled seam-surge bound, not only at seams)**.
4. **Cumulative root yaw at each seam, post −δ correction, stays
   approximately flat** — a ramp means δ was measured wrong systematically
   (not a numerical drift issue — see Step 2).
5. **Per-seam dof discontinuity under ~0.05 rad** **(SUPERSEDED as an
   absolute bound → excess over the source's natural 0.3259 rad step)**,
   validating the shrunk 6-8 frame crossfade window empirically.
6. **Ground penetration**: minimum foot world-z ≥ ~−5mm everywhere in the
   output, and swing-foot clearance stays within the source clip's own
   observed range throughout. This is the classic blend artifact and most
   likely to appear inside the two crossfade regions and around the
   z-snapped stop stance — check those regions specifically, not just a
   global min.
7. **Arrival assert (NEW this round)**: measured end position within
   ~0.4m of `(dx, dy)`, and settled final heading approximately matches
   the final travel direction **(the implied chord-based travel-direction
   estimator is SUPERSEDED → the DECEL-REGION displacement, for the
   documented curvature-bias reason)**. The turn bookkeeping (residual vectors,
   post-turn re-aiming) is the newest, least-tested part of this pipeline —
   an arrival miss (e.g. a sign error in the turn direction, or a residual
   distance computed before the turn instead of after) is invisible to
   checks #1-6, which only look at local smoothness/skate/penetration —
   the clip can be perfectly smooth and land 2m off target.

Only after all 7 pass does viewing in MuJoCo
(`motion/motion_builder/combined/walk_and_wave/scripts/view_npz.py`) make
sense.

## Two deployment-behavior facts, verified directly from source this round

Neither blocks milestone #1 (offline clip generation + MuJoCo verification),
but both matter for real-hardware expectations later:

1. **Heading alignment is entry-only, not continuous.** Confirmed via the
   actual log message in `observation_evaluator.py`: `"Motion yaw alignment
   captured at motion entry"` — the alignment offset is computed **once**
   when tracking starts, cached, then reapplied unchanged on every
   subsequent call (`_align_ref_quat_for_motion_entry` doesn't recompute
   it). Consequence: any yaw drift baked into the generated clip will
   **not** be corrected during tracking on the real robot — it's entirely
   on generation-time correctness (the `−δ` per-cycle guard in Step 2) to
   get this right, not something the deployment stack will paper over.
2. **Whether the tracking policy's training data included turning motions
   is unverifiable right now** — no policy has been trained in this repo
   yet (motion_tracking training hasn't been run at all so far this
   project). If turns end up out-of-distribution for whatever policy
   eventually gets trained, a clip that plays perfectly in MuJoCo could
   still stumble on real hardware. Not a blocker for building/verifying
   this generator now; a real concern to revisit once training actually
   happens.

## What's NOT wanted here (ruled out this session)

- A live closed-loop navigation controller (LocoClient + real-time position
  feedback) — explicitly declined; this is a pre-built reference clip
  generated per target, not a runtime control loop. The in-clip turn (Step
  1) is still generation-time clip editing, not this.
- Per-frame inverse kinematics of any kind.
- Uniform root-path rescaling to hit exact distance — causes foot skate.
- Analytic/on-paper computation of gait-cycle count — causes systematic
  undershoot from uncounted crossfade-seam travel loss.
- Rotating the whole clip's saved orientation to "aim" it — confirmed this
  round to do nothing at real deployment (see "Critical finding" above);
  replaced by Step 1's actual in-clip turn.

`motion/walk_to_point/build_walk_to_point.py` **was** a complete working
implementation of three of the bullets above at once (aim-rotate the saved
clip + uniform root-path rescale + sqrt-warp decel + an overlap-and-remove
`crossfade` with `CROSSFADE_WINDOW = 20`). It was imported by nothing and
carried a stale duplicate copy of every seam primitive — a live regression
hazard for anyone grepping for `align_segment` — and has been **deleted**.
Its only still-true contribution, the timewarp, lives on (with its integration
bug fixed) in `stop_tail.timewarp_decelerate`.

---

## RESULTS / DEVIATIONS (measured supersessions)

Every row is a plan literal that the real source data or a measured bug
invalidated. The **Why** is summarised here; the full measurement lives in the
code comment named in **Where**, which is the authoritative version.

| Plan said | Actually shipped | Why (measured) | Where |
|---|---|---|---|
| Step 0 duty factor 0.55-0.7 | 0.42-0.7 | This clip's true single-foot duty is ~0.5 — a brisk gait with essentially zero double support (per-foot intervals exactly abut). Stored contacts and independent FK thresholding agree, so the floor was relaxed rather than the signal fudged. | `heel_strike._print_report` duty comment |
| FK check: z within ~1cm, speed < 5cm/s | interval CORE (trim `FK_TRIM=3`), z-range < 2.5cm, speed < 12cm/s | The plan assumes a point contact; what is measured is the ankle_roll **link center**, which rocks 2-6cm and moves 10-50cm/s during the 1-3 heel-strike/toe-off transition frames while the contact point stays planted. | `common.py` FK tolerance note (single origin; cited by Steps 1/3 and QA) |
| Mid-stride start → synthesize an ease-in timewarp-accelerate | Use the clip's own uncut ramp-up `[a, ca)` | "First right step slides", found in the viewer and still present after 3 rounds of bridge tuning: `start_segment` ends at frame 18 still accelerating, `steady_cycle` starts at 52 already at cruise, so the bridge was asked to invent a real ~0.6 m/s acceleration. Using the skipped frames makes the handoff two ADJACENT real frames. | `generate()` and `assemble_straight()` `[a, ca)` comments |
| Seam dof discontinuity < ~0.05 rad (absolute) | excess over the source's natural max step < 0.05 rad | Unsatisfiable by the source: same-phase heel-strike poses differ 0.06-0.28 rad between cycles and the clip's own per-frame step reaches 0.3259 rad. | `assemble_straight`'s "MEASURED FACT" comment; QA #1, #5 |
| 6-8 frame crossfade window (fixed width) | `CROSSFADE_WINDOW_CAP = 7` is a CAP; `used_window` is adaptive (often 1-2) | A fixed 7-frame bridge diluted one frame's worth of leg motion across 7 while root xy kept accelerating — measured 9-11 frame both-feet-airborne runs at every seam. | `crossfade` docstring, failure #3 |
| Overlap-and-remove blending | INSERT a bridge, remove nothing | Overlap-and-remove structurally compresses ~2×window frames of real motion into `window` output frames — a measured 2-2.5× root-speed spike at every seam, unfixable by ANY interpolation method (three were tried). This is why four files index seams as `[len_a, len_a+used_w)`. | `crossfade` docstring, failures #1/#2 |
| QA #2 skate < 2-3 cm/s (absolute) | contact-core excess over the source's 13.36 cm/s baseline < 1.5 cm/s; stop blend/hold still < 3 cm/s absolute | Same link-center-vs-contact-point measurement issue: the untouched source already reaches 5-12 cm/s during contact cores. | `qa_check` module docstring; `turn_injector.__main__`'s "MEASURED FACT" |
| QA #3 "across every seam" | global check against a modeled seam-surge bound `6·G·FPS/W² + natural` | Each seam consumes real travel that plays out as a C1-smooth smoothstep surge; a discontinuity is a `dv` exceeding that. The bug this caught (sqrt warp) measured 7.67 m/s/frame vs a ~1.2 bound. | QA #3 comment |
| Stop-stance source #1 (`walk_straight[0]`) | `stair_climbing[0]` | `walk_straight` frame 0 is double-support but NOT idle — root already translating 0.151 m/s. The preference check stays live, it just fails. | `pick_stop_stance` |
| Stop seam via `crossfade()` | dedicated `_stance_blend()` (overlap, position blend) | `crossfade`'s velocity-eased root bridge measured a real 23 cm/s stance-foot drag; the stance foot must stay PINNED throughout, not merely match at the endpoints. A plain position blend is safe *here* only because `timewarp_decelerate` already made both curves near-static. | `_stance_blend` docstring |
| Adjacency cut z tolerance 2-5cm | ladder `0.05..0.13` + best-combined-score fallback | Best adjacency point sits at swing height 10.7-12cm under the insert-based bridge, rising to 13.2cm for a 75° total-turn case. The fallback is what makes heavily-turned targets generate at all. | `_find_adjacency_cut` docstring |
| sqrt(s) ease-out timewarp | `1-(1-s)^k`, `k = extend_factor` | sqrt has infinite slope at s=0: the first warped frame compressed ~3.7 source frames into one (root speed 1.16 → 8.83 m/s, 0.80 rad dof step) in EVERY generated clip. | `timewarp_decelerate`'s "INTEGRATION BUG FOUND BY QA" |
| QA #7 travel direction from a pre-decel chord | displacement over the DECEL region | The 30-frame chord is curvature-BIASED — measured on (0,5) it lagged the end tangent by ~16.6°, flunking motions whose settled yaw was within 0.4-1.7° of their real final travel. | QA #7 comment |
| (implicit) match appended-cycle z to the previous frame | `align_z=False` for all same-clip appends | Pelvis z is PERIODIC; prev's last frame is gait phase `cb-1` while the cycle starts at phase `ca`, a measured -4.08 mm gap. Matching bakes it in permanently: (25,0) sank -10 cm over 20 cycles and poisoned `contact_from_fk`'s global-min classifier; one stop-cycle append gave -5.76 mm penetration at (1.8,0)/(2,0). | `align_segment` docstring |
| (implicit) scalar `\|residual - stop_travel\|` stop rule | vector landing-point prediction, GATED to \|heading err\| < 30° | The stop travels a VECTOR: on (5,5) the scalar rule broke predicting a 0.204 m miss where the actual was 0.396 m. Ungated, the vector rule fires while still turning — (0,5) rejected at miss 5.28 m. | `generate()`'s break-rule comment, `_stop_landing_miss` |

### Known non-equivalences deliberately left alone

Three pairs in this module look like duplication and are not. Merging any of
them silently changes output:

1. `crossfade()` (INSERT, velocity-eased root, actively walking) vs
   `_stance_blend()` (OVERLAP, pinned-foot position blend, near-static).
2. `np.gradient` foot speed (centred) vs `np.diff` root speed (forward) —
   different quantities with different endpoint behaviour.
3. `stop_tail.__main__`'s `np.roll` contact-mask erosion — `np.roll` **wraps**,
   so it is genuinely not `common.interval_core` trimming at the array edges.
   It is self-test-only and affects the weakest of three overlapping checks;
   fixing it would change a printed number, so it stays as written.
