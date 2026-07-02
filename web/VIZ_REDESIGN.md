# Dashboard Object-Recognition Redesign (3D sphere + 2D ring)

_11-agent research/design workflow, 2026-07-02._

# Unitree G1 LiDAR Dashboard — Object-Recognition Redesign

## Problem restated
`obstacle3d.js` collapses every 0.13 m ground cell to its **tallest** return (`voxelize()` keeps `max h`, line 193) and renders one height-colored `InstancedMesh` box per cell. That single-column-per-cell collapse throws away all vertical structure and never groups cells, so a wall, a person and a pole become the same blocky, same-ramp field. The operator cannot tell objects apart because the render encodes **height + proximity only — never identity or true shape**.

## Decision summary (3D sphere = primary)
- **Rendering: switch from voxel columns to a shape-preserving `THREE.Points` cloud** of `msg.points`, opaque + depth-tested, with a **per-cluster footprint patch** kept underneath as an anti-vanish floor for sparse/low objects (hybrid, not additive). This is the base every other change sits on.
- **Segmentation: cluster CLIENT-SIDE in JS** — grid connected-components (union-find) over the occupied cells `voxelize()` already produces. No backend GPU touched, sub-ms on ≤~1500 cells. A bounded backend export is *optional future work*, not needed for v1 (see §Backend).
- **Per-object identity = categorical hue (Glasbey/golden-angle) + temporal-stable IDs + pooled AABB box + centroid label**. Height demoted to a secondary lightness cue inside each cluster's hue; proximity/threat re-encoded as box outline weight/halo so the safety cue isn't lost.
- **2D ring (secondary): adjacent-sector merge into per-object arcs (ships immediately on `msg.ring`) + a Cartesian cluster-footprint overlay sharing the 3D palette**, drawn on top of the existing honest 3-state free/unknown/occupied wedges (kept as a coverage underlay).

Only `realtime_in_browser=true`, adopt/pilot techniques are included. Dropped items are listed at the end.

---

## 1. 3D sphere rendering — `web/obstacle3d.js` (primary rewrite)

### 1a. Point cloud instead of columns (adopt — biggest shape win, lowest risk)
Replace the `columns` `InstancedMesh` + `voxelize`-max-height render (lines 159-167, 250-276) with the **proven `THREE.Points` path already in `lidar.js`** (56-69):
- `BufferGeometry` with dynamic `position` + `color` attributes, growable in place via an `ensureCapacity()` clone of `lidar.js:134-139` (zero per-frame GC).
- Material: reuse `lidar.js` `makeDiscTexture()` (102-116) for soft round depth-correct discs, `depthWrite:true`, `alphaTest:0.45`, **`transparent` but NOT additive** — additive removes occlusion and smears overlapping objects into one blob (the opposite of the goal).
- **Anti-vanish (mandatory):** a strided ~3000-pt cloud in a 3 m dome is sparse, so a curb (`z≈0.1`) can disappear. Mitigate with (i) a clamped point size — a minimal custom `ShaderMaterial` with `gl_PointSize = clamp(sizeAtten, MIN_PX≈2.0, MAX_PX)`, and (ii) the **per-cluster footprint patch** from §1c so every object keeps a solid ground signature even when its points are thin. Keep the point cloud, drop the tall columns.
- Ingest: reuse the existing stride-3 loop (211-215) as the point feed; keep the ring-sector stub fallback (216-230) for when `msg.points` is absent.

Result: a person's torso/arms, a chair, a pole, a flat wall read as **distinct silhouettes** instead of identical columns — the literal "is that a person" fix.

### 1b. Client-side clustering (adopt — highest leverage for identity)
Add a `computeClusters(msg)` pass (new shared module, §4) run **once per frame, cached by `msg.seq`** so both panels reuse it:
1. Reuse `voxelize()` to get the `Map("i,j" → cell)` of occupied 0.13 m cells (ground already removed by backend).
2. **Union-find / BFS connected-components** over 8-connected `(i,j)` neighbours → a `clusterId` per cell. O(occupied cells), sub-ms.
3. **Min-cluster-size gate** (~5–8 cells / points): reject 1–2 cell specks so the display isn't peppered with phantom objects (the cheap, high-value half of Euclidean-extraction — the full KD-tree is dropped).
4. Point→cluster: each point's `("i,j")` key looks up its cell's `clusterId` (O(n)); colour the cloud by it.

**Known caveat handled:** top-down `(i,j)` connectivity merges a person standing against a wall and can split a thin pole across gaps. v1 ships 2D CC because it gives one **footprint identity shared by both the sphere and the BEV** (a stated goal). Documented upgrade path: add a coarse z-band to the key (`"i,j,kz"`, `CELL_Z≈0.2 m`, 26-conn) to split stacked/leaning objects — a one-line key change, do it only if merges prove problematic in the field.

### 1c. Per-object encoding (adopt)
- **Categorical palette:** a fixed ~16-hue Glasbey-style array (or `THREE.Color.setHSL` golden-angle for unbounded counts), indexed by stable track ID. This is what makes "these points are one thing" readable; the old cyan→red height ramp painted every object alike. Reuse the exact same palette in the 2D BEV for cross-panel correspondence.
- **Height as secondary channel:** keep the height cue by modulating **lightness within the cluster's hue** (dark base at `z≈0` → lighter toward `H_MAX`), reusing the `heightColor`/RAMP interpolation shape but as a value axis, not the hue. Operators keep "how tall" without losing "which object".
- **Per-object overlays (pooled, allocation-free):** a fixed pool (~24) of `THREE.LineSegments` AABB boxes built once from cluster extents (min/max x,y,z), `visible=false` when idle — matches the file's `_m/_p/_s` scratch discipline and the `lidar.js` fan-buffer pattern (249-262). Box **proportions** answer wall (long/low) vs pole (tall/thin) vs person (~0.5 m×~1.7 m). 
- **Centroid label:** one `Sprite` per pooled ID showing `#id · 1.4 m · 1.7 m` (id, distance, height). **Cache one `CanvasTexture` per ID, redraw only when the text changes** — a new texture per frame leaks/GC-stutters the WS loop. Gate labels behind a toggle; cull overlapping labels in dense scenes.
- **Threat re-encoding (safety):** don't lose the proximity cue the old amber→red tint carried. Re-express it as **box outline weight + a pulsing halo** for clusters inside `SLOW_M`/`STOP_M`, reusing the `occAt()` + prox math (235-269). Identity = hue; danger = outline/halo — two independent channels.
- **Oriented (PCA/min-area) box — PILOT:** upgrade only high-aspect (structural) clusters; skip/smooth heading for near-square (person) hulls where yaw flips ±90° frame-to-frame. Ship AABB first.

### 1d. Temporal ID stability (adopt — necessary, not optional)
Per-frame CC yields arbitrary IDs, so colours/boxes/labels **strobe at 10 Hz** and "recognize the object" breaks under motion. Add a module-scope `{id, centroid, extent, lastSeen}` list (like `lastMsg`) and a **greedy nearest-centroid association** each frame; retire IDs after K missed frames. ~30 lines, negligible cost. Two real gaps to handle: (1) split/merge ID swaps — bias matching by size + centroid; (2) the frame is **robot-relative**, so when the robot walks every static object translates — use a generous gate (~0.5–0.7 m) or ego-motion-compensate later with odometry. Without this, §1c actively harms recognition.

### 1e. Ground grid / legend / depth cues (adopt as adjunct)
- Keep all orientation/threat decor: `GridHelper`, stop/slow/RANGE ground rings (rescaled live from `stop_m`/`slow_m`), D435i FOV cone, translucent dome, robot body+arrow.
- Add `FogExp2` + gradient backdrop from `lidar.js` (11-17): near objects crisp, far ones recede — a cheap, strong foreground/background cue. Both points and boxes honour fog.
- Add **numeric distance labels** on the stop/slow/range rings (Sprite `CanvasTexture`, cached) for scale judgement.
- Port the corner **orientation gizmo** (`lidar.js:49-54,217-228`) — persistent up/forward reference while orbiting.
- Add a small **cluster legend / count chip** (`N objects`, nearest distance) using the existing `obs-` CSS tokens.

---

## 2. 2D ring / BEV — `web/obstacle.js` (secondary)

- **Adjacent-sector merge (adopt — ship immediately, standalone):** in `drawRadarGeom` (321-387), collapse runs of adjacent occupied 5° sectors of similar range into one **per-object arc/span** so a wall reads as a continuous band instead of 8 identical bars. Pure post-process on `msg.ring`, no clustering dependency. Keep the raw per-sector min-distance underneath as a safety gauge so de-cluttering never hides a close return.
- **Cartesian cluster-footprint overlay (pilot):** draw each cluster's **convex-hull footprint** (tiny inline `polygonHull`, fallback to a dot for <3 pts) filled in its **shared 3D palette hue**, using the existing ego→screen mapping `screenX=cx−r·sin(a), screenY=cy−r·cos(a)` (`sector()` 299-313) so BEV and sphere agree geometrically. Draw it **on top of** the existing honest 3-state free/unknown/occupied wedges — those encode blind-spot/log-odds info `msg.points` (occupied-only) cannot reconstruct, so keep them as a faint coverage underlay. Inherits the sparse-jaggedness + flicker risks → depends on §1b/§1d.
- **Perceptual sector distance ramp (pilot/polish):** clamp/normalise the occupied-wedge colour to a fixed 0–3 m window; low value, do last.

---

## 3. Backend — `obstacle/obstacle_node.py`
**No change required for v1.** The `msg.points` contract (`_viz_points`, ~line 965: flat `[x,y,z]`, z = height above fitted floor, strided ≤ `viz_max_points=3000`) is already sufficient to cluster client-side, and the node is pure-CPU numpy with the GPU owned by the pose service — keep heavy work off it.

**Optional future (only if the robot dashboard's CPU can't afford client CC):** the node already maintains a `PolarOccupancyGrid` log-odds ring; add a bounded numpy connected-components pass over that grid and export `ring.cluster[]` (per-sector integer label) alongside `ring.state/prob`. Cheap, no GPU, keeps the payload bounded. Prefer client-side first — it's sub-ms and avoids a contract change.

---

## 4. Shared module + wiring
`obstacle.js` and `obstacle3d.js` are separate IIFEs but receive the **same `msg`** (pushed from `controller.js:94-95` and `phone.js:136-137`). Factor clustering into a small `web/cluster.js` exposing `window.ObstacleClusters.get(msg)` that:
- computes once, **caches by `msg.seq`**, returns `{ clusters:[{id, hue, cells, centroid, aabb, hull, pointIds}], cellToId }`;
- owns the temporal tracker state.
Whichever panel renders first computes; the second reads the cache — one clustering pass per frame for both views, guaranteeing identical IDs/colours. Keep the `window.Obstacle3D.{update,mountTo}` and `window.Obstacle.{init,handle,setBigRadar}` signatures so the rewrite is drop-in for desktop and phone.

---

## 5. Performance budget (~10 Hz data / 60 fps render)
- Point cloud: 1 draw call, ~3000 pts. In-place typed-array overwrite, `setDrawRange`, update only touched attributes — no per-frame geometry alloc.
- Clustering: `voxelize` O(3000) + union-find over ≤~1500 cells + point→cell lookup O(3000) → sub-ms.
- Tracker: O(clusters²), <~30 objects → trivial.
- Overlays: ~24 pooled `LineSegments` boxes + ~24 Sprites with **cached** textures (redraw on text change only).
- `renderer.setPixelRatio(Math.min(dpr, 1.5))` — fill-rate/overdraw, not point count, is the real cost on the robot's modest GPU and the phone. Opaque + depth-tested, no post-process passes.
Total main-thread cost low-single-ms/frame; comfortably inside 10 Hz updates rendered at 60 fps. (OffscreenCanvas/Web-Worker offload is unnecessary — dropped.)

---

## 6. Phased build order (biggest legibility win first)
1. **Points cloud swap** (§1a) + hygiene (§5): replace columns with the `lidar.js` `THREE.Points` pattern, opaque, clamped size, footprint-patch anti-vanish. Immediate shape recovery, lowest risk.
2. **Clustering + categorical palette** (§1b, §1c colour): grid CC + size gate + hue-per-cluster recolour, height→lightness. Delivers per-object separation.
3. **Temporal ID stability** (§1d): nearest-centroid tracker. Mandatory companion to step 2 — stops colour strobe.
4. **Per-object overlays** (§1c boxes/labels/threat): pooled AABB + cached centroid labels + outline/halo threat. Operator reads "one object at 1.4 m, 1.7 m tall". Pilot oriented boxes for structural clusters.
5. **Ground/legend polish** (§1e): fog, ring distance labels, gizmo, cluster legend chip.
6. **2D BEV** (§2): ship sector-merge immediately; pilot the Cartesian footprint overlay sharing the palette once steps 2–3 are stable.

---

## 7. Explicitly DROPPED (fancy-but-not-legible / cost without payoff)
- **Eye-Dome Lighting (EDL):** needs dense neighbour pixels; ~3000 sparse pts barely form creases. Requires vendoring EffectComposer/depth-target (importmap exposes only `three`+`OrbitControls`) + the never-merged near-plane fix. High effort, redundant once cluster colour + occlusion land. Drop.
- **DBSCAN / full KD-tree Euclidean extraction on raw points:** grid-equivalent grouping for more per-frame cost on already-voxelized data; keep only the cheap **min-cluster-size gate**. Drop the trees.
- **Turbo/Viridis DataTexture LUT as the primary channel:** competes with categorical hue for the colour channel and only improves height, not object separation. Height stays as lightness. Drop as primary.
- **BEV density/occupancy heatmap background:** redundant with the existing log-odds 3-state wedges, muddies the identity palette on the dark theme. Drop.
- **OffscreenCanvas + Web-Worker offload:** pure perf insurance, unnecessary at 3000 pts/10 Hz; `msg.points` arrives as a JSON Array so "zero-copy transfer" needs a copy anyway. Revisit only if profiling shows jank.

---

## Sources
- Repo: `web/obstacle3d.js` (voxel-column render, `voxelize` max-height collapse L185-196, InstancedMesh L159-276), `web/lidar.js` (`THREE.Points` + disc texture + fog + growable buffers + `LineSegments` fan pool + gizmo), `web/obstacle.js` (`drawRadarGeom`/`sector` polar 3-state radar, `RC` palette), `obstacle/obstacle_node.py` (`_viz_points` flat height-above-floor export L965, payload assembly L560-596).
- Verified findings (this task): point-cloud-render, client-clustering, viz-encoding, bev-2d dimensions — adopt/pilot vs drop verdicts with `realtime_in_browser` and `improves_recognition` flags, incl. the sparse-vanish, additive-blending, temporal-flicker, and threat-cue-loss caveats.

---

## Top changes (phased)

**1. Replace voxel columns with a shape-preserving point cloud** (effort low, impact high)

- Files: `web/obstacle3d.js`, `web/lidar.js`
- Swap the max-height InstancedMesh columns (voxelize L185-196 + render L250-276) for a THREE.Points cloud of msg.points, copying the proven lidar.js pattern (BufferGeometry pos+color, makeDiscTexture round discs, opaque depthWrite, growable in-place buffers). Add a clamped minimum point size (small custom ShaderMaterial or size floor) plus a per-cluster ground footprint patch so sparse/low objects never vanish. NOT additive blending.
- Why: The max-height-per-cell collapse is the root cause: it discards all vertical structure so wall/person/pole become identical columns. Restoring the real 3D point distribution makes silhouettes distinguishable immediately, and it's a near-verbatim copy of code already running in lidar.js.

**2. Client-side grid connected-components clustering + categorical per-object palette** (effort medium, impact high)

- Files: `web/obstacle3d.js`, `web/cluster.js`
- Reuse voxelize()'s occupied-cell Map, run union-find 8-connected components with a min-cluster-size gate, map each point to its cell's clusterId (O(n)), and colour the cloud by a fixed Glasbey/golden-angle hue per ID. Demote height to lightness within the hue. Factor into a shared cluster.js cached by msg.seq so the 2D panel reuses identical IDs.
- Why: Distinct hue per object is the strongest 'these points are one thing' cue and the highest-leverage change for the actual goal (identity). Sub-ms on <=1500 cells, zero backend/GPU change, and the grid it needs already exists.

**3. Temporal cluster-ID stability (nearest-centroid tracker)** (effort low, impact high)

- Files: `web/obstacle3d.js`, `web/cluster.js`
- Carry a module-scope {id,centroid,extent,lastSeen} list and greedily match this frame's centroids to last frame's (generous ~0.5-0.7 m gate for robot-relative ego-motion), retiring IDs after K missed frames.
- Why: Without stable IDs the per-object hues/boxes/labels strobe every 10 Hz frame, which actively disorients the operator and negates the clustering work. ~30 lines, negligible cost, mandatory companion to the palette change.

**4. Pooled per-object AABB boxes + cached centroid labels + threat halo** (effort medium, impact high)

- Files: `web/obstacle3d.js`
- A fixed pool (~24) of THREE.LineSegments AABB boxes built once from cluster extents (visible=false when idle) plus one Sprite label per ID (#id, distance, height) with a CanvasTexture cached and redrawn only on text change. Re-encode the old amber->red proximity threat as box outline weight + pulsing halo using the existing occAt/prox math.
- Why: Box proportions give the AV-viewer 'discrete entity + extent' read (wall long/low vs pole tall/thin vs person), the label answers 'what and how far', and the halo preserves the safety-critical proximity cue that categorical hue would otherwise discard. Pooling matches the file's allocation-free discipline.

**5. 2D BEV: adjacent-sector merge arcs + shared-palette cluster footprints** (effort medium, impact medium)

- Files: `web/obstacle.js`, `web/cluster.js`
- In drawRadarGeom, merge runs of adjacent occupied 5deg sectors of similar range into per-object arcs (ships standalone on msg.ring). Then overlay each cluster's convex-hull footprint filled in the SAME palette hue as the 3D sphere, using the existing ego->screen sector() mapping, drawn on top of the retained faint free/unknown/occupied coverage wedges.
- Why: Delivers the secondary goal and 3D<->2D correspondence: a wall reads as one continuous span instead of 8 identical bars, and footprints colour-match the sphere. The sector-merge half is cheap and dependency-free; the footprint layer reuses the shared clustering already computed.

