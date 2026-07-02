// 3D obstacle view for the obstacle panel.
//
// ES module — uses the page importmap (three + OrbitControls), like lidar.js.
// Renders the ACTUAL kept obstacle points the guard perceives (post ground/self/
// range filter) as a shape-preserving POINT CLOUD, then groups the points into
// distinct OBJECTS (client-side connected-components clustering) and paints each
// object its own stable colour with an axis-aligned bounding box. So a wall, a
// person and a pole read as three separate coloured things with recognisable
// shape + extent -- instead of the old identical height-coloured columns.
//
// Frame mirrors lidar.js: X forward, Y left, Z up; 1 scene unit = 1 metre.
// Point z = height ABOVE the fitted floor, so ground-level returns sit near 0.
//
// Data: controller.js calls window.Obstacle3D.update(msg) every obstacle frame.
// We read msg.points (flat [x0,y0,z0, ...]) when present, else fall back to the
// coarse msg.ring sectors. Self-contained: mounts into <div id="obs-sphere">.
// Never throws into the WS loop.

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

(function () {
  "use strict";

  const RANGE_M = 3.0;                 // detection dome radius (matches the radar)
  const DEF_STOP = 0.75, DEF_SLOW = 1.75;
  let STOP_M = DEF_STOP, SLOW_M = DEF_SLOW;
  const H_MAX = 1.9;                   // height (m) mapped to the top of the lightness ramp
  const CELL = 0.13;                   // clustering / voxel cell size (m)
  const MAX_POINTS = 4096;             // point-cloud buffer cap
  const MAX_CELLS = 1500;              // columns-mode instanced-column cap
  const MAX_BOXES = 24;                // pooled per-object bounding boxes
  const MIN_CLUSTER_CELLS = 4;         // reject specks (< this many cells = noise, no box/colour)
  const MIN_H = 0.06;                  // columns mode: min drawn column height (low hits stay visible)
  const POINT_SIZE = 0.18;             // points mode: disc size (m) -- bigger = fuller/easier to see
  const TRACK_GATE_M = 0.65;           // nearest-centroid association gate (robot-relative frame)
  const TRACK_MISS_MAX = 6;            // retire a track after this many unmatched frames
  const D435I_FOV_DEG = 87;
  const COL = { grid: 0x232c3e, dome: 0x35507d, robot: 0x5c8df2,
                stop: 0xff4747, slow: 0xff8636, noise: 0x5b647a };

  let scene, camera, renderer, controls, cloud, cloudPos, cloudCol, boxes = [], columns;
  let stopRing, slowRing, mount, booted = false, lastMsg = null, toggleBtn = null;
  // render mode: "points" (per-object coloured cloud) or "columns" (legacy height-coloured
  // voxel columns). Remembered across reloads.
  let mode = "points";
  try { const s = localStorage.getItem("obs3dMode"); if (s === "columns" || s === "points") mode = s; } catch (e) {}
  const _red = new THREE.Color(COL.stop), _amber = new THREE.Color(COL.slow),
        _c = new THREE.Color(), _c2 = new THREE.Color();
  const _m = new THREE.Matrix4(), _p = new THREE.Vector3(),
        _q = new THREE.Quaternion(), _s = new THREE.Vector3();
  function clampv(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  function applyMode() {
    if (cloud) cloud.visible = (mode === "points");
    if (columns) columns.visible = (mode === "columns");
    if (toggleBtn) toggleBtn.textContent = (mode === "points" ? "◾ Columns" : "• Points");
  }
  function setMode(m) {
    if (m !== "points" && m !== "columns") return;
    mode = m;
    try { localStorage.setItem("obs3dMode", m); } catch (e) {}
    applyMode();
    if (lastMsg) refresh(lastMsg);
  }

  // --- temporal object tracker: keeps a colour/ID stable across frames so the
  //     per-object hues don't strobe at 10 Hz (which would defeat recognition). ---
  let tracks = [];                     // [{id, cx, cy, missed}]
  let nextId = 1;

  // stable hue per object id (golden-angle -> maximally distinct categorical colours)
  function hueColor(id, out) {
    return out.setHSL((id * 0.61803398875) % 1, 0.68, 0.56);
  }

  // columns mode: legacy height ramp cyan(ground) -> green -> amber -> red(tall)
  const _RAMP = [[0x28e0d0], [0x3fd39b], [0xff8636], [0xff4747]].map((c) => new THREE.Color(c[0]));
  function heightColor(h, out) {
    const t = clampv(h / H_MAX, 0, 1) * (_RAMP.length - 1);
    const i = Math.min(_RAMP.length - 2, Math.floor(t));
    return out.copy(_RAMP[i]).lerp(_RAMP[i + 1], t - i);
  }

  // soft round sprite for the points (depth-correct discs, not square splats)
  function discTexture() {
    const s = 64, cv = document.createElement("canvas");
    cv.width = cv.height = s;
    const g = cv.getContext("2d").createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(0.55, "rgba(255,255,255,0.9)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    const ctx = cv.getContext("2d");
    ctx.fillStyle = g; ctx.fillRect(0, 0, s, s);
    const t = new THREE.CanvasTexture(cv); t.needsUpdate = true;
    return t;
  }

  function groundRing(radius, color) {
    const segs = 96, pts = [];
    for (let i = 0; i <= segs; i++) {
      const a = (i / segs) * Math.PI * 2;
      pts.push(Math.cos(a) * radius, Math.sin(a) * radius, 0);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color }));
  }

  function build() {
    if (booted) return true;
    mount = document.getElementById("obs-sphere") || document.getElementById("bigSphere");
    if (!mount) return false;
    const w = Math.max(120, mount.clientWidth || 200);
    const h = Math.max(120, mount.clientHeight || 200);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c1019);
    scene.fog = new THREE.FogExp2(0x0c1019, 0.06);   // near crisp, far recedes (depth cue)

    camera = new THREE.PerspectiveCamera(50, w / h, 0.05, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(-3.4, -3.8, 3.0);
    camera.lookAt(0, 0, 0.3);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));  // fill-rate bound
    renderer.setSize(w, h, false);
    mount.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.3);
    controls.enablePan = false;
    controls.minDistance = 2.0;
    controls.maxDistance = 14;
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.9));

    const grid = new THREE.GridHelper(RANGE_M * 2, 12, COL.grid, COL.grid);
    grid.rotation.x = Math.PI / 2;
    grid.material.opacity = 0.35; grid.material.transparent = true;
    scene.add(grid);

    stopRing = groundRing(1, COL.stop); stopRing.scale.set(STOP_M, STOP_M, 1); scene.add(stopRing);
    slowRing = groundRing(1, COL.slow); slowRing.scale.set(SLOW_M, SLOW_M, 1); scene.add(slowRing);
    scene.add(groundRing(RANGE_M, COL.dome));

    const fovHalf = ((D435I_FOV_DEG / 2) * Math.PI) / 180, fovPts = [];
    [-fovHalf, fovHalf].forEach((a) => {
      fovPts.push(0, 0, 0.02, RANGE_M * Math.cos(a), RANGE_M * Math.sin(a), 0.02);
    });
    const fovGeo = new THREE.BufferGeometry();
    fovGeo.setAttribute("position", new THREE.Float32BufferAttribute(fovPts, 3));
    scene.add(new THREE.LineSegments(fovGeo,
      new THREE.LineBasicMaterial({ color: COL.robot, transparent: true, opacity: 0.18 })));

    const dome = new THREE.Mesh(
      new THREE.SphereGeometry(RANGE_M, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: COL.dome, wireframe: true, transparent: true, opacity: 0.10 }));
    scene.add(dome);

    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.5, 16),
      new THREE.MeshBasicMaterial({ color: COL.robot }));
    body.rotation.x = Math.PI / 2; body.position.set(0, 0, 0.25); scene.add(body);
    const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.28, 16),
      new THREE.MeshBasicMaterial({ color: COL.robot }));
    arrow.rotation.z = -Math.PI / 2; arrow.position.set(0.4, 0, 0.25); scene.add(arrow);

    // --- obstacle POINT CLOUD (per-object coloured) ---
    const geo = new THREE.BufferGeometry();
    cloudPos = new Float32Array(MAX_POINTS * 3);
    cloudCol = new Float32Array(MAX_POINTS * 3);
    geo.setAttribute("position", new THREE.BufferAttribute(cloudPos, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute("color", new THREE.BufferAttribute(cloudCol, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setDrawRange(0, 0);
    cloud = new THREE.Points(geo, new THREE.PointsMaterial({
      size: POINT_SIZE, sizeAttenuation: true, vertexColors: true, map: discTexture(),
      transparent: true, alphaTest: 0.5, depthWrite: true,
    }));
    scene.add(cloud);

    // --- legacy voxel COLUMNS (toggle target): one height-coloured box per occupied cell ---
    const boxGeo = new THREE.BoxGeometry(1, 1, 1); boxGeo.translate(0, 0, 0.5);  // grow from z=0
    columns = new THREE.InstancedMesh(boxGeo, new THREE.MeshLambertMaterial(), MAX_CELLS);
    columns.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    columns.count = 0;
    scene.add(columns);
    scene.add(new THREE.DirectionalLight(0xffffff, 0.6).translateZ(5));  // shading for the boxes

    // --- pooled per-object bounding boxes (unit cube edges, scaled/placed per frame) ---
    const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    for (let i = 0; i < MAX_BOXES; i++) {
      const b = new THREE.LineSegments(edges,
        new THREE.LineBasicMaterial({ transparent: true, opacity: 0.9 }));
      b.visible = false; scene.add(b); boxes.push(b);
    }

    // mode toggle: a small fixed corner button + the 'c' key (choice persists).
    toggleBtn = document.createElement("button");
    toggleBtn.id = "obs3d-mode"; toggleBtn.title = "Toggle 3D view: points / columns (c)";
    toggleBtn.style.cssText = "position:fixed;bottom:10px;left:10px;z-index:9999;" +
      "padding:4px 9px;font:600 12px system-ui,sans-serif;color:#E9EDF6;" +
      "background:rgba(20,26,38,0.85);border:1px solid #5C8DF2;border-radius:6px;cursor:pointer;";
    toggleBtn.onclick = () => setMode(mode === "points" ? "columns" : "points");
    document.body.appendChild(toggleBtn);
    window.addEventListener("keydown", (e) => {
      const tag = (e.target && e.target.tagName) || "";
      if ((e.key === "c" || e.key === "C") && !/INPUT|TEXTAREA/.test(tag))
        setMode(mode === "points" ? "columns" : "points");
    });

    booted = true;
    applyMode();
    if (lastMsg) refresh(lastMsg);
    animate();
    window.addEventListener("resize", onResize);
    return true;
  }

  function onResize() {
    if (!booted || !mount) return;
    const w = Math.max(120, mount.clientWidth || 200);
    const h = Math.max(120, mount.clientHeight || 200);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }

  // Bin (x,y,height) samples into cells; each cell keeps its tallest height, integer
  // (i,j) index and the list of contributing point indices (for per-point recolour).
  function voxelize(addSamples) {
    const cells = new Map();
    addSamples((x, y, hz, pIdx) => {
      const i = Math.round(x / CELL), j = Math.round(y / CELL), key = i + "," + j;
      let c = cells.get(key);
      if (!c) { c = { i, j, cx: i * CELL, cy: j * CELL, h: hz, pts: [] }; cells.set(key, c); }
      else if (hz > c.h) c.h = hz;
      if (pIdx >= 0) c.pts.push(pIdx);
    });
    return cells;
  }

  // 8-connected connected-components (union-find) over the occupied cells.
  const NEIGH = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
  function clusterCells(cells) {
    const parent = new Map();
    for (const k of cells.keys()) parent.set(k, k);
    const find = (k) => { while (parent.get(k) !== k) { parent.set(k, parent.get(parent.get(k))); k = parent.get(k); } return k; };
    const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent.set(ra, rb); };
    for (const [k, c] of cells) {
      for (const [di, dj] of NEIGH) {
        const nk = (c.i + di) + "," + (c.j + dj);
        if (cells.has(nk)) union(k, nk);
      }
    }
    const groups = new Map();
    for (const k of cells.keys()) { const r = find(k); (groups.get(r) || groups.set(r, []).get(r)).push(k); }
    const clusters = [];
    for (const ks of groups.values()) {
      if (ks.length < MIN_CLUSTER_CELLS) continue;
      let sx = 0, sy = 0, minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9, maxh = 0;
      for (const k of ks) {
        const c = cells.get(k); sx += c.cx; sy += c.cy;
        if (c.cx < minx) minx = c.cx; if (c.cx > maxx) maxx = c.cx;
        if (c.cy < miny) miny = c.cy; if (c.cy > maxy) maxy = c.cy;
        if (c.h > maxh) maxh = c.h;
      }
      clusters.push({ keys: ks, cx: sx / ks.length, cy: sy / ks.length,
                      minx, maxx, miny, maxy, maxh, cells: ks.length, id: 0 });
    }
    return clusters;
  }

  // greedy nearest-centroid association -> stable ids across frames
  function assignIds(clusters) {
    const matched = new Array(tracks.length).fill(false);
    for (const cl of clusters) {
      let best = -1, bd = TRACK_GATE_M * TRACK_GATE_M;
      for (let t = 0; t < tracks.length; t++) {
        if (matched[t]) continue;
        const dx = tracks[t].cx - cl.cx, dy = tracks[t].cy - cl.cy, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = t; }
      }
      if (best >= 0) { cl.id = tracks[best].id; tracks[best].cx = cl.cx; tracks[best].cy = cl.cy; tracks[best].missed = 0; matched[best] = true; }
      else { cl.id = nextId++; tracks.push({ id: cl.id, cx: cl.cx, cy: cl.cy, missed: 0 }); matched.push(true); }
    }
    for (let t = tracks.length - 1; t >= 0; t--) {
      if (!matched[t]) { tracks[t].missed++; if (tracks[t].missed > TRACK_MISS_MAX) tracks.splice(t, 1); }
    }
  }

  function refresh(msg) {
    if (!booted) return;
    const ns = (msg && typeof msg.stop_m === "number" && isFinite(msg.stop_m)) ? msg.stop_m : DEF_STOP;
    const nl = (msg && typeof msg.slow_m === "number" && isFinite(msg.slow_m)) ? msg.slow_m : DEF_SLOW;
    if (ns !== STOP_M) { STOP_M = ns; if (stopRing) stopRing.scale.set(STOP_M, STOP_M, 1); }
    if (nl !== SLOW_M) { SLOW_M = nl; if (slowRing) slowRing.scale.set(SLOW_M, SLOW_M, 1); }

    const ring = msg && msg.ring, pts = msg && msg.points;
    // gather (x,y,h) samples + keep a parallel raw-XYZ list for point rendering
    const px = [], py = [], ph = [];
    if (Array.isArray(pts) && pts.length >= 3) {
      const m = Math.min(pts.length - (pts.length % 3), MAX_POINTS * 3);
      for (let i = 0; i < m; i += 3) { px.push(pts[i]); py.push(pts[i + 1]); ph.push(pts[i + 2]); }
    } else if (ring && Array.isArray(ring.dist) && isFinite(ring.bin_deg) && isFinite(ring.start_deg)) {
      const st = Array.isArray(ring.state) ? ring.state : null;
      for (let i = 0; i < ring.dist.length; i++) {
        const d = ring.dist[i];
        if ((st ? st[i] !== 2 : d == null) || d == null) continue;
        const a = ((ring.start_deg + (i + 0.5) * ring.bin_deg) * Math.PI) / 180;
        const dd = Math.min(d, RANGE_M);
        px.push(dd * Math.cos(a)); py.push(dd * Math.sin(a)); ph.push(0.4);
      }
    }

    const cells = voxelize((emit) => { for (let i = 0; i < px.length; i++) emit(px[i], py[i], ph[i], i); });
    const clusters = clusterCells(cells);
    assignIds(clusters);

    // point index -> cluster (via its cell); build a cell-key -> cluster map
    const keyToCl = new Map();
    for (const cl of clusters) for (const k of cl.keys) keyToCl.set(k, cl);

    if (mode === "points") {
      // POINTS: colour each point by its object's stable hue, height as a lightness cue;
      // ungrouped (noise) points -> dim steel.
      const n = Math.min(px.length, MAX_POINTS);
      for (let i = 0; i < n; i++) {
        cloudPos[i * 3] = px[i]; cloudPos[i * 3 + 1] = py[i]; cloudPos[i * 3 + 2] = ph[i];
        const key = Math.round(px[i] / CELL) + "," + Math.round(py[i] / CELL);
        const cl = keyToCl.get(key);
        if (cl) {
          hueColor(cl.id, _c);
          _c.getHSL(_hsl);
          _c.setHSL(_hsl.h, _hsl.s, clampv(0.30 + 0.5 * (ph[i] / H_MAX), 0.22, 0.82));
        } else { _c.setHex(COL.noise); }
        cloudCol[i * 3] = _c.r; cloudCol[i * 3 + 1] = _c.g; cloudCol[i * 3 + 2] = _c.b;
      }
      cloud.geometry.setDrawRange(0, n);
      cloud.geometry.attributes.position.needsUpdate = true;
      cloud.geometry.attributes.color.needsUpdate = true;
    } else {
      // COLUMNS (legacy): one height-coloured box per occupied cell.
      let ci = 0;
      for (const cc of cells.values()) {
        if (ci >= MAX_CELLS) break;
        _s.set(CELL, CELL, Math.max(MIN_H, cc.h));
        _m.compose(_p.set(cc.cx, cc.cy, 0), _q, _s);
        columns.setMatrixAt(ci, _m);
        heightColor(cc.h, _c); columns.setColorAt(ci, _c);
        ci++;
      }
      columns.count = ci;
      columns.instanceMatrix.needsUpdate = true;
      if (columns.instanceColor) columns.instanceColor.needsUpdate = true;
    }

    // per-object bounding boxes, coloured by the object hue, tinted toward red when the
    // object is inside the slow/stop band (threat) so the safety cue survives.
    let bi = 0;
    for (const cl of clusters) {
      if (bi >= MAX_BOXES) break;
      const b = boxes[bi++];
      const wx = Math.max(CELL, cl.maxx - cl.minx + CELL);
      const wy = Math.max(CELL, cl.maxy - cl.miny + CELL);
      const wz = Math.max(0.12, cl.maxh);
      b.position.set((cl.minx + cl.maxx) / 2, (cl.miny + cl.maxy) / 2, wz / 2);
      b.scale.set(wx, wy, wz);
      hueColor(cl.id, _c);
      const r = Math.hypot(cl.cx, cl.cy);
      const prox = clampv((SLOW_M - r) / ((SLOW_M - STOP_M) || 1), 0, 1);
      if (prox > 0) { _c2.copy(_amber).lerp(_red, prox); _c.lerp(_c2, 0.6 * prox); }
      b.material.color.copy(_c);
      b.material.opacity = 0.55 + 0.4 * prox;
      b.visible = true;
    }
    for (; bi < MAX_BOXES; bi++) boxes[bi].visible = false;
  }
  const _hsl = { h: 0, s: 0, l: 0 };

  function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    renderer.render(scene, camera);
  }

  window.Obstacle3D = {
    update: function (msg) {
      try {
        lastMsg = msg || null;
        if (!booted && !build()) return;
        refresh(lastMsg);
      } catch (e) { /* defensive: never break the WS loop */ }
    },
    mountTo: function (target) {
      try {
        if (!target) return;
        if (!booted && !build()) return;
        if (renderer && renderer.domElement.parentElement !== target) {
          target.appendChild(renderer.domElement); mount = target; onResize();
        }
      } catch (e) { /* defensive */ }
    },
    setMode: function (m) { try { setMode(m); } catch (e) {} },
    toggleMode: function () { try { setMode(mode === "points" ? "columns" : "points"); } catch (e) {} },
    getMode: function () { return mode; },
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();
