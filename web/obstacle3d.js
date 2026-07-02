// 3D obstacle view for the obstacle panel.
//
// ES module — uses the page importmap (three + OrbitControls), like lidar.js.
// Renders the ACTUAL kept obstacle points the guard perceives (post ground/self/
// range filter) as a SOLID voxel-column field: points are binned into ground
// cells and each occupied cell becomes a height-coloured block standing on the
// floor. Solid + dense (no scattered-dot gaps), and a low object on the ground
// shows up as a short block instead of a few invisible dots. This is the
// high-precision "what does it see / what is it blind to" companion to the flat
// 2D polar radar in obstacle.js.
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
  // Ground danger rings. Defaults MATCH the node (0.75 / 1.75, NOT 0.7/2.0); the
  // real values arrive on msg.stop_m / msg.slow_m and rescale the rings live.
  const DEF_STOP = 0.75, DEF_SLOW = 1.75;
  let STOP_M = DEF_STOP, SLOW_M = DEF_SLOW;
  const H_MAX = 1.9;                   // height (m) at the top of the colour ramp
  const CELL = 0.13;                   // voxel cell size (m) — smaller = finer/denser
  const MIN_H = 0.06;                  // min drawn column height so low hits stay visible
  const MAX_CELLS = 1500;              // instanced-column cap (bounded GPU work)
  const D435I_FOV_DEG = 87;            // forward depth-camera cone hint
  const COL = { grid: 0x232c3e, dome: 0x35507d, robot: 0x5c8df2,
                stop: 0xff4747, slow: 0xff8636 };

  let scene, camera, renderer, controls, columns, stopRing, slowRing;
  let mount, booted = false, lastMsg = null;
  // Preallocated scratch — reused every frame so the update loop never allocates
  // (per-frame `new` here causes GC stutter / instance flicker).
  const _m = new THREE.Matrix4(), _p = new THREE.Vector3(),
        _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _col = new THREE.Color();
  const _voxCol = new THREE.Color(), _dc = new THREE.Color(),
        _amber = new THREE.Color(COL.slow), _red = new THREE.Color(COL.stop);
  function clampv(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  // Height -> colour ramp: ground (cyan) -> waist (green) -> tall (amber) -> high (red).
  const RAMP = [
    [0.00, 0x39d6ff], [0.30, 0x3fd39b], [0.62, 0xffd24a], [1.00, 0xff5a47],
  ];
  const _ca = new THREE.Color(), _cb = new THREE.Color();
  function heightColor(h) {
    let t = h / H_MAX; t = t < 0 ? 0 : t > 1 ? 1 : t;
    let i = 0;
    while (i < RAMP.length - 2 && t > RAMP[i + 1][0]) i++;
    const a = RAMP[i], b = RAMP[i + 1];
    let f = (t - a[0]) / (b[0] - a[0] || 1); f = f < 0 ? 0 : f > 1 ? 1 : f;
    _ca.setHex(a[1]); _cb.setHex(b[1]);
    return _col.copy(_ca).lerp(_cb, f);
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
    // The small in-panel mount (#obs-sphere) was removed; the sphere now lives only
    // in the big top-right feed. Build into #obs-sphere if it still exists, else the
    // big feed mount (#bigSphere). feedview.js re-parents the single canvas as the
    // operator switches feeds; a null target there is a no-op, so it parks here.
    mount = document.getElementById("obs-sphere") || document.getElementById("bigSphere");
    if (!mount) return false;
    const w = Math.max(120, mount.clientWidth || 200);
    const h = Math.max(120, mount.clientHeight || 200);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c1019);

    camera = new THREE.PerspectiveCamera(50, w / h, 0.05, 100);
    camera.up.set(0, 0, 1);                 // Z up
    camera.position.set(-3.4, -3.8, 3.0);   // behind + left, looking forward & down
    camera.lookAt(0, 0, 0.3);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h, false);
    mount.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.3);
    controls.enablePan = false;
    controls.minDistance = 2.0;
    controls.maxDistance = 14;
    controls.autoRotate = false;            // no auto-spin (operator preference)
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.78));
    const dir = new THREE.DirectionalLight(0xffffff, 0.55);
    dir.position.set(2, -3, 5);
    scene.add(dir);

    // ground grid (GridHelper lies in XZ by default -> rotate into XY for Z-up).
    const grid = new THREE.GridHelper(RANGE_M * 2, 12, COL.grid, COL.grid);
    grid.rotation.x = Math.PI / 2;
    grid.material.opacity = 0.35;
    grid.material.transparent = true;
    scene.add(grid);

    // stop / slow / range reference rings on the ground (danger context). The stop
    // and slow rings are built at unit radius and SCALED to the live thresholds, so
    // refresh() can rescale them when the guard reports its real stop_m / slow_m.
    stopRing = groundRing(1, COL.stop); stopRing.scale.set(STOP_M, STOP_M, 1); scene.add(stopRing);
    slowRing = groundRing(1, COL.slow); slowRing.scale.set(SLOW_M, SLOW_M, 1); scene.add(slowRing);
    scene.add(groundRing(RANGE_M, COL.dome));

    // D435i forward depth-FOV hint (~87 deg cone) — faint boundary lines on the ground.
    const fovHalf = ((D435I_FOV_DEG / 2) * Math.PI) / 180;
    const fovPts = [];
    [-fovHalf, fovHalf].forEach((a) => {
      fovPts.push(0, 0, 0.02, RANGE_M * Math.cos(a), RANGE_M * Math.sin(a), 0.02);
    });
    const fovGeo = new THREE.BufferGeometry();
    fovGeo.setAttribute("position", new THREE.Float32BufferAttribute(fovPts, 3));
    scene.add(new THREE.LineSegments(fovGeo,
      new THREE.LineBasicMaterial({ color: COL.robot, transparent: true, opacity: 0.18 })));

    // translucent detection dome — upper hemisphere "sphere".
    const dome = new THREE.Mesh(
      new THREE.SphereGeometry(RANGE_M, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2),
      new THREE.MeshBasicMaterial({
        color: COL.dome, wireframe: true, transparent: true, opacity: 0.10,
      }));
    scene.add(dome);

    // robot body + forward arrow at the centre, so orientation is obvious.
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.14, 0.14, 0.5, 16),
      new THREE.MeshLambertMaterial({ color: COL.robot }));
    body.rotation.x = Math.PI / 2;          // cylinder axis Y -> Z (stand upright)
    body.position.set(0, 0, 0.25);
    scene.add(body);
    const arrow = new THREE.Mesh(
      new THREE.ConeGeometry(0.11, 0.28, 16),
      new THREE.MeshLambertMaterial({ color: COL.robot }));
    arrow.rotation.z = -Math.PI / 2;        // cone apex +Y -> +X (forward), laid flat
    arrow.position.set(0.4, 0, 0.25);
    scene.add(arrow);

    // obstacle voxel columns — InstancedMesh of unit boxes whose base sits on the
    // floor (z=0) and grows +Z when scaled, so scale.z == column height.
    const boxGeo = new THREE.BoxGeometry(1, 1, 1);
    boxGeo.translate(0, 0, 0.5);
    columns = new THREE.InstancedMesh(
      boxGeo, new THREE.MeshLambertMaterial(), MAX_CELLS);
    columns.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    columns.count = 0;
    scene.add(columns);

    booted = true;
    if (lastMsg) refresh(lastMsg);
    animate();
    window.addEventListener("resize", onResize);
    return true;
  }

  function onResize() {
    if (!booted || !mount) return;
    const w = Math.max(120, mount.clientWidth || 200);
    const h = Math.max(120, mount.clientHeight || 200);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }

  // Bin a list of (x,y,height) samples into ground cells -> tallest height per cell.
  function voxelize(addSamples) {
    const cells = new Map();              // "i,j" -> {cx, cy, h}
    addSamples((x, y, hz) => {
      const i = Math.round(x / CELL), j = Math.round(y / CELL);
      const key = i + "," + j;
      const prev = cells.get(key);
      if (prev === undefined) cells.set(key, { cx: i * CELL, cy: j * CELL, h: hz });
      else if (hz > prev.h) prev.h = hz;
    });
    return cells;
  }

  function refresh(msg) {
    if (!booted) return;

    // thresholds: prefer the guard's real values; else node defaults. Rescale the
    // ground rings only when they actually change (cheap; avoids per-frame work).
    const ns = (msg && typeof msg.stop_m === "number" && isFinite(msg.stop_m)) ? msg.stop_m : DEF_STOP;
    const nl = (msg && typeof msg.slow_m === "number" && isFinite(msg.slow_m)) ? msg.slow_m : DEF_SLOW;
    if (ns !== STOP_M) { STOP_M = ns; if (stopRing) stopRing.scale.set(STOP_M, STOP_M, 1); }
    if (nl !== SLOW_M) { SLOW_M = nl; if (slowRing) slowRing.scale.set(SLOW_M, SLOW_M, 1); }

    const ring = msg && msg.ring;
    const pts = msg && msg.points;
    let cells;
    if (Array.isArray(pts) && pts.length >= 3) {
      const m = pts.length - (pts.length % 3);
      cells = voxelize((emit) => {
        for (let i = 0; i < m; i += 3) emit(pts[i], pts[i + 1], pts[i + 2]);
      });
    } else {
      // Fallback: no point export -> a stub column per OCCUPIED ring sector.
      cells = voxelize((emit) => {
        if (!ring || !Array.isArray(ring.dist) ||
            !isFinite(ring.bin_deg) || !isFinite(ring.start_deg)) return;
        const st = Array.isArray(ring.state) ? ring.state : null;
        for (let i = 0; i < ring.dist.length; i++) {
          const d = ring.dist[i];
          const occupied = st ? (st[i] === 2) : (d != null);
          if (!occupied || d == null) continue;
          const a = ((ring.start_deg + (i + 0.5) * ring.bin_deg) * Math.PI) / 180;
          const dd = Math.min(d, RANGE_M);
          emit(dd * Math.cos(a), dd * Math.sin(a), 0.5);   // unknown height -> stub
        }
      });
    }

    // Occupancy lookup by angle from the ring (state/prob) -> tints voxels by
    // proximity * occupancy-probability ON TOP of the height colour. Absent -> height only.
    const rState = (ring && Array.isArray(ring.state)) ? ring.state : null;
    const rProb  = (ring && Array.isArray(ring.prob)) ? ring.prob : null;
    const haveOcc = !!(ring && (rState || rProb) && Array.isArray(ring.dist) &&
                       isFinite(ring.bin_deg) && isFinite(ring.start_deg));
    const rN = haveOcc ? (ring.n || ring.dist.length) : 0;
    function occAt(cx, cy) {
      if (!haveOcc || rN <= 0) return 0;
      const deg = Math.atan2(cy, cx) * 180 / Math.PI;
      let idx = Math.floor((deg - ring.start_deg) / ring.bin_deg);
      idx = ((idx % rN) + rN) % rN;
      const occupied = rState ? (rState[idx] === 2) : (ring.dist[idx] != null);
      if (!occupied) return 0;
      return rProb ? clampv(rProb[idx], 0, 1) : 1;
    }

    let k = 0;
    cells.forEach((c) => {
      if (k >= MAX_CELLS) return;
      const hDraw = Math.max(MIN_H, c.h);
      _p.set(c.cx, c.cy, 0);
      _s.set(CELL * 0.92, CELL * 0.92, hDraw);
      _m.compose(_p, _q, _s);
      columns.setMatrixAt(k, _m);

      _voxCol.copy(heightColor(c.h));
      if (haveOcc) {
        const r = Math.hypot(c.cx, c.cy);
        // proximity: 1 at/inside stop, 0 beyond slow.
        const prox = clampv((SLOW_M - r) / ((SLOW_M - STOP_M) || 1), 0, 1);
        const threat = prox * occAt(c.cx, c.cy);
        if (threat > 0) {
          _dc.copy(_amber).lerp(_red, prox);     // amber (far) -> red (close)
          _voxCol.lerp(_dc, 0.65 * threat);
        }
      }
      columns.setColorAt(k, _voxCol);
      k++;
    });
    columns.count = k;
    columns.instanceMatrix.needsUpdate = true;
    if (columns.instanceColor) columns.instanceColor.needsUpdate = true;
  }

  function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();        // keep controls current (no auto-spin)
    renderer.render(scene, camera);
  }

  window.Obstacle3D = {
    update: function (msg) {
      try {
        lastMsg = msg || null;
        if (!booted && !build()) return;     // panel not laid out yet -> try next frame
        refresh(lastMsg);
      } catch (e) { /* defensive: never break the WS loop */ }
    },
    // Move the single renderer canvas into `target` (the big top-right feed) or
    // back to the small panel (#obs-sphere). One WebGL context, re-parented +
    // resized — no second context. Builds first if telemetry hasn't arrived yet.
    mountTo: function (target) {
      try {
        if (!target) return;
        if (!booted && !build()) return;
        if (renderer && renderer.domElement.parentElement !== target) {
          target.appendChild(renderer.domElement);
          mount = target;
          onResize();
        }
      } catch (e) { /* defensive */ }
    },
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
