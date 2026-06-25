// G1 3D LiDAR / map viewer — point cloud coloured by height, with a ground grid,
// XYZ axes, OrbitControls, and analysis tools (top-down, fit, point size, colour).
// World frame is Z-up; data over /ws/lidar: uint32 N, uint8 stride, 3 pad, N*3 f32.

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

const container = document.getElementById("lidarView");
const badge = document.getElementById("lidarBadge");

const scene = new THREE.Scene();
// Vertical gradient backdrop + gentle exponential depth fog: the cloud reads as a
// volume (near points crisp, far ones melting into the haze) instead of flat dots
// on a slab. Fog colour matches the lower backdrop tone so the far edge is clean.
const FOG_COLOR = 0x0b0d14;
scene.background = makeGradientBackdrop("#171c28", "#0d1019", "#080910");
scene.fog = new THREE.FogExp2(FOG_COLOR, 0.014);

const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);  // FOV matches the 3D sphere
camera.up.set(0, 0, 1);                 // Z is up
camera.position.set(-3.4, -3.8, 3.0);   // match the 3D sphere's framing

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
// Sphere-like feel: locked onto the cloud (no pan, so you can't slide away and get
// lost), crisp 1:1 orbit that STOPS the instant you release (no damping drift), and
// bounded zoom. The camera only moves when YOU move it -- the data-driven auto-
// switch that used to yank the view is removed (see updateObstacle below).
controls.enableDamping = false;
controls.enablePan = false;
controls.minDistance = 0.5;
controls.maxDistance = 80;
controls.target.set(0, 0, 0.3);         // match the 3D sphere's target (robot-centric)
controls.maxPolarAngle = Math.PI / 2;   // never orbit below the floor (Z-up frame)

// Larger, dimmer ground grid: the fog fades the far lines so it reads as an
// endless floor rather than a hard 20 m square. 1 m spacing.
const grid = new THREE.GridHelper(60, 60, 0x3a4156, 0x171b27);
grid.rotation.x = Math.PI / 2;          // make it the XY (ground) plane
grid.material.transparent = true;
grid.material.opacity = 0.55;
scene.add(grid);
const axes = new THREE.AxesHelper(0.6);
scene.add(axes);

// Orientation gizmo: tiny axis triad that mirrors the main camera's rotation,
// drawn in the bottom-left corner so up/down stays clear while orbiting.
const gizmoScene = new THREE.Scene();
gizmoScene.add(new THREE.AxesHelper(1));            // R=X(fwd), G=Y(left), B=Z(up)
const gizmoCam = new THREE.PerspectiveCamera(50, 1, 0.1, 10);
const _gdir = new THREE.Vector3();

// Point cloud (per-vertex colour by height).
const geometry = new THREE.BufferGeometry();
let capacity = 0;
geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
let pointSize = 0.05;
const material = new THREE.PointsMaterial({
  size: pointSize, vertexColors: true, sizeAttenuation: true,
  map: makeDiscTexture(),     // round, soft points instead of hard squares
  alphaTest: 0.45,            // cut the sprite's transparent corners -> depth-correct discs
  transparent: true, depthWrite: true,
});
const points = new THREE.Points(geometry, material);
scene.add(points);

// Cloud extent (updated each frame) for fit / top-down framing.
let center = new THREE.Vector3(2, 0, 0.8);
let radius = 4;
let didAutoFit = false;
let mapView = false;             // live vs loaded-map view (drives the auto-framing)
let colorByHeight = true;

// Dynamic height range for the colour ramp (smoothed).
let zLo = 0, zHi = 2.2;

// Height -> colour. A refined cool->warm sweep (blue floor ... red overhead):
// deeper and less neon than a raw turbo, but still reads height at a glance.
const _RAMP = [
  [0.00, 0x2b4cf0], [0.22, 0x1fb6c9], [0.45, 0x37d67a],
  [0.68, 0xc9d62e], [0.85, 0xf7a93b], [1.00, 0xff5a4a],
];
function turbo(t) {
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  let i = 0;
  while (i < _RAMP.length - 2 && t > _RAMP[i + 1][0]) i++;
  const a = _RAMP[i], b = _RAMP[i + 1];
  const f = (t - a[0]) / (b[0] - a[0] || 1);
  const ca = a[1], cb = b[1];
  return [
    (((ca >> 16) & 255) + ((((cb >> 16) & 255) - ((ca >> 16) & 255)) * f)) / 255,
    (((ca >> 8) & 255) + ((((cb >> 8) & 255) - ((ca >> 8) & 255)) * f)) / 255,
    (((ca) & 255) + ((((cb) & 255) - ((ca) & 255)) * f)) / 255,
  ];
}

// Round, soft point sprite (replaces the default hard square) — drawn once.
function makeDiscTexture() {
  const s = 64, c = document.createElement("canvas");
  c.width = c.height = s;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  grd.addColorStop(0.0, "rgba(255,255,255,1)");
  grd.addColorStop(0.55, "rgba(255,255,255,1)");
  grd.addColorStop(0.80, "rgba(255,255,255,0.55)");
  grd.addColorStop(1.0, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, s, s);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// Vertical gradient backdrop texture (top -> mid -> bottom CSS colours).
function makeGradientBackdrop(top, mid, bottom) {
  const c = document.createElement("canvas");
  c.width = 4; c.height = 256;
  const g = c.getContext("2d");
  const grd = g.createLinearGradient(0, 0, 0, 256);
  grd.addColorStop(0, top);
  grd.addColorStop(0.6, mid);
  grd.addColorStop(1, bottom);
  g.fillStyle = grd;
  g.fillRect(0, 0, 4, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function ensureCapacity(n) {
  if (n <= capacity) return;
  capacity = Math.max(n, capacity * 2, 4096);
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
}

function updateCloud(buffer) {
  const view = new DataView(buffer);
  const n = view.getUint32(0, true);
  if (n === 0) { geometry.setDrawRange(0, 0); return; }
  const f = new Float32Array(buffer, 8, n * 3);
  ensureCapacity(n);
  const pos = geometry.getAttribute("position").array;

  // First pass: bounds.
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity, zMin = Infinity, zMax = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = f[i * 3], y = f[i * 3 + 1], z = f[i * 3 + 2];
    if (x < xMin) xMin = x; if (x > xMax) xMax = x;
    if (y < yMin) yMin = y; if (y > yMax) yMax = y;
    if (z < zMin) zMin = z; if (z > zMax) zMax = z;
  }
  // Smooth the height-colour range toward the actual z span (robust to outliers-ish).
  zLo += (zMin - zLo) * 0.2;
  zHi += (zMax - zHi) * 0.2;
  const span = Math.max(0.3, zHi - zLo);

  const col = geometry.getAttribute("color").array;
  const flat = [1.0, 0.72, 0.30];
  for (let i = 0; i < n; i++) {
    const x = f[i * 3], y = f[i * 3 + 1], z = f[i * 3 + 2];
    pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
    const c = colorByHeight ? turbo((z - zLo) / span) : flat;
    col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2];
  }
  geometry.getAttribute("position").needsUpdate = true;
  geometry.getAttribute("color").needsUpdate = true;
  geometry.setDrawRange(0, n);

  center.set((xMin + xMax) / 2, (yMin + yMax) / 2, (zMin + zMax) / 2);
  radius = Math.max(1, 0.5 * Math.hypot(xMax - xMin, yMax - yMin, zMax - zMin));
  // First frame (and on a live<->map switch): a big map is framed to fit; the live
  // cloud gets the 3D-sphere's robot-centric framing so the two views match.
  if (!didAutoFit) { fitView(); didAutoFit = true; }
}

// --- view controls -------------------------------------------------------

function fitView() {
  controls.target.copy(center);
  const d = radius * 1.8;
  camera.position.set(center.x - d * 0.7, center.y - d * 0.7, center.z + d * 0.7);
  camera.updateProjectionMatrix();
  controls.update();
}

// No in-view buttons: the feed auto-frames the cloud (fitView) and is mouse/touch
// orbit only. Height colours and the ground grid are always on.

function resize() {
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  // Pulse the nearest-obstacle ring while in the STOP zone (set by the overlay).
  if (typeof nearPulse !== "undefined" && nearPulse && nearMarker.visible) {
    nearMat.opacity = 0.45 + 0.45 * (0.5 + 0.5 * Math.sin(performance.now() * 0.008));
  }

  // Main scene fills the whole canvas.
  renderer.setViewport(0, 0, container.clientWidth, container.clientHeight);
  renderer.render(scene, camera);

  // Orientation gizmo (bottom-left corner) — axes mirror the main camera's rotation,
  // so the blue Z axis always points the same way "up" does in the main view.
  _gdir.copy(camera.position).sub(controls.target).normalize();
  gizmoCam.position.copy(_gdir).multiplyScalar(3);
  gizmoCam.up.copy(camera.up);
  gizmoCam.lookAt(0, 0, 0);
  const s = 92;
  renderer.setScissorTest(true);
  renderer.setViewport(10, 10, s, s);
  renderer.setScissor(10, 10, s, s);
  renderer.render(gizmoScene, gizmoCam);
  renderer.setScissorTest(false);
}

// --- obstacle-guard overlay ----------------------------------------------
// A visual aid layered on the existing point cloud: the live front depth
// profile it "tracks" (B), a planned-path arrow toward the chosen gap (A),
// and subtle side ticks (C). Only visible while the guard is enabled+live.
// All drawn near the ground plane (z ~ 0.05), robot origin = (0,0). Frame:
// angle 0 = straight ahead (+X), + = LEFT (+Y), - = RIGHT (-Y).
const obstacleGroup = new THREE.Group();
obstacleGroup.visible = false;
scene.add(obstacleGroup);

const OBS_Z = 0.05;                     // sit just above the grid
const STOP_M = 0.7, SLOW_M = 2.0;       // distance thresholds for colour ramp
const COL_STOP = 0xff3b30, COL_SLOW = 0xffb020, COL_CLEAR = 0x32d74b;
const COL_SIDE = 0xff8c1a;

// Reused materials (lines rebuild geometry each message; materials persist).
const profileMatStop  = new THREE.LineBasicMaterial({ color: COL_STOP });
const profileMatSlow  = new THREE.LineBasicMaterial({ color: COL_SLOW });
const profileMatClear = new THREE.LineBasicMaterial({ color: COL_CLEAR });
const sideMat = new THREE.LineBasicMaterial({ color: COL_SIDE });

// Profile "fan": radial lines from origin to each sensed bin (B).
const profileLines = new THREE.Group();
obstacleGroup.add(profileLines);

// Side / back ticks (C).
const sideTicks = new THREE.Group();
obstacleGroup.add(sideTicks);

// Nearest-obstacle marker ring (B): a flat ring laid on the ground, recoloured
// and pulsed by zone. Built once, repositioned per message.
const nearGeo = new THREE.RingGeometry(0.14, 0.20, 28);
const nearMat = new THREE.MeshBasicMaterial({ color: COL_STOP, transparent: true, opacity: 0.9, side: THREE.DoubleSide });
const nearMarker = new THREE.Mesh(nearGeo, nearMat);
nearMarker.visible = false;
obstacleGroup.add(nearMarker);
let nearPulse = false;                  // animate opacity when STOP

// Planned-path arrow (A): a shaft (curved toward the turn) + a cone tip.
const pathMat = new THREE.LineBasicMaterial({ color: COL_CLEAR, linewidth: 3 });
const pathLine = new THREE.Line(new THREE.BufferGeometry(), pathMat);
obstacleGroup.add(pathLine);
const tipGeo = new THREE.ConeGeometry(0.09, 0.26, 16);
const tipMat = new THREE.MeshBasicMaterial({ color: COL_CLEAR });
const pathTip = new THREE.Mesh(tipGeo, tipMat);
obstacleGroup.add(pathTip);

function distMat(d) {
  if (d == null || d >= SLOW_M) return profileMatClear;
  return d < STOP_M ? profileMatStop : profileMatSlow;
}

// Dispose every child geometry of a Group and empty it (avoid GPU leaks).
function emptyGroup(g) {
  for (let i = g.children.length - 1; i >= 0; i--) {
    const c = g.children[i];
    if (c.geometry) c.geometry.dispose();
    g.remove(c);
  }
}

function updateObstacle(msg) {
  if (!msg) return;

  // The camera no longer auto-switches to the driving view when the guard turns on
  // -- it was yanking the view out from under you while you were orbiting. Use the
  // "Bot view" button to jump to the first-person driving view yourself.

  const visible = !!(msg.enabled && msg.live) && !msg.fault && !msg.auto_disabled;
  obstacleGroup.visible = visible;
  if (!visible) { nearPulse = false; return; }

  // --- (B) profile fan + nearest obstacle ---------------------------------
  emptyGroup(profileLines);
  const ring = msg.ring;
  let dists, N, angOf;
  if (ring && Array.isArray(ring.dist)) {
    dists = ring.dist; N = dists.length;
    angOf = (i) => (ring.start_deg + (i + 0.5) * ring.bin_deg) * Math.PI / 180;
  } else {                                  // legacy front-only fan
    dists = Array.isArray(msg.profile) ? msg.profile : []; N = dists.length;
    angOf = (i) => (60 - (i + 0.5) * (120 / N)) * Math.PI / 180;
  }
  let near = null, nearAng = 0;
  for (let i = 0; i < N; i++) {
    const d = dists[i];
    if (d == null) continue;
    const a = angOf(i);
    const ex = d * Math.cos(a), ey = d * Math.sin(a);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(
      new Float32Array([0, 0, OBS_Z, ex, ey, OBS_Z]), 3));
    profileLines.add(new THREE.Line(g, distMat(d)));
    if (near == null || d < near) { near = d; nearAng = a; }
  }

  // Nearest-obstacle marker: closest profile point, else front_m straight ahead.
  const zone = msg.zone || "CLEAR";
  let markD = near, markA = nearAng;
  if (markD == null && msg.front_m != null) { markD = msg.front_m; markA = 0; }
  if (markD != null) {
    nearMarker.visible = true;
    nearMarker.position.set(markD * Math.cos(markA), markD * Math.sin(markA), OBS_Z + 0.005);
    const zc = zone === "STOP" ? COL_STOP : zone === "SLOW" ? COL_SLOW : COL_CLEAR;
    nearMat.color.setHex(zc);
    nearPulse = zone === "STOP";
    if (!nearPulse) nearMat.opacity = 0.9;
  } else {
    nearMarker.visible = false;
    nearPulse = false;
  }

  // --- (A) planned-path arrow ---------------------------------------------
  const gap = msg.gap || {};
  const passable = gap.passable !== false && gap.state !== "BLOCKED";
  const arrowCol = passable ? COL_CLEAR : COL_STOP;
  pathMat.color.setHex(arrowCol);
  tipMat.color.setHex(arrowCol);

  const aimA = ((gap.center_deg || 0)) * Math.PI / 180;
  const sScale = (typeof msg.speed_scale === "number") ? msg.speed_scale : 1;
  const len = 2.0 * (0.4 + 0.6 * Math.max(0, Math.min(1, sScale)));  // shorten if slow
  const dim = 0.35 + 0.65 * Math.max(0, Math.min(1, sScale));        // dim if slow

  // Shallow quadratic bend toward gap.yaw_cmd (+ = turn left). Build the shaft
  // as a polyline that curves off the straight aim heading by a small lateral
  // offset that grows quadratically along its length.
  const yaw = (typeof gap.yaw_cmd === "number") ? gap.yaw_cmd : 0;
  const bend = Math.max(-0.5, Math.min(0.5, yaw * 0.4));   // metres of side-shift at tip
  const fwd = new THREE.Vector2(Math.cos(aimA), Math.sin(aimA));     // aim direction
  const lat = new THREE.Vector2(-fwd.y, fwd.x);                      // +90 deg = left
  const STEPS = 12;
  const pos = new Float32Array((STEPS + 1) * 3);
  let tipX = 0, tipY = 0, tipPX = 0, tipPY = 0;
  for (let i = 0; i <= STEPS; i++) {
    const t = i / STEPS;
    const off = bend * t * t;            // quadratic lateral bend
    const px = fwd.x * (len * t) + lat.x * off;
    const py = fwd.y * (len * t) + lat.y * off;
    pos[i * 3] = px; pos[i * 3 + 1] = py; pos[i * 3 + 2] = OBS_Z;
    if (i === STEPS) { tipX = px; tipY = py; }
    if (i === STEPS - 1) { tipPX = px; tipPY = py; }
  }
  const oldPath = pathLine.geometry;
  pathLine.geometry = new THREE.BufferGeometry();
  pathLine.geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  if (oldPath) oldPath.dispose();
  pathMat.opacity = dim; pathMat.transparent = true;

  // Cone tip: oriented along the last segment, pulled back half its height (0.13 m)
  // so the apex lands exactly at the shaft end instead of overshooting it.
  const tdx = tipX - tipPX, tdy = tipY - tipPY;
  const tdLen = Math.hypot(tdx, tdy) || 1;
  const ux = tdx / tdLen, uy = tdy / tdLen;
  pathTip.position.set(tipX - ux * 0.13, tipY - uy * 0.13, OBS_Z);
  // Cone default points +Y; rotate it to face the heading in the XY plane.
  pathTip.rotation.set(0, 0, Math.atan2(tdy, tdx) - Math.PI / 2);
  tipMat.opacity = dim; tipMat.transparent = true;

  // --- (C) side ticks ------------------------------------------------------
  emptyGroup(sideTicks);
  const side = msg.side || {};
  function sideTick(yVal) {
    if (yVal == null) return;
    const g = new THREE.BufferGeometry();
    // a short cross tick at (0, yVal, z), oriented along forward (X).
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
      -0.12, yVal, OBS_Z,  0.12, yVal, OBS_Z,
    ]), 3));
    sideTicks.add(new THREE.Line(g, sideMat));
  }
  if (side.left && msg.left_m != null) sideTick(+msg.left_m);
  if (side.right && msg.right_m != null) sideTick(-msg.right_m);
}

function clearObstacle() {
  obstacleGroup.visible = false;
  nearMarker.visible = false;
  nearPulse = false;
  emptyGroup(profileLines);
  emptyGroup(sideTicks);
}

window.LidarOverlay = { updateObstacle, clear: clearObstacle };

function setBadge(text, live) {
  if (!badge) return;
  badge.textContent = text;
  badge.classList.toggle("live", !!live);
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/lidar`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => setBadge("3D ●", true);
  ws.onmessage = (e) => {
    if (typeof e.data === "string") {
      try {
        const m = JSON.parse(e.data);
        if (m.type === "lidar_meta") {
          mapView = (m.view === "map");
          setBadge(mapView ? "MAP ●" : "LIVE ●", true);
          const info = document.getElementById("infoView");
          if (info) info.textContent = mapView ? "map" : "live";
          // Re-frame when switching between live and a (much larger) map view.
          didAutoFit = false;
        }
      } catch (_) {}
      return;
    }
    updateCloud(e.data);
  };
  ws.onclose = () => { setBadge("○ reconnecting…", false); setTimeout(connect, 1000); };
  ws.onerror = () => ws.close();
}
connect();

// Start the render loop LAST: animate() references the obstacle-overlay objects
// (nearPulse / nearMarker) defined above. Calling it before they are initialised
// throws a TDZ ReferenceError that aborts the whole module -> blank LiDAR feed.
animate();
