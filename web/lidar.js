// G1 3D LiDAR / map viewer — point cloud coloured by height, with a ground grid,
// XYZ axes, OrbitControls, and analysis tools (top-down, fit, point size, colour).
// World frame is Z-up; data over /ws/lidar: uint32 N, uint8 stride, 3 pad, N*3 f32.

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

const container = document.getElementById("lidarView");
const badge = document.getElementById("lidarBadge");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e0f15);

const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 500);
camera.up.set(0, 0, 1);                 // Z is up
camera.position.set(-4, -4, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(2.0, 0, 0.8);
controls.maxPolarAngle = Math.PI / 2;   // never orbit below the floor (Z-up frame)

const grid = new THREE.GridHelper(20, 40, 0x2a2e3d, 0x1b1e29);
grid.rotation.x = Math.PI / 2;          // make it the XY (ground) plane
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
const material = new THREE.PointsMaterial({ size: pointSize, vertexColors: true, sizeAttenuation: true });
const points = new THREE.Points(geometry, material);
scene.add(points);

// Cloud extent (updated each frame) for fit / top-down framing.
let center = new THREE.Vector3(2, 0, 0.8);
let radius = 4;
let didAutoFit = false;
let colorByHeight = true;

// Dynamic height range for the colour ramp (smoothed).
let zLo = 0, zHi = 2.2;

function turbo(t) {
  t = Math.min(1, Math.max(0, t));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(2 * t - 1.5) * 1.5));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(2 * t - 1.0) * 1.5));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(2 * t - 0.5) * 1.5));
  return [r, g, b];
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

function topView() {
  controls.target.copy(center);
  camera.up.set(1, 0, 0);                 // robot-forward points up on screen
  camera.position.set(center.x, center.y, center.z + radius * 2.4);
  camera.updateProjectionMatrix();
  controls.update();
}

function view3D() {
  camera.up.set(0, 0, 1);
  fitView();
}

function setPointSize(mult) {
  pointSize = Math.min(0.3, Math.max(0.01, pointSize * mult));
  material.size = pointSize;
}

function bind(id, fn) {
  const el = document.getElementById(id);
  if (el) el.addEventListener("click", fn);
}
bind("viewHome", view3D);   // view3D() sets camera.up=(0,0,1) then fitView()
bind("view3d", view3D);
bind("viewTop", topView);
bind("viewFit", fitView);
bind("ptMinus", () => setPointSize(0.7));
bind("ptPlus", () => setPointSize(1.4));
bind("viewColor", () => { colorByHeight = !colorByHeight; });
bind("viewGrid", () => { grid.visible = !grid.visible; });

// --- on-screen navigation (laptop-friendly orbit/tilt/zoom) --------------

const _off = new THREE.Vector3();
function orbit(daz, del) {
  _off.copy(camera.position).sub(controls.target);
  const r = _off.length();
  let theta = Math.atan2(_off.y, _off.x);                  // azimuth around Z
  let phi = Math.acos(Math.min(1, Math.max(-1, _off.z / r)));// polar from +Z
  theta += daz;
  phi = Math.min(Math.PI - 0.05, Math.max(0.05, phi + del));
  _off.set(r * Math.sin(phi) * Math.cos(theta),
           r * Math.sin(phi) * Math.sin(theta),
           r * Math.cos(phi));
  camera.position.copy(controls.target).add(_off);
  camera.up.set(0, 0, 1);
  camera.lookAt(controls.target);
  controls.update();
}
function dolly(factor) {
  _off.copy(camera.position).sub(controls.target);
  _off.setLength(Math.min(200, Math.max(0.4, _off.length() * factor)));
  camera.position.copy(controls.target).add(_off);
  controls.update();
}

// Press-and-hold: fire once on press, then repeat while held.
function holdRepeat(id, fn) {
  const el = document.getElementById(id);
  if (!el) return;
  let timer = null;
  const start = (e) => { e.preventDefault(); fn(); timer = setInterval(fn, 60); };
  const stop = () => { if (timer) { clearInterval(timer); timer = null; } };
  el.addEventListener("pointerdown", start);
  el.addEventListener("pointerup", stop);
  el.addEventListener("pointerleave", stop);
  el.addEventListener("pointercancel", stop);
}
const AZ = 0.05, EL = 0.05;
holdRepeat("navLeft",  () => orbit(+AZ, 0));
holdRepeat("navRight", () => orbit(-AZ, 0));
holdRepeat("navUp",    () => orbit(0, -EL));
holdRepeat("navDown",  () => orbit(0, +EL));
holdRepeat("navZoomIn",  () => dolly(0.94));
holdRepeat("navZoomOut", () => dolly(1.06));

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
animate();

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
          setBadge(m.view === "map" ? "MAP ●" : "LIVE ●", true);
          const info = document.getElementById("infoView");
          if (info) info.textContent = m.view === "map" ? "map" : "live";
          // Re-fit when switching between live and a (much larger) map view.
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
