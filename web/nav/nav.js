// Nav tab — top-down live-position view for the autonomous circle-patrol
// (g1_nav track: circle_patrol.py / patrol_supervisor.py / sim_runner.py).
//
// ES module (page importmap: three + OrbitControls) -- same setup as
// web/obstacle3d.js. Frame: X forward, Y left, Z up; 1 scene unit = 1 metre --
// same convention as obstacle3d.js/lidar.js, so this tab reads consistently
// with the rest of the dashboard.
//
// Data: controller.js's WS dispatch calls window.Nav.update(msg) for every
// {"type":"nav_state", source:"real"|"sim", x, y, yaw_deg, path:{cx,cy,r},
// phase, persons:[{x,y,dist,id}], ts} frame (see the shared contract in
// scripts/robot_web_controller.py once that side lands -- this module works
// standalone against the documented shape even before the server emits it,
// it just won't receive real frames yet).
//
// The Real/Simulation toggle buttons live as static markup in index.html; we
// just wire their clicks to window.send({type:"nav_mode", mode}) -- the SAME
// send() helper (and owner-gated control-lock plumbing) every other mutating
// control on this dashboard already uses (controller.js is a classic script,
// so its top-level `function send` is a plain window global).
//
// SAFETY: the "SIMULATED" watermark / dashed border is driven ONLY by the
// confirmed msg.source of the last-received nav_state frame -- never by which
// toggle button was clicked. Clicking just requests a mode switch; until the
// server confirms it by tagging its next frame, the view keeps showing
// whatever source it last actually saw. An operator must never be shown "real"
// styling while actually looking at a sim run (or vice versa).
//
// Exposed as window.Nav (matches window.Obstacle3D's global-attach convention).

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

(function () {
  "use strict";

  const MAX_PERSONS = 16;
  const COL = {
    grid: 0x232c3e, robot: 0x5c8df2, circle: 0x35507d,
    person: 0xffd166,
  };
  const PHASE_LABEL = {
    circling: "Circling", pausing: "Pausing", facing: "Facing person",
    waving: "Waving", waiting: "Waiting for interaction", resuming: "Resuming",
    fault: "FAULT", manual_override: "Manual override",
    paused_odom_degraded: "Odometry degraded — paused",
  };
  const PHASE_WARN = new Set(["fault", "manual_override", "paused_odom_degraded"]);

  // Exhibition-mode "activity" sub-status (additive, may not exist yet -- see
  // file header / the plan's Track A Extension). Bare (no ":" suffix) values
  // get a fixed label; "kind:rest" values are parsed generically so any
  // gesture name or dance fsm_id renders without code changes here.
  const ACTIVITY_LABEL = { roam: "🚶 roaming", idle: "⏸ idle" };
  const ACTIVITY_PREFIX = {
    gesture: function (rest) { return "👋 gesture: " + rest; },
    dance: function (rest) { return "🕺 dancing (FSM " + rest + ")"; },
  };
  function formatActivity(activity) {
    if (typeof activity !== "string" || !activity) return null;
    const i = activity.indexOf(":");
    if (i === -1) return ACTIVITY_LABEL[activity] || null;
    const kind = activity.slice(0, i), rest = activity.slice(i + 1);
    const fmt = ACTIVITY_PREFIX[kind];
    return (fmt && rest) ? fmt(rest) : null;
  }

  let scene, camera, renderer, controls, mount, booted = false;
  let robotGroup = null;
  let fieldGroup = null, fieldR = null, fieldCx = null, fieldCy = null;
  let personMeshes = [];
  let lastMsg = null;

  // DOM chrome lives as static markup in index.html; we just look it up + wire it.
  let navWindowEl = null, simWatermarkEl = null, badgeEl = null, phaseChipEl = null;
  let activityChipEl = null;
  let btnReal = null, btnSim = null;

  function el(id) { return document.getElementById(id); }
  function finite(v, d) { return (typeof v === "number" && isFinite(v)) ? v : d; }

  function groundRing(cx, cy, radius, color, z) {
    const segs = 96, pts = [];
    for (let i = 0; i <= segs; i++) {
      const a = (i / segs) * Math.PI * 2;
      pts.push(cx + Math.cos(a) * radius, cy + Math.sin(a) * radius, z || 0.02);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color }));
  }

  // Small canvas-texture text label (person track id), billboarded via THREE.Sprite.
  function textSprite(text, color) {
    const cv = document.createElement("canvas");
    cv.width = 128; cv.height = 64;
    const ctx = cv.getContext("2d");
    ctx.font = "700 42px system-ui, sans-serif";
    ctx.fillStyle = color || "#E9EDF6";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(text, 64, 34);
    const tex = new THREE.CanvasTexture(cv);
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
    const spr = new THREE.Sprite(mat);
    spr.scale.set(0.55, 0.28, 1);
    return spr;
  }

  // The static patrol circle + ground grid, centred/sized from path.cx/cy/r.
  // Only rebuilt when the path actually changes (it's fixed for the life of a patrol).
  function ensureField(cx, cy, r) {
    cx = finite(cx, 0); cy = finite(cy, 0);
    const rr = finite(r, 3.0) > 0.1 ? finite(r, 3.0) : 3.0;
    if (fieldGroup && Math.abs(fieldR - rr) < 0.01 &&
        Math.abs(fieldCx - cx) < 0.01 && Math.abs(fieldCy - cy) < 0.01) return;
    if (fieldGroup) { scene.remove(fieldGroup); }
    fieldGroup = new THREE.Group();
    const size = Math.max(8, rr * 2 + 4);
    const grid = new THREE.GridHelper(size, Math.max(4, Math.round(size)), COL.grid, COL.grid);
    grid.rotation.x = Math.PI / 2;   // GridHelper lies in XZ by default -> rotate into our XY ground plane
    grid.material.opacity = 0.3; grid.material.transparent = true;
    fieldGroup.add(grid);
    fieldGroup.add(groundRing(0, 0, rr, COL.circle, 0.015));   // the patrol circle itself
    fieldGroup.add(groundRing(0, 0, 0.12, COL.circle, 0.02));  // small centre marker
    fieldGroup.position.set(cx, cy, 0);
    scene.add(fieldGroup);
    fieldR = rr; fieldCx = cx; fieldCy = cy;
  }

  function makePersonMarker() {
    const group = new THREE.Group();
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 16, 12),
      new THREE.MeshBasicMaterial({ color: COL.person }));
    dot.position.z = 0.3;
    group.add(dot);
    const stick = new THREE.Mesh(
      new THREE.CylinderGeometry(0.02, 0.02, 0.3, 6),
      new THREE.MeshBasicMaterial({ color: COL.person, transparent: true, opacity: 0.55 }));
    stick.rotation.x = Math.PI / 2; stick.position.z = 0.15;
    group.add(stick);
    group.add(groundRing(0, 0, 0.22, COL.person, 0.015));
    group.visible = false;
    return { group, label: null, lastLabel: null };
  }

  function updateLabel(pm, text) {
    if (pm.label) {
      pm.group.remove(pm.label);
      pm.label.material.map.dispose();
      pm.label.material.dispose();
      pm.label = null;
    }
    const spr = textSprite("#" + text, "#ffd166");
    spr.position.set(0, 0, 0.85);
    pm.group.add(spr);
    pm.label = spr;
  }

  function wireToggle() {
    btnReal = el("navModeReal");
    btnSim = el("navModeSim");
    if (btnReal) btnReal.addEventListener("click", () => {
      window.send && window.send({ type: "nav_mode", mode: "real" });
    });
    if (btnSim) btnSim.addEventListener("click", () => {
      window.send && window.send({ type: "nav_mode", mode: "sim" });
    });
  }

  function build() {
    if (booted) return true;
    mount = el("navView");
    if (!mount) return false;
    navWindowEl = el("navWindow");
    simWatermarkEl = el("navSimWatermark");
    badgeEl = el("navBadge");
    phaseChipEl = el("navPhaseChip");
    activityChipEl = el("navActivityChip");
    wireToggle();

    const w = Math.max(120, mount.clientWidth || 240);
    const h = Math.max(120, mount.clientHeight || 240);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c1019);

    // Slightly-angled top-down "tactical map" view (not a dead-straight bird's
    // eye — a little angle keeps height/depth of the robot marker readable).
    camera = new THREE.PerspectiveCamera(45, w / h, 0.05, 200);
    camera.up.set(0, 0, 1);
    camera.position.set(0, -6.5, 8.5);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(w, h, false);
    mount.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enablePan = true;
    controls.minDistance = 3;
    controls.maxDistance = 30;
    controls.maxPolarAngle = Math.PI * 0.48;   // keep it top-down-ish; never dip under the floor
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.95));
    const dl = new THREE.DirectionalLight(0xffffff, 0.5); dl.position.set(0, 0, 6); scene.add(dl);

    ensureField(0, 0, 3.0);

    // --- robot marker: body + heading wedge (arrow), same look/colour as obstacle3d.js
    //     so the robot reads as "the same robot" across tabs. ---
    robotGroup = new THREE.Group();
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.5, 16),
      new THREE.MeshBasicMaterial({ color: COL.robot }));
    body.rotation.x = Math.PI / 2; body.position.z = 0.25;
    robotGroup.add(body);
    const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.13, 0.32, 16),
      new THREE.MeshBasicMaterial({ color: COL.robot }));
    arrow.rotation.z = -Math.PI / 2; arrow.position.set(0.42, 0, 0.25);   // points +X (forward, yaw=0)
    robotGroup.add(arrow);
    scene.add(robotGroup);

    // --- pooled person markers (dot + stick + ring), hidden until data arrives ---
    for (let i = 0; i < MAX_PERSONS; i++) {
      const pm = makePersonMarker();
      scene.add(pm.group);
      personMeshes.push(pm);
    }

    booted = true;
    if (lastMsg) refresh(lastMsg);
    animate();
    window.addEventListener("resize", onResize);
    return true;
  }

  function onResize() {
    if (!booted || !mount) return;
    const w = Math.max(120, mount.clientWidth || 240);
    const h = Math.max(120, mount.clientHeight || 240);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }

  function updateChrome(msg) {
    const source = msg.source === "sim" ? "sim" : (msg.source === "real" ? "real" : null);
    if (badgeEl) {
      badgeEl.textContent = source ? source.toUpperCase() : "…";
      badgeEl.classList.toggle("live", source === "real");
    }
    const isSim = source === "sim";
    if (navWindowEl) navWindowEl.classList.toggle("sim", isSim);
    if (simWatermarkEl) simWatermarkEl.hidden = !isSim;
    // Toggle-button highlight reflects the CONFIRMED source only (see file header) --
    // never optimistically set from the click itself.
    if (btnReal) btnReal.classList.toggle("current", source === "real");
    if (btnSim) btnSim.classList.toggle("current", source === "sim");

    const phase = msg.phase;
    if (phaseChipEl) {
      phaseChipEl.textContent = PHASE_LABEL[phase] || (phase ? String(phase) : "phase —");
      phaseChipEl.classList.toggle("warn", PHASE_WARN.has(phase));
    }

    // Sub-status line: only shown once we have a recognized msg.activity --
    // null/absent/unrecognized (server piece not landed yet, or Real mode
    // pre-conductor) must render nothing, never an empty/broken-looking chip.
    if (activityChipEl) {
      const label = formatActivity(msg.activity);
      activityChipEl.textContent = label || "";
      activityChipEl.hidden = !label;
    }
  }

  function refresh(msg) {
    if (!booted || !msg) return;
    const path = msg.path || {};
    ensureField(path.cx, path.cy, path.r);

    const x = finite(msg.x, 0), y = finite(msg.y, 0);
    const yawRad = (finite(msg.yaw_deg, 0) * Math.PI) / 180;
    robotGroup.position.set(x, y, 0);
    robotGroup.rotation.z = yawRad;

    const persons = Array.isArray(msg.persons) ? msg.persons : [];
    for (let i = 0; i < personMeshes.length; i++) {
      const pm = personMeshes[i];
      if (i < persons.length) {
        const p = persons[i] || {};
        pm.group.position.set(finite(p.x, 0), finite(p.y, 0), 0);
        pm.group.visible = true;
        const idTxt = (p.id !== undefined && p.id !== null) ? String(p.id) : "?";
        if (pm.lastLabel !== idTxt) { updateLabel(pm, idTxt); pm.lastLabel = idTxt; }
      } else {
        pm.group.visible = false;
      }
    }

    updateChrome(msg);
  }

  function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    renderer.render(scene, camera);
  }

  window.Nav = {
    update: function (msg) {
      try {
        lastMsg = msg || null;
        if (!booted && !build()) return;
        refresh(lastMsg);
      } catch (e) { /* defensive: never break the WS loop */ }
    },
    resize: function () { try { onResize(); } catch (e) {} },
    onActivate: function () { try { onResize(); } catch (e) {} },
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();
