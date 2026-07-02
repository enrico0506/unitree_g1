// G1 Web Controller — frontend logic

// Velocity caps -- defaults; overwritten by the server's "config" message
// (sourced from config/robot.yaml) as soon as the WebSocket connects.
let MAX_VX = 1.5;
let MAX_VY = 0.7;   // match config/robot.yaml max_vy so the ellipse readout is right pre-config
let MAX_VYAW = 2.0;
let SLOW_SCALE = 0.4;

const SEND_HZ = 30;
const SEND_INTERVAL = 1000 / SEND_HZ;

// --- Verified FSM IDs (must match server) ---
const FSM_ZERO_TORQUE  = 0;
const FSM_DAMPING      = 1;
const FSM_READY_STAND  = 4;
const FSM_MAIN_CONTROL = 802;

const FSM_NAMES = {
  [FSM_ZERO_TORQUE]:  "zero_torque",
  [FSM_DAMPING]:      "damping",
  [FSM_READY_STAND]:  "ready_stand",
  [FSM_MAIN_CONTROL]: "main_control",
  812: "climb",   // whole-body combo targets (display only)
  503: "dance",
};

const UI_TO_FSM = {
  zero_torque: FSM_ZERO_TORQUE,
  damp:        FSM_DAMPING,
  stand:       FSM_READY_STAND,
  walk:        FSM_MAIN_CONTROL,
};

const MODE_HINTS = {
  zero_torque: 'Mode "zero torque" (FSM 0): motors OFF. Robot collapses unless on gantry.',
  damp:        'Mode "damp" (FSM 1): joints relaxed. Robot will not respond to movement.',
  stand:       'Mode "stand" (FSM 4 = ready_stand): robot stands rigidly. Press "Walk" to enable balanced locomotion.',
  walk:        'Mode "walk" (FSM 802 = main_control): hold WASD / arrows. Hold SHIFT for slow mode.',
  dance:       'Dancing (FSM 503)… it returns to Walk on its own. To stop early press Stand (then Walk) — exit never damps.',
  climb:       'Climb mode (FSM 812)… press Stand then Walk to exit — exit never damps.',
};

// --- Connection ---
const wsUrl = `ws://${location.host}/ws`;
let ws = null;
let connected = false;

// --- Single-controller lock (shared with the phone page over the same /ws) ---
// Identity is one id per page LOAD (survives reconnects; not persisted). The laptop
// never auto-grabs the lock -- the server assigns it on connect only if it's free,
// so a lone laptop drives exactly as before. When another device takes control, our
// outbound commands are dropped (see send()) and the control chip explains why.
const CLIENT_ID = (window.crypto && crypto.randomUUID)
  ? crypto.randomUUID()
  : "laptop-" + Math.random().toString(36).slice(2);
const CLIENT_LABEL = "Laptop";
const ALWAYS_ALLOWED_MSGS = new Set(["hello", "take_control"]);
let hasControl = false;
let ownerLabel = null;

const statusEl = document.getElementById("status");
const statusText = document.getElementById("statusText");

function connect() {
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    connected = true;
    // Announce identity so the server can arbitrate the control lock (take-if-free).
    send({ type: "hello", client_id: CLIENT_ID, label: CLIENT_LABEL });
    statusEl.classList.add("connected");
    statusText.textContent = "Connected";
    setInfo("infoConn", "connected");
    logEvent("connected to robot", "ok");
    if (window.Obstacle) window.Obstacle.init(send);
    if (window.StepMode) window.StepMode.init(send);
    if (window.DriveMode) window.DriveMode.init();
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "fsm_state") updateFsmState(msg);
      else if (msg.type === "config") applyConfig(msg);
      else if (msg.type === "map_status") updateMapStatus(msg);
      else if (msg.type === "telemetry") {
        updateTelemetry(msg);
        window.StepMode && window.StepMode.handle(msg);   // step_mode / step_estimated chip
      }
      else if (msg.type === "step_mode") window.StepMode && window.StepMode.handle(msg);
      else if (msg.type === "obstacle") {
        window.Obstacle && window.Obstacle.handle(msg);
        window.LidarOverlay && window.LidarOverlay.updateObstacle(msg);
        window.Obstacle3D && window.Obstacle3D.update(msg);
      }
      else if (msg.type === "control") updateControl(msg);
    } catch (e) { /* ignore */ }
  };

  ws.onclose = () => {
    connected = false;
    // Zero held keys + recentre the thumbsticks so a still-held input can't
    // re-assert a stale velocity the instant the socket reconnects.
    clearAllHeld();
    hasControl = false; ownerLabel = null; renderControlChip();
    statusEl.classList.remove("connected");
    statusText.textContent = "Disconnected — retrying...";
    setInfo("infoConn", "disconnected");
    logEvent("disconnected — retrying…", "warn");
    setTimeout(connect, 1000);
  };

  ws.onerror = () => { ws.close(); };
}

function send(obj) {
  if (!connected || !ws || ws.readyState !== WebSocket.OPEN) return;
  // Read-only device: while another device holds the control lock, drop every
  // mutating command (move/mode/obstacle/step/map/cmd). hello + take_control pass.
  if (obj && obj.type && !ALWAYS_ALLOWED_MSGS.has(obj.type) && !hasControl) return;
  ws.send(JSON.stringify(obj));
}

connect();

// --- Mode state ---
let uiMode = "damp";
let fsmId = null;
let transitioning = false;
let slowMode = false;   // Shift = precise slow control (~40% speed)

function speedScale() { return slowMode ? SLOW_SCALE : 1.0; }

function applyConfig(msg) {
  if (typeof msg.max_vx === "number") MAX_VX = msg.max_vx;
  if (typeof msg.max_vy === "number") MAX_VY = msg.max_vy;
  if (typeof msg.max_vyaw === "number") MAX_VYAW = msg.max_vyaw;
  if (typeof msg.slow_scale === "number") SLOW_SCALE = msg.slow_scale;
  setInfo("infoSpeeds", `vx≤${MAX_VX} · vy≤${MAX_VY} · vyaw≤${MAX_VYAW} m/s`);

  // Whole-body combo buttons (dance/climb) only work once their FSM id has been
  // captured into config/mapping.yaml; enable/disable them from the server flag.
  if (msg.mode_combos && typeof msg.mode_combos === "object") {
    for (const [name, ready] of Object.entries(msg.mode_combos)) {
      document.querySelectorAll(`button[data-combo="${name}"]`).forEach(btn => {
        btn.disabled = !ready;
        btn.title = ready
          ? `${name} — remote combo`
          : `${name}: FSM id not captured yet — run scripts/capture_combo_fsm.py`;
      });
    }
  }

  // Forward the initial obstacle-guard flags to the obstacle panel.
  if (window.Obstacle && msg.obstacle) {
    window.Obstacle.handle(Object.assign({ type: "obstacle" }, msg.obstacle));
  }

  // Forward the initial step-size selector state ('Normal' on load by default).
  if (window.StepMode && msg.step) {
    window.StepMode.handle(Object.assign({ type: "step_mode" }, msg.step));
  }
}

function fsmDisplay(id) {
  if (id === null || id === undefined) return "—";
  const name = FSM_NAMES[id] || "unknown";
  return `${id} (${name})`;
}

function updateFsmState(msg) {
  const prevUiMode = uiMode;          // for rising-edge detection into walk
  uiMode = msg.ui_mode;
  fsmId  = msg.fsm_id;
  transitioning = !!msg.transitioning;

  // Drive the console's accent temperature from the robot's actual mode: the UI
  // stays cool/inert until "walk" (live), where it warms. Pure data attribute —
  // all theming lives in CSS (body[data-mode=...]).
  document.body.dataset.mode = uiMode || "";
  document.body.dataset.transitioning = transitioning ? "1" : "";

  document.querySelectorAll(".mode-btn").forEach(btn => {
    const isCurrent = btn.dataset.mode === uiMode;
    btn.classList.toggle("current", isCurrent);
    btn.classList.toggle("transitioning", isCurrent && transitioning);
  });

  const hintEl = document.getElementById("modeHint");
  if (hintEl) {
    if (transitioning) {
      hintEl.textContent =
        `Transitioning to "${uiMode}"... robot at FSM ${fsmDisplay(fsmId)}`;
    } else {
      hintEl.textContent = MODE_HINTS[uiMode] || "";
    }
  }

  setInfo("infoMode", uiMode);   // transition state is shown by .transit-flag (CSS)

  const mismatchEl = document.getElementById("fsmMismatch");
  if (mismatchEl) {
    const expected = UI_TO_FSM[uiMode];
    if (!transitioning && fsmId !== null && expected !== undefined
        && fsmId !== expected) {
      mismatchEl.textContent =
        `⚠ commanded "${uiMode}" → expected FSM ${expected}, robot reports ${fsmId}`;
      mismatchEl.classList.add("warn");
    } else {
      mismatchEl.textContent = "";
      mismatchEl.classList.remove("warn");
    }
  }

  const moveDisabled = uiMode !== "walk" || transitioning;
  document.querySelectorAll("button[data-hold]").forEach(btn => {
    btn.classList.toggle("disabled", moveDisabled);
  });

  if (moveDisabled) {
    clearAllHeld();
  } else if (prevUiMode !== "walk" && window.DriveMode && window.DriveMode.reset) {
    // Rising edge INTO live walk (even if the server skipped a transitioning
    // message): zero any thumbstick deflected during the change so the first
    // 30 Hz tick can't fire a stale full-speed command.
    window.DriveMode.reset();
  }
}

function requestMode(name) {
  if (name === "zero_torque") {
    const ok = confirm(
      "ZERO TORQUE will cut motor power entirely. " +
      "The robot will collapse if it is not on the gantry.\n\n" +
      "Continue?"
    );
    if (!ok) return;
  }
  logEvent(`mode → ${name}`);
  send({ type: "mode", name });
}

// =========================================================================
// SINGLE-CONTROLLER LOCK (multi-device arbitration; shared with the phone page)
// =========================================================================

function updateControl(msg) {
  const ownerId = msg.owner_id || null;
  const prev = hasControl;
  hasControl = (ownerId !== null && ownerId === CLIENT_ID);
  ownerLabel = msg.owner_label || null;
  renderControlChip();
  // The laptop is the DEFAULT driver: whenever the lock is FREE (nobody driving),
  // reclaim it automatically so a lone laptop -- or the laptop after a phone leaves
  // -- keeps working exactly as before. It never auto-steals from another device
  // (acts only when ownerId is null); a phone must always take control explicitly.
  if (ownerId === null) {
    send({ type: "take_control", client_id: CLIENT_ID });
    return;
  }
  if (prev && !hasControl) {
    // Just lost control -> drop held keys / recentre sticks so nothing is latched.
    clearAllHeld();
    logEvent(ownerLabel ? `control taken by ${ownerLabel}` : "control taken by another device", "warn");
  } else if (!prev && hasControl) {
    logEvent("you have control", "ok");
  }
}

function renderControlChip() {
  const chip = document.getElementById("controlChip");
  if (!chip) return;
  chip.hidden = false;
  chip.classList.toggle("mine", hasControl);
  chip.classList.toggle("theirs", !hasControl);
  chip.textContent = hasControl
    ? "● You have control"
    : (ownerLabel ? `○ ${ownerLabel} has control · Take back` : "○ Take control");
}

const _controlChip = document.getElementById("controlChip");
if (_controlChip) {
  _controlChip.addEventListener("click", () => {
    // Clicking when you already hold control is a no-op (avoids a needless velocity reset).
    if (!hasControl) send({ type: "take_control", client_id: CLIENT_ID });
  });
}

// =========================================================================
// HELD INPUT TRACKING (multi-source: kbd + touch independently)
// =========================================================================

const held = {
  w:    new Set(),
  s:    new Set(),
  a:    new Set(),
  d:    new Set(),
  rotL: new Set(),
  rotR: new Set(),
};

function isHeld(action) {
  return held[action] && held[action].size > 0;
}

function setHeld(action, source, on) {
  if (!held[action]) return;
  if (on) held[action].add(source);
  else    held[action].delete(source);
  refreshButtonVisual(action);
}

function clearAllHeld() {
  for (const action in held) held[action].clear();
  document.querySelectorAll("button[data-hold]").forEach(btn => {
    btn.classList.remove("active");
  });
  // Also recentre the analog thumbsticks so a held stick can't latch velocity
  // through a Stop / mode change / disconnect.
  if (window.DriveMode && window.DriveMode.reset) window.DriveMode.reset();
}

function refreshButtonVisual(action) {
  document.querySelectorAll(`button[data-hold="${action}"]`).forEach(btn => {
    btn.classList.toggle("active", isHeld(action));
  });
}

function computeVelocity() {
  if (uiMode !== "walk" || transitioning) return { vx: 0, vy: 0, vyaw: 0 };
  const scale = speedScale();
  let vx = 0, vy = 0, vyaw = 0;
  if (isHeld("w"))    vx += MAX_VX * scale;
  if (isHeld("s"))    vx -= MAX_VX * scale;
  if (isHeld("a"))    vy += MAX_VY * scale;
  if (isHeld("d"))    vy -= MAX_VY * scale;
  if (isHeld("rotL")) vyaw += MAX_VYAW * scale;
  if (isHeld("rotR")) vyaw -= MAX_VYAW * scale;
  // Analog thumbsticks (mobile mode): each axis is a fraction in [-1, 1] of its cap.
  // axes() returns null in laptop mode, so the keyboard/D-pad stay authoritative.
  const ax = window.DriveMode && window.DriveMode.axes && window.DriveMode.axes();
  if (ax) {
    vx   += MAX_VX   * scale * ax.fwd;
    vy   += MAX_VY   * scale * ax.strafe;
    vyaw += MAX_VYAW * scale * ax.turn;
  }
  // Combining a stick with a key could overshoot a single cap -> clamp each axis
  // back to its limit before normalizing (the server clamps too; this keeps the
  // on-screen readout honest).
  vx   = Math.max(-MAX_VX,   Math.min(MAX_VX,   vx));
  vy   = Math.max(-MAX_VY,   Math.min(MAX_VY,   vy));
  vyaw = Math.max(-MAX_VYAW, Math.min(MAX_VYAW, vyaw));
  // Normalize a diagonal onto the speed ELLIPSE so W+A isn't faster than a straight
  // axis (the server re-normalizes too; doing it here keeps the on-screen readout honest).
  if (MAX_VX > 0 && MAX_VY > 0) {
    const m = (vx / MAX_VX) ** 2 + (vy / MAX_VY) ** 2;
    if (m > 1) { const s = 1 / Math.sqrt(m); vx *= s; vy *= s; }
  }
  return { vx, vy, vyaw };
}

// =========================================================================
// SEND LOOP -- always send in walk mode, even at zero velocity
// =========================================================================

const vxEl   = document.getElementById("vxVal");
const vyEl   = document.getElementById("vyVal");
const vyawEl = document.getElementById("vyawVal");

function fmt(v) {
  return (v >= 0 ? "+" : "") + v.toFixed(2);
}

setInterval(() => {
  const { vx, vy, vyaw } = computeVelocity();
  if (vxEl)   vxEl.textContent   = fmt(vx);
  if (vyEl)   vyEl.textContent   = fmt(vy);
  if (vyawEl) vyawEl.textContent = fmt(vyaw);

  if (uiMode === "walk" && !transitioning) {
    send({ type: "move", vx, vy, vyaw });
  }
}, SEND_INTERVAL);

// =========================================================================
// KEYBOARD -- layout-independent via event.code
// =========================================================================

function codeToAction(code) {
  switch (code) {
    case "KeyW":       return "w";
    case "KeyS":       return "s";
    case "KeyA":       return "a";
    case "KeyD":       return "d";
    case "ArrowLeft":  return "rotL";
    case "ArrowRight": return "rotR";
  }
  return null;
}

document.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;

  if (["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Space"].includes(e.code)) {
    e.preventDefault();
  }

  if (e.code === "ShiftLeft" || e.code === "ShiftRight") {
    slowMode = true;
    return;
  }

  const action = codeToAction(e.code);
  if (action) {
    setHeld(action, "kbd", true);
    return;
  }

  if (e.repeat) return;

  switch (e.code) {
    case "Space":  clearAllHeld(); send({ type: "stop" }); break;
    case "Digit1": requestMode("zero_torque"); break;
    case "Digit2": requestMode("damp"); break;
    case "Digit3": requestMode("stand"); break;
    case "Digit4": requestMode("walk"); break;
    case "Digit9": send({ type: "cmd", name: "high_wave" }); break;  // working "Wave" (old "wave" did nothing)
    case "Digit0": send({ type: "cmd", name: "shake" });  break;
  }
});

document.addEventListener("keyup", (e) => {
  if (e.code === "ShiftLeft" || e.code === "ShiftRight") {
    slowMode = false;
    return;
  }
  const action = codeToAction(e.code);
  if (action) setHeld(action, "kbd", false);
});

// =========================================================================
// MODE BUTTONS
// =========================================================================

document.querySelectorAll("button[data-mode]").forEach(btn => {
  btn.addEventListener("click", () => requestMode(btn.dataset.mode));
});

// =========================================================================
// TOUCH / MOUSE BUTTONS
// =========================================================================

document.querySelectorAll("button[data-hold]").forEach(btn => {
  const action = btn.dataset.hold;

  const press = (e) => {
    e.preventDefault();
    if (uiMode !== "walk" || transitioning) return;
    setHeld(action, "touch", true);
    if (e.pointerId !== undefined && btn.setPointerCapture) {
      try { btn.setPointerCapture(e.pointerId); } catch (_) {}
    }
  };
  const release = (e) => {
    e.preventDefault();
    setHeld(action, "touch", false);
  };

  if (window.PointerEvent) {
    btn.addEventListener("pointerdown",   press);
    btn.addEventListener("pointerup",     release);
    btn.addEventListener("pointercancel", release);
    btn.addEventListener("pointerleave",  release);
  } else {
    btn.addEventListener("mousedown",   press);
    btn.addEventListener("mouseup",     release);
    btn.addEventListener("mouseleave",  release);
    btn.addEventListener("touchstart",  press,   { passive: false });
    btn.addEventListener("touchend",    release);
    btn.addEventListener("touchcancel", release);
  }
});

// =========================================================================
// ONE-SHOT GESTURE BUTTONS
// =========================================================================

document.querySelectorAll("button[data-cmd]").forEach(btn => {
  btn.addEventListener("click", () => {
    const name = btn.dataset.cmd;
    // Arm gestures (data-arm) only make sense once the robot is up: block them
    // in zero_torque / damp so a press isn't a confusing no-op.
    if (btn.dataset.arm && uiMode !== "stand" && uiMode !== "walk") {
      logEvent(`"${name}" — Stand or Walk first`, "warn");
      return;
    }
    // Whole-body behaviors (dance/climb) move the entire robot — confirm first.
    if (btn.dataset.combo) {
      const ok = confirm(
        `This triggers the robot's ${name.toUpperCase()} behavior — the whole ` +
        `body will move. Ensure clear space and keep the e-stop in reach.\n\n` +
        `Press Stand or Walk afterwards to exit.\n\nContinue?`
      );
      if (!ok) return;
    }
    send({ type: "cmd", name });
    logEvent(`gesture → ${name}`);
    // hands_up is a toggle; reflect raise/lower state on the button.
    if (btn.dataset.toggle) btn.classList.toggle("active");
    // Releasing the arms clears any latched toggle (e.g. Hands Up).
    if (name === "release_arm") {
      document.querySelectorAll("button[data-cmd][data-toggle]")
        .forEach(b => b.classList.remove("active"));
    }
  });
});

// Gesture dropdown (header): choosing an item runs it immediately by clicking the
// matching hidden data-cmd button -- so the arm-gating / Hands-Up toggle / logging
// above is reused. We then reset the select back to the "Gesture…" placeholder so
// the SAME gesture can be picked again (a change event only fires on a new value).
const _gestureSelect = document.getElementById("gestureSelect");
if (_gestureSelect) {
  _gestureSelect.addEventListener("change", () => {
    const name = _gestureSelect.value;
    _gestureSelect.value = "";            // snap back to the placeholder label
    if (!name) return;
    const btn = document.querySelector(`.gesture-src button[data-cmd="${name}"]`);
    if (btn) btn.click();
  });
}

// =========================================================================
// TAP BUTTONS (stop)
// =========================================================================

document.querySelectorAll("button[data-tap]").forEach(btn => {
  btn.addEventListener("click", () => {
    clearAllHeld();
    send({ type: btn.dataset.tap });
  });
});

// =========================================================================
// SAFETY: stop only on actual page close, not on focus loss
// =========================================================================

window.addEventListener("pagehide", () => {
  clearAllHeld();
  send({ type: "stop" });
});

// =========================================================================
// MAP CONTROLS (build / save / load / discard)
// =========================================================================

let mapActive = false;

const mapToggle  = document.getElementById("mapToggle");
const mapSave    = document.getElementById("mapSave");
const mapName    = document.getElementById("mapName");
const mapDiscard = document.getElementById("mapDiscard");
const mapLoad    = document.getElementById("mapLoad");
const mapStatus  = document.getElementById("mapStatus");

function updateMapStatus(msg) {
  mapActive = !!msg.active;

  mapToggle.textContent = mapActive ? "■ Stop Mapping" : "● Start Mapping";
  mapToggle.classList.toggle("active", mapActive);
  mapToggle.dataset.map = mapActive ? "stop" : "start";

  const pts = (msg.points || 0).toLocaleString();
  if (mapActive) {
    mapStatus.textContent = `mapping… ${pts} pts · unsaved`;
  } else if (msg.points > 0) {
    mapStatus.textContent = msg.saved
      ? `map "${msg.loaded || "?"}" · ${pts} pts`
      : `stopped · ${pts} pts · UNSAVED`;
  } else {
    mapStatus.textContent = "live view";
  }
  mapStatus.classList.toggle("unsaved", !msg.saved && msg.points > 0);

  // Populate the Load dropdown (preserve the placeholder + current selection).
  const cur = mapLoad.value;
  const maps = msg.maps || [];
  mapLoad.innerHTML = '<option value="">Load map…</option>' +
    maps.map(m => `<option value="${m}">${m}</option>`).join("");
  if (maps.includes(cur)) mapLoad.value = cur;

  setInfo("infoMap", mapStatus.textContent);
}

if (mapToggle) {
  mapToggle.addEventListener("click", () => {
    const action = mapActive ? "stop" : "start";
    logEvent(`mapping → ${action}`);
    send({ type: "map", action });
  });
  mapSave.addEventListener("click", () => {
    const name = (mapName.value || "").trim();
    if (!name) { mapName.focus(); return; }
    logEvent(`map save "${name}"`, "ok");
    send({ type: "map", action: "save", name });
  });
  mapDiscard.addEventListener("click", () => {
    logEvent("map discarded");
    send({ type: "map", action: "clear" });
  });
  mapLoad.addEventListener("change", () => {
    if (mapLoad.value) { logEvent(`map load "${mapLoad.value}"`); send({ type: "map", action: "load", name: mapLoad.value }); }
  });
}

// =========================================================================
// TABS + INFO + EVENT LOG
// =========================================================================

function setInfo(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

const eventLog = document.getElementById("eventLog");
function logEvent(text, level) {
  if (!eventLog) return;
  const t = new Date().toLocaleTimeString();
  const line = document.createElement("div");
  line.className = "log-line" + (level ? " " + level : "");
  line.textContent = `${t}  ${text}`;
  eventLog.prepend(line);
  while (eventLog.childElementCount > 200) eventLog.lastElementChild.remove();
}
const logClearBtn = document.getElementById("logClear");
if (logClearBtn) logClearBtn.addEventListener("click", () => { eventLog.innerHTML = ""; });

function updateBattery(soc, volts) {
  const pill = document.getElementById("battery");
  const text = document.getElementById("batteryText");
  const fill = pill && pill.querySelector(".batt-fill");
  if (soc === null || soc === undefined) {     // BMS not reporting yet
    if (text) text.textContent = "—";
    if (fill) fill.style.width = "0%";
    if (pill) pill.className = "battery";
    setInfo("infoBattery", "n/a");
    return;
  }
  const pct = Math.max(0, Math.min(100, Math.round(soc)));
  if (text) text.textContent = pct + "%";
  if (fill) fill.style.width = pct + "%";
  const level = pct <= 15 ? "crit" : pct <= 35 ? "warn" : "ok";
  if (pill) pill.className = "battery " + level;
  setInfo("infoBattery", pct + "%" + (volts != null ? ` · ${volts.toFixed(1)} V` : ""));
}

function updateTelemetry(msg) {
  setInfo("poseX", (msg.x ?? 0).toFixed(2));
  setInfo("poseY", (msg.y ?? 0).toFixed(2));
  setInfo("poseYaw", (msg.yaw_deg ?? 0).toFixed(1));
  setInfo("infoConn", msg.odom_live ? "connected · odom live" : "connected · no odom");
  updateBattery(msg.battery_soc, msg.battery_v);
}

// Tab switching (keeps both panels in the DOM so the WS/feeds keep running).
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    const name = tab.dataset.tab;
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("current", t === tab));
    document.querySelectorAll(".tab-panel").forEach(p => {
      p.classList.toggle("current", p.id === `tab-${name}`);
    });
    // The 3D canvas needs a resize when its tab becomes visible again.
    if (name === "drive") window.dispatchEvent(new Event("resize"));
  });
});