// G1 phone teleop -- stripped-down, touch-first controller.
//
// Speaks the SAME /ws JSON protocol as the laptop console (controller.js) and
// shares the server-side single-controller lock. Surface is intentionally tiny:
//   - two analog thumbsticks (Move = fwd/strafe, Turn = yaw)
//   - one obstacle-avoidance toggle
//   - a Zero-Torque -> Damp -> Stand -> Walk mode ladder, each change gated behind
//     a SWIPE-to-confirm so a mis-tap can't switch the robot's mode
//   - a "Take control" affordance (lock arbitration across devices)
// No camera / map / lidar / gestures -- those stay on the laptop console.

(function () {
  "use strict";

  // ---- speed caps (overwritten by the server's "config" message) ----
  let MAX_VX = 1.5, MAX_VY = 0.7, MAX_VYAW = 2.0;   // SLOW_SCALE unused (no Shift on a phone)
  const SEND_HZ = 30, SEND_INTERVAL = 1000 / SEND_HZ;

  // ---- thumbstick shaping (tuned for FINE, granular control) ----
  // A LOW top speed (PHONE_SPEED) is what makes it precise: the whole stick travel maps
  // to a small speed range, so every bit of push is a distinguishable "step". EXPO is
  // only a mild curve (gentle near centre, but the mid-range stays usable -- a strong
  // expo made the middle nearly dead, which felt like just a few speeds). Long TRAVEL =
  // lots of thumb distance per unit speed. Raise PHONE_SPEED for more top-end; raise
  // EXPO for a softer centre; lower EXPO toward 0 for a perfectly linear feel.
  const DEADZONE = 0.04, TRAVEL = 0.90, EXPO = 0.45;
  const PHONE_SPEED = 0.5;    // fraction of the robot's max speed used from the phone

  // ---- identity: ONE id per page load. Not persisted, so two tabs are two
  // distinct clients (the lock arbitrates between them); a module const so a ws
  // reconnect keeps the same identity. ----
  // crypto.randomUUID exists only in a SECURE context; over plain http:// on a LAN IP
  // it's undefined, so the fallback must be bulletproof (and never throw, or the whole
  // module dies and the page half-works).
  const CLIENT_ID = (function () {
    try { if (window.crypto && crypto.randomUUID) return crypto.randomUUID(); } catch (_) {}
    return "phone-" + Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2);
  })();
  const LABEL = "Phone";

  // ---- live state ----
  const axes = { fwd: 0, strafe: 0, turn: 0 };
  let uiMode = "damp", transitioning = false;
  let hasControl = false, ownerLabel = null;
  let robotMoving = false;   // fsm_state.moving — you can only take the drive lock while stopped
  // Arm "greeting" gestures any client may fire WITHOUT the drive lock (match the server's
  // OPEN_GESTURES). Driving, modes and Dance/Climb still take control first.
  const OPEN_GESTURES = new Set(["high_wave", "high_five", "clap", "shake", "hug", "heart", "kiss"]);
  let actionsOpen = false;                 // Actions surface shown -> deck inert
  let comboReady = { dance: false, climb: false };   // server-captured whole-body FSM ids
  let gestureBusyUntil = 0;                // optimistic double-fire lock (monotonic ms)
  let obstacleEnabled = false;
  let connected = false, ws = null;
  let lastObs = null;          // last obstacle frame, for on-demand radar redraw
  let curFeed = "radar";       // "radar" | "sphere" | "cloud"
  let currentStep = "continuous";  // discrete-step size (server-authoritative)
  const sticks = [];   // {reset} handles

  const $ = (id) => document.getElementById(id);
  const el = {
    mode: $("phMode"), transit: $("phTransit"),
    status: $("phStatus"), statusText: $("phStatusText"),
    batt: $("phBatt"), battText: $("phBattText"), battFill: $("phBattFill"),
    chip: $("phControlChip"), fsBtn: $("phFsBtn"), landExit: $("phLandExit"),
    obsView: $("phObsView"), bigRadar: $("bigRadar"),
    obsBtn: $("phObsBtn"), obsState: $("phObsState"),
    deckHint: $("phDeckHint"), stop: $("phStop"),
    takeOverlay: $("phTakeOverlay"), takeBtn: $("phTakeBtn"),
    overlayTitle: $("phOverlayTitle"), overlaySub: $("phOverlaySub"),
    sheet: $("phSheet"), sheetTitle: $("phSheetTitle"), sheetNote: $("phSheetNote"),
    sheetX: $("phSheetX"), swipe: $("phSwipe"), swipeKnob: $("phSwipeKnob"),
    swipeFill: $("phSwipeFill"), swipeHint: $("phSwipeHint"),
    actions: $("phActions"), actionsX: $("phActionsX"), actionsStop: $("phActionsStop"),
    actionsHint: $("phActionsHint"), actionsStatus: $("phActionsStatus"),
    gestGrid: $("phGestGrid"), deck: $("phDeck"),
  };

  // =====================================================================
  // Connection
  // =====================================================================
  function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen = () => {
      connected = true;
      setStatus(true, "connected");
      // Announce identity so the server can arbitrate the control lock.
      send({ type: "hello", client_id: CLIENT_ID, label: LABEL });
    };
    ws.onmessage = (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch (_) { return; }
      switch (msg.type) {
        case "config":    applyConfig(msg); break;
        case "fsm_state": applyFsm(msg); break;
        case "telemetry": applyTelemetry(msg); break;
        case "obstacle":  applyObstacle(msg); break;
        case "control":   applyControl(msg); break;
        case "takeover_denied":
          // Visible on BOTH surfaces: the Actions status line AND a brief flash on the
          // always-on header chip (the Drive surface has no actions status).
          setActionStatus("Robot moving — stop it to take control");
          if (el.chip) {
            el.chip.classList.add("denied");
            el.chip.textContent = "○ robot moving — can't take yet";
            setTimeout(() => { el.chip.classList.remove("denied"); renderChip(); }, 1600);
          }
          break;
        case "step_mode": applyStep(msg.name || msg.mode || msg.step_mode); break;
      }
    };
    ws.onclose = () => {
      connected = false;
      setStatus(false, "disconnected — retrying…");
      setControl(false, null);   // lost the socket -> definitely not the owner
      resetSticks();
      setTimeout(connect, 1000);
    };
    ws.onerror = () => { try { ws.close(); } catch (_) {} };
  }
  function send(obj) {
    if (connected && ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  // =====================================================================
  // Inbound messages
  // =====================================================================
  function applyConfig(msg) {
    if (typeof msg.max_vx === "number")   MAX_VX = msg.max_vx;
    if (typeof msg.max_vy === "number")   MAX_VY = msg.max_vy;
    if (typeof msg.max_vyaw === "number") MAX_VYAW = msg.max_vyaw;
    if (msg.obstacle && typeof msg.obstacle.enabled === "boolean") setObstacle(msg.obstacle.enabled);
    if (msg.step && msg.step.mode) applyStep(msg.step.mode);
    // whole-body Dance/Climb are only replayable once the server captured their FSM id
    if (msg.mode_combos && typeof msg.mode_combos === "object") {
      comboReady.dance = !!msg.mode_combos.dance;
      comboReady.climb = !!msg.mode_combos.climb;
      renderActions();
    }
  }

  function applyFsm(msg) {
    uiMode = msg.ui_mode || "";
    transitioning = !!msg.transitioning;
    if (msg.moving !== undefined) { robotMoving = !!msg.moving; renderChip(); }
    document.body.dataset.mode = uiMode;
    document.body.dataset.transitioning = transitioning ? "1" : "";
    if (el.mode) el.mode.textContent = uiMode || "—";
    if (el.transit) el.transit.hidden = !transitioning;
    renderModes();
    renderDeckHint();
    renderActions();                  // arm-gating + whole-body status track the mode
    if (msg.fsm_id === 503) setActionStatus("Dancing… (Stand→Walk to exit)");
    else if (msg.fsm_id === 812) setActionStatus("Climbing… (Stand→Walk to exit)");
    if (!drivable()) resetSticks();   // left walk (or transition began) -> drop velocity
  }

  function applyTelemetry(msg) {
    updateBattery(msg.battery_soc, msg.battery_v);
    if (msg.step_mode) applyStep(msg.step_mode);
  }

  function applyObstacle(msg) {
    if (typeof msg.enabled === "boolean") setObstacle(msg.enabled);
    lastObs = msg;
    // Feed all three views every frame (like the dashboard) so switching is instant.
    if (curFeed === "radar") drawRadarInto(el.bigRadar, msg);
    if (window.LidarOverlay && window.LidarOverlay.updateObstacle) window.LidarOverlay.updateObstacle(msg);
    if (window.Obstacle3D && window.Obstacle3D.update) window.Obstacle3D.update(msg);
  }

  function applyControl(msg) {
    const ownerId = msg.owner_id || null;
    // Auto-claim the lock ONLY when it is FREE (nobody driving) -- so a lone phone,
    // or the phone after the driver leaves / after a server restart, just works
    // instead of getting stuck on "nobody has control". It never auto-steals from an
    // active driver; stealing still needs an explicit "Take control" tap.
    if (ownerId === null) { requestControl(); return; }
    setControl(ownerId === CLIENT_ID, msg.owner_label || null);
  }

  // =====================================================================
  // Control lock UI
  // =====================================================================
  // Repaint the control chip. Shared by setControl (owner changed) and applyFsm (robot
  // started/stopped moving), because the take-over gate depends on BOTH: another device
  // driving AND the robot moving -> you must wait for it to stop.
  function renderChip() {
    if (!el.chip) return;
    const blocked = !hasControl && !!ownerLabel && robotMoving;
    el.chip.classList.toggle("mine", hasControl);
    el.chip.classList.toggle("theirs", !hasControl);
    el.chip.classList.toggle("blocked", blocked);
    el.chip.textContent = hasControl
      ? "● You have control"
      : blocked ? `○ ${ownerLabel} driving — stop to take`
      : (ownerLabel ? `○ ${ownerLabel} driving — tap to take` : "○ Take control");
  }

  function setControl(mine, label) {
    const changed = (mine !== hasControl);
    hasControl = mine;
    ownerLabel = label;
    renderChip();
    // Grab-on-first-tap model: the read-only overlay must NEVER block the deck, or you
    // couldn't tap a control to grab. Keep it hidden; the chip shows who drives, and any
    // stick/mode tap takes control (steals) via withControl().
    if (el.takeOverlay) el.takeOverlay.hidden = true;
    if (changed && !mine) { resetSticks(); closeSheet(); closeActions(); pendingAction = null; }
    if (changed && mine) runPending();   // grab-on-first-tap: fire the queued action now
    renderDeckHint();
    renderActions();          // arm/whole-body gating follows the control lock
  }

  // Only send an explicit take-over when it can succeed: not already owner, and not blocked
  // by another device actively driving (server enforces the same rule -> avoids a futile ask).
  function tryTakeControl() {
    if (!hasControl && !(ownerLabel && robotMoving)) requestControl();
  }

  function requestControl() { send({ type: "take_control", client_id: CLIENT_ID }); }

  // =====================================================================
  // Mode ladder + swipe-to-confirm
  // =====================================================================
  const MODE_META = {
    zero_torque: { title: "Zero Torque", warn: true,
      note: "⚠ Motors switch OFF — the robot will collapse if it is not on the gantry." },
    damp:  { title: "Damp",  warn: false, note: "Joints relax. The robot will not respond to steering." },
    stand: { title: "Stand", warn: false, note: "Robot stands rigidly. Then switch to Walk to drive." },
    walk:  { title: "Walk",  warn: false, note: "Enables locomotion — the thumbsticks go live." },
  };

  function renderModes() {
    document.querySelectorAll(".ph-mbtn").forEach((btn) => {
      const on = btn.dataset.mode === uiMode;
      btn.classList.toggle("current", on);
      btn.classList.toggle("transitioning", on && transitioning);
    });
  }

  function renderDeckHint() {
    // "Switch to Walk to drive" only when we actually hold control but aren't in walk.
    if (el.deckHint) el.deckHint.hidden = !(hasControl && uiMode !== "walk");
  }

  // --- swipe sheet plumbing ---
  let pendingMode = null;
  function openSheet(mode) {
    const meta = MODE_META[mode];
    if (!meta) return;
    pendingMode = mode;
    if (el.sheetTitle) el.sheetTitle.textContent = `Switch to ${meta.title}?`;
    if (el.sheetNote) {
      el.sheetNote.textContent = meta.note;
      el.sheetNote.classList.toggle("warn", !!meta.warn);
    }
    resetSwipe();
    if (el.sheet) el.sheet.hidden = false;
  }
  function closeSheet() {
    pendingMode = null;
    if (el.sheet) el.sheet.hidden = true;
    resetSwipe();
  }
  function resetSwipe() {
    if (el.swipeKnob) { el.swipeKnob.classList.add("snap"); el.swipeKnob.style.transform = "translateX(0)"; }
    if (el.swipeFill) el.swipeFill.style.width = "0px";
    if (el.swipe) el.swipe.classList.remove("armed");
    if (el.swipeHint) el.swipeHint.style.opacity = "1";
  }
  function confirmMode() {
    const mode = pendingMode;
    closeSheet();
    if (!mode) return;
    if (!hasControl) { requestControl(); return; }   // safety: only the owner sends
    send({ type: "mode", name: mode });
  }

  function wireSwipe() {
    const knob = el.swipeKnob, track = el.swipe;
    if (!knob || !track) return;
    let dragging = false, pid = null, startX = 0, startY = 0, maxTravel = 0;
    const THRESH = 0.85;   // must drag >=85% of the track to confirm

    // offsetWidth is the LAYOUT width (transform-independent), so travel is correct
    // even when the surface is CSS-rotated into forced-landscape.
    const geom = () => Math.max(0, track.offsetWidth - knob.offsetWidth - 8);  // 4px inset each side
    // Progress along the track's local X, de-rotated from the raw pointer delta.
    const progress = (e) => Math.max(0, Math.min(maxTravel,
      toLocal(e.clientX - startX, e.clientY - startY).x));
    const setX = (x) => {
      knob.style.transform = `translateX(${x}px)`;
      if (el.swipeFill) el.swipeFill.style.width = (x + knob.offsetWidth) + "px";
      if (el.swipeHint) el.swipeHint.style.opacity = String(1 - Math.min(1, x / (maxTravel || 1)));
    };

    knob.addEventListener("pointerdown", (e) => {
      dragging = true; pid = e.pointerId; startX = e.clientX; startY = e.clientY; maxTravel = geom();
      knob.classList.remove("snap");
      try { knob.setPointerCapture(pid); } catch (_) {}
      e.preventDefault();
    });
    knob.addEventListener("pointermove", (e) => {
      if (!dragging || e.pointerId !== pid) return;
      const x = progress(e);
      setX(x);
      track.classList.toggle("armed", maxTravel > 0 && x / maxTravel >= THRESH);
      e.preventDefault();
    });
    const end = (e) => {
      if (!dragging || e.pointerId !== pid) return;
      dragging = false;
      try { knob.releasePointerCapture(e.pointerId); } catch (_) {}
      const x = progress(e);
      if (maxTravel > 0 && x / maxTravel >= THRESH) confirmMode();
      else resetSwipe();
    };
    knob.addEventListener("pointerup", end);
    knob.addEventListener("pointercancel", end);
    knob.addEventListener("lostpointercapture", end);
  }

  // =====================================================================
  // Obstacle toggle
  // =====================================================================
  function setObstacle(on) {
    obstacleEnabled = !!on;
    if (el.obsBtn) el.obsBtn.classList.toggle("on", obstacleEnabled);
    if (el.obsState) el.obsState.textContent = obstacleEnabled ? "ON" : "OFF";
  }

  // Discrete-step size selector (server-authoritative; mirror the dashboard).
  function applyStep(name) {
    if (!name) return;
    currentStep = name;
    document.querySelectorAll("#phStep [data-step]").forEach((b) =>
      b.classList.toggle("on", b.dataset.step === name));
  }

  // =====================================================================
  // 2D obstacle ring (ported from web/obstacle.js drawRadarGeom/drawRadarInto)
  // =====================================================================
  const Z_STOP = 0.7, Z_SLOW = 2.0, RADAR_RANGE_M = 3.0;
  const RC = { stop: "#FF4747", slow: "#FF8636", clear: "#3FD39B", grid: "#232C3E", dim: "#7E879B" };
  function radarZoneColor(d) {
    if (d == null) return null;
    if (d < Z_STOP) return RC.stop;
    if (d < Z_SLOW) return RC.slow;
    return RC.clear;
  }
  function drawRadarGeom(ctx, size, msg) {
    const cx = size / 2, cy = size / 2, R = (size / 2) * 0.9;
    ctx.clearRect(0, 0, size, size);
    ctx.strokeStyle = RC.grid; ctx.lineWidth = 1;
    [Z_STOP, Z_SLOW, RADAR_RANGE_M].forEach((rm) => {
      ctx.beginPath();
      ctx.arc(cx, cy, R * Math.min(1, rm / RADAR_RANGE_M), 0, Math.PI * 2);
      ctx.stroke();
    });
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy - R); ctx.stroke();  // forward crosshair
    const ring = msg && msg.ring;
    if (ring && Array.isArray(ring.dist)) {
      const n = ring.dist.length, binRad = (ring.bin_deg * Math.PI) / 180;
      for (let i = 0; i < n; i++) {
        const d = ring.dist[i], col = radarZoneColor(d);
        if (!col) continue;
        const cRad = ((ring.start_deg + (i + 0.5) * ring.bin_deg) * Math.PI) / 180;
        const r = R * Math.min(1, d / RADAR_RANGE_M);
        const a0 = cRad - binRad / 2, a1 = cRad + binRad / 2;
        ctx.fillStyle = col; ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx - r * Math.sin(a0), cy - r * Math.cos(a0));
        ctx.lineTo(cx - r * Math.sin(a1), cy - r * Math.cos(a1));
        ctx.closePath(); ctx.fill();
      }
      ctx.globalAlpha = 1;
    } else {
      ctx.fillStyle = RC.dim;
      ctx.font = Math.max(11, Math.round(size / 16)) + "px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("no ring data", cx, cy);
    }
  }
  function drawRadarInto(canvas, msg) {
    if (!canvas) return;
    const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
    if (!cssW || !cssH) return;                 // not laid out / hidden feed
    const size = Math.min(cssW, cssH);
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const needW = Math.round(cssW * DPR), needH = Math.round(cssH * DPR);
    if (canvas.width !== needW || canvas.height !== needH) { canvas.width = needW; canvas.height = needH; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(DPR, 0, 0, DPR, ((cssW - size) / 2) * DPR, ((cssH - size) / 2) * DPR);
    drawRadarGeom(ctx, size, msg);
  }

  // =====================================================================
  // Feed switcher (2D ring / 3D sphere / cloud feed) -- one visible at a time
  // =====================================================================
  function setFeed(mode) {
    if (mode !== "radar" && mode !== "sphere" && mode !== "cloud") return;
    curFeed = mode;
    if (el.obsView) el.obsView.dataset.feed = mode;
    document.body.dataset.feed = mode;
    document.querySelectorAll("[data-feed-btn]").forEach((b) =>
      b.classList.toggle("active", b.dataset.feedBtn === mode));
    // Let the three.js modules size to the now-visible box, then redraw the radar.
    window.dispatchEvent(new Event("resize"));
    if (mode === "radar") drawRadarInto(el.bigRadar, lastObs);
  }

  // =====================================================================
  // Fullscreen + landscape lock (immersive driving surface; ported from
  // web/drive_mode.js). Both require a user gesture, so they're wired to taps.
  // =====================================================================
  function fsElement() { return document.fullscreenElement || document.webkitFullscreenElement || null; }
  // iPhone Safari exposes NO Fullscreen API for elements (only <video>/iPad), so
  // feature-detect rather than assume. When absent, real fullscreen is impossible
  // and the only path is a standalone "Add to Home Screen" launch.
  function fsSupported() {
    const d = document.documentElement;
    return !!(d.requestFullscreen || d.webkitRequestFullscreen);
  }
  function isStandalone() {
    return window.navigator.standalone === true ||
      (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches);
  }
  function requestFS(elm) {
    const fn = elm.requestFullscreen || elm.webkitRequestFullscreen;
    if (!fn) return Promise.reject();
    try { return Promise.resolve(fn.call(elm)); } catch (e) { return Promise.reject(e); }
  }
  function exitFS() {
    const fn = document.exitFullscreen || document.webkitExitFullscreen;
    if (fn) { try { fn.call(document); } catch (_) {} }
  }
  function lockLandscape() {
    try {
      if (screen.orientation && screen.orientation.lock) screen.orientation.lock("landscape").catch(function () {});
    } catch (_) {}
  }
  function unlockOrientation() {
    try { screen.orientation && screen.orientation.unlock && screen.orientation.unlock(); } catch (_) {}
  }
  function showA2HS(on) { const h = $("phA2hs"); if (h) h.hidden = !on; }

  // --- Forced landscape (the ⛶ button) --------------------------------------
  // iOS blocks the Fullscreen API + orientation lock for web pages, so we FAKE a
  // landscape driving screen by CSS-rotating the whole surface 90deg (see phone.css)
  // -- turn the phone sideways with the system rotation lock ON and it reads upright.
  // It only engages while the device is actually portrait; when the device is truly
  // landscape the native layout already applies. Because the surface is rotated,
  // pointer math for the sticks + swipe is de-rotated via toLocal().
  function rotActive() {
    return document.body.classList.contains("ph-landscape")
      && !!(window.matchMedia && window.matchMedia("(orientation: portrait)").matches);
  }
  // Map a screen-space vector into the surface's LOCAL frame (inverse of the CSS
  // rotate(-90deg)); identity when not force-rotated.
  function toLocal(sx, sy) { return rotActive() ? { x: -sy, y: sx } : { x: sx, y: sy }; }

  function setLandscape(on) {
    document.body.classList.toggle("ph-landscape", on);
    if (el.fsBtn) el.fsBtn.classList.toggle("on", on);
    try { localStorage.setItem("g1.phoneLandscape", on ? "1" : ""); } catch (_) {}
    // Real fullscreen + orientation lock where the platform allows it (Android/iPad);
    // on iOS the CSS rotation is what actually delivers landscape.
    if (on) { if (fsSupported()) requestFS(document.documentElement).then(lockLandscape, lockLandscape); }
    else if (fsElement()) { unlockOrientation(); exitFS(); }
    resetSticks();                                   // rotation changes the stick frame
    window.dispatchEvent(new Event("resize"));       // re-fit three.js views + ring
    if (curFeed === "radar") drawRadarInto(el.bigRadar, lastObs);
  }
  function toggleImmersive() { setLandscape(!document.body.classList.contains("ph-landscape")); }
  // Take-control gesture: best-effort REAL fullscreen only (no rotation); no-op on iOS.
  function goImmersive() { if (fsSupported()) requestFS(document.documentElement).then(lockLandscape, lockLandscape); }

  // =====================================================================
  // Thumbsticks (ported shaping from web/drive_mode.js)
  // =====================================================================
  // mild cubic expo: a soft centre but the whole travel stays usable + granular (a
  // quartic/strong expo bunched all the speed into the last bit -> felt like few steps).
  // Reaches 1.0 at full deflection.
  function curveOf(lin) { return EXPO * lin * lin * lin + (1 - EXPO) * lin; }
  function shape(v) {
    const a = Math.abs(v);
    if (a <= DEADZONE) return 0;
    return Math.sign(v) * curveOf((a - DEADZONE) / (1 - DEADZONE));
  }
  function shapeVec(nx, ny) {
    const mag = Math.hypot(nx, ny);
    if (mag <= DEADZONE) return { x: 0, y: 0 };
    const m = Math.min(mag, 1);
    const s = curveOf((m - DEADZONE) / (1 - DEADZONE)) / mag;
    return { x: nx * s, y: ny * s };
  }

  // ---- Actions surface: gestures + whole-body --------------------------------
  const GEST_LABEL = {
    high_wave: "Waving", high_five: "High-fiving", clap: "Clapping", shake: "Shaking hand",
    hug: "Hugging", heart: "Heart", kiss: "Kiss", hands_up: "Hands up", release_arm: "Releasing arms",
    dance: "Dancing", climb: "Climb mode",
  };
  function setActionStatus(txt) { if (el.actionsStatus) el.actionsStatus.textContent = txt || ""; }
  function armReady() { return uiMode === "stand" || uiMode === "walk"; }

  // Repaint tile enabled/greyed state: arm gestures need the robot up (mirror the desktop
  // guard -- the server's apply_cmd does NOT mode-check); whole-body needs its FSM id captured.
  function renderActions() {
    if (el.actionsHint) {
      el.actionsHint.textContent = !hasControl
        ? "Greetings work for anyone · other actions take control"
        : armReady() ? "Robot is up — OK" : "Stand first for arm gestures";
    }
    document.querySelectorAll("#phGestGrid .ph-gest").forEach((b) =>
      b.classList.toggle("disabled", !!b.dataset.arm && !armReady()));
    document.querySelectorAll(".ph-wb").forEach((b) => {
      const ready = !!comboReady[b.dataset.combo];
      b.classList.toggle("disabled", !ready);
      b.title = ready ? "" : b.dataset.combo + ": FSM id not captured yet";
    });
  }

  function openActions() {
    if (actionsOpen) return;
    actionsOpen = true;
    resetSticks(); send({ type: "stop" });            // nothing translates while gesturing
    if (el.actions) el.actions.hidden = false;
    if (el.deck) el.deck.classList.add("inert");
    document.querySelectorAll(".ph-surfbtn").forEach((b) =>
      b.classList.toggle("current", b.dataset.surf === "actions"));
    setActionStatus(""); renderActions();
  }
  function closeActions() {
    if (!actionsOpen) return;
    actionsOpen = false;
    if (el.actions) el.actions.hidden = true;
    if (el.deck) el.deck.classList.remove("inert");
    document.querySelectorAll(".ph-surfbtn").forEach((b) =>
      b.classList.toggle("current", b.dataset.surf === "drive"));
  }

  // Grab-on-first-tap: if we already own control run fn now; otherwise take control
  // (steals the lock) and run fn the moment the grant lands. A phone left idle never
  // steals -- only an actual tap/stick grabs control. Pending action expires in 3 s.
  let pendingAction = null, pendingUntil = 0;
  function withControl(fn) {
    if (hasControl) { fn(); return; }
    pendingAction = fn; pendingUntil = performance.now() + 3000;
    requestControl(); setActionStatus("Taking control…");
  }
  function runPending() {
    const p = pendingAction; pendingAction = null;
    if (p && performance.now() <= pendingUntil) p();
  }

  // one-shot gesture (mirrors controller.js). Arm-gated tiles need the robot up;
  // a short busy-lock stops double-fire.
  function fireGesture(name, isArm) {
    const run = () => {
      if (isArm && !armReady()) { setActionStatus("Stand first for arm gestures"); return; }
      const now = performance.now();
      if (now < gestureBusyUntil) return;
      gestureBusyUntil = now + 900;
      send({ type: "cmd", name });
      setActionStatus("→ " + (GEST_LABEL[name] || name));
      if (navigator.vibrate) navigator.vibrate(15);
    };
    // Shared arm gestures fire WITHOUT the drive lock (the server accepts them from any
    // client). Owner-only actions (hands_up / release_arm) take control first.
    if (OPEN_GESTURES.has(name)) run();
    else withControl(run);
  }

  // whole-body hold-to-fire: press-and-hold ~1.3 s (progress fill); release early aborts.
  function wireHold(btn) {
    const HOLD_MS = 1300;
    const fill = btn.querySelector(".ph-wb-fill"), name = btn.dataset.combo;
    let raf = null, t0 = 0;
    function cancel() {
      if (raf) cancelAnimationFrame(raf); raf = null;
      btn.classList.remove("holding"); if (fill) fill.style.width = "0%";
    }
    function tick() {
      const p = Math.min(1, (performance.now() - t0) / HOLD_MS);
      if (fill) fill.style.width = (p * 100) + "%";
      if (p >= 1) {
        cancel();
        withControl(() => {
          resetSticks(); send({ type: "stop" });
          send({ type: "cmd", name });
          setActionStatus("→ " + (GEST_LABEL[name] || name));
          if (navigator.vibrate) navigator.vibrate([20, 40, 20]);
        });
        return;
      }
      raf = requestAnimationFrame(tick);
    }
    btn.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      if (btn.classList.contains("disabled")) { setActionStatus(name + ": not captured yet"); return; }
      if (!hasControl) requestControl();          // kick off the grab; the hold proceeds
      t0 = performance.now(); btn.classList.add("holding");
      try { btn.setPointerCapture(e.pointerId); } catch (_) {}
      raf = requestAnimationFrame(tick);
    });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach((ev) =>
      btn.addEventListener(ev, cancel));
  }

  // Sticks bite only in live walk AND while this phone owns the control lock AND the
  // Actions surface is closed (opening it makes the deck inert + zeroes velocity).
  function drivable() {
    return hasControl && uiMode === "walk" && !transitioning && !actionsOpen;
  }

  function makeStick(base, onChange) {
    const knob = base.querySelector(".ph-knob");
    let pointerId = null, originX = 0, originY = 0;
    const setKnob = (dx, dy) => { knob.style.transform = `translate(${dx}px, ${dy}px)`; };

    function reset() {
      pointerId = null;
      knob.classList.remove("active");
      setKnob(0, 0);
      onChange(0, 0);
    }
    // RELATIVE origin: "neutral" is wherever the thumb FIRST lands, not the base centre.
    // A fixed-centre stick reads a big deflection the instant a thumb touches off-centre
    // -> "goes to max velocity immediately". Anchoring to the touch-down point makes
    // first contact 0 and every push from there analog + shaped -> smooth/precise, like
    // the desktop mouse (which happens to click near the centre). Deflection is measured
    // from the origin, still de-rotated via toLocal for forced-landscape.
    function track(e) {
      const max = (base.getBoundingClientRect().width / 2) * TRAVEL;
      const L = toLocal(e.clientX - originX, e.clientY - originY);
      let dx = L.x, dy = L.y;
      const dist = Math.hypot(dx, dy);
      if (dist > max && dist > 0) { const s = max / dist; dx *= s; dy *= s; }
      setKnob(dx, dy);
      onChange(max > 0 ? dx / max : 0, max > 0 ? dy / max : 0);
    }
    base.addEventListener("pointerdown", (e) => {
      if (pointerId !== null) return;
      tryTakeControl();   // grab-on-first-touch when free/stopped (drives once granted; a
                          // moving robot held by another device denies -> takeover_denied)
      if (!drivable()) return;
      pointerId = e.pointerId;
      originX = e.clientX; originY = e.clientY;   // anchor neutral to the touch-down point
      knob.classList.add("active");
      try { base.setPointerCapture(pointerId); } catch (_) {}
      setKnob(0, 0); onChange(0, 0);              // start at rest -> no lurch on contact
      e.preventDefault();
    });
    base.addEventListener("pointermove", (e) => {
      if (e.pointerId !== pointerId) return;
      if (!drivable()) { try { base.releasePointerCapture(e.pointerId); } catch (_) {} reset(); return; }
      track(e); e.preventDefault();
    });
    const up = (e) => {
      if (e.pointerId !== pointerId) return;
      try { base.releasePointerCapture(e.pointerId); } catch (_) {}
      reset(); e.preventDefault();
    };
    base.addEventListener("pointerup", up);
    base.addEventListener("pointercancel", up);
    base.addEventListener("lostpointercapture", up);
    return { reset };
  }

  function resetSticks() {
    axes.fwd = axes.strafe = axes.turn = 0;
    sticks.forEach((s) => s.reset());
  }

  // =====================================================================
  // 30 Hz send loop -- only the controller in live walk emits velocity (which
  // also feeds the server's owner-gated motion watchdog).
  // =====================================================================
  function computeVelocity() {
    // PHONE_SPEED caps the top speed so the whole stick range is gentler / finer.
    const cx = MAX_VX * PHONE_SPEED, cy = MAX_VY * PHONE_SPEED, cyaw = MAX_VYAW * PHONE_SPEED;
    let vx = cx * axes.fwd;
    let vy = cy * axes.strafe;
    let vyaw = cyaw * axes.turn;
    vx = Math.max(-cx, Math.min(cx, vx));
    vy = Math.max(-cy, Math.min(cy, vy));
    vyaw = Math.max(-cyaw, Math.min(cyaw, vyaw));
    // Normalize a diagonal onto the (capped) speed ellipse (server re-normalizes too).
    if (cx > 0 && cy > 0) {
      const m = (vx / cx) ** 2 + (vy / cy) ** 2;
      if (m > 1) { const s = 1 / Math.sqrt(m); vx *= s; vy *= s; }
    }
    return { vx, vy, vyaw };
  }

  setInterval(() => {
    if (drivable()) {
      const { vx, vy, vyaw } = computeVelocity();
      send({ type: "move", vx, vy, vyaw });
    }
  }, SEND_INTERVAL);

  // =====================================================================
  // Wire-up
  // =====================================================================
  function build() {
    // thumbsticks
    const moveBase = $("phStickMove"), turnBase = $("phStickTurn");
    if (moveBase) sticks.push(makeStick(moveBase, (nx, ny) => {
      const v = shapeVec(nx, ny);       // radial deadzone + expo -> honest diagonals
      axes.strafe = -v.x;               // screen-left -> strafe-left (+vy), matches drive_mode.js
      axes.fwd    = -v.y;               // screen-up   -> forward (+vx)
    }));
    if (turnBase) sticks.push(makeStick(turnBase, (nx /*, ny */) => {
      axes.turn = shape(-nx);           // screen-left -> turn-left (+vyaw)
    }));

    // STOP: zero velocity now + recentre sticks (server halts, gait stays ready)
    if (el.stop) el.stop.addEventListener("click", () => { resetSticks(); send({ type: "stop" }); });

    // mode ladder -> swipe sheet (needs control; otherwise prompt for it)
    document.querySelectorAll(".ph-mbtn").forEach((btn) => {
      btn.addEventListener("click", () => withControl(() => openSheet(btn.dataset.mode)));
    });

    // Drive | Actions surface selector + the Actions surface controls
    document.querySelectorAll(".ph-surfbtn").forEach((b) =>
      b.addEventListener("click", () => (b.dataset.surf === "actions" ? openActions() : closeActions())));
    if (el.actionsX) el.actionsX.addEventListener("click", closeActions);
    if (el.actionsStop) el.actionsStop.addEventListener("click", () => { resetSticks(); send({ type: "stop" }); });
    document.querySelectorAll("#phGestGrid .ph-gest").forEach((b) =>
      b.addEventListener("click", () => fireGesture(b.dataset.cmd, !!b.dataset.arm)));
    document.querySelectorAll(".ph-wb").forEach(wireHold);

    // obstacle toggle (needs control)
    if (el.obsBtn) el.obsBtn.addEventListener("click", () => {
      if (!hasControl) { requestControl(); return; }
      send({ type: "obstacle", action: obstacleEnabled ? "disable" : "enable" });
    });

    // step-size selector (needs control)
    document.querySelectorAll("#phStep [data-step]").forEach((b) =>
      b.addEventListener("click", () => {
        if (!hasControl) { requestControl(); return; }
        send({ type: "step_mode", name: b.dataset.step });
      }));

    // take-control affordances (taking control is a natural moment to go immersive)
    if (el.takeBtn) el.takeBtn.addEventListener("click", () => { tryTakeControl(); goImmersive(); });
    if (el.chip) el.chip.addEventListener("click", () => { tryTakeControl(); });

    // fullscreen + landscape lock toggle (⛶). Hidden if already launched standalone
    // (Add-to-Home-Screen) since we're already chromeless there.
    // ⤢ toggles forced-landscape (needed even in standalone fullscreen, which is
    // exactly where you want a landscape driving screen).
    if (el.fsBtn) el.fsBtn.addEventListener("click", toggleImmersive);
    // ✕ exits the joysticks-only landscape view (the ⤢ toggle is in the hidden header).
    if (el.landExit) el.landExit.addEventListener("click", () => setLandscape(false));
    // dismiss the iOS "Add to Home Screen" help
    { const x = $("phA2hsX"); if (x) x.addEventListener("click", () => showA2HS(false)); }
    { const h = $("phA2hs"); if (h) h.addEventListener("click", (e) => { if (e.target === h) showA2HS(false); }); }

    // obstacle feed switcher (2D ring / 3D sphere / cloud feed)
    document.querySelectorAll("[data-feed-btn]").forEach((b) =>
      b.addEventListener("click", () => setFeed(b.dataset.feedBtn)));

    // swipe sheet controls
    if (el.sheetX) el.sheetX.addEventListener("click", closeSheet);
    if (el.sheet) el.sheet.addEventListener("click", (e) => { if (e.target === el.sheet) closeSheet(); });
    wireSwipe();

    // redraw the 2D ring when its box resizes (orientation change / fullscreen).
    window.addEventListener("resize", () => { if (curFeed === "radar") drawRadarInto(el.bigRadar, lastObs); });

    // start read-only until the server tells us who owns the lock
    // restore forced-landscape preference (CSS-only on load; a real FS request needs
    // a user gesture, so that only happens when the ⛶ button is tapped).
    try {
      if (localStorage.getItem("g1.phoneLandscape") === "1") {
        document.body.classList.add("ph-landscape");
        if (el.fsBtn) el.fsBtn.classList.add("on");
      }
    } catch (_) {}

    setControl(false, null);
    setStatus(false, "connecting…");
    setFeed("radar");
    applyStep(currentStep);
    renderModes();
    window.dispatchEvent(new Event("resize"));
  }

  // =====================================================================
  // Status + battery chrome
  // =====================================================================
  function setStatus(ok, text) {
    if (el.status) el.status.classList.toggle("on", !!ok);
    if (el.statusText) el.statusText.textContent = text;
  }
  function updateBattery(soc, volts) {
    if (soc === null || soc === undefined) {
      if (el.battText) el.battText.textContent = "—";
      if (el.battFill) el.battFill.style.width = "0%";
      if (el.batt) el.batt.className = "ph-batt";
      return;
    }
    const pct = Math.max(0, Math.min(100, Math.round(soc)));
    if (el.battText) el.battText.textContent = pct + "%";
    if (el.battFill) el.battFill.style.width = pct + "%";
    const level = pct <= 15 ? "crit" : pct <= 35 ? "warn" : "";
    if (el.batt) el.batt.className = "ph-batt" + (level ? " " + level : "");
  }

  // Safety: recentre + stop on page close (not on mere focus loss).
  window.addEventListener("pagehide", () => { resetSticks(); send({ type: "stop" }); });

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
  connect();
})();
