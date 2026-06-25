// Drive-input mode toggle + virtual thumbsticks ("mobile" control).
//
// Adds a Laptop / Mobile switch to the Drive panel. In "laptop" mode the existing
// keyboard + D-pad drive the robot (unchanged). In "mobile" mode the D-pad is
// hidden and TWO analog thumbsticks appear -- laid out like the Unitree G1 hand
// controller:
//
//   LEFT  stick = translation  (up/down -> forward/back vx, left/right -> strafe vy)
//   RIGHT stick = rotation      (left/right -> turn vyaw)
//
// The sticks are ANALOG: how far you push = how fast you go (0..MAX), unlike the
// binary D-pad. They produce normalized axes in [-1, 1] which controller.js folds
// into computeVelocity() exactly like the held keys -- so obstacle gating, the
// speed ellipse, slow mode and the server-side clamp all still apply untouched.
//
// Self-contained, plain DOM, no framework. Mirrors step_mode.js / feedview.js:
// loaded BEFORE controller.js so window.DriveMode exists when the send loop reads
// axes(). Injects its own CSS (namespaced "dm-"), reusing the console theme vars.
//
//   window.DriveMode = {
//     init()      -- optional; controller.js may call it (build is idempotent)
//     axes()      -- {fwd, strafe, turn} each in [-1,1]; null when in laptop mode
//     reset()     -- snap both sticks to centre + zero the axes (safety)
//     mode()      -- "laptop" | "mobile"
//   }

(function () {
  "use strict";

  const LS_KEY = "g1.driveMode";       // remembers the operator's last choice
  const DEADZONE = 0.08;               // ignore a resting thumb's tiny drift
  const TRAVEL = 0.72;                 // knob travels 72% of the base radius
  const EXPO = 0.55;                   // response curve: gentle near centre, full at the edge

  // Analog axes read every frame by controller.js. Each component is in [-1, 1].
  const axes = { fwd: 0, strafe: 0, turn: 0 };

  let mode = "laptop";                 // "laptop" | "mobile"
  let booted = false;
  const sticks = [];                   // {reset} handles, so reset() hits both

  // A coarse pointer / touch screen -> default to the thumbstick layout. Desktops
  // keep the keyboard layout. The explicit toggle always wins and is remembered.
  function defaultMode() {
    try {
      const saved = localStorage.getItem(LS_KEY);
      if (saved === "laptop" || saved === "mobile") return saved;
    } catch (_) {}
    const coarse = (navigator.maxTouchPoints || 0) > 0
      || (window.matchMedia && window.matchMedia("(pointer: coarse)").matches);
    return coarse ? "mobile" : "laptop";
  }

  // Deadzone + rescale so the very first motion past the deadzone starts at 0
  // (no velocity step), and full deflection still reaches 1.
  function shape(v) {
    const a = Math.abs(v);
    if (a <= DEADZONE) return 0;
    const lin = (a - DEADZONE) / (1 - DEADZONE);              // 0..1 past the deadzone
    const curved = EXPO * lin * lin * lin + (1 - EXPO) * lin; // expo: fine near centre, full at edge
    return Math.sign(v) * curved;
  }

  // The sticks may only deflect while the robot is in LIVE walk -- never during
  // stand / transitions. This mirrors the D-pad's `if (uiMode!=='walk'||...)` gate
  // in controller.js so a thumb resting on a stick can't pre-arm a velocity that
  // would fire the instant Walk engages. controller.js mirrors the FSM onto the
  // <body> data attributes, so we read those (no cross-module import needed).
  function drivable() {
    return !!document.body
      && document.body.dataset.mode === "walk"
      && document.body.dataset.transitioning !== "1";
  }

  // ----------------------------------------------------------------- styles
  const CSS = `
  /* Mode toggle sits inline in the "Drive" panel heading. */
  #drive-mode-toggle { display: inline-flex; vertical-align: middle; margin-left: 10px; }
  .dm-seg { display: inline-flex; border-radius: 8px; overflow: hidden;
    border: 1px solid var(--line, #232C3E); }
  .dm-btn {
    font-family: inherit; font-size: 11px; font-weight: 600; letter-spacing: .04em;
    color: var(--steel, #8A93A6); background: var(--surface-2, #161C2A);
    border: 0; border-right: 1px solid var(--line, #232C3E);
    padding: 5px 12px; cursor: pointer; min-height: 30px;
    transition: color .15s, background .15s;
  }
  .dm-btn:last-child { border-right: 0; }
  .dm-btn:hover { color: var(--chalk, #E9EDF6); }
  .dm-btn.on { color: #06281C; background: var(--live, #3FD39B); }

  /* The two-thumbstick deck. Hidden unless body[data-drive-mode="mobile"]. */
  #joystick-panel { display: none; }
  body[data-drive-mode="mobile"] .drive-main { display: none; }
  body[data-drive-mode="mobile"] #joystick-panel {
    display: flex; flex: 1 1 280px; align-items: flex-end; justify-content: space-around;
    gap: 14px; flex-wrap: wrap; padding: 6px 0 2px;
  }
  /* In mobile the step-size selector is a HORIZONTAL row pinned ON TOP of the
     sticks, so it never sits under the thumbs. */
  body[data-drive-mode="mobile"] .drive-body { flex-direction: column; align-items: stretch; }
  body[data-drive-mode="mobile"] #step-panel { order: -1; align-self: center; max-width: none; }
  body[data-drive-mode="mobile"] #step-panel .step-row { flex-direction: row; align-items: center; gap: 8px; }
  body[data-drive-mode="mobile"] .step-seg { flex-direction: row; }
  body[data-drive-mode="mobile"] .step-btn { border-bottom: 0; border-right: 1px solid var(--line, #232C3E); padding: 7px 14px; }
  body[data-drive-mode="mobile"] .step-btn:last-child { border-right: 0; }
  .dm-stick-wrap { display: flex; flex-direction: column; align-items: center; gap: 8px; }
  .dm-stick {
    position: relative; width: clamp(120px, 38vw, 168px);
    /* explicit square height first: aspect-ratio is unsupported on iOS<=14 /
       older Android, where a width-only box would collapse to 0 height. */
    height: clamp(120px, 38vw, 168px); min-height: 120px; aspect-ratio: 1 / 1;
    border-radius: 50%; touch-action: none; user-select: none; -webkit-user-select: none;
    background:
      radial-gradient(circle at 50% 50%, color-mix(in srgb, var(--surface-2,#161C2A) 70%, transparent) 0 58%, transparent 60%),
      var(--inset, #0C1019);
    border: 1px solid var(--line, #232C3E);
    box-shadow: inset 0 2px 10px rgba(0,0,0,.45);
    cursor: grab;
  }
  /* Cross-hair guides so the rest position reads as centre. */
  .dm-stick::before, .dm-stick::after {
    content: ""; position: absolute; background: var(--line, #232C3E); opacity: .6;
  }
  .dm-stick::before { left: 12%; right: 12%; top: 50%; height: 1px; transform: translateY(-.5px); }
  .dm-stick::after  { top: 12%; bottom: 12%; left: 50%; width: 1px; transform: translateX(-.5px); }
  .dm-knob {
    /* width + height both 38% of the (square) base -> a circle without relying on
       aspect-ratio; margin -19% (always relative to base width) re-centres it. */
    position: absolute; left: 50%; top: 50%; width: 38%; height: 38%; aspect-ratio: 1 / 1;
    margin-left: -19%; margin-top: -19%; border-radius: 50%;
    background: linear-gradient(180deg, var(--surface-2,#161C2A), var(--surface,#10141E));
    border: 1px solid var(--line, #232C3E);
    box-shadow: 0 4px 12px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.04);
    transform: translate(0px, 0px); transition: transform .08s ease-out;
    pointer-events: none;
  }
  .dm-knob.active { transition: none; border-color: var(--live, #3FD39B);
    box-shadow: 0 4px 14px rgba(0,0,0,.55), 0 0 0 1px var(--live, #3FD39B); }
  .dm-stick-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: .14em;
    color: var(--steel, #8A93A6); text-align: center;
  }
  .dm-stick-label small { display: block; font-family: var(--font-data, monospace);
    font-size: 9px; letter-spacing: .03em; color: var(--steel-dim, #7E879B); margin-top: 2px; text-transform: none; }
  /* Sticks only "bite" in walk; dim AND disable input otherwise so a resting
     thumb can't pre-arm a deflection that would fire the moment Walk engages. */
  body:not([data-mode="walk"]) #joystick-panel,
  body[data-transitioning="1"] #joystick-panel { opacity: .45; pointer-events: none; }

  /* ===== Focused driving layouts (set by JS on switching to Mobile) =====
     wide  -> iPad / desktop: drop the obstacle box, give Drive the full width.
     phone -> small screens: a landscape, controller-only surface.            */

  /* --- wide: tablets & bigger --- */
  body[data-drive-focus="wide"] #tab-drive .panel:not(.buttons) { display: none; }  /* obstacle box gone */
  body[data-drive-focus="wide"] #tab-drive .panel.buttons { grid-column: 1 / -1; }
  body[data-drive-focus="wide"] .dm-stick {
    width: clamp(150px, 26vw, 250px); height: clamp(150px, 26vw, 250px);
  }
  @media (min-width: 760px) {
    body[data-drive-focus="wide"] main.controls { grid-template-columns: 1fr; }       /* Drive spans full width */
  }
  /* the "Show obstacle / Full width" toggle only makes sense in the wide focus */
  .dm-obstacle-toggle { margin-left: 8px; }
  body:not([data-drive-focus="wide"]) .dm-obstacle-toggle { display: none; }
  /* on demand: bring the obstacle box back and narrow the drive panel again */
  body[data-drive-focus="wide"][data-drive-wide-obstacle="1"] #tab-drive .panel:not(.buttons) { display: block; }
  body[data-drive-focus="wide"][data-drive-wide-obstacle="1"] #tab-drive .panel.buttons { grid-column: auto; }
  @media (min-width: 760px) {
    body[data-drive-focus="wide"][data-drive-wide-obstacle="1"] main.controls { grid-template-columns: 1.15fr 1fr; }
  }

  /* --- phone: landscape controller-only surface --- */
  body[data-drive-focus="phone"] { padding: 0 !important; overflow: hidden; }
  /* strip the page chrome down to the live mode bar + the Drive controls */
  body[data-drive-focus="phone"] .windows,
  body[data-drive-focus="phone"] #tab-drive .panel:not(.buttons),
  body[data-drive-focus="phone"] #tab-setup,
  body[data-drive-focus="phone"] #tab-info { display: none !important; }
  body[data-drive-focus="phone"] > header {
    padding: 6px 12px; gap: 8px; border: 0; border-radius: 0; background: var(--ink, #0A0D13);
  }
  body[data-drive-focus="phone"] > header .wordmark,
  body[data-drive-focus="phone"] > header .tabs,
  body[data-drive-focus="phone"] > header .gesture-select,
  body[data-drive-focus="phone"] > header .gesture-src,
  body[data-drive-focus="phone"] > header .header-spacer,
  body[data-drive-focus="phone"] > header #status { display: none !important; }
  body[data-drive-focus="phone"] #tab-drive,
  body[data-drive-focus="phone"] main.controls { display: block; margin: 0; padding: 0; }
  body[data-drive-focus="phone"] .panel.buttons {
    border: 0; border-radius: 0; box-shadow: none; background: transparent; padding: 2px 14px 8px;
  }
  body[data-drive-focus="phone"] .panel.buttons > h2 {
    display: flex; align-items: center; gap: 10px; margin: 0 0 2px;
  }
  body[data-drive-focus="phone"] .drive-body {
    display: flex; flex-direction: column; flex-wrap: nowrap;
    height: calc(100vh - 96px); gap: 4px;
  }
  body[data-drive-focus="phone"] #step-panel { align-self: center; }
  body[data-drive-focus="phone"] .step-seg { flex-direction: row; }
  body[data-drive-focus="phone"] .step-btn { border-bottom: 0; border-right: 1px solid var(--line, #232C3E); }
  body[data-drive-focus="phone"] .step-btn:last-child { border-right: 0; }
  /* sticks pinned to the bottom corners, sized to the landscape height */
  body[data-drive-focus="phone"] #joystick-panel {
    margin-top: auto; display: flex; justify-content: space-between; align-items: flex-end; padding: 0;
  }
  body[data-drive-focus="phone"] .dm-stick {
    width: min(34vw, 56vh); height: min(34vw, 56vh); min-height: 120px;
  }

  /* portrait nudge: tell the operator to rotate (phones where we can't force it) */
  .dm-rotate-hint { display: none; }
  @media (orientation: portrait) {
    body[data-drive-focus="phone"] .dm-rotate-hint {
      display: flex; align-items: center; justify-content: center; text-align: center;
      position: fixed; inset: 0; z-index: 200; padding: 24px; gap: 12px;
      background: var(--ink, #0A0D13); color: var(--chalk, #E9EDF6);
      font-size: 17px; line-height: 1.5; letter-spacing: .02em;
    }
  }
  `;

  function injectStyle() {
    if (document.getElementById("drive-mode-style")) return;
    const s = document.createElement("style");
    s.id = "drive-mode-style";
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  // ----------------------------------------------------------- one thumbstick
  // onChange(nx, ny) receives the knob's normalized offset, nx/ny in [-1, 1],
  // where +x is screen-right and +y is screen-DOWN.
  function makeStick(container, label, sub, onChange) {
    const wrap = document.createElement("div");
    wrap.className = "dm-stick-wrap";

    const base = document.createElement("div");
    base.className = "dm-stick";
    const knob = document.createElement("div");
    knob.className = "dm-knob";
    base.appendChild(knob);

    const cap = document.createElement("div");
    cap.className = "dm-stick-label";
    cap.innerHTML = `${label}<small>${sub}</small>`;

    wrap.appendChild(base);
    wrap.appendChild(cap);
    container.appendChild(wrap);

    let pointerId = null;

    function setKnob(dx, dy) { knob.style.transform = `translate(${dx}px, ${dy}px)`; }

    function reset() {
      pointerId = null;
      knob.classList.remove("active");
      setKnob(0, 0);
      onChange(0, 0);
    }

    // FIXED (sticky) stick: the neutral point is always the base CENTRE -- it never
    // floats to the thumb, so the reference doesn't wander. Deflection = thumb
    // position relative to that fixed centre, clamped to the travel radius; a full
    // release springs the knob back to centre and zeroes the axes.
    function track(e) {
      const rect = base.getBoundingClientRect();
      const r = rect.width / 2;
      const max = r * TRAVEL;
      let dx = e.clientX - (rect.left + r);
      let dy = e.clientY - (rect.top + r);
      const dist = Math.hypot(dx, dy);
      if (dist > max && dist > 0) { const s = max / dist; dx *= s; dy *= s; }
      setKnob(dx, dy);
      onChange(max > 0 ? dx / max : 0, max > 0 ? dy / max : 0);
    }

    base.addEventListener("pointerdown", (e) => {
      if (pointerId !== null) return;           // one finger per stick
      if (!drivable()) return;                  // can't arm outside live walk
      pointerId = e.pointerId;
      knob.classList.add("active");
      try { base.setPointerCapture(pointerId); } catch (_) {}
      track(e);
      e.preventDefault();
    });
    base.addEventListener("pointermove", (e) => {
      if (e.pointerId !== pointerId) return;
      if (!drivable()) {                        // walk ended mid-drag -> let go
        try { base.releasePointerCapture(e.pointerId); } catch (_) {}
        reset();
        return;
      }
      track(e);
      e.preventDefault();
    });
    const up = (e) => {
      if (e.pointerId !== pointerId) return;
      try { base.releasePointerCapture(e.pointerId); } catch (_) {}
      reset();
      e.preventDefault();
    };
    base.addEventListener("pointerup", up);
    base.addEventListener("pointercancel", up);
    // Capture yanked out from under us (OS gesture, mode flip) -> spring to 0 too,
    // so "full release -> zero" holds even when no pointerup ever arrives.
    base.addEventListener("lostpointercapture", up);

    return { reset };
  }

  // ------------------------------------------------------------------- build
  function build() {
    if (booted) return;
    const togMount = document.getElementById("drive-mode-toggle");
    const padMount = document.getElementById("joystick-panel");
    if (!togMount || !padMount) return;        // markup not ready yet
    booted = true;
    injectStyle();

    // Portrait "rotate to landscape" hint (CSS shows it only in phone focus + portrait).
    if (!document.querySelector(".dm-rotate-hint")) {
      const hint = document.createElement("div");
      hint.className = "dm-rotate-hint";
      hint.textContent = "↻  Rotate to landscape to drive";
      document.body.appendChild(hint);
    }

    // --- segmented Laptop / Mobile toggle ---
    const seg = document.createElement("div");
    seg.className = "dm-seg";
    const mkBtn = (m, text) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "dm-btn";
      b.dataset.driveMode = m;
      b.textContent = text;
      b.addEventListener("click", () => selectMode(m, true));   // user gesture -> may fullscreen
      seg.appendChild(b);
      return b;
    };
    const btnLaptop = mkBtn("laptop", "Laptop");
    const btnMobile = mkBtn("mobile", "Mobile");
    togMount.appendChild(seg);

    // "Less wide" toggle: only visible in the wide (tablet/desktop) focus, where the
    // obstacle box is auto-hidden. Tapping brings the obstacle box back and narrows
    // the drive panel; tapping again returns to full width. (CSS hides it otherwise.)
    const obsToggle = document.createElement("button");
    obsToggle.type = "button";
    obsToggle.className = "dm-btn dm-obstacle-toggle";
    const renderObsToggle = () => {
      const on = document.body.dataset.driveWideObstacle === "1";
      obsToggle.textContent = on ? "Full width" : "Show obstacle";
      obsToggle.classList.toggle("on", on);
    };
    obsToggle.addEventListener("click", () => {
      const on = document.body.dataset.driveWideObstacle === "1";
      document.body.dataset.driveWideObstacle = on ? "" : "1";
      renderObsToggle();
    });
    togMount.appendChild(obsToggle);
    renderObsToggle();

    // --- the two thumbsticks ---
    // LEFT stick: forward/back + strafe. Screen-up (ny<0) -> forward (+vx);
    // screen-left (nx<0) -> strafe-left (+vy, matching the 'A' key).
    sticks.push(makeStick(padMount, "Move", "fwd / strafe", (nx, ny) => {
      axes.strafe = shape(-nx);
      axes.fwd    = shape(-ny);
    }));
    // RIGHT stick: turn only. Screen-left (nx<0) -> turn-left (+vyaw, like Arrow-Left).
    sticks.push(makeStick(padMount, "Turn", "yaw", (nx /*, ny */) => {
      axes.turn = shape(-nx);
    }));

    function renderToggle() {
      btnLaptop.classList.toggle("on", mode === "laptop");
      btnMobile.classList.toggle("on", mode === "mobile");
    }

    // setMode is closed over the buttons; expose via outer ref.
    setModeImpl = function (m) {
      if (m !== "laptop" && m !== "mobile") return;
      mode = m;
      document.body.dataset.driveMode = m;
      renderToggle();
      try { localStorage.setItem(LS_KEY, m); } catch (_) {}
      // Leaving mobile (or re-entering) must not leave a stale velocity latched.
      reset();
    };

    selectMode(defaultMode(), false);   // non-gesture: sets layout, never forces fullscreen
  }

  // setMode is wired inside build() once the buttons exist; before that it is a
  // no-op stash so an early caller can't crash.
  let setModeImpl = null;
  function setMode(m) { if (setModeImpl) setModeImpl(m); }

  function reset() {
    axes.fwd = axes.strafe = axes.turn = 0;
    sticks.forEach((s) => s.reset());
  }

  // ----------------------------------------------------- focused driving layout
  // Small physical screens get a landscape, controller-only surface; tablets and
  // bigger just widen (dropping the obstacle box). A phone is the smaller screen
  // dimension <= 600 CSS px (iPhone ~390, iPad >=768), orientation-independent.
  function isPhoneScreen() {
    return Math.min(screen.width || 9999, screen.height || 9999) <= 600;
  }
  function fsElement() {
    return document.fullscreenElement || document.webkitFullscreenElement || null;
  }
  function requestFS(el) {
    const fn = el.requestFullscreen || el.webkitRequestFullscreen;
    if (!fn) return Promise.reject();              // e.g. iOS Safari: CSS focus only
    try { return Promise.resolve(fn.call(el)); } catch (e) { return Promise.reject(e); }
  }
  function exitFS() {
    const fn = document.exitFullscreen || document.webkitExitFullscreen;
    if (fn) { try { fn.call(document); } catch (_) {} }
  }
  function lockLandscape() {
    try {
      if (screen.orientation && screen.orientation.lock) {
        screen.orientation.lock("landscape").catch(function () {});
      }
    } catch (_) {}
  }
  function unlockOrientation() {
    try { screen.orientation && screen.orientation.unlock && screen.orientation.unlock(); } catch (_) {}
  }

  // Apply the focused layout for the current mode + screen. True fullscreen and
  // the orientation lock require a user gesture, so they only fire when
  // userGesture is true (the toggle click); otherwise we set the CSS state only.
  function applyFocus(userGesture) {
    if (mode !== "mobile") { clearFocus(); return; }
    if (isPhoneScreen()) {
      if (!userGesture) { document.body.dataset.driveFocus = ""; return; }  // wait for an explicit tap
      document.body.dataset.driveFocus = "phone";
      requestFS(document.documentElement).then(lockLandscape, lockLandscape);
    } else {
      document.body.dataset.driveFocus = "wide";          // iPad/desktop: widen, drop obstacle box
    }
  }
  function clearFocus() {
    document.body.dataset.driveFocus = "";
    unlockOrientation();
    if (fsElement()) exitFS();
  }

  // The Laptop/Mobile buttons call this (userGesture=true); load calls it false.
  function selectMode(m, userGesture) {
    setMode(m);
    applyFocus(userGesture);   // clears focus for laptop, applies it for mobile
  }

  // Leaving fullscreen (Back / Esc / system gesture) from the phone surface exits
  // the driving mode cleanly back to the laptop layout.
  function onFsChange() {
    if (!fsElement() && document.body.dataset.driveFocus === "phone") {
      selectMode("laptop", true);
    }
  }
  document.addEventListener("fullscreenchange", onFsChange);
  document.addEventListener("webkitfullscreenchange", onFsChange);

  // -------------------------------------------------------------- public API
  window.DriveMode = {
    init: build,
    mode: function () { return mode; },
    reset: reset,
    // Only steer while in mobile mode; null lets controller.js skip the analog
    // contribution entirely in laptop mode (keyboard/D-pad stay authoritative).
    axes: function () { return mode === "mobile" ? axes : null; },
  };

  // Apply the resolved layout BEFORE first paint to avoid a D-pad -> joystick
  // flash on coarse-pointer devices. This script runs at end-of-<body>, so
  // document.body/head already exist; build() re-applies the same value later.
  if (document.body) {
    injectStyle();
    mode = defaultMode();
    document.body.dataset.driveMode = mode;
    // Pre-paint the tablet/desktop "wide" focus so the obstacle box doesn't flash
    // in before build() runs. Phone focus waits for an explicit tap (needs a gesture).
    if (mode === "mobile" && !isPhoneScreen()) document.body.dataset.driveFocus = "wide";
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
