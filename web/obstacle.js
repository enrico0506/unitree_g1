// Obstacle-avoidance UI module (CONTRACT 6).
//
// Builds the entire obstacle panel inside the empty <div id="obstacle-panel">
// that index.html provides. Exposes window.Obstacle = {init(send), handle(msg)}:
//
//   init(send)  -- controller.js calls this once after the WS opens, passing its
//                  JSON send() helper. We stash it; the toggle buttons use it to
//                  emit {type:"obstacle", action, on}.
//   handle(msg) -- controller.js calls this for every {type:"obstacle"} telemetry
//                  message, AND once on connect with the "config" message's
//                  .obstacle sub-object (mode may be absent then -> DISABLED).
//
// Loaded BEFORE controller.js, so we must define window.Obstacle at parse time
// (controller.js wires it up once it has a live socket). Plain DOM, no framework,
// no external libs — mirrors detect.js / pose.js conventions. CSS namespaced
// "obs-"; styles injected here so the module is fully self-contained.

(function () {
  "use strict";

  let send = function () {};        // controller.js hands us the real one in init()
  let enabled = false;              // mirrors the guard's operator toggle
  let booted = false;               // panel built yet?
  let lastMsg = null;               // last telemetry, for an on-demand radar redraw

  // --- elements we update from handle() (filled in by build()) -------------
  const el = {};

  // ----------------------------------------------------------------- styles
  const CSS = `
  #obstacle-panel { display: flex; flex-direction: column; gap: 12px; }

  .obs-banner {
    display: none; padding: 9px 12px; border-radius: 8px;
    font-size: 12px; font-weight: 600; letter-spacing: .02em;
    border: 1px solid transparent;
  }
  .obs-banner.show { display: block; }
  .obs-banner.fault {
    color: #fff; background: rgba(255,71,71,0.16);
    border-color: var(--stop, #FF4747);
    animation: obs-pulse 1.1s ease-in-out infinite;
  }
  .obs-banner.autodis {
    color: var(--caution, #FF8636); background: rgba(255,134,54,0.12);
    border-color: var(--caution, #FF8636);
  }
  .obs-banner.starting, .obs-banner.nodata {
    color: var(--steel, #8A93A6); background: rgba(138,147,166,0.08);
    border-color: var(--line, #232C3E); font-weight: 500;
    animation: none;
  }
  @keyframes obs-pulse { 50% { opacity: .55; } }

  .obs-toggles { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .obs-btn {
    font-family: inherit; font-size: 13px; font-weight: 500;
    color: var(--chalk, #E9EDF6);
    background: var(--surface-2, #161C2A);
    border: 1px solid var(--line, #232C3E); border-radius: 8px;
    padding: 8px 14px; cursor: pointer;
    transition: color .15s, background .15s, border-color .15s, opacity .15s;
  }
  .obs-btn:hover { color: #fff; border-color: var(--focus, #5C8DF2); }
  .obs-btn.primary.on {
    color: #06281C; background: var(--live, #3FD39B); border-color: var(--live, #3FD39B);
  }
  .obs-btn.sub { font-size: 12px; padding: 6px 12px; }
  .obs-btn.sub.on {
    color: var(--chalk, #E9EDF6); background: rgba(92,141,242,0.18);
    border-color: var(--focus, #5C8DF2);
  }
  .obs-btn.disabled, .obs-btn:disabled {
    opacity: .4; cursor: not-allowed; pointer-events: none;
  }

  .obs-readouts {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px 16px; align-items: center;
  }
  .obs-row { display: flex; align-items: center; gap: 8px; }
  .obs-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: .14em;
    color: var(--steel, #8A93A6);
  }
  .obs-val {
    font-family: var(--font-data, ui-monospace, monospace);
    font-variant-numeric: tabular-nums;
    font-size: 15px; color: var(--chalk, #E9EDF6);
  }

  .obs-chip {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: .06em;
    border: 1px solid var(--line, #232C3E); color: var(--steel, #8A93A6);
    background: var(--inset, #0C1019);
  }
  .obs-chip.clear { color: #06281C; background: var(--live, #3FD39B); border-color: var(--live, #3FD39B); }
  .obs-chip.slow  { color: #2A1300; background: var(--caution, #FF8636); border-color: var(--caution, #FF8636); }
  .obs-chip.stop  { color: #fff;    background: var(--stop, #FF4747);    border-color: var(--stop, #FF4747); }

  .obs-bar {
    flex: 1; height: 8px; min-width: 80px; border-radius: 999px;
    background: var(--inset, #0C1019); border: 1px solid var(--line, #232C3E);
    overflow: hidden;
  }
  .obs-bar-fill {
    height: 100%; width: 100%; border-radius: 999px;
    background: var(--live, #3FD39B); transition: width .2s ease, background .2s ease;
  }

  .obs-side {
    font-family: var(--font-data, ui-monospace, monospace);
    font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 6px;
    border: 1px solid var(--line, #232C3E); color: var(--steel-dim, #7E879B);
    background: var(--inset, #0C1019);
  }
  .obs-side.hit { color: #2A1300; background: var(--caution, #FF8636); border-color: var(--caution, #FF8636); }

  .obs-gap {
    font-family: var(--font-data, ui-monospace, monospace);
    font-size: 12px; color: var(--steel, #8A93A6);
  }
  .obs-gap b { color: var(--chalk, #E9EDF6); }
  .obs-gap .pass  { color: var(--live, #3FD39B); }
  .obs-gap .block { color: var(--stop, #FF4747); }

  .obs-muted { color: var(--steel-dim, #7E879B); }
  `;

  function injectStyle() {
    if (document.getElementById("obs-style")) return;
    const s = document.createElement("style");
    s.id = "obs-style";
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  // ------------------------------------------------------------------- build
  function build() {
    const root = document.getElementById("obstacle-panel");
    if (!root || booted) return;
    booted = true;
    injectStyle();
    root.innerHTML = "";

    // Banners (only one shown at a time; priority handled in render()).
    el.banner = document.createElement("div");
    el.banner.className = "obs-banner";
    root.appendChild(el.banner);

    // Toggle row.
    const toggles = document.createElement("div");
    toggles.className = "obs-toggles";

    el.primary = makeBtn("obs-btn primary", "Obstacle Guard: Off", () => {
      // Optimistic visual; server telemetry is authoritative and will correct.
      send({ type: "obstacle", action: enabled ? "disable" : "enable" });
    });
    el.gapBtn = makeBtn("obs-btn sub", "Gap-follow steering", () => {
      send({ type: "obstacle", action: "gap_follow", on: !btnOn(el.gapBtn) });
    });
    el.recBtn = makeBtn("obs-btn sub", "Search recovery", () => {
      send({ type: "obstacle", action: "recovery", on: !btnOn(el.recBtn) });
    });
    // Depth fusion is independent of the motion guard (toggle it to VALIDATE the D435i
    // near-ground reading even with the guard off) -- so it is never greyed out.
    el.depthBtn = makeBtn("obs-btn sub", "Depth fusion (D435i)", () => {
      send({ type: "obstacle", action: "depth_fusion", on: !btnOn(el.depthBtn) });
    });

    toggles.appendChild(el.primary);
    toggles.appendChild(el.gapBtn);
    toggles.appendChild(el.recBtn);
    toggles.appendChild(el.depthBtn);
    root.appendChild(toggles);

    // Readouts.
    const ro = document.createElement("div");
    ro.className = "obs-readouts";

    el.front = mkReadout(ro, "front", '<span class="obs-val obs-muted">--</span>');
    el.front = el.front.querySelector(".obs-val");

    // Depth fusion (D435i) near-ground reading -- the validation readout.
    el.depth = mkReadout(ro, "depth (D435i)", '<span class="obs-val obs-muted">off</span>');
    el.depth = el.depth.querySelector(".obs-val");

    // Zone chip.
    const zoneRow = mkRow(ro, "zone");
    el.zone = document.createElement("span");
    el.zone.className = "obs-chip";
    el.zone.textContent = "--";
    zoneRow.appendChild(el.zone);

    // Speed-scale bar.
    const spRow = mkRow(ro, "speed");
    const bar = document.createElement("div");
    bar.className = "obs-bar";
    el.bar = document.createElement("div");
    el.bar.className = "obs-bar-fill";
    bar.appendChild(el.bar);
    spRow.appendChild(bar);
    el.barPct = document.createElement("span");
    el.barPct.className = "obs-val";
    el.barPct.style.fontSize = "12px";
    el.barPct.textContent = "100%";
    spRow.appendChild(el.barPct);

    // L / R side indicators.
    const sideRow = mkRow(ro, "sides");
    el.left = document.createElement("span");
    el.left.className = "obs-side";
    el.left.textContent = "L";
    el.right = document.createElement("span");
    el.right.className = "obs-side";
    el.right.textContent = "R";
    sideRow.appendChild(el.left);
    sideRow.appendChild(el.right);

    root.appendChild(ro);

    // Gap line (full width).
    el.gap = document.createElement("div");
    el.gap.className = "obs-gap obs-muted";
    el.gap.textContent = "gap: --";
    root.appendChild(el.gap);

    // No in-panel visualizations: the 2D ring and 3D sphere live in the top-right
    // LiDAR window's feed toggle (drawn big into #bigRadar / #bigSphere). The panel
    // still feeds those big views via drawRadar() + Obstacle3D.update() in render().

    setEnabledVisual(false);
  }

  // --------------------------------------------------------------- DOM helpers
  function makeBtn(cls, label, onClick) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    b.textContent = label;
    b.addEventListener("click", () => {
      if (b.classList.contains("disabled") || b.disabled) return;
      onClick();
    });
    return b;
  }
  function btnOn(b) { return b.classList.contains("on"); }

  function mkRow(parent, label) {
    const row = document.createElement("div");
    row.className = "obs-row";
    const l = document.createElement("span");
    l.className = "obs-label";
    l.textContent = label;
    row.appendChild(l);
    parent.appendChild(row);
    return row;
  }
  function mkReadout(parent, label, html) {
    const row = mkRow(parent, label);
    const v = document.createElement("span");
    v.innerHTML = html;
    row.appendChild(v);
    return row;
  }

  // ------------------------------------------------------------ enabled visual
  function setEnabledVisual(on) {
    enabled = !!on;
    if (!el.primary) return;
    el.primary.textContent = "Obstacle Guard: " + (on ? "On" : "Off");
    el.primary.classList.toggle("on", on);
    // Sub-toggles only make sense while the guard is enabled.
    el.gapBtn.classList.toggle("disabled", !on);
    el.recBtn.classList.toggle("disabled", !on);
  }

  // -------------------------------------------------------------------- render
  function num(v) { return (typeof v === "number" && isFinite(v)); }

  function fmtDist(v) { return num(v) ? v.toFixed(2) + " m" : "--"; }

  const RADAR_SIZE = 180, RADAR_R = 80;       // CSS px; R leaves a margin
  const Z_STOP = 0.7, Z_SLOW = 2.0;           // thresholds (match lidar.js)
  const RADAR_RANGE_M = 3.0;
  const RC = { stop: "#FF4747", slow: "#FF8636", clear: "#3FD39B",
               grid: "#232C3E", dim: "#7E879B" };

  function radarZoneColor(d) {
    if (d == null) return null;               // empty sector -> not drawn
    if (d < Z_STOP) return RC.stop;
    if (d < Z_SLOW) return RC.slow;
    return RC.clear;
  }

  // Draw the polar radar into a 2D context, centred in a `size`-CSS-px square.
  // Frame-agnostic so the same routine fills the small panel canvas AND the big
  // top-right feed canvas (just a different size).
  function drawRadarGeom(ctx, size, msg) {
    const cx = size / 2, cy = size / 2, R = (size / 2) * 0.9;
    ctx.clearRect(0, 0, size, size);

    // concentric range rings at stop / slow / max.
    ctx.strokeStyle = RC.grid; ctx.lineWidth = 1;
    [Z_STOP, Z_SLOW, RADAR_RANGE_M].forEach((rm) => {
      ctx.beginPath();
      ctx.arc(cx, cy, R * Math.min(1, rm / RADAR_RANGE_M), 0, Math.PI * 2);
      ctx.stroke();
    });
    // forward crosshair (up = +X forward).
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy - R); ctx.stroke();

    const ring = msg && msg.ring;
    if (ring && Array.isArray(ring.dist)) {
      const n = ring.dist.length;
      const binRad = (ring.bin_deg * Math.PI) / 180;
      for (let i = 0; i < n; i++) {
        const d = ring.dist[i];
        const col = radarZoneColor(d);
        if (!col) continue;                   // clear/empty sector
        const cDeg = ring.start_deg + (i + 0.5) * ring.bin_deg;
        const cRad = (cDeg * Math.PI) / 180;
        const r = R * Math.min(1, d / RADAR_RANGE_M);
        const a0 = cRad - binRad / 2, a1 = cRad + binRad / 2;
        // robot frame (x fwd, y left) -> screen (canvas y is DOWN):
        //   screenX = cx - r*sin(angle)  (+left -> screen-left)
        //   screenY = cy - r*cos(angle)  (+fwd  -> screen-up)
        ctx.fillStyle = col; ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx - r * Math.sin(a0), cy - r * Math.cos(a0));
        ctx.lineTo(cx - r * Math.sin(a1), cy - r * Math.cos(a1));
        ctx.closePath();
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    } else {
      ctx.fillStyle = RC.dim;
      ctx.font = Math.max(11, Math.round(size / 16)) + "px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("no ring data", cx, cy);
    }
  }

  // Size a canvas backing store to its CSS box (DPR-aware) and draw the radar,
  // centred within a possibly-wide box. Skips hidden (display:none) canvases.
  function drawRadarInto(canvas, msg) {
    if (!canvas) return;
    const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
    if (!cssW || !cssH) return;               // not laid out / hidden
    const size = Math.min(cssW, cssH);
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const needW = Math.round(cssW * DPR), needH = Math.round(cssH * DPR);
    if (canvas.width !== needW || canvas.height !== needH) {
      canvas.width = needW; canvas.height = needH;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(DPR, 0, 0, DPR,
      ((cssW - size) / 2) * DPR, ((cssH - size) / 2) * DPR);
    drawRadarGeom(ctx, size, msg);
  }

  function drawRadar(msg) {
    // Small in-panel radar was removed; only the big top-right feed draws now (and
    // only while it's the active feed — el.bigRadar is null otherwise).
    if (el.bigRadar) drawRadarInto(el.bigRadar, msg);
  }

  function render(msg) {
    if (!booted) build();
    if (!booted) return;   // panel container not in the DOM (shouldn't happen)

    lastMsg = msg;         // remembered so a feed-toggle can redraw immediately

    const mode = msg.mode || "DISABLED";
    const live = msg.live === undefined ? true : !!msg.live;

    // Operator toggle state. On the connect "config" message mode is absent, so
    // fall back to the explicit enabled / gap_follow / recovery flags.
    const isEnabled = (msg.enabled !== undefined)
      ? !!msg.enabled
      : (mode !== "DISABLED" && mode !== "FAULT");
    setEnabledVisual(isEnabled);

    // Sub-toggle states.
    if (msg.gap_follow !== undefined) el.gapBtn.classList.toggle("on", !!msg.gap_follow);
    if (msg.recovery   !== undefined) el.recBtn.classList.toggle("on", !!msg.recovery);

    // Depth fusion (D435i): toggle state + the validation readout (distance + pts).
    const depth = msg.depth;
    if (depth && el.depthBtn) {
      el.depthBtn.classList.toggle("on", !!depth.enabled);
      if (el.depth) {
        if (!depth.enabled) {
          el.depth.textContent = "off";
          el.depth.classList.add("obs-muted");
        } else if (num(depth.front_near_m)) {
          el.depth.textContent = depth.front_near_m.toFixed(2) + " m ("
            + (depth.n_near || 0) + " pts)";
          el.depth.classList.remove("obs-muted");
        } else {
          el.depth.textContent = (depth.live ? "clear" : "no depth") + " ("
            + (depth.n_near || 0) + " pts)";
          el.depth.classList.add("obs-muted");
        }
      }
    }

    // --- banner priority: FAULT > auto_disabled > no-data > starting > none ---
    const b = el.banner;
    b.className = "obs-banner";
    if (mode === "FAULT" || msg.fault) {
      b.classList.add("show", "fault");
      b.textContent = "⚠ OBSTACLE DETECTION FAILED — forward motion stopped";
    } else if (msg.auto_disabled) {
      b.classList.add("show", "autodis");
      b.textContent = "Guard auto-disabled — re-enable when the sensor is back.";
    } else if (isEnabled && !live) {
      b.classList.add("show", "nodata");
      b.textContent = "No data from obstacle node…";
    } else if (mode === "STARTING") {
      b.classList.add("show", "starting");
      b.textContent = "Starting… waiting for the first LiDAR frame.";
    }

    // --- front distance (prefer filtered; fall back to raw) ---
    const front = num(msg.front_filt_m) ? msg.front_filt_m : msg.front_m;
    el.front.textContent = fmtDist(front);
    el.front.classList.toggle("obs-muted", !num(front));

    // --- zone chip ---
    const zone = (msg.zone || "").toUpperCase();
    el.zone.className = "obs-chip";
    if (zone === "CLEAR")     { el.zone.classList.add("clear"); el.zone.textContent = "CLEAR"; }
    else if (zone === "SLOW") { el.zone.classList.add("slow");  el.zone.textContent = "SLOW"; }
    else if (zone === "STOP") { el.zone.classList.add("stop");  el.zone.textContent = "STOP"; }
    else                      { el.zone.textContent = "--"; }

    // --- speed-scale bar ---
    let scale = num(msg.speed_scale) ? msg.speed_scale : 1.0;
    scale = Math.max(0, Math.min(1, scale));
    const pct = Math.round(scale * 100);
    el.bar.style.width = pct + "%";
    el.bar.style.background = pct >= 99 ? "var(--live, #3FD39B)"
      : pct <= 1 ? "var(--stop, #FF4747)" : "var(--caution, #FF8636)";
    el.barPct.textContent = pct + "%";

    // --- L / R side indicators ---
    const side = msg.side || {};
    el.left.classList.toggle("hit", !!side.left);
    el.right.classList.toggle("hit", !!side.right);

    // --- gap line ---
    const gap = msg.gap || {};
    if (gap.state) {
      const state = String(gap.state).toUpperCase();
      const center = num(gap.center_deg) ? gap.center_deg.toFixed(0) : "--";
      // sign: + = left, - = right (angle = atan2(y,x), y = LEFT).
      const dirTxt = num(gap.center_deg)
        ? (gap.center_deg > 0 ? center + "° L" : center.replace("-", "") + "° R")
        : "--";
      const passable = gap.passable;
      const passHtml = passable === undefined ? ""
        : passable ? ' <span class="pass">PASS</span>'
                   : ' <span class="block">BLOCK</span>';
      el.gap.classList.remove("obs-muted");
      el.gap.innerHTML = "gap: <b>" + state + "</b> @ " + dirTxt + passHtml;
    } else {
      el.gap.classList.add("obs-muted");
      el.gap.textContent = "gap: --";
    }

    drawRadar(msg);
  }

  // -------------------------------------------------------------- public API
  window.Obstacle = {
    init: function (sendFn) {
      if (typeof sendFn === "function") send = sendFn;
      build();
    },
    handle: function (msg) {
      if (!msg || typeof msg !== "object") return;
      try { render(msg); } catch (e) { /* defensive: never break the WS loop */ }
    },
    // Register (or clear with null) a big radar canvas for the top-right feed view.
    // The small panel radar always draws; this mirrors it onto the big canvas and
    // redraws the last frame immediately so the toggle feels instant.
    setBigRadar: function (canvas) {
      el.bigRadar = canvas || null;
      if (el.bigRadar) { try { drawRadarInto(el.bigRadar, lastMsg); } catch (e) {} }
    },
  };

  // Build as soon as the container exists, so the panel is populated even before
  // controller.js connects (init() also calls build(), idempotently).
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
