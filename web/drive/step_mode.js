// Step-size selector UI module.
//
// Builds a self-contained "Step Size" panel inside the empty <div id="step-panel">
// that index.html provides, mirroring obstacle.js. Exposes
// window.StepMode = {init(send), handle(msg)}:
//
//   init(send)  -- controller.js calls this once after the WS opens, passing its
//                  JSON send() helper. The segmented buttons use it to emit
//                  {type:"step_mode", name:"continuous"|"medium"|"small"}.
//   handle(msg) -- controller.js calls this for every {type:"telemetry"} message
//                  (step_mode -> highlight, step_estimated -> measured/estimated chip)
//                  AND once on connect with the "config" message's .step sub-object
//                  (sets the selected button so "Normal" is on at load).
//
// THE KEY UX: the operator picks a step SIZE here (one mutually-exclusive group), and
// it governs WHICHEVER arrow / WASD they then hold -- forward, back, strafe L/R,
// rotate L/R. They do NOT pick a direction; one size selector, all six directions.
//
// Loaded BEFORE controller.js, so we define window.StepMode at parse time (controller
// wires it once it has a live socket). Plain DOM, no framework. CSS namespaced "step-",
// reusing the obs-btn visual conventions (CSS vars --live etc) so it matches the panel.

(function () {
  "use strict";

  let send = function () {};        // controller.js hands us the real one in init()
  let booted = false;               // panel built yet?
  let current = "continuous";       // server-authoritative selected mode

  const el = {};

  // The three sizes, left-to-right. WS name + operator label + helper line.
  const MODES = [
    { name: "continuous", label: "Normal",
      help: "Continuous walking (default)." },
    { name: "medium", label: "Medium",
      help: "Discrete medium steps in every direction." },
    { name: "small", label: "Small",
      help: "Smallest discrete steps for tight maneuvering." },
  ];

  // ----------------------------------------------------------------- styles
  const CSS = `
  /* Vertical selector, stacked down the right edge of the Drive box. */
  #step-panel { display: flex; flex-direction: column; gap: 8px; max-width: 170px; }
  .step-row { display: flex; flex-direction: column; align-items: stretch; gap: 6px; }
  .step-title {
    font-size: 10px; text-transform: uppercase; letter-spacing: .14em;
    color: var(--steel, #8A93A6);
  }
  .step-seg { display: flex; flex-direction: column; gap: 0; border-radius: 8px;
    overflow: hidden; border: 1px solid var(--line, #232C3E); }
  .step-btn {
    font-family: inherit; font-size: 13px; font-weight: 500; text-align: center;
    color: var(--chalk, #E9EDF6);
    background: var(--surface-2, #161C2A);
    border: 0; border-bottom: 1px solid var(--line, #232C3E);
    padding: 9px 16px; cursor: pointer;
    transition: color .15s, background .15s;
  }
  .step-btn:last-child { border-bottom: 0; }
  .step-btn:hover { color: #fff; }
  .step-btn.on {
    color: #06281C; background: var(--live, #3FD39B);
  }
  .step-help { font-size: 11.5px; line-height: 1.4; color: var(--steel, #8A93A6); }
  .step-chip {
    display: none; padding: 2px 9px; border-radius: 999px;
    font-size: 10px; font-weight: 700; letter-spacing: .06em;
    border: 1px solid var(--line, #232C3E); color: var(--steel, #8A93A6);
    background: var(--inset, #0C1019);
  }
  .step-chip.show { display: inline-block; }
  .step-chip.measured { color: #06281C; background: var(--live, #3FD39B);
    border-color: var(--live, #3FD39B); }
  .step-chip.estimated { color: var(--caution, #FF8636);
    border-color: var(--caution, #FF8636); }
  `;

  function injectStyle() {
    if (document.getElementById("step-style")) return;
    const s = document.createElement("style");
    s.id = "step-style";
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  // ------------------------------------------------------------------- build
  function build() {
    const root = document.getElementById("step-panel");
    if (!root || booted) return;
    booted = true;
    injectStyle();
    root.innerHTML = "";

    // Segmented selector row.
    const row = document.createElement("div");
    row.className = "step-row";

    const title = document.createElement("span");
    title.className = "step-title";
    title.textContent = "Step size";
    row.appendChild(title);

    const seg = document.createElement("div");
    seg.className = "step-seg";
    el.btns = {};
    MODES.forEach((m) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "step-btn";
      b.textContent = m.label;
      b.dataset.mode = m.name;
      b.addEventListener("click", () => select(m.name));
      el.btns[m.name] = b;
      seg.appendChild(b);
    });
    row.appendChild(seg);

    // (The odom measured/estimated chip is intentionally not rendered -- setChip()
    // no-ops without el.chip, so the "odom: measured/estimated" text never shows.)

    root.appendChild(row);

    // (No per-mode helper line -- setSelected() no-ops without el.help.)

    setSelected(current);
  }

  // -------------------------------------------------------- selection / render
  function select(name) {
    // Optimistic highlight; the server confirm (step_mode / telemetry) is authoritative.
    setSelected(name);
    send({ type: "step_mode", name: name });
  }

  function setSelected(name) {
    if (!MODES.some((m) => m.name === name)) name = "continuous";
    current = name;
    if (!el.btns) return;
    MODES.forEach((m) => {
      el.btns[m.name].classList.toggle("on", m.name === name);
    });
    const m = MODES.find((x) => x.name === name);
    if (el.help) el.help.textContent = m ? m.help : "";
  }

  // estimated/measured chip: only shown when the server reports a step_estimated flag
  // (distance_quantize on). measured = closed-loop on odom; estimated = fell back to time.
  function setChip(estimated, hasFlag) {
    if (!el.chip) return;
    // Show only in a discrete mode AND when the server actually reports the flag.
    if (!hasFlag || current === "continuous") {
      el.chip.classList.remove("show");
      return;
    }
    el.chip.classList.add("show");
    el.chip.classList.toggle("estimated", !!estimated);
    el.chip.classList.toggle("measured", !estimated);
    el.chip.textContent = estimated ? "odom: estimated" : "odom: measured";
  }

  // -------------------------------------------------------------- public API
  window.StepMode = {
    init: function (sendFn) {
      if (typeof sendFn === "function") send = sendFn;
      build();
    },
    handle: function (msg) {
      if (!msg || typeof msg !== "object") return;
      try {
        if (!booted) build();
        // Accept both the connect 'config' .step sub-object {mode,...} and the
        // confirm/telemetry messages {name|step_mode,...}; server is authoritative.
        const mode = msg.mode || msg.name || msg.step_mode;
        if (mode) setSelected(mode);
        const hasFlag = Object.prototype.hasOwnProperty.call(msg, "step_estimated");
        setChip(msg.step_estimated, hasFlag);
      } catch (e) { /* defensive: never break the WS loop */ }
    },
  };

  // Build as soon as the container exists (init() also calls build(), idempotently).
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
