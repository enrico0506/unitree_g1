// Motion tab — record → pause → recreate against the Phase-1 STUB replay provider.
//
// One clean camera feed (bare <img src="/camera/stream">, NO canvas/overlay) plus a
// single morphing primary button that walks the job state machine:
//
//   idle/ready  → "● Record"      POST /motion/record/start
//   recording   → "❚❚ Pause"      POST /motion/record/stop
//   recorded    → "Recreate"      POST /motion/jobs/{id}/recreate
//   processing  → "Processing…"   (disabled; a Pose→GMR→SONIC stepper advances)
//
// Reads/writes only /motion/* and /camera/status. It never re-implements the control
// lock: controller.js owns arbitration and tells us who holds it via a `g1:control`
// CustomEvent (detail.isOwner) and by publishing the page's id on window.G1_CLIENT_ID.
// The Motion tab reuses that SAME id over HTTP as the `X-Client-Id` header on every
// gated POST, so the server gates record/recreate to the current controller.
//
// Plain classic script, no framework (mirrors camera.js / feedview.js). Loaded from
// index.html; wires up on DOMContentLoaded. A no-op if the Motion tab isn't present.

(function () {
  "use strict";

  const POLL_JOBS_MS = 1500;   // refresh the takes list (open GET, cheap)
  const POLL_STATUS_MS = 700;  // fast poll while the active job is RECORDING/PROCESSING
  const CAM_POLL_MS = 3000;    // camera liveness badge
  const STAGES = ["POSE", "GMR", "SONIC"];   // fixed stepper order

  // Button label per data-state (the vocab the contract pins on #motionPrimary).
  const LABELS = {
    idle: "● Record",
    recording: "❚❚ Pause",
    recorded: "Recreate",
    processing: "Processing…",
    ready: "● Record",
  };

  // ---- DOM handles (resolved in init) ------------------------------------
  let feed, camBadge, primary, stepper, takesEl, resultEl, origVid, replayVid, hintEl;

  // ---- state -------------------------------------------------------------
  let isOwner = false;       // last value from the g1:control event
  let activeJobId = null;    // job the primary button acts on (record/recreate target)
  let viewingJobId = null;   // job shown in the result view (for row highlight only)
  let activeState = "idle";  // current #motionPrimary data-state
  let busy = false;          // in-flight POST guard (blocks double submit)
  let statusTimer = null;    // fast per-job status poll
  let jobsTimer = null;      // takes-list poll
  let errorUntil = 0;        // suppress the generic hint while a fresh error is shown

  function $(id) { return document.getElementById(id); }

  function init() {
    feed = $("motionFeed");
    camBadge = $("motionCamBadge");
    primary = $("motionPrimary");
    stepper = $("motionStepper");
    takesEl = $("motionTakes");
    resultEl = $("motionResult");
    origVid = $("motionOriginal");
    replayVid = $("motionReplay");
    hintEl = $("motionHint");
    if (!primary) return;   // Motion tab markup not on the page -> nothing to do.

    primary.addEventListener("click", onPrimaryClick);

    // Enable/disable in lock-step with the control chip. controller.js dispatches
    // this on every control update; our listener is registered before the first
    // (async) WS message arrives, so we never miss the initial ownership signal.
    window.addEventListener("g1:control", function (e) {
      isOwner = !!(e && e.detail && e.detail.isOwner);
      refreshPrimary();
    });

    wireCamBadge();

    // Takes list: paint now, then keep it live.
    pollJobs();
    jobsTimer = setInterval(pollJobs, POLL_JOBS_MS);

    setState("idle");
  }

  // ---- primary button ----------------------------------------------------

  function onPrimaryClick() {
    if (busy) return;
    switch (activeState) {
      case "idle":
      case "ready":      startRecord(); break;
      case "recording":  stopRecord();  break;
      case "recorded":   recreate();    break;
      // "processing" is disabled; ignore.
    }
  }

  // Set the button's data-state + label, toggle the stepper, and refresh gating.
  function setState(s) {
    activeState = s;
    primary.dataset.state = s;
    primary.textContent = LABELS[s] || LABELS.idle;
    if (stepper) stepper.hidden = (s !== "processing");
    refreshPrimary();
  }

  // Disabled state + the guidance hint are both derived from (isOwner, state, busy).
  function refreshPrimary() {
    const processing = activeState === "processing";
    const disabled = !isOwner || processing || busy;
    primary.disabled = disabled;
    primary.classList.toggle("disabled", disabled);

    if (!isOwner) {
      setHint("Take control to record.");
    } else if (processing) {
      setHint("Recreating — Pose → GMR → SONIC…");
    } else if (activeState === "recording") {
      setHint("Recording — press Pause to stop.");
    } else if (activeState === "recorded") {
      setHint("Recorded. Press Recreate to reproduce the motion.");
    } else {
      setHint("Press Record to capture a take.");
    }
  }

  // ---- record / recreate actions ----------------------------------------

  async function startRecord() {
    busy = true; refreshPrimary();
    try {
      const res = await motionPost("/motion/record/start");
      if (!(await ok(res))) return;
      const data = await res.json();          // { id, state: "RECORDING" }
      activeJobId = data.id;
      hideResult();
      resetStepper();
      setState("recording");
      startStatusPoll();
      pollJobs();
    } catch (e) {
      setError("Record failed: " + e);
    } finally {
      busy = false; refreshPrimary();
    }
  }

  async function stopRecord() {
    busy = true; refreshPrimary();
    try {
      const res = await motionPost("/motion/record/stop");
      // The recorder can fail cleanly (e.g. no frames captured off-robot): the route
      // moves the job to ERROR and returns 500 with { id, state:"ERROR", error }.
      if (res.status === 500) {
        let err = "recording failed";
        try { const j = await res.json(); err = j.error || err; } catch (_) {}
        setError("Recording failed: " + err);
        stopStatusPoll();
        activeJobId = null;
        setState("idle");
        pollJobs();
        return;
      }
      if (!(await ok(res))) return;
      const status = await res.json();         // full status.json (state RECORDED)
      stopStatusPoll();
      applyStatus(status);
      pollJobs();
    } catch (e) {
      setError("Stop failed: " + e);
    } finally {
      busy = false; refreshPrimary();
    }
  }

  async function recreate() {
    if (!activeJobId) return;
    busy = true; refreshPrimary();
    try {
      const res = await motionPost("/motion/jobs/" + encodeURIComponent(activeJobId) + "/recreate");
      if (!(await ok(res))) return;
      // { id, state: "PROCESSING" } — the daemon thread advances the stages server-side.
      resetStepper();
      setState("processing");
      startStatusPoll();
      pollJobs();
    } catch (e) {
      setError("Recreate failed: " + e);
    } finally {
      busy = false; refreshPrimary();
    }
  }

  // ---- status polling / reducer -----------------------------------------

  function startStatusPoll() {
    stopStatusPoll();
    statusTimer = setInterval(pollStatus, POLL_STATUS_MS);
    pollStatus();
  }
  function stopStatusPoll() {
    if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
  }

  async function pollStatus() {
    if (!activeJobId) { stopStatusPoll(); return; }
    try {
      const res = await fetch("/motion/jobs/" + encodeURIComponent(activeJobId) + "/status",
                              { cache: "no-store" });
      if (!res.ok) return;                    // 404 can be transient right after start
      applyStatus(await res.json());
    } catch (_) { /* transient network error — next tick retries */ }
  }

  // Single reducer: drive the button + stepper + result view from a status.json body.
  function applyStatus(status) {
    if (!status || !status.state) return;

    if (status.processing && Array.isArray(status.processing.stages)) {
      renderStepper(status.processing.stages);
    }

    switch (status.state) {
      case "RECORDING":
        setState("recording");
        break;
      case "RECORDED":
        stopStatusPoll();
        setState("recorded");
        break;
      case "PROCESSING":
        setState("processing");
        if (status.processing && status.processing.stage) {
          setHint("Recreating — " + prettyStage(status.processing.stage) + "…");
        }
        break;
      case "READY":
        stopStatusPoll();
        setState("ready");
        showResult(activeJobId);
        break;
      case "ERROR":
        stopStatusPoll();
        setError(status.error || "job error");
        // Keep a recorded clip so the user can retry Recreate; otherwise start fresh.
        setState(status.artifacts && status.artifacts.clip ? "recorded" : "idle");
        break;
      default: // IDLE
        setState("idle");
    }
  }

  // ---- stepper -----------------------------------------------------------

  function renderStepper(stages) {
    if (!stepper) return;
    const byName = {};
    stages.forEach(function (s) { byName[s.name] = s.status; });
    STAGES.forEach(function (name) {
      const el = stepper.querySelector('.motion-step[data-step="' + name + '"]');
      if (!el) return;
      const s = byName[name] || "pending";
      el.classList.toggle("active", s === "active");
      el.classList.toggle("done", s === "done");
      el.classList.toggle("error", s === "error");
    });
  }

  function resetStepper() {
    if (!stepper) return;
    stepper.querySelectorAll(".motion-step").forEach(function (el) {
      el.classList.remove("active", "done", "error");
    });
  }

  function prettyStage(name) {
    return name === "POSE" ? "Pose" : name;   // GMR / SONIC stay upper-case
  }

  // ---- takes list --------------------------------------------------------

  async function pollJobs() {
    try {
      const res = await fetch("/motion/jobs", { cache: "no-store" });
      if (!res.ok) return;
      const data = await res.json();
      renderTakes((data && data.jobs) || []);
    } catch (_) { /* router may not be mounted — leave the placeholder as-is */ }
  }

  function renderTakes(jobs) {
    if (!takesEl) return;
    if (!jobs.length) {
      takesEl.innerHTML = '<span class="people-empty">no takes yet</span>';
      return;
    }
    const frag = document.createDocumentFragment();
    jobs.forEach(function (j) { frag.appendChild(takeRow(j)); });
    takesEl.replaceChildren(frag);
  }

  function takeRow(j) {
    const row = document.createElement("div");
    row.className = "motion-take";
    row.dataset.job = j.id;
    if (j.id === activeJobId) row.classList.add("current");
    if (j.id === viewingJobId) row.classList.add("viewing");
    // Selecting a row makes it the primary button's target (resume Recreate / re-view).
    row.addEventListener("click", function () { selectTake(j.id); });

    const id = document.createElement("span");
    id.className = "motion-take-id";
    id.textContent = shortId(j.id);

    const state = document.createElement("span");
    state.className = "motion-take-state";
    state.dataset.state = String(j.state || "").toLowerCase();
    state.textContent = j.state || "?";

    const fps = document.createElement("span");
    fps.className = "motion-take-fps";
    fps.textContent = (typeof j.measured_fps === "number")
      ? j.measured_fps.toFixed(1) + " fps" : "—";

    row.appendChild(id);
    row.appendChild(state);
    row.appendChild(fps);

    if (j.has_replay) {
      const view = document.createElement("button");
      view.className = "motion-take-view";
      view.type = "button";
      view.textContent = "View";
      view.addEventListener("click", function (e) {
        e.stopPropagation();               // view-only: don't retarget the primary button
        showResult(j.id);
      });
      row.appendChild(view);
    }
    return row;
  }

  // Adopt an existing take as the active job and sync the button to its live state.
  async function selectTake(jobId) {
    activeJobId = jobId;
    try {
      const res = await fetch("/motion/jobs/" + encodeURIComponent(jobId) + "/status",
                              { cache: "no-store" });
      if (!res.ok) return;
      const status = await res.json();
      applyStatus(status);
      if (status.state === "RECORDING" || status.state === "PROCESSING") startStatusPoll();
    } catch (_) { /* ignore — pollJobs will re-render shortly */ }
    pollJobs();   // repaint highlights
  }

  // Human-friendly label from a "YYYYMMDD-HHMMSS-hex" job id.
  function shortId(id) {
    const m = /^\d{8}-(\d{2})(\d{2})(\d{2})-(\w+)$/.exec(id || "");
    return m ? (m[1] + ":" + m[2] + ":" + m[3] + " · " + m[4]) : (id || "?");
  }

  // ---- result view -------------------------------------------------------

  // Phase-1 stub: the "replay" IS the original clip (StubProvider copies clip→replay),
  // and only replay.mp4 is a served route, so both panes point at the same source.
  function showResult(jobId) {
    if (!jobId || !resultEl) return;
    const url = "/motion/jobs/" + encodeURIComponent(jobId) + "/replay.mp4";
    if (origVid && origVid.getAttribute("src") !== url) origVid.src = url;
    if (replayVid && replayVid.getAttribute("src") !== url) replayVid.src = url;
    resultEl.hidden = false;
    viewingJobId = jobId;
  }

  function hideResult() {
    if (!resultEl) return;
    resultEl.hidden = true;
    viewingJobId = null;
    if (origVid) { origVid.removeAttribute("src"); origVid.load(); }
    if (replayVid) { replayVid.removeAttribute("src"); replayVid.load(); }
  }

  // ---- camera badge + MJPEG reconnect (mirrors camera.js wireFeed) --------

  function wireCamBadge() {
    async function poll() {
      try {
        const r = await fetch("/camera/status", { cache: "no-store" });
        const s = await r.json();
        if (camBadge) {
          camBadge.textContent = s.live ? (s.backend + " ●") : "no signal ○";
          camBadge.classList.toggle("live", !!s.live);
        }
        if (feed) feed.classList.toggle("offline", !s.live);
      } catch (_) {
        if (camBadge) { camBadge.textContent = "offline ○"; camBadge.classList.remove("live"); }
      }
    }
    poll();
    setInterval(poll, CAM_POLL_MS);

    // If the MJPEG stream drops (e.g. no camera on the workstation), nudge <img> back.
    if (feed) {
      feed.addEventListener("error", function () {
        setTimeout(function () { feed.src = "/camera/stream?" + Date.now(); }, 1000);
      });
    }
  }

  // ---- fetch helpers -----------------------------------------------------

  // POST to a /motion route carrying the page's control-lock id as a header (never a
  // query param, so the id never lands in a URL). Read the id LAZILY: controller.js
  // publishes it once the WS control message lands, which is well before any click.
  function motionPost(path) {
    return fetch(path, {
      method: "POST",
      headers: { "X-Client-Id": window.G1_CLIENT_ID || "" },
      cache: "no-store",
    });
  }

  // Translate a POST response into "proceed?"; surfaces gate/error messages in the hint.
  async function ok(res) {
    if (res.ok) return true;
    if (res.status === 403) {
      setError("You don't hold control — tap the control chip.");
      return false;
    }
    if (res.status === 409) {
      setError(await detailOf(res, "Busy — can't do that right now."));
      return false;
    }
    // 404 / 500 / anything else.
    setError(await detailOf(res, "Request failed (" + res.status + ")."));
    return false;
  }

  // Pull a message out of a FastAPI error body ({detail:…}) or our own ({error:…}).
  async function detailOf(res, fallback) {
    try {
      const j = await res.json();
      return (j && (j.detail || j.error)) || fallback;
    } catch (_) {
      return fallback;
    }
  }

  // ---- hint (transient errors win for a few seconds) ----------------------

  function setError(msg) {
    if (!hintEl) return;
    errorUntil = Date.now() + 6000;
    hintEl.textContent = msg;
    hintEl.classList.add("warn");
  }

  function setHint(msg) {
    if (!hintEl) return;
    if (Date.now() < errorUntil) return;   // keep a fresh error visible
    hintEl.textContent = msg;
    hintEl.classList.remove("warn");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
