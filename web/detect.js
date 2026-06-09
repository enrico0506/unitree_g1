// Object-detection overlay toggle (YOLO-World, open-vocabulary).
//
// "Object Detection" turns the box overlay on/off INDEPENDENTLY of the skeleton
// overlay — both can be on at once. We don't touch the camera <img>; we poll the
// detection geometry (/camera/detect/objects) and hand it to CamOverlay, which
// draws the boxes on the shared canvas. Polling also heartbeats the detect
// container so its GPU only runs while the overlay is on.
//
// The "Objects" bar lists detected class names with counts.
// Loaded after cam-overlay.js (the canvas compositor). Mirrors pose.js, minus the
// per-id naming (that's a pose-only feature).

(function () {
  const toggle = document.getElementById("detectToggle");
  const objBar = document.getElementById("objBar");
  const objList = document.getElementById("objList");
  const detectBadge = document.getElementById("detectBadge");
  if (!toggle || !objBar || !window.CamOverlay) return;

  let detectOn = false;
  let objTimer = null;
  let statusTimer = null;

  function renderObjects(items) {
    const counts = {};
    for (const d of items) counts[d.cls] = (counts[d.cls] || 0) + 1;
    const keys = Object.keys(counts).sort();
    objList.innerHTML = "";
    if (!keys.length) {
      const s = document.createElement("span");
      s.className = "people-empty";
      s.textContent = "nothing detected";
      objList.appendChild(s);
      return;
    }
    for (const k of keys) {
      const chip = document.createElement("span");
      chip.className = "obj-chip";
      chip.textContent = `${k} ×${counts[k]}`;
      objList.appendChild(chip);
    }
  }

  async function pollObjects() {
    try {
      const data = await (await fetch("/camera/detect/objects", { cache: "no-store" })).json();
      window.CamOverlay.setDetectData(data);
      renderObjects(data.items || []);
    } catch (e) { /* detect service may be down; ignore */ }
  }

  async function pollStatus() {
    try {
      const s = await (await fetch("/camera/detect/status", { cache: "no-store" })).json();
      if (detectBadge) {
        detectBadge.textContent = s.live ? "● live" : "○ no signal";
        detectBadge.classList.toggle("live", !!s.live);
      }
    } catch (e) {
      if (detectBadge) { detectBadge.textContent = "○ offline"; detectBadge.classList.remove("live"); }
    }
  }

  function applyDetect(on) {
    detectOn = on;
    toggle.classList.toggle("active", on);
    objBar.hidden = !on;
    window.CamOverlay.setDetect(on);
    if (on) {
      pollObjects(); pollStatus();
      objTimer = setInterval(pollObjects, 120);   // responsive box tracking
      statusTimer = setInterval(pollStatus, 3000);
    } else {
      clearInterval(objTimer); objTimer = null;
      clearInterval(statusTimer); statusTimer = null;
    }
  }

  toggle.addEventListener("click", () => applyDetect(!detectOn));
})();
