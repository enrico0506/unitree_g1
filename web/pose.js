// People-skeleton overlay toggle + manual ID labeling.
//
// "Skeleton" turns the pose overlay on/off INDEPENDENTLY of object detection —
// both can be on at once. We don't touch the camera <img>; we poll the pose
// geometry (/camera/pose/tracks) and hand it to CamOverlay, which draws the
// skeletons on the shared canvas. Polling also heartbeats the pose container so
// its GPU only runs while the overlay is on.
//
// The "People" bar lists tracked IDs; typing a name labels that skeleton
// (POST /camera/pose/label) until the track is lost.
//
// Loaded after cam-overlay.js (the canvas compositor).

(function () {
  const toggle = document.getElementById("poseToggle");
  const peopleBar = document.getElementById("peopleBar");
  const peopleList = document.getElementById("peopleList");
  const poseBadge = document.getElementById("poseBadge");
  if (!toggle || !peopleBar || !window.CamOverlay) return;

  let poseOn = false;
  let tracksTimer = null;
  let statusTimer = null;
  const rows = new Map(); // id -> { row, input }

  function postLabel(id, name) {
    fetch("/camera/pose/label", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: id, name: name }),
    }).catch(() => {});
  }

  function renderTracks(items) {
    const empty = peopleList.querySelector(".people-empty");
    const seen = new Set();
    // Only people with a stable track id can be named (the bar is the naming UI);
    // untracked skeletons still draw on the canvas, they just aren't listed here.
    const named = items.filter((t) => t.id != null);
    for (const t of named) {
      seen.add(t.id);
      let r = rows.get(t.id);
      if (!r) {
        const row = document.createElement("div");
        row.className = "people-row";
        const tag = document.createElement("span");
        tag.className = "people-id";
        tag.textContent = "#" + t.id;
        const input = document.createElement("input");
        input.type = "text";
        input.maxLength = 32;
        input.placeholder = "name…";
        input.autocomplete = "off";
        input.value = t.name || "";
        input.addEventListener("change", () => postLabel(t.id, input.value.trim()));
        input.addEventListener("keydown", (e) => { if (e.key === "Enter") input.blur(); });
        row.appendChild(tag);
        row.appendChild(input);
        peopleList.appendChild(row);
        r = { row: row, input: input };
        rows.set(t.id, r);
      } else if (document.activeElement !== r.input) {
        r.input.value = t.name || "";   // reflect server state, but never while typing
      }
    }
    for (const [id, r] of rows) {
      if (!seen.has(id)) { r.row.remove(); rows.delete(id); }
    }
    if (empty) empty.style.display = named.length ? "none" : "";
  }

  async function pollTracks() {
    try {
      const data = await (await fetch("/camera/pose/tracks", { cache: "no-store" })).json();
      window.CamOverlay.setPoseData(data);
      renderTracks(data.items || []);
    } catch (e) { /* pose service may be down; ignore */ }
  }

  async function pollStatus() {
    try {
      const s = await (await fetch("/camera/pose/status", { cache: "no-store" })).json();
      if (poseBadge) {
        poseBadge.textContent = s.live ? "● live" : "○ no signal";
        poseBadge.classList.toggle("live", !!s.live);
      }
    } catch (e) {
      if (poseBadge) { poseBadge.textContent = "○ offline"; poseBadge.classList.remove("live"); }
    }
  }

  function applyPose(on) {
    poseOn = on;
    toggle.classList.toggle("active", on);
    peopleBar.hidden = !on;
    window.CamOverlay.setPose(on);
    if (on) {
      pollTracks(); pollStatus();
      tracksTimer = setInterval(pollTracks, 80);   // ~12 Hz, matches the pose service
      statusTimer = setInterval(pollStatus, 3000);
      if (window.HandsOverlay) window.HandsOverlay.start();   // fingers ride the Skeleton toggle
    } else {
      clearInterval(tracksTimer); tracksTimer = null;
      clearInterval(statusTimer); statusTimer = null;
      if (window.HandsOverlay) window.HandsOverlay.stop();
    }
  }

  toggle.addEventListener("click", () => applyPose(!poseOn));
})();
