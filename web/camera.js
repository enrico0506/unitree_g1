// G1 media badges — polls /camera/status and /depth/status and reflects
// liveness on each window. The frames are rendered natively by the <img> tags.

function wireFeed(statusUrl, badgeId, imgId, streamUrl) {
  const badge = document.getElementById(badgeId);
  const img = document.getElementById(imgId);
  // statusUrl is mutable via .current so a toggle (e.g. Infrared) can redirect
  // this same poller instead of stacking a second setInterval on top of it.
  const urls = { current: statusUrl };

  async function poll() {
    try {
      const r = await fetch(urls.current, { cache: "no-store" });
      const s = await r.json();
      if (badge) {
        badge.textContent = s.live ? `${s.backend} ●` : "no signal ○";
        badge.classList.toggle("live", !!s.live);
      }
      if (img) img.classList.toggle("offline", !s.live);
      const info = document.getElementById("infoCam");
      if (info) info.textContent = s.live ? `${s.backend} · live` : "no signal";
    } catch (e) {
      if (badge) { badge.textContent = "offline ○"; badge.classList.remove("live"); }
    }
  }
  poll();
  setInterval(poll, 3000);

  // If the MJPEG stream drops, nudge the <img> to reconnect. Reconnect to the
  // currently-selected stream (the Infrared toggle may have swapped it).
  if (img) {
    img.addEventListener("error", () => {
      setTimeout(() => {
        img.src = (img.dataset.stream || streamUrl) + "?" + Date.now();
      }, 1000);
    });
  }
  return urls;
}

const camFeedUrls = wireFeed("/camera/status", "camBadge", "camFeed", "/camera/stream");

// Infrared toggle -- swaps the <img> between the RGB stream (videohub) and the
// D435i's wider-vertical-FOV infrared/stereo node (/dev/video2), so the two
// can be visually compared. Purely a source swap; skeleton/detect overlays
// keep drawing on the canvas on top of whichever feed is showing.
(function () {
  const btn = document.getElementById("irToggle");
  const img = document.getElementById("camFeed");
  if (!btn || !img) return;

  let irOn = false;

  btn.addEventListener("click", () => {
    irOn = !irOn;
    btn.classList.toggle("active", irOn);
    img.dataset.stream = irOn ? "/camera/ir/stream" : "/camera/stream";
    img.src = img.dataset.stream + "?" + Date.now();
    camFeedUrls.current = irOn ? "/camera/ir/status" : "/camera/status";
    // ir_service.py (the D435i's third video stream) only actually runs while
    // someone has this on -- tell the server now (see robot_web_controller.py
    // mtype "ir_toggle"). Without this it'd just show "no signal" forever.
    if (window.send) window.send({ type: "ir_toggle", on: irOn });
  });
})();
