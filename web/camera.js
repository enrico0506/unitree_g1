// G1 media badges — polls /camera/status and /depth/status and reflects
// liveness on each window. The frames are rendered natively by the <img> tags.

function wireFeed(statusUrl, badgeId, imgId, streamUrl) {
  const badge = document.getElementById(badgeId);
  const img = document.getElementById(imgId);

  async function poll() {
    try {
      const r = await fetch(statusUrl, { cache: "no-store" });
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
  // currently-selected stream (pose.js may have swapped it to the pose feed).
  if (img) {
    img.addEventListener("error", () => {
      setTimeout(() => {
        img.src = (img.dataset.stream || streamUrl) + "?" + Date.now();
      }, 1000);
    });
  }
}

wireFeed("/camera/status", "camBadge", "camFeed", "/camera/stream");
