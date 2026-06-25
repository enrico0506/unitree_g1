// Feed-mode switcher for the top-right LiDAR window.
//
// The window can show one of three things, full-size, chosen by a small toggle:
//   cloud  -> the live 3D LiDAR point cloud (lidar.js, default)
//   radar  -> the 2D obstacle ring (obstacle.js drawn big into #bigRadar)
//   sphere -> the 3D obstacle sphere (obstacle3d.js renderer re-parented to #bigSphere)
//
// We don't create second renderers: the 2D radar just draws into a bigger canvas,
// and the 3D sphere's single WebGL canvas is moved between the small panel mount
// (#obs-sphere) and the big mount (#bigSphere). The cloud-only overlays (nav pad,
// view bar, map bar, legend) are hidden via the window's data-feed attribute (CSS).
//
// Plain classic script, no framework. Loaded after obstacle.js (classic) and the
// modules; wires up on DOMContentLoaded when window.Obstacle / Obstacle3D exist.

(function () {
  "use strict";

  function init() {
    const win = document.querySelector(".lidar-window");
    if (!win) return;
    const btns = win.querySelectorAll("[data-feed-btn]");
    const bigRadar = document.getElementById("bigRadar");
    const bigSphere = document.getElementById("bigSphere");
    const smallSphere = document.getElementById("obs-sphere");
    if (!btns.length) return;

    function setMode(mode) {
      win.setAttribute("data-feed", mode);
      btns.forEach((b) => b.classList.toggle("active", b.dataset.feedBtn === mode));

      // 2D radar: hand obstacle.js the big canvas only while it's the active view.
      if (window.Obstacle && window.Obstacle.setBigRadar) {
        window.Obstacle.setBigRadar(mode === "radar" ? bigRadar : null);
      }
      // 3D sphere: move the single renderer to the big mount, else back to small.
      if (window.Obstacle3D && window.Obstacle3D.mountTo) {
        window.Obstacle3D.mountTo(mode === "sphere" ? bigSphere : smallSphere);
      }
      // Let three.js / the radar resize to their new boxes.
      window.dispatchEvent(new Event("resize"));
    }

    btns.forEach((b) =>
      b.addEventListener("click", () => setMode(b.dataset.feedBtn)));

    setMode("cloud");   // default: live cloud, sphere parked in the small panel
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
