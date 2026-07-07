// Feed-mode switcher for the top-right LiDAR window.
//
// The window can show one of three things, full-size, chosen by a small toggle:
//   cloud  -> live 3D feed: the FAST-LIO map while mapping/loaded, else the 3D
//             obstacle sphere (obstacle3d.js) re-parented into #cloudSphere
//   radar  -> the 2D obstacle ring (obstacle.js drawn big into #bigRadar)
//   sphere -> the 3D obstacle sphere (obstacle3d.js renderer re-parented to #bigSphere)
//
// Two client inputs drive the window: the feed toggle (cloud/radar/sphere) and
// mapShown, mirrored from map_status (active OR points>0) by controller.js. When
// feed=cloud and NOT mapShown the window shows the obstacle sphere (idle look,
// identical to the 3D sphere view — same single renderer, same telemetry); when
// mapShown it shows the FAST-LIO map (lidar.js /ws/lidar). We never stand up a
// second WebGL context: the sphere's single canvas is re-parented between the big
// mount (#bigSphere) and the cloud-idle mount (#cloudSphere); when neither wants
// it, it's left parked and hidden by CSS. The 2D radar just draws bigger.
//
// Plain classic script, no framework. Loaded after obstacle.js (classic) and the
// modules; wires up on DOMContentLoaded when window.Obstacle / Obstacle3D exist.

(function () {
  "use strict";

  // Module-level so setMapShown (from a map_status) and setMode (from a click)
  // both feed the one idempotent applyLayers(); last event wins, no split-brain.
  let feed = "cloud";
  let mapShown = false;

  function init() {
    const win = document.querySelector(".lidar-window");
    if (!win) return;
    const btns = win.querySelectorAll("[data-feed-btn]");
    const bigRadar = document.getElementById("bigRadar");
    const bigSphere = document.getElementById("bigSphere");
    const cloudSphere = document.getElementById("cloudSphere");
    if (!btns.length) return;

    // Recompute the window's data-* attributes, the 2D radar canvas, and where the
    // single 3D-sphere renderer is mounted, from (feed, mapShown). Idempotent.
    function applyLayers() {
      win.setAttribute("data-feed", feed);
      win.setAttribute("data-cloud", mapShown ? "map" : "sphere");
      btns.forEach((b) => b.classList.toggle("active", b.dataset.feedBtn === feed));

      // 2D radar: hand obstacle.js the big canvas only while it's the active view.
      if (window.Obstacle && window.Obstacle.setBigRadar) {
        window.Obstacle.setBigRadar(feed === "radar" ? bigRadar : null);
      }
      // 3D sphere: the single renderer goes to the big mount for the sphere view,
      // to the cloud-idle mount when cloud is showing the sphere, else stays put
      // (parked + hidden by CSS). mountTo(null) leaves the canvas where it is.
      const target =
        feed === "sphere" ? bigSphere :
        (feed === "cloud" && !mapShown) ? cloudSphere :
        null;
      if (target && window.Obstacle3D && window.Obstacle3D.mountTo) {
        window.Obstacle3D.mountTo(target);
      }
      // Let three.js / the radar resize to their new boxes.
      window.dispatchEvent(new Event("resize"));
    }

    function setMode(mode) { feed = mode; applyLayers(); }
    function setMapShown(b) { mapShown = !!b; applyLayers(); }

    btns.forEach((b) =>
      b.addEventListener("click", () => setMode(b.dataset.feedBtn)));

    // Exposed so controller.js's updateMapStatus() can flip the cloud window
    // between the idle sphere and the FAST-LIO map as mapping starts/stops/loads.
    window.FeedView = { setMode: setMode, setMapShown: setMapShown };

    setMode("cloud");   // default: cloud feed, sphere mounted into #cloudSphere
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
