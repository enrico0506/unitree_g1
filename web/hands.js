// Hand-landmark overlay poller.
//
// Hands are NOT a separate toggle: they ride the "Skeleton" toggle. pose.js
// calls window.HandsOverlay.start()/stop() from applyPose, so the 21 finger
// landmarks appear exactly when the body skeleton does. We poll the hand
// geometry (/camera/hands/tracks) and hand it to CamOverlay, which draws it on
// the shared canvas. Polling also heartbeats the g1-hands container so its GPU
// only runs while the Skeleton overlay is on.
//
// Loaded after cam-overlay.js (needs window.CamOverlay).

(function () {
  if (!window.CamOverlay) return;

  let handsTimer = null;

  async function pollHands() {
    try {
      const data = await (await fetch("/camera/hands/tracks", { cache: "no-store" })).json();
      window.CamOverlay.setHandsData(data);
    } catch (e) { /* hands service may be down; ignore */ }
  }

  window.HandsOverlay = {
    start() {
      if (handsTimer) return;
      pollHands();
      handsTimer = setInterval(pollHands, 80);   // ~12 Hz, matches the pose poll
    },
    stop() {
      clearInterval(handsTimer); handsTimer = null;
      window.CamOverlay.setHandsData(null);
    },
  };
})();
