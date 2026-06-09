// Canvas compositor for the camera overlays.
//
// The camera <img> (#camFeed) always shows the RAW feed. Skeletons (pose) and
// object-detection boxes are drawn as vectors on a <canvas> (#camCanvas) sitting
// exactly over the image, from geometry JSON the two services publish. Because
// the overlays are vector layers — not baked into the video — BOTH can be shown
// at the same time; each toggle just flips whether its layer is drawn.
//
// pose.js / detect.js feed data in via window.CamOverlay.{setPose,setPoseData,
// setDetect,setDetectData}. Must load AFTER camera.js and BEFORE pose.js/detect.js.

(function () {
  const camFeed = document.getElementById("camFeed");
  const canvas = document.getElementById("camCanvas");
  if (!camFeed || !canvas) return;
  const ctx = canvas.getContext("2d");

  // camera.js reconnects the <img> using dataset.stream; keep it on the raw feed.
  camFeed.dataset.stream = "/camera/stream";

  let poseOn = false, detectOn = false;
  let poseData = null, detectData = null;   // each: { w, h, items:[...] }

  // COCO-17 skeleton bones (index pairs into the keypoint array).
  const BONES = [
    [5, 7], [7, 9], [6, 8], [8, 10],          // arms
    [11, 13], [13, 15], [12, 14], [14, 16],   // legs
    [5, 6], [11, 12], [5, 11], [6, 12],       // torso
    [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6],  // head
  ];
  const KP_MIN = 0.3;
  const POSE_COLOR = "#39d98a";
  const DETECT_COLOR = "#ffb74d";

  // Map source-frame pixel (px,py) -> canvas CSS px, honouring the <img>'s
  // object-fit:contain letterboxing. Returns null if no frame size is known.
  function projector(data, cw, ch) {
    const w = (data && data.w) || 0, h = (data && data.h) || 0;
    if (!w || !h) return null;
    const scale = Math.min(cw / w, ch / h);
    const ox = (cw - w * scale) / 2, oy = (ch - h * scale) / 2;
    return (px, py) => [ox + px * scale, oy + py * scale];
  }

  function drawDetect(data, cw, ch) {
    const P = projector(data, cw, ch);
    if (!P) return;
    ctx.lineWidth = 2;
    ctx.strokeStyle = DETECT_COLOR;
    ctx.font = "13px ui-monospace, Menlo, monospace";
    ctx.textBaseline = "bottom";
    for (const d of data.items || []) {
      if (!d.box) continue;
      const [ax, ay] = P(d.box[0], d.box[1]);
      const [bx, by] = P(d.box[2], d.box[3]);
      ctx.strokeRect(ax, ay, bx - ax, by - ay);
      const pct = d.conf != null ? " " + Math.round(d.conf * 100) + "%" : "";
      const label = d.cls + pct;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(ax, ay - 15, tw + 8, 15);
      ctx.fillStyle = DETECT_COLOR;
      ctx.fillText(label, ax + 4, ay - 2);
    }
  }

  function drawPose(data, cw, ch) {
    const P = projector(data, cw, ch);
    if (!P) return;
    ctx.font = "13px ui-monospace, Menlo, monospace";
    ctx.textBaseline = "bottom";
    for (const p of data.items || []) {
      const k = p.kpts || [];
      ctx.strokeStyle = POSE_COLOR;
      ctx.lineWidth = 2;
      for (const [a, b] of BONES) {
        if (k[a] && k[b] && k[a][2] > KP_MIN && k[b][2] > KP_MIN) {
          const [ax, ay] = P(k[a][0], k[a][1]);
          const [bx, by] = P(k[b][0], k[b][1]);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
        }
      }
      ctx.fillStyle = POSE_COLOR;
      for (const kp of k) {
        if (kp[2] > KP_MIN) {
          const [x, y] = P(kp[0], kp[1]);
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      // name (or #id) above the person — only when there's something to show
      // (a track id may be absent, but the skeleton above is always drawn).
      const label = p.name || (p.id != null ? "#" + p.id : "");
      if (p.box && label) {
        const [x1, y1] = P(p.box[0], p.box[1]);
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = p.name ? "rgba(57,217,138,0.85)" : "rgba(0,0,0,0.6)";
        ctx.fillRect(x1, y1 - 15, tw + 8, 15);
        ctx.fillStyle = p.name ? "#06281a" : "#fff";
        ctx.fillText(label, x1 + 4, y1 - 2);
      }
    }
  }

  function frame() {
    requestAnimationFrame(frame);
    const cw = camFeed.clientWidth, ch = camFeed.clientHeight;
    if (!cw || !ch) return;

    // Keep the canvas exactly over the <img> (which sits below the title/bar).
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(ch * dpr);
    }
    canvas.style.left = camFeed.offsetLeft + "px";
    canvas.style.top = camFeed.offsetTop + "px";
    canvas.style.width = cw + "px";
    canvas.style.height = ch + "px";

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    if (detectOn && detectData) drawDetect(detectData, cw, ch);
    if (poseOn && poseData) drawPose(poseData, cw, ch);   // skeletons on top

    // On-canvas readout: proves the overlay layer is rendering and shows the live
    // count the browser is receiving (skeleton people / detected objects).
    if (poseOn || detectOn) {
      const parts = [];
      if (poseOn) parts.push("skeleton: " + (poseData && poseData.items ? poseData.items.length : "…"));
      if (detectOn) parts.push("objects: " + (detectData && detectData.items ? detectData.items.length : "…"));
      const txt = parts.join("    ");
      ctx.font = "14px ui-monospace, Menlo, monospace";
      ctx.textBaseline = "top";
      const tw = ctx.measureText(txt).width;
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(8, 8, tw + 14, 24);
      ctx.fillStyle = "#39d98a";
      ctx.fillText(txt, 15, 13);
    }
  }
  requestAnimationFrame(frame);

  window.CamOverlay = {
    setPose(on) { poseOn = !!on; if (!on) poseData = null; },
    setDetect(on) { detectOn = !!on; if (!on) detectData = null; },
    setPoseData(d) { poseData = d; },
    setDetectData(d) { detectData = d; },
  };
})();
