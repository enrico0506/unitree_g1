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
  let poseData = null, detectData = null, handsData = null;   // each: { w, h, items:[...] }

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
  const HAND_COLOR = "#5ac8fa";   // fingers, a distinct blue vs. the green skeleton

  // MediaPipe 21-landmark hand connections (index pairs into the 21-point list).
  const HAND_BONES = [
    [0, 1], [1, 2], [2, 3], [3, 4],            // thumb
    [0, 5], [5, 6], [6, 7], [7, 8],            // index
    [5, 9], [9, 10], [10, 11], [11, 12],       // middle
    [9, 13], [13, 14], [14, 15], [15, 16],     // ring
    [13, 17], [17, 18], [18, 19], [19, 20],    // pinky
    [0, 17],                                   // palm base
  ];

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
    const items = data.items || [];

    // Pass 1: translucent filled segmentation masks UNDER the boxes. Each mask is
    // a list of polygons; each polygon is a flat [x0,y0,x1,y1,...] in source-frame
    // pixels — the SAME space as the box — so it maps through the identical
    // projector P. Detectors without masks (YOLO-World) simply omit the field.
    ctx.fillStyle = "rgba(255,183,77,0.22)";
    ctx.strokeStyle = "rgba(255,183,77,0.7)";
    ctx.lineWidth = 1.5;
    for (const d of items) {
      if (!d.mask) continue;
      for (const poly of d.mask) {
        if (!poly || poly.length < 6 || poly.length % 2) continue;   // need >= 3 (x,y) points
        ctx.beginPath();
        const [sx, sy] = P(poly[0], poly[1]);
        ctx.moveTo(sx, sy);
        for (let i = 2; i < poly.length; i += 2) {
          const [px, py] = P(poly[i], poly[i + 1]);
          ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    }

    // Pass 2: boxes + labels on top, so they stay legible over the fills.
    ctx.lineWidth = 2;
    ctx.strokeStyle = DETECT_COLOR;
    ctx.font = "13px ui-monospace, Menlo, monospace";
    ctx.textBaseline = "bottom";
    for (const d of items) {
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

  function drawHands(data, cw, ch) {
    const P = projector(data, cw, ch);
    if (!P) return;
    ctx.strokeStyle = HAND_COLOR;
    ctx.fillStyle = HAND_COLOR;
    // MediaPipe always returns a full 21-point hand (occluded joints are
    // estimated), so unlike the skeleton there is no per-point confidence gate —
    // we draw every landmark of every detected hand.
    for (const hand of data.items || []) {
      const lm = hand.landmarks || [];
      if (lm.length < 21) continue;
      ctx.lineWidth = 2;
      for (const [a, b] of HAND_BONES) {
        const [ax, ay] = P(lm[a][0], lm[a][1]);
        const [bx, by] = P(lm[b][0], lm[b][1]);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }
      for (const pt of lm) {
        const [x, y] = P(pt[0], pt[1]);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
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
    if (poseOn && handsData) drawHands(handsData, cw, ch); // fingers ride the Skeleton toggle

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
    // Skeleton toggle owns BOTH the body skeleton and the hand landmarks.
    setPose(on) { poseOn = !!on; if (!on) { poseData = null; handsData = null; } },
    setDetect(on) { detectOn = !!on; if (!on) detectData = null; },
    setPoseData(d) { poseData = d; },
    setDetectData(d) { detectData = d; },
    setHandsData(d) { handsData = d; },
  };
})();
