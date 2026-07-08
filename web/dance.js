// G1 Dance chooser + Dance Lab
// =============================================================================
// The header "Dance" button opens a modal that lets the operator CHOOSE which
// whole-body dance to run (named tiles, each showing the clear space it needs)
// instead of firing the one hard-coded routine. A dance is played by sending
// { type:"dance", fsm_id } — the server maps it to LocoClient.SetFsmId(fsm_id).
//
// DANCE LAB (supervised discovery): because there is no public menu of G1 dance
// FSM ids, the only safe way to find MORE working dances is to probe candidate
// ids one at a time with the robot standing + space clear, watch the camera, and
// SAVE the ones that dance (name + measured clear radius). Saved dances persist
// server-side (config/dances.yaml) and appear as tiles on every connected client.
//
// All three actions (dance / dance_probe / dance_save) mutate the robot, so
// controller.js's send() gates them on holding the control lock — a read-only
// device can open the chooser but its Play/Fire/Save are dropped by the server.
// =============================================================================

window.Dance = (function () {
  let send = null;
  let dances = [];           // [{name, fsm_id, space_m, note}]
  let uiMode = "damp";
  let fsmId = null;
  let transitioning = false;
  let built = false;
  let els = {};

  // A dance/probe is only safe with the robot upright + settled (stand or walk).
  function upright() { return (uiMode === "stand" || uiMode === "walk") && !transitioning; }

  // fsm ids that ARE a dance routine (so we can tell "the robot is dancing now").
  function danceFsms() { return new Set(dances.map(d => d.fsm_id)); }

  function build() {
    if (built) return;
    built = true;

    const overlay = document.createElement("div");
    overlay.id = "danceModal";
    overlay.className = "dance-modal";
    overlay.hidden = true;
    overlay.innerHTML = `
      <div class="dance-card" role="dialog" aria-modal="true" aria-label="Choose a dance">
        <div class="dance-head">
          <h2>Dances</h2>
          <button class="dance-close" type="button" aria-label="Close">✕</button>
        </div>
        <div class="dance-warn" id="danceWarn" hidden>Stand or Walk the robot first — a dance needs it upright.</div>
        <div class="dance-grid" id="danceGrid"></div>
        <div class="dance-foot">
          <button class="dance-stop" id="danceStop" type="button" title="Exit the routine (returns to Stand)">■ Stop → Stand</button>
        </div>

        <details class="dance-lab" id="danceLab">
          <summary>🧪 Dance Lab — find &amp; add new dances (supervised)</summary>
          <p class="dance-lab-note">
            No public list of G1 dance ids exists, so discover them safely: robot
            <b>standing</b>, ~2.5&nbsp;m clear all round, e-stop in reach. Type a candidate
            FSM id, <b>Fire once</b>, watch the camera. If it dances, name it + the clear
            radius it needed and <b>Save</b>. If not, <b>Stop → Stand</b> and try another.
          </p>
          <div class="dance-lab-row">
            <label>FSM id <input type="number" id="labProbeId" min="0" max="9999" step="1" placeholder="e.g. 504"></label>
            <button id="labFire" type="button" class="lab-fire">▶ Fire once</button>
            <span class="lab-cur">robot now at FSM <b id="labCurFsm">—</b></span>
          </div>
          <div class="dance-lab-save">
            <span class="lab-save-label">Danced? Save it:</span>
            <label>Name <input type="text" id="labSaveName" maxlength="40" placeholder="e.g. Twist"></label>
            <label>Needs <input type="number" id="labSaveSpace" min="0" max="10" step="0.5" value="1.5"> m clear</label>
            <button id="labSave" type="button" class="lab-save">Save dance</button>
          </div>
        </details>
      </div>`;
    document.body.appendChild(overlay);

    els = {
      overlay,
      grid: overlay.querySelector("#danceGrid"),
      warn: overlay.querySelector("#danceWarn"),
      stop: overlay.querySelector("#danceStop"),
      close: overlay.querySelector(".dance-close"),
      lab: overlay.querySelector("#danceLab"),
      probeId: overlay.querySelector("#labProbeId"),
      fire: overlay.querySelector("#labFire"),
      curFsm: overlay.querySelector("#labCurFsm"),
      saveName: overlay.querySelector("#labSaveName"),
      saveSpace: overlay.querySelector("#labSaveSpace"),
      save: overlay.querySelector("#labSave"),
    };

    // Open/close wiring.
    const opener = document.getElementById("danceOpen");
    if (opener) opener.addEventListener("click", open);
    els.close.addEventListener("click", close);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !overlay.hidden) close();
    });

    // Footer: exit the routine by returning to Stand (routes through 802 server-side).
    els.stop.addEventListener("click", () => {
      if (window.logEvent) window.logEvent("dance → stop (Stand)");
      send && send({ type: "mode", name: "stand" });
    });

    // Dance Lab: supervised probe.
    els.fire.addEventListener("click", () => {
      const id = parseInt(els.probeId.value, 10);
      if (!Number.isInteger(id) || id < 0 || id > 9999) { els.probeId.focus(); return; }
      if (!upright()) { flashWarn(); return; }
      const ok = confirm(
        `DANCE LAB — SUPERVISED PROBE\n\n` +
        `About to run SetFsmId(${id}) on the real robot. This is an UNKNOWN motion: ` +
        `it could dance, kick, jump, crouch or stumble.\n\n` +
        `Confirm: robot standing, ~2.5 m clear all round, e-stop in hand.\n\nFire FSM ${id}?`
      );
      if (!ok) return;
      if (window.logEvent) window.logEvent(`dance-lab → probe FSM ${id}`, "warn");
      // Pre-fill the Save NAME (value, not placeholder) so a good result is one
      // click from saved — the Save handler requires a non-empty value.
      if (!els.saveName.value) els.saveName.value = `Dance ${id}`;
      send && send({ type: "dance_probe", fsm_id: id });
    });

    // Dance Lab: save a verified dance.
    els.save.addEventListener("click", () => {
      // Prefer the just-probed id; fall back to whatever the robot is in now.
      let id = parseInt(els.probeId.value, 10);
      if (!Number.isInteger(id)) id = fsmId;
      const name = (els.saveName.value || "").trim();
      const space = parseFloat(els.saveSpace.value);
      if (!Number.isInteger(id) || id < 0 || id > 9999) { els.probeId.focus(); return; }
      if (!name) { els.saveName.focus(); return; }
      send && send({ type: "dance_save", fsm_id: id, name,
                     space_m: Number.isFinite(space) ? space : 2.0 });
      if (window.logEvent) window.logEvent(`dance-lab → save "${name}" (FSM ${id})`, "ok");
      els.saveName.value = "";
    });
  }

  function flashWarn() {
    if (!els.warn) return;
    els.warn.hidden = false;
    els.warn.classList.remove("flash");
    void els.warn.offsetWidth;      // restart the CSS animation
    els.warn.classList.add("flash");
  }

  function open() { build(); els.overlay.hidden = false; render(); }
  function close() { if (els.overlay) els.overlay.hidden = true; }

  // A VERIFIED dance -> Play it (confirmed working). fsm_id sent to the server.
  function playDance(d) {
    if (!upright()) { flashWarn(); return; }
    const ok = confirm(
      `Play “${d.name}” — the whole body will move and it needs about ` +
      `${d.space_m} m of clear space all round.\n\n` +
      `Keep the e-stop in reach. Press Stand/Walk (or Stop) to exit.\n\nContinue?`
    );
    if (!ok) return;
    if (window.logEvent) window.logEvent(`dance → ${d.name}`);
    send && send({ type: "dance", fsm_id: d.fsm_id });
  }

  // A HUB dance (is_hub) -> "Enter Dance Mode". Fires the hub fsm (e.g. 503) from
  // upright; once the robot is in it, the child dances below unlock. Mirrors R1+B.
  function enterHub(d) {
    if (!upright()) { flashWarn(); return; }
    const ok = confirm(
      `Enter “${d.name}” — the robot hands its whole body to dance mode and needs ` +
      `about ${d.space_m} m of clear space all round. Left alone it plays the ` +
      `classic dance; or press a dance button once it's in.\n\n` +
      `Keep the e-stop in reach. Press Stand/Walk (or Stop) to exit.\n\nContinue?`
    );
    if (!ok) return;
    if (window.logEvent) window.logEvent(`dance → enter ${d.name}`);
    send && send({ type: "dance", fsm_id: d.fsm_id });
  }

  // A CHILD dance (parent_fsm set) -> only fireable while the robot is in that hub
  // fsm (dance mode). No heavy confirm: we're already committed to dancing, this
  // just switches which dance. Mirrors pressing a direction after R1+B.
  function playChild(d) {
    if (fsmId !== d.parent_fsm) { flashWarn(); return; }
    if (window.logEvent) window.logEvent(`dance → ${d.name} (from mode ${d.parent_fsm})`);
    send && send({ type: "dance", fsm_id: d.fsm_id });
  }

  // A CANDIDATE (unverified) tile -> Test it exactly like a Dance Lab probe: same
  // supervised confirm, and prefill the Lab's Save form so a good result is one click
  // from becoming a verified dance.
  function testDance(d) {
    if (!upright()) { flashWarn(); return; }
    const ok = confirm(
      `TEST “${d.name}” (FSM ${d.fsm_id}) — this is an UNVERIFIED candidate. It could ` +
      `dance, kick, jump, crouch or stumble.\n\n` +
      `Confirm: robot standing, ~2.5 m clear all round, e-stop in hand.\n\nTest FSM ${d.fsm_id}?`
    );
    if (!ok) return;
    if (window.logEvent) window.logEvent(`dance-lab → test ${d.name} (FSM ${d.fsm_id})`, "warn");
    if (els.lab) els.lab.open = true;             // reveal the Lab so Save is visible
    els.probeId.value = d.fsm_id;
    if (!els.saveName.value) els.saveName.value = d.name;
    send && send({ type: "dance_probe", fsm_id: d.fsm_id });
  }

  function removeDance(d) {
    if (!confirm(`Remove “${d.name}” (FSM ${d.fsm_id}) from the list?`)) return;
    if (window.logEvent) window.logEvent(`dance-lab → remove ${d.name}`);
    send && send({ type: "dance_delete", fsm_id: d.fsm_id });
  }

  let _gridSig = null;   // dirty-check: only rebuild the grid DOM when it changes

  function render() {
    if (!built) return;
    const opener = document.getElementById("danceOpen");
    if (opener) {
      opener.disabled = false;
      const verified = dances.filter(d => d.verified).length;
      opener.title = dances.length
        ? `Choose a dance (${verified} ready · ${dances.length - verified} to test)`
        : "Open the Dance chooser / Dance Lab";
    }
    if (els.overlay.hidden) return;   // only paint while visible

    const up = upright();
    const dancingNow = fsmId != null && danceFsms().has(fsmId);
    els.warn.hidden = up;
    els.stop.hidden = !dancingNow;
    els.curFsm.textContent = fsmId == null ? "—" : String(fsmId);
    els.fire.disabled = !up;

    // Rebuild the tile grid ONLY when something visible changed (dance list, which
    // one is active, or the upright gate) — a periodic fsm_state must not tear down
    // the DOM mid-click (dropped Play) ~2x/sec.
    const sig = JSON.stringify(dances) + "|" + fsmId + "|" + up;
    if (sig === _gridSig) return;
    _gridSig = sig;

    if (!dances.length) {
      els.grid.innerHTML =
        `<p class="dance-empty">No dances yet — open the Dance Lab below to find one.</p>`;
      return;
    }
    const dis = up ? "" : "disabled";

    // Partition into: hubs (dance modes you ENTER), their children (dances that
    // fire from inside a hub), and standalone tiles (flat dances / candidates). A
    // child whose parent isn't actually a hub falls back to standalone so it stays
    // reachable rather than vanishing.
    const hubs = dances.filter(d => d.is_hub);
    const hubFsms = new Set(hubs.map(h => h.fsm_id));
    const childrenOf = (fsm) => dances.filter(d => d.parent_fsm === fsm);
    const standalone = dances.filter(d =>
      !d.is_hub && !(d.parent_fsm != null && hubFsms.has(d.parent_fsm)));

    const hubHtml = (h) => {
      const inMode = fsmId === h.fsm_id;                 // robot IS in this dance mode
      const enterDis = (up && !inMode) ? "" : "disabled";
      const kids = childrenOf(h.fsm_id);
      const kidsHtml = kids.map((c) => {
        const cactive = fsmId === c.fsm_id;             // this child is dancing now
        const cdis = (inMode && !cactive) ? "" : "disabled";
        return `<button class="dt-sub${cactive ? " active" : ""}" data-fsm="${c.fsm_id}" ${cdis}
                  title="${esc(c.note || "")}">
              <span class="dt-sub-name">${cactive ? "● " : ""}${esc(c.name)}</span>
              <span class="dt-sub-space">~${c.space_m} m</span>
            </button>`;
      }).join("");
      return `<div class="dance-hub${inMode ? " active" : ""}">
          <div class="dance-hub-head">
            <div class="dt-name">${esc(h.name)}</div>
            <div class="dt-space">needs ~${h.space_m} m clear</div>
          </div>
          ${h.note ? `<div class="dt-note" title="${esc(h.note)}">${esc(h.note)}</div>` : ""}
          <button class="dt-enter" data-fsm="${h.fsm_id}" ${enterDis}>${inMode ? "● in dance mode" : "▶ Enter Dance Mode"}</button>
          ${kids.length ? `<div class="dance-subwrap">
              <div class="dance-sub-label">${inMode ? "Pick a dance:" : "Enter dance mode to unlock these:"}</div>
              <div class="dance-subrow">${kidsHtml}</div>
            </div>` : ""}
        </div>`;
    };

    const tileHtml = (d) => {
      const active = d.fsm_id === fsmId;
      const action = d.verified
        ? `<button class="dt-play" data-fsm="${d.fsm_id}" ${dis}>${active ? "● dancing" : "▶ Play"}</button>`
        : `<button class="dt-test" data-fsm="${d.fsm_id}" ${dis}>${active ? "● testing" : "🧪 Test"}</button>`;
      return `<div class="dance-tile${active ? " active" : ""}${d.verified ? "" : " candidate"}">
          <button class="dt-del" data-fsm="${d.fsm_id}" title="Remove">✕</button>
          <div class="dt-name">${esc(d.name)}</div>
          <div class="dt-space">needs ~${d.space_m} m clear${d.verified ? "" : " · candidate"}</div>
          ${d.note ? `<div class="dt-note" title="${esc(d.note)}">${esc(d.note)}</div>` : ""}
          ${action}
        </div>`;
    };

    els.grid.innerHTML = hubs.map(hubHtml).join("") + standalone.map(tileHtml).join("");

    const find = (btn) => dances.find(x => x.fsm_id === parseInt(btn.dataset.fsm, 10));
    els.grid.querySelectorAll(".dt-enter").forEach(b =>
      b.addEventListener("click", () => { const d = find(b); if (d) enterHub(d); }));
    els.grid.querySelectorAll(".dt-sub").forEach(b =>
      b.addEventListener("click", () => { const d = find(b); if (d) playChild(d); }));
    els.grid.querySelectorAll(".dt-play").forEach(b =>
      b.addEventListener("click", () => { const d = find(b); if (d) playDance(d); }));
    els.grid.querySelectorAll(".dt-test").forEach(b =>
      b.addEventListener("click", () => { const d = find(b); if (d) testDance(d); }));
    els.grid.querySelectorAll(".dt-del").forEach(b =>
      b.addEventListener("click", () => { const d = find(b); if (d) removeDance(d); }));
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  // Build the modal + wire the opener at load time (this script is at the end of
  // <body>, so the DOM exists) — otherwise the "Dance ▾" button would be inert
  // until the first WebSocket connect ran init(). send stays null until init(),
  // so any pre-connect Play/Test/Save is a safe no-op.
  build();

  return {
    init(sendFn) { send = sendFn; build(); render(); },
    applyConfig(cfg) {
      if (cfg && Array.isArray(cfg.dances)) dances = cfg.dances;
      render();
    },
    handleDances(msg) {
      if (msg && Array.isArray(msg.dances)) dances = msg.dances;
      render();
    },
    setState(id, mode, transit) {
      fsmId = (id === undefined ? fsmId : id);
      uiMode = mode || uiMode;
      transitioning = !!transit;
      render();
    },
  };
})();
