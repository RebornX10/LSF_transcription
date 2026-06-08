// Browser-camera controller for the LSF interpreter.
//
//   getUserMedia → <video> → throttled JPEG → POST /process
//   ← landmarks + glosses → draw skeleton on <canvas> overlay + transcript
//
// The video stays local (smooth) and only small frames go up / tiny JSON comes
// back, so it runs on CPU-only hosts (Hugging Face) as well as locally.

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const video = $("cam");
  const overlay = $("overlay");
  const stage = $("stage");
  const camHint = $("cam-hint");
  const stream = $("transcript-stream");
  const wordList = $("word-list");
  const search = $("search");
  const categorySel = $("category");
  const addForm = $("add-form");
  const addFr = $("add-fr");
  const trainBanner = $("train-banner");
  const trainWordEl = $("train-word");
  const pillDepth = $("pill-depth");
  const pillFps = $("pill-fps");
  const pillTrained = $("pill-trained");
  const vocabCount = $("vocab-count");

  const ctx = overlay.getContext("2d");
  const proc = document.createElement("canvas");   // offscreen capture canvas
  const pctx = proc.getContext("2d");

  // --- Tuning (max speed for CPU hosts like HF free tier) ---
  // Lower PROC_WIDTH = much faster MediaPipe + smaller uploads. Raise it (e.g.
  // 480) on a fast machine for crisper landmarks.
  const PROC_WIDTH = 320;        // frame width sent to the server
  const JPEG_QUALITY = 0.5;
  const TARGET_FPS = 15;         // capture/processing cap (self-paced, no pile-up)
  const FRAME_BUDGET = 1000 / TARGET_FPS;

  let words = [];
  let recordingGloss = null;
  let inFlight = false;
  let running = false;

  // MediaPipe connection topology (indices), for drawing the skeleton in JS.
  const HAND = [
    [0,1],[1,2],[2,3],[3,4], [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12], [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20], [0,17],
  ];
  const POSE = [ // upper-body subset: shoulders, arms, torso, face line
    [11,12],[11,13],[13,15],[12,14],[14,16],
    [11,23],[12,24],[23,24],[0,11],[0,12],
  ];
  // Face mesh contours (standard MediaPipe indices): eyes, brows, lips, oval.
  const FACE = [
    // left eye
    [33,7],[7,163],[163,144],[144,145],[145,153],[153,154],[154,155],[155,133],
    [33,246],[246,161],[161,160],[160,159],[159,158],[158,157],[157,173],[173,133],
    // right eye
    [362,382],[382,381],[381,380],[380,374],[374,373],[373,390],[390,249],[249,263],
    [263,466],[466,388],[388,387],[387,386],[386,385],[385,384],[384,398],[398,362],
    // eyebrows
    [70,63],[63,105],[105,66],[66,107],[46,53],[53,52],[52,65],[65,55],
    [300,293],[293,334],[334,296],[296,336],[276,283],[283,282],[282,295],[295,285],
    // outer lips
    [61,146],[146,91],[91,181],[181,84],[84,17],[17,314],[314,405],[405,321],[321,375],[375,291],
    [61,185],[185,40],[40,39],[39,37],[37,0],[0,267],[267,269],[269,270],[270,409],[409,291],
    // inner lips
    [78,95],[95,88],[88,178],[178,87],[87,14],[14,317],[317,402],[402,318],[318,324],[324,308],
    [78,191],[191,80],[80,81],[81,82],[82,13],[13,312],[312,311],[311,310],[310,415],[415,308],
    // face oval
    [10,338],[338,297],[297,332],[332,284],[284,251],[251,389],[389,356],[356,454],[454,323],
    [323,361],[361,288],[288,397],[397,365],[365,379],[379,378],[378,400],[400,377],[377,152],
    [152,148],[148,176],[176,149],[149,150],[150,136],[136,172],[172,58],[58,132],[132,93],
    [93,234],[234,127],[127,162],[162,21],[21,54],[54,103],[103,67],[67,109],[109,10],
  ];

  function post(url, body) {
    return fetch(url, { method: "POST", body: new URLSearchParams(body || {}) })
      .then((r) => r.json());
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function toast(msg) {
    let el = document.querySelector(".toast");
    if (!el) { el = document.createElement("div"); el.className = "toast"; document.body.appendChild(el); }
    el.textContent = msg; el.classList.add("show");
    clearTimeout(el._t); el._t = setTimeout(() => el.classList.remove("show"), 2200);
  }

  // ---- Camera ----
  async function initCamera() {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      video.srcObject = s;
      await video.play();
      camHint.classList.add("hidden");
      sizeCanvas();
      detectDualPixel(s.getVideoTracks()[0]);
      running = true;
      loop();
    } catch (e) {
      camHint.textContent = "Caméra refusée ou indisponible : " + (e.message || e.name);
    }
  }

  function sizeCanvas() {
    const ar = (video.videoWidth || 4) / (video.videoHeight || 3);
    stage.style.aspectRatio = ar.toFixed(4);
    overlay.width = stage.clientWidth;
    overlay.height = stage.clientHeight;
    proc.width = PROC_WIDTH;
    proc.height = Math.round(PROC_WIDTH / ar);
  }
  window.addEventListener("resize", () => { if (running) sizeCanvas(); });

  // ---- Dual-pixel / depth detection (client device, e.g. Pixel) ----
  async function detectDualPixel(track) {
    let supported = false, reason = "", label = (track && track.label) || "";
    try {
      const caps = track.getCapabilities ? track.getCapabilities() : {};
      const sup = navigator.mediaDevices.getSupportedConstraints
        ? navigator.mediaDevices.getSupportedConstraints() : {};
      const labelHit = /pixel|depth|tof|truedepth/i.test(label);
      const depthConstraint = !!(sup.depthNear || sup.depthFar);
      const focusCap = !!caps.focusDistance;
      let depthDevice = false;
      try {
        const devs = await navigator.mediaDevices.enumerateDevices();
        depthDevice = devs.some((d) => /depth/i.test(d.label));
      } catch (_) {}
      supported = labelHit || depthConstraint || depthDevice;
      const hits = [labelHit && "label", depthConstraint && "depth-constraint",
                    focusCap && "focusDistance", depthDevice && "depth-device"].filter(Boolean);
      reason = supported
        ? "Capteur de profondeur probable (" + hits.join(", ") + ")"
        : 'Aucun capteur dual-pixel détecté (label="' + (label || "?") + '")';
    } catch (e) { reason = "Échec de la détection : " + e.message; }

    post("/depth", { dual_pixel: supported ? "1" : "0", reason, label })
      .then((r) => updateStatus(r.status))
      .catch(() => {});
  }

  // ---- Capture → process loop ----
  function captureBlob() {
    return new Promise((resolve) => {
      pctx.drawImage(video, 0, 0, proc.width, proc.height);
      proc.toBlob((b) => resolve(b), "image/jpeg", JPEG_QUALITY);
    });
  }

  async function loop() {
    if (!running) return;
    const start = performance.now();
    if (!inFlight && video.readyState >= 2) {
      inFlight = true;
      try {
        const blob = await captureBlob();
        const res = await fetch("/process", { method: "POST", body: blob });
        const data = await res.json();
        if (data.ok) {
          drawSkeleton(data.landmarks);
          if (data.reco) renderReco(data.reco);
          updateStatus(data.status);
        }
      } catch (_) { /* transient network/CPU hiccup; keep going */ }
      inFlight = false;
    }
    const wait = Math.max(0, FRAME_BUDGET - (performance.now() - start));
    setTimeout(loop, wait);
  }

  // ---- Drawing ----
  function drawSkeleton(lm) {
    const W = overlay.width, H = overlay.height;
    ctx.clearRect(0, 0, W, H);
    if (!lm) return;
    const draw = (pts, conns, color, dots, lw) => {
      if (!pts) return;
      ctx.strokeStyle = color; ctx.lineWidth = lw || 2;
      ctx.beginPath();
      for (const [a, b] of conns) {
        if (!pts[a] || !pts[b]) continue;
        ctx.moveTo(pts[a][0] * W, pts[a][1] * H);
        ctx.lineTo(pts[b][0] * W, pts[b][1] * H);
      }
      ctx.stroke();
      if (!dots) return;                  // skip joint dots for pose/face (clutter)
      ctx.fillStyle = color;
      for (const p of pts) { ctx.beginPath(); ctx.arc(p[0] * W, p[1] * H, 2.5, 0, 6.283); ctx.fill(); }
    };
    // Face first (thin, underneath), then body, then hands on top.
    draw(lm.face, FACE, "rgba(150,210,230,0.55)", false, 1);
    draw(lm.pose, POSE, "#5ad1ff", false, 2);
    draw(lm.left_hand, HAND, "#2ee6a6", true, 2);
    draw(lm.right_hand, HAND, "#ffd166", true, 2);
  }

  function renderReco(reco) {
    const placeholder = stream.querySelector(".placeholder");
    if (placeholder) placeholder.remove();
    const el = document.createElement("span");
    el.className = "gloss";
    el.dataset.source = reco.source;
    el.title = `source: ${reco.source}`;
    el.innerHTML = `${escapeHtml(reco.gloss)} <span class="conf">${Math.round(reco.confidence * 100)}%</span>`;
    stream.appendChild(el);
    stream.scrollTop = stream.scrollHeight;
  }

  function updateStatus(s) {
    if (!s) return;
    pillDepth.textContent = `depth: ${s.depth_source}`;
    pillDepth.title = s.depth_reason || "";
    pillFps.textContent = `fps: ${s.fps}`;
  }

  // ---- Vocabulary (unchanged behaviour) ----
  function loadLexicon() {
    return fetch("/lexicon").then((r) => r.json()).then((data) => {
      words = data.words || [];
      fillCategories(data.categories || []);
      renderWords();
      const trained = words.filter((w) => w.samples > 0).length;
      pillTrained.textContent = `appris: ${trained}/${words.length}`;
      vocabCount.textContent = `${trained} appris sur ${words.length}`;
    }).catch(() => {});
  }
  function fillCategories(cats) {
    const current = categorySel.value;
    categorySel.innerHTML = '<option value="">Toutes catégories</option>';
    for (const c of cats) {
      const o = document.createElement("option");
      o.value = c; o.textContent = c; categorySel.appendChild(o);
    }
    categorySel.value = current;
  }
  function renderWords() {
    const q = (search.value || "").trim().toLowerCase();
    const cat = categorySel.value;
    wordList.innerHTML = "";
    const shown = words.filter((w) => {
      if (cat && w.category !== cat) return false;
      if (!q) return true;
      return w.fr.toLowerCase().includes(q) || w.gloss.toLowerCase().includes(q) ||
             (w.en || "").toLowerCase().includes(q);
    });
    if (!shown.length) { wordList.innerHTML = '<p class="placeholder">Aucun mot.</p>'; return; }
    for (const w of shown) {
      const trained = w.samples > 0;
      const card = document.createElement("div");
      card.className = "word" + (trained ? " trained" : "");
      const badge = trained
        ? `<span class="badge ok">✓ ${w.samples} échantillon${w.samples > 1 ? "s" : ""}</span>`
        : `<span class="badge">non appris</span>`;
      const meta = [w.gloss, w.en, w.category].filter(Boolean).map(escapeHtml).join(" · ");
      const detail = [];
      detail.push(w.tip
        ? `<p class="word-tip">${escapeHtml(w.tip)}</p>`
        : `<p class="word-tip muted">Pas de description écrite — voir la vidéo.</p>`);
      if (w.ref) detail.push(
        `<a class="word-ref" href="${escapeHtml(w.ref)}" target="_blank" rel="noopener">▶ Voir le signe en vidéo (Elix)</a>`);
      card.innerHTML = `
        <div class="word-row">
          <div class="word-main">
            <div class="word-fr">${escapeHtml(w.fr)} ${badge} <span class="chev">▸</span></div>
            <div class="word-meta">${meta}</div>
          </div>
          <div class="word-actions">
            <button class="btn btn-sm" data-train="${escapeHtml(w.gloss)}" data-fr="${escapeHtml(w.fr)}">Entraîner</button>
            ${trained ? `<button class="btn btn-sm btn-danger" data-del="${escapeHtml(w.gloss)}">✕</button>` : ""}
          </div>
        </div>
        <div class="word-detail">${detail.join("")}</div>`;
      wordList.appendChild(card);
    }
  }

  // ---- Training ----
  function startTraining(gloss, fr) {
    if (!running) { toast("Activez d'abord la caméra."); return; }
    post("/record/start", { label: gloss }).then((res) => {
      if (res.ok) {
        recordingGloss = gloss;
        trainWordEl.textContent = fr + " (" + gloss + ")";
        trainBanner.classList.remove("hidden");
      }
    });
  }
  function saveTraining() {
    post("/record/stop").then((res) => {
      trainBanner.classList.add("hidden"); recordingGloss = null;
      toast(res.saved ? `Signe appris : ${res.saved}` : "Trop court — bougez les mains et réessayez.");
      loadLexicon();
    });
  }
  function cancelTraining() {
    post("/record/cancel").then(() => { trainBanner.classList.add("hidden"); recordingGloss = null; });
  }

  // ---- Events ----
  search.addEventListener("input", renderWords);
  categorySel.addEventListener("change", renderWords);
  addForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const fr = (addFr.value || "").trim();
    if (!fr) return;
    post("/vocab/add", { fr }).then(() => { addFr.value = ""; loadLexicon().then(() => toast(`Mot ajouté : ${fr}`)); });
  });
  wordList.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (btn) {
      if (btn.dataset.train) startTraining(btn.dataset.train, btn.dataset.fr);
      else if (btn.dataset.del && confirm(`Oublier les échantillons de « ${btn.dataset.del} » ?`))
        post("/vocab/delete", { gloss: btn.dataset.del }).then(loadLexicon);
      return;
    }
    if (e.target.closest("a")) return;
    const card = e.target.closest(".word");
    if (card) card.classList.toggle("open");
  });
  $("btn-save").addEventListener("click", saveTraining);
  $("btn-cancel").addEventListener("click", cancelTraining);
  $("btn-clear").addEventListener("click", () => {
    post("/transcript/clear").then(() => {
      stream.innerHTML = '<p class="placeholder">La transcription apparaîtra ici…</p>';
    });
  });

  // ---- Boot ----
  loadLexicon();
  initCamera();
})();
