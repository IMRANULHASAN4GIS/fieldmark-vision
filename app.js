/* Fieldmark — live object recognition prototype
 *
 * Pipeline:
 *   camera frame -> drop-if-busy gate -> COCO-SSD detect (async)
 *     -> IoU tracker (stable IDs) -> box EMA + label vote + hysteresis
 *     -> overlay render on rAF
 *
 * The tracker/smoothing layer is the difference between a jittery demo
 * and something that feels like a product. Turn "Unsmoothed output" on
 * in Tune to see the raw detector for comparison.
 */

const COCO_CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"];

/* ---------- tuning ---------- */
const CFG = {
  maxBoxes: 20,
  minHits: 3,        // frames before a track is drawn (hysteresis in)
  maxMisses: 9,      // frames before a track is dropped (hysteresis out)
  iouGate: 0.30,     // below this, a detection can't match a track
  boxAlpha: 0.38,    // EMA weight for new box coords
  scoreAlpha: 0.30,
  voteWindow: 12,    // label history depth
  speakCooldown: 6000,
  maxLabels: 12,
};

/* ---------- element handles ---------- */
const $ = id => document.getElementById(id);
const video = $('video'), cv = $('overlay'), ctx = cv.getContext('2d');
const elFps = $('fps'), elMs = $('ms'), elStatus = $('statusText'), elTally = $('tally');

/* ---------- state ---------- */
let model = null, modelBase = 'lite_mobilenet_v2';
let stream = null, facing = 'environment';
let running = false, frozen = false, busy = false, speak = false;
let wakeLock = null;
let scoreThreshold = 0.50;
let showBoxes = true, showScores = true, rawMode = false;
let rawDets = [];
let lastSpoken = new Map();
let fpsHist = [], lastFrameT = 0, inferMs = 0;

/* ---------- geometry helpers ---------- */
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const w = x2 - x1, h = y2 - y1;
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter);
}

/* ---------- tracker ----------
 * Greedy IoU association. Not Kalman/ByteTrack, but enough to give every
 * object a stable identity across frames, which is what the smoothing needs.
 */
class Tracker {
  constructor() { this.tracks = []; this.nextId = 1; }

  update(dets) {
    const pairs = [];
    for (let t = 0; t < this.tracks.length; t++)
      for (let d = 0; d < dets.length; d++) {
        const s = iou(this.tracks[t].box, dets[d].bbox);
        if (s >= CFG.iouGate) pairs.push({ t, d, s });
      }
    pairs.sort((a, b) => b.s - a.s);

    const usedT = new Set(), usedD = new Set();
    for (const p of pairs) {
      if (usedT.has(p.t) || usedD.has(p.d)) continue;
      usedT.add(p.t); usedD.add(p.d);
      this.hit(this.tracks[p.t], dets[p.d]);
    }

    dets.forEach((d, i) => { if (!usedD.has(i)) this.spawn(d); });

    this.tracks.forEach((tr, i) => {
      if (!usedT.has(i)) { tr.misses++; tr.hits = Math.max(0, tr.hits - 1); }
    });

    this.tracks = this.tracks.filter(tr => tr.misses <= CFG.maxMisses);
  }

  spawn(d) {
    this.tracks.push({
      id: this.nextId++,
      box: d.bbox.slice(),
      votes: [d.class],
      label: d.class,
      score: d.score,
      hits: 1, misses: 0, shown: false,
    });
  }

  hit(tr, d) {
    const a = CFG.boxAlpha;
    for (let i = 0; i < 4; i++) tr.box[i] = tr.box[i] * (1 - a) + d.bbox[i] * a;
    tr.score = tr.score * (1 - CFG.scoreAlpha) + d.score * CFG.scoreAlpha;
    tr.votes.push(d.class);
    if (tr.votes.length > CFG.voteWindow) tr.votes.shift();
    tr.hits++; tr.misses = 0;

    // majority label over the vote window — stops "chair/couch/chair" flicker
    const tally = {};
    let best = tr.label, bestN = 0;
    for (const v of tr.votes) {
      tally[v] = (tally[v] || 0) + 1;
      if (tally[v] > bestN) { bestN = tally[v]; best = v; }
    }
    tr.label = best;
  }

  visible() {
    return this.tracks.filter(tr => {
      if (tr.hits >= CFG.minHits) tr.shown = true;
      if (tr.misses > 3 && tr.hits < CFG.minHits) tr.shown = false;
      return tr.shown;
    });
  }
}
const tracker = new Tracker();

/* ---------- colour: stable hue per class ---------- */
const hueCache = new Map();
function hueFor(label) {
  if (hueCache.has(label)) return hueCache.get(label);
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) % 360;
  // steer away from the amber chrome so UI and data stay distinguishable
  if (h > 28 && h < 62) h = (h + 150) % 360;
  hueCache.set(label, h);
  return h;
}

/* ---------- camera ---------- */
async function openCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  const constraints = {
    audio: false,
    video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } }
  };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await video.play();
}

async function keepAwake() {
  try { if ('wakeLock' in navigator) wakeLock = await navigator.wakeLock.request('screen'); }
  catch (_) { /* not critical */ }
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && running && !wakeLock) keepAwake();
});

/* ---------- model ---------- */
async function loadModel(base) {
  setStatus('Loading model');
  model = null;
  model = await cocoSsd.load({ base });
  modelBase = base;
  // warm-up pass: the first inference compiles shaders and is always slow
  const warm = tf.zeros([1, 300, 300, 3], 'int32');
  await model.detect(warm);
  warm.dispose();
  setStatus('Live');
}

/* ---------- canvas sizing + cover transform ---------- */
let view = { scale: 1, dx: 0, dy: 0 };
function sizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  // must match CSS object-fit: cover, or every box lands in the wrong place
  const scale = Math.max(w / vw, h / vh);
  view = { scale, dx: (w - vw * scale) / 2, dy: (h - vh * scale) / 2, w, h, vw };
}
addEventListener('resize', sizeCanvas);
addEventListener('orientationchange', () => setTimeout(sizeCanvas, 300));

function toScreen(b) {
  const mirror = facing === 'user';
  let x = b[0] * view.scale + view.dx;
  const y = b[1] * view.scale + view.dy;
  const w = b[2] * view.scale, h = b[3] * view.scale;
  if (mirror) x = view.w - (x + w);
  return [x, y, w, h];
}

/* ---------- main loop ---------- */
async function loop() {
  if (!running) return;

  const now = performance.now();
  if (lastFrameT) {
    fpsHist.push(1000 / (now - lastFrameT));
    if (fpsHist.length > 30) fpsHist.shift();
  }
  lastFrameT = now;

  if (model && !busy && !frozen && video.readyState >= 2) {
    busy = true;
    (async () => {
      try {
        const t0 = performance.now();
        const dets = await model.detect(video, CFG.maxBoxes, scoreThreshold);
        inferMs = performance.now() - t0;
        rawDets = dets;
        tracker.update(dets);
      } catch (e) {
        console.error(e);
      } finally { busy = false; }
    })();
  }

  render();
  requestAnimationFrame(loop);
}

/* ---------- rendering ---------- */
function render() {
  if (!view.w) sizeCanvas();
  ctx.clearRect(0, 0, view.w, view.h);

  const items = rawMode
    ? rawDets.map(d => ({ box: d.bbox, label: d.class, score: d.score, id: -1 }))
    : tracker.visible();

  const ordered = items
    .slice()
    .sort((a, b) => (b.box[2] * b.box[3]) - (a.box[2] * a.box[3]))
    .slice(0, CFG.maxLabels);

  const placed = [];
  for (const it of ordered) {
    const [x, y, w, h] = toScreen(it.box);
    const hue = hueFor(it.label);
    const stroke = `hsl(${hue} 88% 66%)`;

    if (showBoxes) {
      ctx.save();
      ctx.strokeStyle = 'rgba(8,11,14,.5)';
      ctx.lineWidth = 4;
      corners(x, y, w, h);
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2;
      corners(x, y, w, h);
      ctx.restore();
    }

    // label pill, nudged down if it would collide with one already drawn
    const text = showScores
      ? `${it.label}  ${Math.round(it.score * 100)}`
      : it.label;
    ctx.font = '500 13px "IBM Plex Sans", system-ui, sans-serif';
    const tw = ctx.measureText(text).width;
    const pw = tw + 18, ph = 22;
    let px = Math.max(6, Math.min(x, view.w - pw - 6));
    let py = y - ph - 5;
    if (py < 4) py = y + 5;
    for (let guard = 0; guard < 8; guard++) {
      const clash = placed.find(r => px < r.x + r.w && px + pw > r.x && py < r.y + r.h && py + ph > r.y);
      if (!clash) break;
      py = clash.y + clash.h + 3;
    }
    placed.push({ x: px, y: py, w: pw, h: ph });

    ctx.fillStyle = 'rgba(8,11,14,.82)';
    ctx.fillRect(px, py, pw, ph);
    ctx.fillStyle = stroke;
    ctx.fillRect(px, py, 2, ph);
    ctx.fillStyle = '#EEF3F6';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, px + 9, py + ph / 2 + .5);
    if (showScores) {
      const nw = ctx.measureText(String(Math.round(it.score * 100))).width;
      ctx.fillStyle = 'rgba(238,243,246,.5)';
      ctx.fillText('%', px + 9 + tw + 1, py + ph / 2 + .5);
      void nw;
    }
  }

  updateReadouts(ordered);
}

// open-corner brackets rather than a closed rectangle: less visual noise
// over a busy scene, and it reads as an instrument sight
function corners(x, y, w, h) {
  const c = Math.max(8, Math.min(22, Math.min(w, h) * 0.24));
  ctx.beginPath();
  ctx.moveTo(x, y + c); ctx.lineTo(x, y); ctx.lineTo(x + c, y);
  ctx.moveTo(x + w - c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + c);
  ctx.moveTo(x + w, y + h - c); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - c, y + h);
  ctx.moveTo(x + c, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - c);
  ctx.stroke();
}

let lastTally = '';
function updateReadouts(items) {
  const fps = fpsHist.length ? fpsHist.reduce((a, b) => a + b) / fpsHist.length : 0;
  elFps.textContent = fps ? fps.toFixed(0) : '—';
  elMs.textContent = inferMs ? inferMs.toFixed(0) : '—';

  const counts = new Map();
  for (const it of items) counts.set(it.label, (counts.get(it.label) || 0) + 1);
  const key = [...counts.entries()].map(([k, v]) => k + v).join('|');
  if (key !== lastTally) {
    lastTally = key;
    elTally.innerHTML = '';
    if (!counts.size) {
      const s = document.createElement('span');
      s.className = 'empty';
      s.textContent = frozen ? 'Frozen' : 'Nothing recognised yet';
      elTally.appendChild(s);
    } else {
      for (const [label, n] of [...counts].sort((a, b) => b[1] - a[1])) {
        const s = document.createElement('span');
        const hue = hueFor(label);
        s.style.borderColor = `hsl(${hue} 80% 62%)`;
        s.style.color = `hsl(${hue} 90% 76%)`;
        s.textContent = n > 1 ? `${label} ×${n}` : label;
        elTally.appendChild(s);
      }
    }
  }
  if (speak) announce(counts);
}

function announce(counts) {
  const t = Date.now();
  for (const label of counts.keys()) {
    if (t - (lastSpoken.get(label) || 0) < CFG.speakCooldown) continue;
    lastSpoken.set(label, t);
    try {
      const u = new SpeechSynthesisUtterance(label);
      u.rate = 1.05;
      speechSynthesis.speak(u);
    } catch (_) {}
  }
}

/* ---------- status + errors ---------- */
function setStatus(s) { elStatus.textContent = s; }
function fail(msg) {
  document.body.classList.remove('live');
  document.body.classList.add('error');
  setStatus('Stopped');
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
}

/* ---------- start ---------- */
$('start').addEventListener('click', async () => {
  const note = $('introNote');
  $('start').disabled = true;
  $('start').textContent = 'Starting…';
  try {
    note.textContent = 'Asking for camera permission';
    await openCamera();
    $('intro').remove();
    document.body.classList.remove('idle');
    document.body.classList.add('live');
    sizeCanvas();
    video.addEventListener('loadedmetadata', sizeCanvas);
    keepAwake();
    running = true;
    loop();
    await loadModel(modelBase);
  } catch (e) {
    console.error(e);
    $('start').disabled = false;
    $('start').textContent = 'Open the camera';
    if (e && (e.name === 'NotAllowedError' || e.name === 'SecurityError')) {
      note.textContent = 'Camera blocked. Allow it in your browser settings, then try again.';
    } else if (!isSecure()) {
      note.textContent = 'The camera needs an https:// address. Open this page over https.';
    } else {
      note.textContent = 'Could not start: ' + (e.message || e.name || 'unknown error');
    }
  }
});

function isSecure() {
  return location.protocol === 'https:' || ['localhost', '127.0.0.1'].includes(location.hostname);
}
if (!isSecure()) {
  $('introNote').textContent = 'This page must be served over https for the camera to work.';
}

/* ---------- controls ---------- */
$('btnFreeze').addEventListener('click', e => {
  frozen = !frozen;
  e.currentTarget.textContent = frozen ? 'Resume' : 'Freeze';
  e.currentTarget.setAttribute('aria-pressed', String(frozen));
  setStatus(frozen ? 'Frozen' : 'Live');
});

$('btnFlip').addEventListener('click', async () => {
  facing = facing === 'environment' ? 'user' : 'environment';
  video.style.transform = facing === 'user' ? 'scaleX(-1)' : 'none';
  tracker.tracks = [];
  try { await openCamera(); sizeCanvas(); }
  catch (e) { fail('That camera is not available on this device.'); }
});

$('btnSpeak').addEventListener('click', e => {
  speak = !speak;
  e.currentTarget.setAttribute('aria-pressed', String(speak));
  if (speak) { lastSpoken.clear(); try { speechSynthesis.cancel(); } catch (_) {} }
});

/* ---------- settings sheet ---------- */
const sheet = $('sheet'), scrim = $('scrim');
const openSheet = () => { sheet.classList.add('open'); scrim.classList.add('open'); };
const closeSheet = () => { sheet.classList.remove('open'); scrim.classList.remove('open'); };
$('btnSettings').addEventListener('click', openSheet);
$('close').addEventListener('click', closeSheet);
scrim.addEventListener('click', closeSheet);

$('conf').addEventListener('input', e => {
  scoreThreshold = e.target.value / 100;
  $('confOut').textContent = e.target.value + '%';
});
$('patience').addEventListener('input', e => {
  CFG.minHits = +e.target.value;
  $('patienceOut').textContent = e.target.value;
});

function toggleBtn(id, get, set) {
  const b = $(id);
  b.addEventListener('click', () => {
    set(!get());
    b.setAttribute('aria-pressed', String(get()));
    b.textContent = get() ? 'On' : 'Off';
  });
}
toggleBtn('tBox', () => showBoxes, v => showBoxes = v);
toggleBtn('tScore', () => showScores, v => showScores = v);
toggleBtn('tRaw', () => rawMode, v => rawMode = v);

async function switchModel(base, pressedFast) {
  if (base === modelBase || !running) return;
  $('mFast').setAttribute('aria-pressed', String(pressedFast));
  $('mAcc').setAttribute('aria-pressed', String(!pressedFast));
  tracker.tracks = [];
  try { await loadModel(base); }
  catch (e) { fail('Could not load that model. Check your connection.'); }
}
$('mFast').addEventListener('click', () => switchModel('lite_mobilenet_v2', true));
$('mAcc').addEventListener('click', () => switchModel('mobilenet_v2', false));

/* vocabulary list */
const vocab = $('vocab');
for (const c of COCO_CLASSES) {
  const s = document.createElement('span');
  s.textContent = c;
  vocab.appendChild(s);
}

/* ---------- service worker (makes it installable + faster on relaunch) ---------- */
if ('serviceWorker' in navigator && isSecure()) {
  addEventListener('load', () => navigator.serviceWorker.register('sw.js').catch(() => {}));
}
