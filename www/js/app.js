/* Fieldmark — orchestration, rendering and UI.
 *
 * The one architectural decision that matters here: detection and
 * segmentation both want the GPU, and if they overlap the whole thing
 * stutters. So they are serialised through a single scheduler, and
 * segmentation's interval adapts to how slow it is actually running on
 * this particular phone.
 */

import { Tracker, TRACK_CFG } from './tracker.js';
import {
  PERCEPTION, initBackend, loadDetector, loadSegmenter,
  analyseSegmentation, disposeModels, DETECTOR_COVERS, COCO_CLASSES, ADE20K_CLASSES,
} from './perception.js';
import { loadObjectRuntime, loadSurfaceRuntime, serviceWorkerReady } from './runtime.js';

const $ = id => document.getElementById(id);
const video = $('video');
const cv = $('overlay');
const ctx = cv.getContext('2d');

/* ---------------- state ---------------- */
const S = {
  mode: 'objects',         // objects | scene | all
  running: false,
  starting: false,
  frozen: false,
  busy: false,
  modelLoading: false,
  surfacesLoading: false,
  speak: false,
  facing: 'environment',
  stream: null,

  scoreThreshold: 0.50,
  maxBoxes: 20,
  maxLabels: 12,
  showBoxes: true,
  showScores: true,
  rawMode: false,

  tint: 0.32,
  minShare: 0.012,
  segPace: 'auto',         // slow | auto | fast
  segInterval: 1100,
  lastSegAt: 0,
  segMs: 0,
  detMs: 0,

  rawDets: [],
  seg: null,               // {maskCanvas, regions, width, height}
  fpsHist: [],
  lastDetectionAt: 0,
  spoken: new Map(),
};

const tracker = new Tracker();
const maskCanvas = document.createElement('canvas');
const maskCtx = maskCanvas.getContext('2d');
let frameRequest = 0;

/* ---------------- palette ----------------
 * One stable hue per class name, steered clear of the amber chrome so the
 * interface and the data never read as the same thing. */
const hueCache = new Map();
function hueFor(label) {
  if (hueCache.has(label)) return hueCache.get(label);
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) % 360;
  if (h > 26 && h < 64) h = (h + 152) % 360;
  hueCache.set(label, h);
  return h;
}
const cssFor = (label, l = 66, s = 88) => `hsl(${hueFor(label)} ${s}% ${l}%)`;

function rgbFor(label) {
  const h = hueFor(label) / 360, s = 0.72, l = 0.55;
  const k = n => (n + h * 12) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = n => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))));
  return [f(0), f(8), f(4)];
}

/* ---------------- camera ---------------- */
async function openCamera() {
  if (S.stream) S.stream.getTracks().forEach(t => t.stop());
  S.stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: S.facing }, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  video.srcObject = S.stream;
  await video.play();
}

function stopCamera() {
  if (S.stream) S.stream.getTracks().forEach(track => track.stop());
  S.stream = null;
  video.srcObject = null;
}

let wakeLock = null;
async function keepAwake() {
  try {
    if ('wakeLock' in navigator) {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => { wakeLock = null; }, { once: true });
    }
  }
  catch (_) {}
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && S.running && !wakeLock) keepAwake();
});

/* ---------------- geometry ----------------
 * The canvas sits on top of a video using object-fit: cover. If this
 * transform does not match what CSS is doing, every box lands in the wrong
 * place — the single commonest bug in camera overlays. */
let view = { scale: 1, dx: 0, dy: 0, w: 0, h: 0 };
function sizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const scale = Math.max(w / vw, h / vh);
  view = { scale, dx: (w - vw * scale) / 2, dy: (h - vh * scale) / 2, w, h, vw, vh, dpr };
}
addEventListener('resize', sizeCanvas);
addEventListener('orientationchange', () => setTimeout(sizeCanvas, 300));

const mirrored = () => S.facing === 'user';

function boxToScreen(b) {
  let x = b[0] * view.scale + view.dx;
  const y = b[1] * view.scale + view.dy;
  const w = b[2] * view.scale, h = b[3] * view.scale;
  if (mirrored()) x = view.w - (x + w);
  return [x, y, w, h];
}
function pointToScreen(vx, vy) {
  let x = vx * view.scale + view.dx;
  const y = vy * view.scale + view.dy;
  if (mirrored()) x = view.w - x;
  return [x, y];
}

/* ---------------- scheduler ----------------
 * Only ever one model on the GPU at a time. Segmentation gets priority
 * when its interval is due, because a late segmentation is more visible
 * than a dropped detection frame. */
function loop() {
  if (!S.running) return;

  const now = performance.now();

  if (!S.busy && !S.modelLoading && !S.frozen && video.readyState >= 2) {
    const wantSeg = S.mode !== 'objects' && PERCEPTION.segmenterReady
      && (now - S.lastSegAt) > S.segInterval;
    const wantDet = S.mode !== 'scene' && PERCEPTION.detector;

    if (wantSeg) runSegmentation();
    else if (wantDet) runDetection();
  }

  render();
  frameRequest = requestAnimationFrame(loop);
}

async function runDetection() {
  S.busy = true;
  try {
    const t0 = performance.now();
    const dets = await PERCEPTION.detector.detect(video, S.maxBoxes, S.scoreThreshold);
    const finishedAt = performance.now();
    S.detMs = finishedAt - t0;
    if (S.lastDetectionAt) {
      S.fpsHist.push(1000 / (finishedAt - S.lastDetectionAt));
      if (S.fpsHist.length > 30) S.fpsHist.shift();
    }
    S.lastDetectionAt = finishedAt;
    S.rawDets = dets;
    tracker.update(dets);
  } catch (e) {
    console.error('detect', e);
  } finally {
    S.busy = false;
  }
}

async function runSegmentation() {
  S.busy = true;
  S.lastSegAt = performance.now();
  try {
    const t0 = performance.now();
    const out = await PERCEPTION.segmenter.segment(video);
    const suppress = new Set(
      S.mode === 'all' ? [...DETECTOR_COVERS] : []
    );
    const a = analyseSegmentation(out, rgbFor, { minShare: S.minShare, suppress });

    if (maskCanvas.width !== a.width || maskCanvas.height !== a.height) {
      maskCanvas.width = a.width;
      maskCanvas.height = a.height;
    }
    maskCtx.putImageData(new ImageData(a.mask, a.width, a.height), 0, 0);
    S.seg = { regions: a.regions, width: a.width, height: a.height };
    S.segMs = performance.now() - t0;
    pace();
  } catch (e) {
    console.error('segment', e);
    S.seg = null;
  } finally {
    S.busy = false;
  }
}

/* Thermal throttling is real: sustained inference heats the phone and
 * everything slows down after a few minutes. Rather than let the frame rate
 * collapse, give segmentation proportionally less of the budget. */
function pace() {
  if (S.segPace === 'slow') { S.segInterval = 2600; return; }
  if (S.segPace === 'fast') { S.segInterval = Math.max(500, S.segMs * 1.2); return; }
  const target = S.mode === 'scene' ? S.segMs * 1.3 : S.segMs * 4.5;
  S.segInterval = Math.max(700, Math.min(4000, target));
}

/* ---------------- rendering ---------------- */
function render() {
  if (!view.w) sizeCanvas();
  ctx.clearRect(0, 0, view.w, view.h);

  if (S.mode !== 'objects' && S.seg) drawSurfaces();
  if (S.mode !== 'scene') drawObjects();

  updateReadouts();
}

function drawSurfaces() {
  const { width: mw, height: mh, regions } = S.seg;
  const dw = view.vw * view.scale, dh = view.vh * view.scale;

  if (S.tint > 0) {
    ctx.save();
    ctx.globalAlpha = S.tint;
    ctx.globalCompositeOperation = 'lighten';
    if (mirrored()) { ctx.translate(view.w, 0); ctx.scale(-1, 1); }
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(maskCanvas, view.dx, view.dy, dw, dh);
    ctx.restore();
  }

  // Regions get a tick and a name, not a box. A box around "sky" would be
  // a lie about what the model actually knows.
  ctx.save();
  ctx.font = '500 13.5px Inter, system-ui, sans-serif';
  ctx.textBaseline = 'alphabetic';
  const placed = [];
  for (const r of regions.slice(0, 8)) {
    const [sx, sy] = pointToScreen(r.x / mw * view.vw, r.y / mh * view.vh);
    if (sx < 4 || sx > view.w - 4 || sy < 40 || sy > view.h - 140) continue;

    const text = r.label;
    const tw = ctx.measureText(text).width;
    let x = Math.max(10, Math.min(sx - tw / 2, view.w - tw - 10));
    let y = sy;
    for (let g = 0; g < 6; g++) {
      const clash = placed.find(p => Math.abs(p.y - y) < 26 && x < p.x + p.w && x + tw > p.x);
      if (!clash) break;
      y = clash.y + 28;
    }
    placed.push({ x, y, w: tw });

    ctx.strokeStyle = 'rgba(8,11,14,.72)';
    ctx.lineWidth = 3.5;
    ctx.strokeText(text, x, y);
    ctx.fillStyle = cssFor(r.label, 80);
    ctx.fillText(text, x, y);

    ctx.fillStyle = cssFor(r.label, 62);
    ctx.fillRect(x, y + 5, tw, 1.5);
  }
  ctx.restore();
}

function drawObjects() {
  const items = S.rawMode
    ? S.rawDets.map(d => ({ box: d.bbox, label: d.class, score: d.score }))
    : tracker.visible();

  const ordered = items.slice()
    .sort((a, b) => (b.box[2] * b.box[3]) - (a.box[2] * a.box[3]))
    .slice(0, S.maxLabels);

  const placed = [];
  for (const it of ordered) {
    const [x, y, w, h] = boxToScreen(it.box);
    const stroke = cssFor(it.label);

    if (S.showBoxes) {
      ctx.save();
      ctx.strokeStyle = 'rgba(8,11,14,.55)';
      ctx.lineWidth = 4.5;
      corners(x, y, w, h);
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2;
      corners(x, y, w, h);
      ctx.restore();
    }

    ctx.font = '500 13px Inter, system-ui, sans-serif';
    const text = S.showScores ? `${it.label}  ${Math.round(it.score * 100)}%` : it.label;
    const tw = ctx.measureText(text).width;
    const pw = tw + 18, ph = 22;
    let px = Math.max(6, Math.min(x, view.w - pw - 6));
    let py = y - ph - 5;
    if (py < 46) py = y + 6;
    for (let g = 0; g < 8; g++) {
      const clash = placed.find(r => px < r.x + r.w && px + pw > r.x && py < r.y + r.h && py + ph > r.y);
      if (!clash) break;
      py = clash.y + clash.h + 3;
    }
    placed.push({ x: px, y: py, w: pw, h: ph });

    ctx.fillStyle = 'rgba(8,11,14,.84)';
    ctx.fillRect(px, py, pw, ph);
    ctx.fillStyle = stroke;
    ctx.fillRect(px, py, 2, ph);
    ctx.fillStyle = '#EEF3F6';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, px + 9, py + ph / 2 + 0.5);
  }
}

// Open corner brackets rather than a closed rectangle: far less visual noise
// over a busy scene, and four corners read as a sight rather than a sticker.
function corners(x, y, w, h) {
  const c = Math.max(8, Math.min(22, Math.min(w, h) * 0.24));
  ctx.beginPath();
  ctx.moveTo(x, y + c); ctx.lineTo(x, y); ctx.lineTo(x + c, y);
  ctx.moveTo(x + w - c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + c);
  ctx.moveTo(x + w, y + h - c); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - c, y + h);
  ctx.moveTo(x + c, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - c);
  ctx.stroke();
}

/* ---------------- readouts ---------------- */
let lastKey = '';
function updateReadouts() {
  const fps = S.mode !== 'scene' && S.fpsHist.length
    ? S.fpsHist.reduce((a, b) => a + b) / S.fpsHist.length
    : 0;
  $('fps').textContent = fps ? fps.toFixed(0) : '—';
  $('ms').textContent = S.mode !== 'scene' && S.detMs ? S.detMs.toFixed(0) : '—';
  $('segms').textContent = S.mode !== 'objects' && S.segMs ? S.segMs.toFixed(0) : '—';

  const things = new Map();
  if (S.mode !== 'scene') {
    for (const t of tracker.visible()) things.set(t.label, (things.get(t.label) || 0) + 1);
  }
  const stuff = (S.mode !== 'objects' && S.seg) ? S.seg.regions.slice(0, 6).map(r => r.label) : [];

  const key = [...things].map(([k, v]) => k + v).join('|') + '::' + stuff.join('|');
  if (key === lastKey) { if (S.speak) announce(things, stuff); return; }
  lastKey = key;

  const el = $('tally');
  el.innerHTML = '';
  if (!things.size && !stuff.length) {
    const s = document.createElement('span');
    s.className = 'empty';
    s.textContent = S.frozen ? 'Frozen' : 'Nothing recognised yet';
    el.appendChild(s);
  } else {
    for (const [label, n] of [...things].sort((a, b) => b[1] - a[1])) {
      el.appendChild(chip(label, n > 1 ? `${label} ×${n}` : label, 'thing'));
    }
    for (const label of stuff) el.appendChild(chip(label, label, 'stuff'));
  }
  if (S.speak) announce(things, stuff);
}

function chip(label, text, kind) {
  const s = document.createElement('span');
  s.className = kind;
  s.style.borderColor = cssFor(label, 60);
  s.style.color = cssFor(label, 78);
  const i = document.createElement('i');
  i.style.background = kind === 'stuff' ? cssFor(label, 62) : 'transparent';
  i.style.border = kind === 'stuff' ? 'none' : `1.5px solid ${cssFor(label, 66)}`;
  s.append(i, document.createTextNode(text));
  return s;
}

function announce(things, stuff) {
  const now = Date.now();
  for (const label of [...things.keys(), ...stuff]) {
    if (now - (S.spoken.get(label) || 0) < 7000) continue;
    S.spoken.set(label, now);
    try {
      const u = new SpeechSynthesisUtterance(label);
      u.rate = 1.05;
      speechSynthesis.speak(u);
    } catch (_) {}
  }
}

/* ---------------- capture ---------------- */
async function capture() {
  const c = document.createElement('canvas');
  c.width = Math.round(view.w * view.dpr);
  c.height = Math.round(view.h * view.dpr);
  const g = c.getContext('2d');
  g.scale(view.dpr, view.dpr);
  g.fillStyle = '#10161C';
  g.fillRect(0, 0, view.w, view.h);

  g.save();
  if (mirrored()) { g.translate(view.w, 0); g.scale(-1, 1); }
  g.drawImage(video, view.dx, view.dy, view.vw * view.scale, view.vh * view.scale);
  g.restore();
  g.drawImage(cv, 0, 0, view.w, view.h);

  const blob = await new Promise(res => c.toBlob(res, 'image/png'));
  if (!blob) return toast('Could not build the image.', true);
  const file = new File([blob], `fieldmark-${Date.now()}.png`, { type: 'image/png' });

  // Share sheet first: on iOS a plain download link does nothing useful.
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try { await navigator.share({ files: [file] }); return; }
    catch (error) { if (error?.name === 'AbortError') return; }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = file.name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  toast('Saved to your downloads.');
}

/* ---------------- status ---------------- */
const setStatus = s => { $('statusText').textContent = s; };
let toastTimer = null;
function toast(msg, bad = false) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.toggle('bad', bad);
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 5200);
}

const isSecure = () =>
  location.protocol === 'https:' || ['localhost', '127.0.0.1'].includes(location.hostname);

/* ---------------- startup ---------------- */
$('start').addEventListener('click', async () => {
  if (S.starting) return;
  const note = $('introNote');
  const btn = $('start');
  S.starting = true;
  btn.disabled = true;
  btn.textContent = 'Starting…';
  try {
    note.textContent = 'Waiting for camera permission';
    await openCamera();
    note.textContent = 'Downloading the object model';
    await loadObjectRuntime();
    note.textContent = 'Optimising recognition for this device';
    await initBackend();
    await loadDetector(PERCEPTION.detectorBase);

    document.body.classList.remove('idle');
    document.body.classList.add('live');
    video.addEventListener('loadedmetadata', sizeCanvas);
    sizeCanvas();
    keepAwake();
    S.running = true;
    setStatus(`Live · objects · ${tf.getBackend()}`);
    loop();
    $('intro').remove();
  } catch (e) {
    console.error(e);
    S.running = false;
    if (frameRequest) cancelAnimationFrame(frameRequest);
    frameRequest = 0;
    stopCamera();
    disposeModels();
    btn.disabled = false;
    btn.textContent = 'Open the camera';
    if (e && (e.name === 'NotAllowedError' || e.name === 'SecurityError')) {
      note.textContent = 'Camera blocked. Allow it in your browser settings, then try again.';
    } else if (!isSecure()) {
      note.textContent = 'The camera needs an https:// address. Open this page over https.';
    } else {
      note.textContent = `Could not start: ${e.message || e.name || 'unknown error'}. Check your connection and try again.`;
    }
  } finally {
    S.starting = false;
  }
});

if (!isSecure()) {
  $('introNote').textContent = 'This page must be served over https for the camera to work.';
}

serviceWorkerReady.then(ready => {
  document.documentElement.dataset.offlineReady = String(ready);
});

/* ---------------- controls ---------------- */
function applyMode(m) {
  const previousMode = S.mode;
  S.mode = m;
  $('mdObjects').setAttribute('aria-pressed', String(m === 'objects'));
  $('mdScene').setAttribute('aria-pressed', String(m === 'scene'));
  $('mdAll').setAttribute('aria-pressed', String(m === 'all'));
  if (m === 'objects') S.seg = null;
  if (m !== 'objects') S.lastSegAt = 0;
  if (previousMode === 'scene' && m !== 'scene') tracker.reset();
  lastKey = '';
  pace();
}

const waitForInference = async () => {
  while (S.busy) await new Promise(resolve => setTimeout(resolve, 25));
};

async function ensureSurfaces() {
  if (PERCEPTION.segmenterReady) return true;
  if (S.surfacesLoading) return false;
  S.surfacesLoading = true;
  $('modes').setAttribute('aria-busy', 'true');
  [...$('modes').querySelectorAll('button')].forEach(button => button.disabled = true);
  setStatus('Downloading surfaces');
  try {
    // Keep object detection responsive while the small library downloads,
    // then take exclusive ownership of the model runtime for initialization.
    await loadSurfaceRuntime();
    S.modelLoading = true;
    await waitForInference();
    await loadSegmenter(2);
    setStatus('Live · surfaces ready');
    return true;
  } catch (error) {
    console.error(error);
    setStatus('Live · objects only');
    toast('Surface recognition could not load. Check your connection and try again.', true);
    return false;
  } finally {
    S.modelLoading = false;
    S.surfacesLoading = false;
    $('modes').removeAttribute('aria-busy');
    [...$('modes').querySelectorAll('button')].forEach(button => button.disabled = false);
  }
}

async function requestMode(mode) {
  if (mode !== 'objects' && !(await ensureSurfaces())) {
    applyMode('objects');
    return;
  }
  applyMode(mode);
  setStatus(mode === 'objects' ? 'Live · objects' : mode === 'scene' ? 'Live · surfaces' : 'Live · objects + surfaces');
}

$('mdObjects').addEventListener('click', () => requestMode('objects'));
$('mdScene').addEventListener('click', () => requestMode('scene'));
$('mdAll').addEventListener('click', () => requestMode('all'));

$('btnFreeze').addEventListener('click', e => {
  S.frozen = !S.frozen;
  e.currentTarget.textContent = S.frozen ? 'Resume' : 'Freeze';
  e.currentTarget.setAttribute('aria-pressed', String(S.frozen));
  setStatus(S.frozen ? 'Frozen' : 'Live');
});

$('btnShot').addEventListener('click', capture);

$('btnFlip').addEventListener('click', async () => {
  const previousFacing = S.facing;
  S.facing = S.facing === 'environment' ? 'user' : 'environment';
  video.style.transform = mirrored() ? 'scaleX(-1)' : 'none';
  tracker.reset();
  S.seg = null;
  try { await openCamera(); sizeCanvas(); }
  catch (_) {
    S.facing = previousFacing;
    video.style.transform = mirrored() ? 'scaleX(-1)' : 'none';
    try { await openCamera(); sizeCanvas(); } catch (_) {}
    toast('That camera is not available on this device.', true);
  }
});

$('btnSpeak').addEventListener('click', e => {
  S.speak = !S.speak;
  e.currentTarget.setAttribute('aria-pressed', String(S.speak));
  if (S.speak) { S.spoken.clear(); try { speechSynthesis.cancel(); } catch (_) {} }
});

/* ---------------- tuning sheet ---------------- */
const sheet = $('sheet'), scrim = $('scrim');
let settingsReturnFocus = null;
const focusableSelector = 'button:not([disabled]), input:not([disabled]), a[href]';
$('btnSettings').addEventListener('click', event => {
  settingsReturnFocus = event.currentTarget;
  sheet.inert = false;
  sheet.setAttribute('aria-hidden', 'false');
  sheet.classList.add('open');
  scrim.classList.add('open');
  $('close').focus();
});
const closeSheet = () => {
  sheet.classList.remove('open');
  scrim.classList.remove('open');
  sheet.setAttribute('aria-hidden', 'true');
  sheet.inert = true;
  settingsReturnFocus?.focus();
};
$('close').addEventListener('click', closeSheet);
scrim.addEventListener('click', closeSheet);
sheet.addEventListener('keydown', event => {
  if (event.key === 'Escape') { closeSheet(); return; }
  if (event.key !== 'Tab') return;
  const focusable = [...sheet.querySelectorAll(focusableSelector)].filter(el => !el.hidden);
  if (!focusable.length) return;
  const first = focusable[0], last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
  if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
});

$('conf').addEventListener('input', e => {
  S.scoreThreshold = e.target.value / 100;
  $('confOut').textContent = e.target.value + '%';
});
$('patience').addEventListener('input', e => {
  TRACK_CFG.minHits = +e.target.value;
  $('patienceOut').textContent = e.target.value;
});
$('tint').addEventListener('input', e => {
  S.tint = e.target.value / 100;
  $('tintOut').textContent = e.target.value + '%';
});
$('minshare').addEventListener('input', e => {
  S.minShare = e.target.value / 1000;
  $('minshareOut').textContent = (e.target.value / 10).toFixed(1) + '%';
  S.lastSegAt = 0;
});

function toggle(id, get, set) {
  const b = $(id);
  b.addEventListener('click', () => {
    set(!get());
    b.setAttribute('aria-pressed', String(get()));
    b.textContent = get() ? 'On' : 'Off';
  });
}
toggle('tBox', () => S.showBoxes, v => S.showBoxes = v);
toggle('tScore', () => S.showScores, v => S.showScores = v);
toggle('tRaw', () => S.rawMode, v => S.rawMode = v);

async function switchDetector(base, fast) {
  if (base === PERCEPTION.detectorBase || !S.running) return;
  if (S.modelLoading) return toast('Another model is still loading.');
  S.modelLoading = true;
  await waitForInference();
  tracker.reset();
  setStatus('Loading objects');
  try {
    await loadDetector(base);
    $('mFast').setAttribute('aria-pressed', String(fast));
    $('mAcc').setAttribute('aria-pressed', String(!fast));
    setStatus('Live · objects');
  } catch (error) {
    console.error(error);
    setStatus('Live · previous detector');
    toast('Could not load that detector. The previous model is still active.', true);
  } finally {
    S.modelLoading = false;
  }
}
$('mFast').addEventListener('click', () => switchDetector('lite_mobilenet_v2', true));
$('mAcc').addEventListener('click', () => switchDetector('mobilenet_v2', false));

function setPace(p) {
  S.segPace = p;
  $('sSlow').setAttribute('aria-pressed', String(p === 'slow'));
  $('sAuto').setAttribute('aria-pressed', String(p === 'auto'));
  $('sFast').setAttribute('aria-pressed', String(p === 'fast'));
  pace();
}
$('sSlow').addEventListener('click', () => setPace('slow'));
$('sAuto').addEventListener('click', () => setPace('auto'));
$('sFast').addEventListener('click', () => setPace('fast'));

/* vocabulary lists */
for (const [el, list] of [[$('vocabThings'), COCO_CLASSES], [$('vocabStuff'), ADE20K_CLASSES]]) {
  const seen = new Set();
  for (const c of list) {
    if (seen.has(c)) continue;
    seen.add(c);
    const s = document.createElement('span');
    s.textContent = c;
    el.appendChild(s);
  }
}

/* ---------------- lifecycle ---------------- */
addEventListener('pagehide', event => {
  if (event.persisted) return;
  S.running = false;
  if (frameRequest) cancelAnimationFrame(frameRequest);
  frameRequest = 0;
  stopCamera();
  disposeModels();
  try { speechSynthesis.cancel(); } catch (_) {}
});

addEventListener('offline', () => toast('Offline · previously loaded models remain available.'));
addEventListener('online', () => toast('Back online.'));
