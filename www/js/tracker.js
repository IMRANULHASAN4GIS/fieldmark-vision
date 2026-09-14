/* Multi-object tracker.
 *
 * Raw detector output is unusable as-is: boxes jump several pixels between
 * frames, labels flip between neighbouring classes, and objects blink out
 * for a frame whenever they are partly occluded. This module gives every
 * object a stable identity so those three problems can be fixed:
 *
 *   identity  -> greedy IoU association against last frame's tracks
 *   jitter    -> exponential moving average on box coordinates
 *   flicker   -> majority vote on the class over a sliding window
 *   blinking  -> hysteresis, so appearing and disappearing use different
 *                thresholds
 */

export const TRACK_CFG = {
  minHits: 3,       // consecutive frames before a track is drawn
  maxMisses: 9,     // frames without a match before a track is dropped
  iouGate: 0.30,    // minimum overlap for a detection to match a track
  boxAlpha: 0.38,   // EMA weight on new box coordinates
  scoreAlpha: 0.30,
  voteWindow: 12,   // how many frames of label history to vote over
};

export function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]);
  const y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const w = x2 - x1, h = y2 - y1;
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter);
}

export class Tracker {
  constructor() {
    this.tracks = [];
    this.nextId = 1;
  }

  reset() {
    this.tracks = [];
  }

  update(detections) {
    // score every track/detection pair, then take them best-first.
    // Hungarian assignment would be optimal; greedy is within a rounding
    // error at these object counts and costs nothing.
    const pairs = [];
    for (let t = 0; t < this.tracks.length; t++) {
      for (let d = 0; d < detections.length; d++) {
        const s = iou(this.tracks[t].box, detections[d].bbox);
        if (s >= TRACK_CFG.iouGate) pairs.push({ t, d, s });
      }
    }
    pairs.sort((a, b) => b.s - a.s);

    const usedTracks = new Set();
    const usedDets = new Set();
    for (const p of pairs) {
      if (usedTracks.has(p.t) || usedDets.has(p.d)) continue;
      usedTracks.add(p.t);
      usedDets.add(p.d);
      this._matched(this.tracks[p.t], detections[p.d]);
    }

    // Count the existing tracks BEFORE spawning. A track created this frame
    // has no index in usedTracks, so if it is included in the miss loop below
    // it is penalised for not matching the detection that created it.
    const existing = this.tracks.length;

    detections.forEach((d, i) => {
      if (!usedDets.has(i)) this._spawn(d);
    });

    for (let i = 0; i < existing; i++) {
      if (usedTracks.has(i)) continue;
      const tr = this.tracks[i];
      tr.misses++;
      tr.hits = Math.max(0, tr.hits - 1);
    }

    this.tracks = this.tracks.filter(tr => tr.misses <= TRACK_CFG.maxMisses);
  }

  _spawn(d) {
    this.tracks.push({
      id: this.nextId++,
      box: d.bbox.slice(),
      votes: [d.class],
      label: d.class,
      score: d.score,
      hits: 1,
      misses: 0,
      shown: false,
      born: performance.now(),
    });
  }

  _matched(tr, d) {
    const a = TRACK_CFG.boxAlpha;
    for (let i = 0; i < 4; i++) {
      tr.box[i] = tr.box[i] * (1 - a) + d.bbox[i] * a;
    }
    tr.score = tr.score * (1 - TRACK_CFG.scoreAlpha) + d.score * TRACK_CFG.scoreAlpha;
    tr.hits++;
    tr.misses = 0;

    tr.votes.push(d.class);
    if (tr.votes.length > TRACK_CFG.voteWindow) tr.votes.shift();

    // majority label over the window stops the "chair / couch / chair" churn
    const tally = Object.create(null);
    let best = tr.label, bestCount = 0;
    for (const v of tr.votes) {
      tally[v] = (tally[v] || 0) + 1;
      if (tally[v] > bestCount) { bestCount = tally[v]; best = v; }
    }
    tr.label = best;
  }

  // Hysteresis: different thresholds for appearing and disappearing.
  // A track earns its way on after minHits consecutive matches, and once
  // promoted it keeps its place until maxMisses drops it from the list
  // entirely. That asymmetry is what stops labels strobing when an object
  // is briefly occluded or motion-blurred.
  visible() {
    const out = [];
    for (const tr of this.tracks) {
      if (!tr.shown && tr.hits >= TRACK_CFG.minHits) tr.shown = true;
      if (tr.shown) out.push(tr);
    }
    return out;
  }
}
