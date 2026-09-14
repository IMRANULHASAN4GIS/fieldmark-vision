/* Model loading and segmentation post-processing.
 *
 * Two models run here, for two genuinely different problems:
 *
 *   COCO-SSD  detects THINGS - countable objects with edges. A person, a
 *             cup, a car. Output is boxes.
 *
 *   DeepLab   segments STUFF - regions with no instance boundary. Grass,
 *   (ADE20K)  sky, road, a tree canopy. Output is a per-pixel class map.
 *             You cannot draw a meaningful box around "grass", which is
 *             exactly why detection alone can never answer that question.
 *
 * Together they cover about 200 concepts. Segmentation is far slower, so
 * it runs on a long interval while detection runs every frame.
 */

/* ADE20K uses some awkward names. Rename to what a person would say. */
const RENAME = {
  earth: 'ground',
  windowpane: 'window',
  skyscraper: 'building',
  'pool table': 'pool table',
  ashcan: 'bin',
  minibike: 'motorbike',
  'crt screen': 'screen',
  hovel: 'hut',
  plaything: 'toy',
  apparel: 'clothing',
  bannister: 'railing',
  'trade name': 'sign',
  signboard: 'sign',
  conveyer: 'conveyor',
  'chest of drawers': 'drawers',
  swivel: 'swivel chair',
  palm: 'palm tree',
  land: 'ground',
  field: 'field',
  base: 'pedestal',
  case: 'display case',
  step: 'step',
  screen: 'screen',
  glass: 'glass',
  hood: 'extractor hood',
};

/* Classes the detector already handles well. Segmentation still tints them,
 * but does not add a second label on top of the detector's. */
const DETECTOR_COVERS = new Set([
  'person', 'car', 'bus', 'truck', 'boat', 'airplane', 'bicycle', 'motorbike',
  'chair', 'sofa', 'bed', 'toilet', 'television', 'refrigerator', 'oven',
  'microwave', 'sink', 'book', 'clock', 'vase', 'bottle', 'bench',
  'traffic light', 'animal', 'ball', 'bag',
]);

export const PERCEPTION = {
  detector: null,
  detectorBase: 'lite_mobilenet_v2',
  segmenter: null,
  segmenterQuantization: 2,
  segmenterReady: false,
  segmenterError: null,
};

export async function initBackend() {
  try {
    await tf.setBackend('webgl');
    await tf.ready();
  } catch (e) {
    await tf.ready();
  }
  return tf.getBackend();
}

export async function loadDetector(base = 'lite_mobilenet_v2') {
  if (typeof cocoSsd === 'undefined') throw new Error('object-detection library did not load');
  const m = await cocoSsd.load({ base });
  // cocoSsd.load() already compiles the model with a correctly shaped
  // internal warm-up tensor. Calling detect() with its 4-D warm-up tensor
  // would add a second batch dimension and fail with a rank-5 input.
  const previous = PERCEPTION.detector;
  PERCEPTION.detector = m;
  PERCEPTION.detectorBase = base;
  if (previous && previous !== m) previous.dispose();
  return m;
}

export async function loadSegmenter(quantizationBytes = 2) {
  if (PERCEPTION.segmenterReady && PERCEPTION.segmenterQuantization === quantizationBytes) {
    return PERCEPTION.segmenter;
  }
  const previous = PERCEPTION.segmenter;
  const previousQuantization = PERCEPTION.segmenterQuantization;
  PERCEPTION.segmenterReady = false;
  PERCEPTION.segmenterError = null;
  try {
    if (typeof deeplab === 'undefined') throw new Error('segmentation library did not load');
    const m = await deeplab.load({ base: 'ade20k', quantizationBytes });
    PERCEPTION.segmenter = m;
    PERCEPTION.segmenterQuantization = quantizationBytes;
    PERCEPTION.segmenterReady = true;
    if (previous && previous !== m) previous.dispose();
    return m;
  } catch (e) {
    PERCEPTION.segmenter = previous;
    PERCEPTION.segmenterQuantization = previousQuantization;
    PERCEPTION.segmenterReady = Boolean(previous);
    PERCEPTION.segmenterError = e;
    throw e;
  }
}

export function disposeModels() {
  if (PERCEPTION.detector) PERCEPTION.detector.dispose();
  if (PERCEPTION.segmenter) PERCEPTION.segmenter.dispose();
  PERCEPTION.detector = null;
  PERCEPTION.segmenter = null;
  PERCEPTION.segmenterReady = false;
}

/* ---------------------------------------------------------------- *
 * Segmentation post-processing
 *
 * deeplab.segment() hands back an RGBA colour map plus a legend of
 * {className: [r,g,b]}. We need three things it does not give us:
 *   1. our own colours, so the tint matches the label chips
 *   2. the share of frame each class occupies, to drop noise
 *   3. a label anchor point that actually sits inside the region
 * ---------------------------------------------------------------- */

export function analyseSegmentation(seg, colorForLabel, opts = {}) {
  const minShare = opts.minShare ?? 0.012;     // ignore anything under ~1.2% of frame
  const suppress = opts.suppress ?? new Set();
  const { legend, width, height, segmentationMap } = seg;

  // packed rgb -> {index, label}
  const byColor = new Map();
  const labels = [];
  let n = 0;
  for (const raw of Object.keys(legend)) {
    const [r, g, b] = legend[raw];
    const label = RENAME[raw] || raw;
    byColor.set((r << 16) | (g << 8) | b, n);
    labels.push(label);
    n++;
  }

  const total = width * height;
  const idMap = new Uint8Array(total);
  const out = new Uint8ClampedArray(total * 4);
  const counts = new Int32Array(n);
  const sumX = new Float64Array(n);
  const sumY = new Float64Array(n);

  // our palette, resolved once per class rather than per pixel
  const pr = new Uint8Array(n), pg = new Uint8Array(n), pb = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = colorForLabel(labels[i]);
    pr[i] = r; pg[i] = g; pb[i] = b;
  }

  let lastKey = -1, lastIdx = 0;
  for (let p = 0, i = 0; p < total; p++, i += 4) {
    const key = (segmentationMap[i] << 16) | (segmentationMap[i + 1] << 8) | segmentationMap[i + 2];
    let idx;
    if (key === lastKey) {
      idx = lastIdx;                 // regions are contiguous, so this hits often
    } else {
      idx = byColor.get(key);
      if (idx === undefined) idx = 0;
      lastKey = key; lastIdx = idx;
    }
    idMap[p] = idx;
    out[i] = pr[idx]; out[i + 1] = pg[idx]; out[i + 2] = pb[idx]; out[i + 3] = 255;
    counts[idx]++;
    sumX[idx] += p % width;
    sumY[idx] += (p / width) | 0;
  }

  const regionByLabel = new Map();
  for (let i = 0; i < n; i++) {
    const share = counts[i] / total;
    if (share < minShare) continue;
    const label = labels[i];
    if (suppress.has(label)) continue;
    const cx = sumX[i] / counts[i];
    const cy = sumY[i] / counts[i];
    const anchor = anchorInside(idMap, width, height, i, cx, cy);
    const existing = regionByLabel.get(label);
    if (!existing) {
      regionByLabel.set(label, { label, share, x: anchor.x, y: anchor.y, index: i, anchorShare: share });
    } else {
      existing.share += share;
      if (share > existing.anchorShare) {
        existing.x = anchor.x;
        existing.y = anchor.y;
        existing.index = i;
        existing.anchorShare = share;
      }
    }
  }
  const regions = [...regionByLabel.values()];
  for (const region of regions) delete region.anchorShare;
  regions.sort((a, b) => b.share - a.share);

  return { mask: out, width, height, regions, idMap };
}

/* The centroid of a concave region can sit outside it - the centre of a ring
 * of sky is the building in the middle. Walk outward until we land on a
 * pixel that really belongs to the class. */
function anchorInside(idMap, w, h, idx, cx, cy) {
  let x = Math.round(cx), y = Math.round(cy);
  x = Math.max(0, Math.min(w - 1, x));
  y = Math.max(0, Math.min(h - 1, y));
  if (idMap[y * w + x] === idx) return { x, y };

  const step = Math.max(2, Math.round(Math.min(w, h) / 64));
  for (let r = step; r < Math.max(w, h); r += step) {
    for (let a = 0; a < 16; a++) {
      const t = (a / 16) * Math.PI * 2;
      const px = Math.round(x + Math.cos(t) * r);
      const py = Math.round(y + Math.sin(t) * r);
      if (px < 0 || py < 0 || px >= w || py >= h) continue;
      if (idMap[py * w + px] === idx) return { x: px, y: py };
    }
  }
  return { x, y };
}

export { DETECTOR_COVERS };

/* The full vocabulary, for the "what it can see" panel. */
export const COCO_CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"];

export const ADE20K_CLASSES = ["wall","building","sky","floor","tree","ceiling","road","bed","window","grass","cabinet","sidewalk","person","ground","door","table","mountain","plant","curtain","chair","car","water","painting","sofa","shelf","house","sea","mirror","rug","field","armchair","seat","fence","desk","rock","wardrobe","lamp","bathtub","railing","cushion","pedestal","box","column","sign","drawers","counter","sand","sink","building","fireplace","refrigerator","grandstand","path","stairs","runway","display case","pool table","pillow","screen door","stairway","river","bridge","bookcase","blind","coffee table","toilet","flower","book","hill","bench","countertop","stove","palm tree","kitchen island","computer","swivel chair","boat","bar","arcade machine","hut","bus","towel","light","truck","tower","chandelier","awning","streetlight","booth","television","airplane","dirt track","clothing","pole","ground","railing","escalator","ottoman","bottle","buffet","poster","stage","van","ship","fountain","conveyor belt","canopy","washer","toy","swimming pool","stool","barrel","basket","waterfall","tent","bag","motorbike","cradle","oven","ball","food","step","tank","sign","microwave","pot","animal","bicycle","lake","dishwasher","screen","blanket","sculpture","extractor hood","sconce","vase","traffic light","tray","bin","fan","pier","screen","plate","monitor","bulletin board","shower","radiator","glass","clock","flag"];
