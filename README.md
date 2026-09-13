# Fieldmark — prototype 1

Live camera object recognition that installs to a phone home screen. Everything
runs on the device; no frame is ever uploaded.

## Getting it onto your phone

The camera API only works over `https://`, so the folder has to be served — you
cannot just open `index.html` from your files. Any of these work:

**Netlify Drop (fastest, no account needed)**
1. Go to `app.netlify.com/drop`
2. Drag this whole folder in (or upload `fieldmark.zip` — it accepts zips)
3. You get an `https://….netlify.app` address. Open it on your phone.

**GitHub Pages** — push these files to a repo, then Settings → Pages → deploy
from branch `main`, folder `/root`.

**Your own machine** — `npx serve` gives you a local address, but phones need
https, so use `npx localtunnel --port 3000` or `ngrok http 3000` to expose it.

## Installing it

- **Android / Chrome** — open the address, tap ⋮ → *Add to Home screen* (or take
  the install prompt when it appears).
- **iPhone / Safari** — open the address in **Safari** (not Chrome), tap the
  share button → *Add to Home Screen*.

It then launches fullscreen with its own icon, no browser bars.

## Files

| | |
|---|---|
| `index.html` | markup and all styling |
| `app.js` | camera, inference loop, tracker, smoothing, rendering |
| `sw.js` | service worker — caches shell and model so relaunch is offline |
| `manifest.webmanifest` | makes it installable |
| `icons/` | home screen icons |

## What is actually running

COCO-SSD (MobileNet backbone) via TensorFlow.js, WebGL backend. Two weights
available in **Tune → Model**: `lite_mobilenet_v2` (fast, default) and
`mobilenet_v2` (slower, more accurate). 80 classes — the full list is in
**Tune → What it can see**.

The layer worth reading is the tracker in `app.js`. Raw detector output flickers
badly frame to frame. On top of it sits:

- greedy IoU association giving each object a stable id across frames
- exponential moving average on box coordinates (`CFG.boxAlpha`)
- majority vote on the class label over a 12-frame window
- hysteresis — appear after `minHits` frames, disappear after `maxMisses`

Turn on **Tune → Unsmoothed output** to see the raw detector and compare.

## Known limits of this build

- 80 classes only. No tree, grass, sky, streetlight, desk, or flower — those
  need either a fine-tuned model or an open-vocabulary one.
- Small, distant, occluded, blurred and low-light objects are missed.
- Sustained use heats the phone and the frame rate drops. Not yet adaptive.
- iOS caps web camera resolution and WebGL throughput below what a native app
  gets, so iPhone frame rates will look worse here than they would in a real app.

## Tuning

All constants live in the `CFG` object at the top of `app.js`.
