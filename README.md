# Fieldmark Vision

Private, on-device camera recognition for phones. Fieldmark detects countable
objects with COCO-SSD and, when requested, segments surfaces such as sky,
grass, roads, buildings, and water with DeepLab ADE20K.

Camera frames stay on the device. There are no accounts, analytics, advertising
SDKs, or application-owned image servers.

## Product features

- Fast object detection with stable IDs, label voting, box smoothing, and
  appearance/disappearance hysteresis
- Optional surface segmentation with adaptive pacing for sustained use
- Objects, Surfaces, and combined display modes
- Confidence, smoothing, overlay, and performance controls
- Rear/front camera switching with recovery when a camera is unavailable
- Freeze, local capture/share, spoken labels, wake lock, and portrait PWA mode
- Installable HTTPS web app plus a reproducible Capacitor Android debug build
- Keyboard-accessible settings and an in-app privacy disclosure

## Privacy and offline behaviour

Inference happens locally in TensorFlow.js. On first use, the browser downloads
pinned JavaScript libraries from jsDelivr and model files from Google-hosted
TensorFlow locations. The optional surface model may route through TensorFlow
Hub's Kaggle catalogue. Those providers receive ordinary web connection
metadata, but Fieldmark does not send them camera frames.

The service worker caches the application shell and each model after its first
successful use. A previously loaded mode can then start offline. A model that
has never been loaded still needs a connection once. See `www/privacy.html` for
the user-facing disclosure.

## Run and validate locally

Requirements: Node.js 20 or newer.

```bash
npm ci
npm run validate
npm run serve
```

Open `http://127.0.0.1:4173`. Localhost is treated as a secure browser context,
so the camera API can be tested there. Opening `www/index.html` directly will
not work.

`npm run validate` performs syntax validation and runs the Node test suite for
tracking, IoU, hysteresis, label voting, segmentation shares, suppression,
palette mapping, and region-label anchors.

## Publish with GitHub Pages

1. Push the repository to GitHub.
2. Open **Settings → Pages** and select **GitHub Actions** as the source.
3. Push to `main`, or run **Publish web app** from the Actions tab.

The workflow validates the application before uploading only the `www/`
directory. GitHub Pages supplies the HTTPS origin required by the camera API.

## Build an Android test APK

Open **Actions → Build Android APK → Run workflow**. The workflow installs the
locked dependency tree, runs validation, creates a fresh Capacitor Android
project, adds the camera permission and Fieldmark launcher artwork, then uploads
`fieldmark-apk` as a workflow artifact.

The result is a debug APK for direct testing and sideloading. It is not signed
or configured for Play Store submission. A store release still needs a durable
application ID, release keystore, signing configuration, versioning policy,
store listing, and completed privacy/data-safety declarations.

## Architecture

- `www/js/app.js` — camera lifecycle, task scheduler, rendering, controls
- `www/js/tracker.js` — IoU association, EMA smoothing, voting, hysteresis
- `www/js/perception.js` — model ownership and segmentation analysis
- `www/js/runtime.js` — pinned, retryable, service-worker-aware dependency loader
- `www/sw.js` — scoped application and model caching
- `tests/core.test.js` — deterministic tests for pure recognition logic
- `scripts/patch-android.mjs` — repeatable native permission/icon patch

Detection and segmentation never run concurrently. Segmentation is lazy-loaded
only when Surfaces or Both is selected, so object detection becomes useful
quickly and lower-powered devices do not pay the second model's cost unless the
user asks for it.

## Known limits

- COCO-SSD recognises 80 fixed object classes; DeepLab ADE20K recognises 150
  fixed scene classes.
- Small, distant, occluded, blurred, and poorly lit subjects remain difficult.
- Surface boundaries are coarse and surface classes are not instance counts.
- Sustained dual-model use heats phones; adaptive pacing reduces but cannot
  eliminate thermal throttling.
- Browser and WebView camera/GPU performance differs across devices, so test on
  the actual target phones before wider distribution.

## Release checklist

- Run `npm run validate`.
- Test rear and front cameras on at least one current Android phone and iPhone.
- Test first-load and previously-cached offline startup.
- Test permission denial, retry, camera switching, capture/share, speech, and
  orientation changes.
- Review `www/privacy.html` whenever network behaviour changes.
