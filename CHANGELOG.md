# Changelog

## 2.1.1 — Startup recovery fix

- Allowed TensorFlow.js's required dynamic-code bootstrap while keeping all
  runtime libraries pinned and protected by SHA-384 integrity checks.
- Rejects partially initialized TensorFlow/model libraries instead of treating
  a global placeholder as a successful load.
- Added camera and runtime-download timeouts so startup cannot remain stuck
  indefinitely.
- Added clear guidance for denied, missing, busy, unsupported, and unresponsive
  cameras, including embedded-browser limitations.
- Allowed and cached TensorFlow Hub's current Kaggle redirect path for the
  optional DeepLab surface model.
- Expanded the privacy disclosure for the TensorFlow Hub catalogue route.

## 2.1.0 — Professional test release

### Reliability

- Removed the invalid four-dimensional COCO-SSD warm-up tensor that could stop
  the app during startup.
- Added retryable, lazy model loading and preserved the active model when a
  replacement fails.
- Serialised model changes with inference and cleanly releases cameras, wake
  locks, speech, animation frames, and model memory.
- Restores the previous camera if switching lenses fails.
- Replaced the misleading display-refresh FPS value with completed-inference
  throughput.

### Privacy and security

- Kept camera frames on-device and added a plain-language privacy page.
- Removed remote font requests and all optional cloud-AI endpoints.
- Pinned TensorFlow.js, COCO-SSD, and DeepLab versions and added SHA-384
  Subresource Integrity verification.
- Added a restrictive Content Security Policy and production WebView settings.
- Limited service-worker cleanup to Fieldmark-owned caches.

### Experience and accessibility

- Starts with the faster object model and downloads segmentation only when the
  user selects Surfaces or Both.
- Keeps startup errors recoverable and explains first-use model downloads.
- Added keyboard focus management, Escape handling, a focus trap, and inert
  hidden settings controls.
- Added mask-safe artwork to Android builds.

### Engineering

- Added deterministic tests for IoU, tracking, smoothing, hysteresis, label
  voting, segmentation analysis, duplicate-region merging, and label anchors.
- Added a zero-dependency local development server and locked npm dependency
  tree.
- GitHub Pages and Android workflows now validate the app before publishing or
  building an artifact.
