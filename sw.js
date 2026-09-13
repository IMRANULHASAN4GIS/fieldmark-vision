/* Caches the app shell so relaunch is instant, and the TF.js runtime plus
 * model weights so the second run does not re-download ~6 MB.
 * The camera itself needs no network, so after one successful run the app
 * works with no connection at all. */

const SHELL = 'fieldmark-shell-v1';
const DEPS = 'fieldmark-deps-v1';

const SHELL_FILES = [
  './',
  './index.html',
  './app.js',
  './manifest.webmanifest',
  './icons/icon-192.png',
  './icons/icon-512.png',
];

// model weights + tfjs runtime + fonts: immutable, so cache-first is safe
const DEP_HOSTS = [
  'cdn.jsdelivr.net',
  'storage.googleapis.com',
  'tfhub.dev',
  'www.kaggle.com',
  'fonts.googleapis.com',
  'fonts.gstatic.com',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(SHELL)
      .then(c => c.addAll(SHELL_FILES))
      .then(() => self.skipWaiting())
      .catch(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(k => k !== SHELL && k !== DEPS).map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  if (DEP_HOSTS.some(h => url.hostname.endsWith(h))) {
    e.respondWith(
      caches.open(DEPS).then(async cache => {
        const hit = await cache.match(req);
        if (hit) return hit;
        const res = await fetch(req);
        cache.put(req, res.clone()).catch(() => {});
        return res;
      }).catch(() => fetch(req))
    );
    return;
  }

  if (url.origin === location.origin) {
    // network-first so edits show up without a hard refresh, cache as fallback
    e.respondWith(
      fetch(req)
        .then(res => {
          const copy = res.clone();
          caches.open(SHELL).then(c => c.put(req, copy)).catch(() => {});
          return res;
        })
        .catch(() => caches.match(req).then(r => r || caches.match('./index.html')))
    );
  }
});
