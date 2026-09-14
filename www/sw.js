/* Offline shell plus runtime/model caching.
 * The first successful use of a model fills DEPS; later launches can reuse
 * those exact pinned resources without sending camera frames anywhere. */
const SHELL = 'fieldmark-shell-v2.1';
const DEPS  = 'fieldmark-deps-v2.1';

const SHELL_FILES = [
  './', './index.html', './manifest.webmanifest',
  './privacy.html',
  './js/app.js', './js/tracker.js', './js/perception.js', './js/runtime.js',
  './icons/icon-192.png', './icons/icon-512.png',
  './icons/icon-maskable-512.png',
];

const DEP_HOSTS = [
  'cdn.jsdelivr.net', 'storage.googleapis.com', 'tfhub.dev',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(SHELL).then(c => c.addAll(SHELL_FILES))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys
        .filter(k => k.startsWith('fieldmark-') && k !== SHELL && k !== DEPS)
        .map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // model weights and libraries are immutable, so cache-first is safe
  if (DEP_HOSTS.some(h => url.hostname.endsWith(h))) {
    e.respondWith(caches.open(DEPS).then(async cache => {
      const hit = await cache.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok || res.type === 'opaque') cache.put(req, res.clone()).catch(() => {});
      return res;
    }).catch(() => fetch(req)));
    return;
  }

  // our own files are network-first, so a push goes live without a hard refresh
  if (url.origin === location.origin) {
    e.respondWith(
      fetch(req).then(res => {
        if (res.ok) {
          const copy = res.clone();
          caches.open(SHELL).then(c => c.put(req, copy)).catch(() => {});
        }
        return res;
      }).catch(() => caches.match(req).then(r => {
        if (r) return r;
        if (req.mode === 'navigate') return caches.match('./index.html');
        return new Response('Offline', { status: 503, headers: { 'Content-Type': 'text/plain' } });
      }))
    );
  }
});
