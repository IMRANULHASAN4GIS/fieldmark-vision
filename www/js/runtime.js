/* Runtime dependency loader.
 *
 * The model libraries are loaded only when their feature is requested. On a
 * PWA-capable browser we wait briefly for the service worker to control the
 * page first, allowing the exact pinned files to be cached for later offline
 * launches. Camera frames never pass through this loader or leave the device.
 */

export const RUNTIME_URLS = Object.freeze({
  tensorflow: Object.freeze({
    url: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
    integrity: 'sha384-vE8hbVJ4lezako5rlvE7bY0BVzWlFhZncPlckrqNwcUQpVtgbENTgZ8TBbnPjZre',
  }),
  detector: Object.freeze({
    url: 'https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js',
    integrity: 'sha384-7qLdgfEQyO9ZQi9ArRHigK+IBto4XPk468jAqc+fnsXaZIcMAhQeLwzggRK7aESl',
  }),
  segmenter: Object.freeze({
    url: 'https://cdn.jsdelivr.net/npm/@tensorflow-models/deeplab@0.2.1/dist/deeplab.min.js',
    integrity: 'sha384-EqnLdEgSTEmbYwhnAjJsUXQGYQcxeXI629tnh/97pA05FLPQFuDU/acTWoM8SRQg',
  }),
});

const pending = new Map();
const SCRIPT_TIMEOUT_MS = 45000;

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function registerWorker() {
  if (!('serviceWorker' in navigator) || !window.isSecureContext) return false;
  try {
    await navigator.serviceWorker.register('./sw.js');
    await navigator.serviceWorker.ready;
    if (!navigator.serviceWorker.controller) {
      await Promise.race([
        new Promise(resolve => navigator.serviceWorker.addEventListener('controllerchange', resolve, { once: true })),
        delay(2500),
      ]);
    }
    return Boolean(navigator.serviceWorker.controller);
  } catch (error) {
    console.warn('Offline support is unavailable.', error);
    return false;
  }
}

export const serviceWorkerReady = registerWorker();

function loadScript(name, resource, ready) {
  if (ready()) return Promise.resolve();
  if (pending.has(name)) return pending.get(name);

  const promise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    let settled = false;
    let timer;
    const finish = callback => value => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      callback(value);
    };
    const succeed = finish(resolve);
    const fail = finish(reject);
    script.src = resource.url;
    script.integrity = resource.integrity;
    script.crossOrigin = 'anonymous';
    script.dataset.runtime = name;
    script.onload = () => ready()
      ? succeed()
      : fail(new Error(`${name} loaded without exposing its browser API`));
    script.onerror = () => fail(new Error(`Could not download ${name}`));
    timer = setTimeout(
      () => fail(new Error(`Timed out downloading ${name}`)),
      SCRIPT_TIMEOUT_MS,
    );
    document.head.appendChild(script);
  }).catch(error => {
    pending.delete(name);
    document.querySelector(`script[data-runtime="${name}"]`)?.remove();
    throw error;
  });

  pending.set(name, promise);
  return promise;
}

async function prepareCaching() {
  // A failed or unsupported service worker must never prevent online use.
  await Promise.race([serviceWorkerReady, delay(3000)]);
}

export async function loadObjectRuntime() {
  await prepareCaching();
  await loadScript('TensorFlow.js', RUNTIME_URLS.tensorflow, () => typeof globalThis.tf?.ready === 'function');
  await loadScript('COCO-SSD', RUNTIME_URLS.detector, () => typeof globalThis.cocoSsd?.load === 'function');
}

export async function loadSurfaceRuntime() {
  await prepareCaching();
  await loadScript('TensorFlow.js', RUNTIME_URLS.tensorflow, () => typeof globalThis.tf?.ready === 'function');
  await loadScript('DeepLab', RUNTIME_URLS.segmenter, () => typeof globalThis.deeplab?.load === 'function');
}
