import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { extname, isAbsolute, join, normalize, relative, resolve } from 'node:path';

const root = resolve('www');
const port = Number(process.env.PORT || 4173);
const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.webmanifest': 'application/manifest+json; charset=utf-8',
  '.png': 'image/png',
};

createServer((request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
  const requestedPath = pathname === '/' ? 'index.html' : pathname.slice(1);
  const file = normalize(join(root, requestedPath));
  const fromRoot = relative(root, file);
  if (fromRoot.startsWith('..') || isAbsolute(fromRoot)) { response.writeHead(403).end('Forbidden'); return; }

  try {
    if (!statSync(file).isFile()) throw new Error('not a file');
    response.setHeader('Content-Type', types[extname(file)] || 'application/octet-stream');
    response.setHeader('Cache-Control', 'no-store');
    createReadStream(file).pipe(response);
  } catch {
    response.writeHead(404).end('Not found');
  }
}).listen(port, '127.0.0.1', () => {
  console.log(`Fieldmark Vision: http://127.0.0.1:${port}`);
});
