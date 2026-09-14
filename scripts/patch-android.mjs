/* `npx cap add android` regenerates the native project from scratch, so any
 * manual edit to AndroidManifest.xml is lost on every CI run. This patches it
 * back in automatically.
 *
 * The camera permission is the critical one: without it the WebView's
 * getUserMedia call fails silently and the app looks broken. */
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const MANIFEST = 'android/app/src/main/AndroidManifest.xml';
if (!existsSync(MANIFEST)) {
  console.error('No AndroidManifest.xml — run `npx cap add android` first.');
  process.exit(1);
}

let xml = readFileSync(MANIFEST, 'utf8');

const PERMISSIONS = [
  '<uses-permission android:name="android.permission.CAMERA" />',
  '<uses-permission android:name="android.permission.INTERNET" />',
  '<uses-feature android:name="android.hardware.camera" android:required="true" />',
  '<uses-feature android:name="android.hardware.camera.autofocus" android:required="false" />',
];

for (const p of PERMISSIONS) {
  const tag = p.match(/android:name="([^"]+)"/)[1];
  if (xml.includes(tag)) continue;
  xml = xml.replace('</manifest>', `    ${p}\n</manifest>`);
}

// keep the screen in the app's control and stop the OS killing GPU context
if (!xml.includes('android:hardwareAccelerated')) {
  xml = xml.replace('<application', '<application android:hardwareAccelerated="true"');
}

// Use the tested mask-safe artwork for debug APKs as well as the PWA. Keeping
// it in drawable-nodpi lets Android scale one lossless source consistently.
const ICON_SOURCE = 'www/icons/icon-maskable-512.png';
const ICON_DIR = 'android/app/src/main/res/drawable-nodpi';
const ICON_TARGET = `${ICON_DIR}/fieldmark_icon.png`;
if (existsSync(ICON_SOURCE)) {
  mkdirSync(ICON_DIR, { recursive: true });
  copyFileSync(ICON_SOURCE, ICON_TARGET);
  xml = xml.replace(/android:icon="[^"]+"/, 'android:icon="@drawable/fieldmark_icon"');
  xml = xml.replace(/android:roundIcon="[^"]+"/, 'android:roundIcon="@drawable/fieldmark_icon"');
}

writeFileSync(MANIFEST, xml);
console.log('AndroidManifest.xml patched: camera permission and Fieldmark icon added.');
