import test from 'node:test';
import assert from 'node:assert/strict';
import { Tracker, TRACK_CFG, iou } from '../www/js/tracker.js';
import { analyseSegmentation } from '../www/js/perception.js';

const detection = (x = 10, label = 'person', score = 0.9) => ({
  bbox: [x, 20, 30, 40], class: label, score,
});

test('IoU is 1 for identical boxes', () => {
  assert.equal(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1);
});

test('IoU is 0 for separated and edge-touching boxes', () => {
  assert.equal(iou([0, 0, 10, 10], [20, 20, 5, 5]), 0);
  assert.equal(iou([0, 0, 10, 10], [10, 0, 5, 5]), 0);
});

test('IoU calculates partial overlap', () => {
  assert.equal(iou([0, 0, 10, 10], [5, 5, 10, 10]), 25 / 175);
});

test('new tracks are not penalised with a first-frame miss', () => {
  const tracker = new Tracker();
  tracker.update([detection()]);
  assert.equal(tracker.tracks[0].hits, 1);
  assert.equal(tracker.tracks[0].misses, 0);
});

test('a track appears only after the configured hit threshold', () => {
  const tracker = new Tracker();
  for (let i = 1; i <= TRACK_CFG.minHits; i++) {
    tracker.update([detection()]);
    assert.equal(tracker.visible().length, i === TRACK_CFG.minHits ? 1 : 0);
  }
});

test('a promoted track survives brief misses then expires', () => {
  const tracker = new Tracker();
  for (let i = 0; i < TRACK_CFG.minHits; i++) tracker.update([detection()]);
  assert.equal(tracker.visible().length, 1);
  for (let i = 0; i < TRACK_CFG.maxMisses; i++) tracker.update([]);
  assert.equal(tracker.visible().length, 1);
  tracker.update([]);
  assert.equal(tracker.visible().length, 0);
});

test('matching remains stable when detection order changes', () => {
  const tracker = new Tracker();
  tracker.update([detection(10, 'person'), detection(100, 'car')]);
  const ids = new Map(tracker.tracks.map(track => [track.label, track.id]));
  tracker.update([detection(101, 'car'), detection(11, 'person')]);
  assert.equal(tracker.tracks.find(track => track.label === 'person').id, ids.get('person'));
  assert.equal(tracker.tracks.find(track => track.label === 'car').id, ids.get('car'));
});

test('box coordinates are damped with an exponential moving average', () => {
  const tracker = new Tracker();
  tracker.update([detection(0)]);
  tracker.update([detection(100)]);
  assert.equal(tracker.tracks[0].box[0], 0, 'large jump should create a new track');

  const close = new Tracker();
  close.update([detection(10)]);
  close.update([detection(15)]);
  const expected = 10 * (1 - TRACK_CFG.boxAlpha) + 15 * TRACK_CFG.boxAlpha;
  assert.equal(close.tracks[0].box[0], expected);
});

test('majority voting resists a one-frame label flip', () => {
  const tracker = new Tracker();
  tracker.update([detection(10, 'chair')]);
  tracker.update([detection(10, 'chair')]);
  tracker.update([detection(10, 'couch')]);
  assert.equal(tracker.tracks[0].label, 'chair');
});

test('reset clears tracks and preserves monotonic IDs', () => {
  const tracker = new Tracker();
  tracker.update([detection()]);
  tracker.reset();
  assert.equal(tracker.tracks.length, 0);
  tracker.update([detection()]);
  assert.equal(tracker.tracks[0].id, 2);
});

function segmentationFixture() {
  const red = [255, 0, 0, 255];
  const green = [0, 255, 0, 255];
  return {
    red,
    green,
    input: {
      legend: { sky: red.slice(0, 3), grass: green.slice(0, 3) },
      width: 2,
      height: 2,
      segmentationMap: new Uint8ClampedArray([...red, ...red, ...red, ...green]),
    },
  };
}

const palette = label => label === 'sky' ? [10, 20, 30] : [40, 50, 60];

test('segmentation calculates region shares and sorts largest first', () => {
  const { input } = segmentationFixture();
  const result = analyseSegmentation(input, palette, { minShare: 0 });
  assert.deepEqual(result.regions.map(r => [r.label, r.share]), [['sky', 0.75], ['grass', 0.25]]);
});

test('segmentation applies the display palette', () => {
  const { input } = segmentationFixture();
  const result = analyseSegmentation(input, palette, { minShare: 0 });
  assert.deepEqual([...result.mask.slice(0, 4)], [10, 20, 30, 255]);
  assert.deepEqual([...result.mask.slice(-4)], [40, 50, 60, 255]);
});

test('segmentation filters small and suppressed regions', () => {
  const { input } = segmentationFixture();
  assert.deepEqual(analyseSegmentation(input, palette, { minShare: 0.3 }).regions.map(r => r.label), ['sky']);
  assert.deepEqual(analyseSegmentation(input, palette, { minShare: 0, suppress: new Set(['sky']) }).regions.map(r => r.label), ['grass']);
});

test('every label anchor lies inside its own region', () => {
  const { input } = segmentationFixture();
  const result = analyseSegmentation(input, palette, { minShare: 0 });
  for (const region of result.regions) {
    assert.equal(result.idMap[region.y * result.width + region.x], region.index);
  }
});

test('renamed ADE classes merge into one user-facing region', () => {
  const red = [255, 0, 0, 255], green = [0, 255, 0, 255];
  const result = analyseSegmentation({
    legend: { building: red.slice(0, 3), skyscraper: green.slice(0, 3) },
    width: 2,
    height: 1,
    segmentationMap: new Uint8ClampedArray([...red, ...green]),
  }, palette, { minShare: 0 });
  assert.deepEqual(result.regions.map(region => [region.label, region.share]), [['building', 1]]);
});
