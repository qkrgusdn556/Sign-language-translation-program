import test from "node:test";
import assert from "node:assert/strict";
import { validateSamples, landmarksToVector, distance, classify, createStabilizer, assignHandSlots, faceReference, poseFaceReference, createFaceTracker, faceRelativePoint, avatarWrist, poseArmTargets } from "../recognition.js";

const hand = (x = 0.3) => Array.from({ length: 21 }, (_, i) => ({ x: x + i * 0.003, y: 0.6 - i * 0.008, z: i * -0.002 }));
const vector = landmarksToVector([hand()]);
const samples = (label, v = vector, count = 3) => Array.from({ length: count }, () => ({ label, vector: [...v] }));

test("legacy and versioned backups are accepted; malformed data is rejected", () => {
  assert.equal(validateSamples(samples("hello")).length, 3);
  assert.equal(validateSamples({ version: 1, samples: samples("hello") }).length, 3);
  for (const bad of [null, {}, { version: 3, samples: [] }, [{ label: "", vector }], [{ label: "a", vector: [1] }], [{ label: "a", vector: vector.map(() => NaN) }]]) {
    assert.throws(() => validateSamples(bad));
  }
});

test("normalization is translation and scale invariant", () => {
  const shifted = hand().map((p) => ({ x: p.x * 2 + 0.2, y: p.y * 2 + 0.2, z: p.z * 2 }));
  assert.ok(distance(vector, landmarksToVector([shifted])) < 1e-10);
  assert.equal(landmarksToVector([]), null);
  assert.equal(landmarksToVector(Array(1).fill(Array(21).fill({ x: 0, y: 0, z: 0 }))), null);
});

test("swapping detector order does not change distance; different hand counts are rejected", () => {
  const other = hand().map((p, i) => ({ ...p, x: p.x + i * 0.012 }));
  assert.equal(distance(landmarksToVector([hand(), other]), landmarksToVector([other, hand()])), 0);
  assert.equal(distance(vector, landmarksToVector([hand(), other])), Infinity);
});

test("requires three samples per label, not just three samples in total", () => {
  assert.equal(classify(vector, [...samples("a", vector, 1), ...samples("b", vector, 2)]).status, "untrained");
  assert.equal(classify(vector, samples("a")).status, "match");
});

test("unknown shapes and ambiguous labels are withheld, including a single-class dataset", () => {
  const far = vector.map((n, i) => i < 63 && i > 2 ? n + 2 : n);
  assert.equal(classify(far, samples("a")).status, "unknown");
  assert.equal(classify(vector, [...samples("a"), ...samples("b")]).status, "unknown");
});

test("stable predictions require a continuous hold and reset after hand loss", () => {
  const gate = createStabilizer(350);
  const match = { status: "match", label: "a", similarity: 1 };
  assert.equal(gate.update(match, 0).status, "pending");
  assert.equal(gate.update(match, 360).status, "match");
  gate.update({ status: "unknown" }, 400);
  assert.equal(gate.update(match, 500).status, "pending");
  assert.equal(gate.update({ ...match, label: "b" }, 860).status, "pending");
});

test("single left hand and crossed hands keep their avatar slots", () => {
  const left = hand(0.8), right = hand(0.2);
  const category = (categoryName) => [{ categoryName, score: 0.98 }];
  assert.deepEqual(assignHandSlots({ landmarks: [left], handedness: [category("Left")] }), [left, null]);
  assert.deepEqual(assignHandSlots({ landmarks: [left, right], handedness: [category("Left"), category("Right")] }), [left, right]);
  assert.deepEqual(assignHandSlots({ landmarks: [right] }, [right, left]), [right, null]);
});

const face = { x: 0.5, y: 0.3, scale: 0.2, aspect: 0.5625 };

test("uncrossed mirrored hands connect to their own shoulders", () => {
  const slots = assignHandSlots({landmarks:[hand(0.75),hand(0.25)],handedness:[[{categoryName:"Left",score:0.99}],[{categoryName:"Right",score:0.99}]]});
  assert.ok(avatarWrist(slots[0][0], face).x < 0);
  assert.ok(avatarWrist(slots[1][0], face).x > 0);
  const crossed = assignHandSlots({landmarks:[hand(0.25),hand(0.75)],handedness:[[{categoryName:"Left",score:0.99}],[{categoryName:"Right",score:0.99}]]}, slots);
  assert.ok(avatarWrist(crossed[0][0], face).x > 0);
});

test("face-relative coordinates ignore shared translation and distance changes", () => {
  const point = { x: 0.7, y: 0.5 };
  const shifted = { x: point.x * 0.5 + 0.1, y: point.y * 0.5 + 0.2 };
  const movedFace = { ...face, x: face.x * 0.5 + 0.1, y: face.y * 0.5 + 0.2, scale: face.scale * 0.5 };
  const a = faceRelativePoint(point, face), b = faceRelativePoint(shifted, movedFace);
  assert.ok(a.every((v,i)=>Math.abs(v-b[i])<1e-10));
  assert.equal(avatarWrist({x:face.x,y:face.y}, face).y, 0.82);
});

test("same shape at the face and chest produces different recognition results", () => {
  const near = hand().map(p=>({...p,y:p.y-0.3}));
  const chest = hand();
  const nearVector = landmarksToVector([near], face);
  const chestVector = landmarksToVector([chest], face);
  assert.equal(nearVector.length,130);
  assert.ok(distance(nearVector,chestVector)>0.3);
  assert.equal(classify(nearVector,[...samples("face",nearVector),...samples("chest",chestVector)]).label,"face");
  assert.equal(classify(nearVector,samples("legacy")).status,"untrained");
  assert.equal(validateSamples({version:2,samples:samples("face",nearVector)}).length,3);
});

test("face and shape features swap together when detection order changes", () => {
  const a = hand(0.7), b = hand(0.2);
  assert.equal(distance(landmarksToVector([a,b],face),landmarksToVector([b,a],face)),0);
});

test("missing, multiple or degenerate faces do not establish a reference", () => {
  const detection = {boundingBox:{originX:400,originY:100,width:200,height:200}};
  assert.equal(faceReference([],1000,600),null);
  assert.equal(faceReference([detection,detection],1000,600),null);
  assert.equal(faceReference([{boundingBox:{width:0,height:0}}],1000,600),null);
  assert.equal(faceReference([detection],1000,600).x,0.5);
});

test("pose face fallback smooths brief detector loss without keeping it indefinitely", () => {
  const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.99, presence: 0.99 }));
  pose[0] = { x: 0.5, y: 0.2, visibility: 0.99, presence: 0.99 };
  pose[11] = { x: 0.7, y: 0.5, visibility: 0.99, presence: 0.99 };
  pose[12] = { x: 0.3, y: 0.5, visibility: 0.99, presence: 0.99 };
  const fallback = poseFaceReference(pose, 1000, 600);
  assert.equal(fallback.source, "pose");
  assert.equal(fallback.box, null);

  const tracker = createFaceTracker({ maxLostMs: 420, smoothing: 0.5 });
  tracker.update({ x: 0.5, y: 0.3, scale: 0.2, aspect: 0.6, source: "face" }, 0);
  const smoothed = tracker.update({ x: 0.7, y: 0.5, scale: 0.4, aspect: 0.6, source: "face" }, 10);
  assert.equal(smoothed.x, 0.6);
  assert.equal(tracker.update(null, 300).stale, true);
  assert.equal(tracker.update(null, 500), null);
});

test("pose arm targets retain shoulder, elbow and wrist order in mirror space", () => {
  const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.99, presence: 0.99 }));
  pose[11] = { x: 0.72, y: 0.45, visibility: 0.99, presence: 0.99 };
  pose[13] = { x: 0.76, y: 0.6, visibility: 0.99, presence: 0.99 };
  pose[15] = { x: 0.8, y: 0.38, visibility: 0.99, presence: 0.99 };
  pose[12] = { x: 0.28, y: 0.45, visibility: 0.99, presence: 0.99 };
  pose[14] = { x: 0.24, y: 0.6, visibility: 0.99, presence: 0.99 };
  pose[16] = { x: 0.2, y: 0.38, visibility: 0.99, presence: 0.99 };
  const arms = poseArmTargets(pose, face);
  assert.equal(arms.length, 2);
  assert.ok(arms[0].shoulder.x < arms[1].shoulder.x);
  assert.equal(arms[0].wrist.source, pose[15]);
  pose[13].visibility = 0.1;
  assert.equal(poseArmTargets(pose, face), null);
});
