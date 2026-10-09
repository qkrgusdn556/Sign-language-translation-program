import test from "node:test";
import assert from "node:assert/strict";
import { validateSamples, landmarksToVector, mouthFeatures, distance, classify, classifySequence, motionMagnitude, dtwDistance, createStabilizer, assignHandSlots, faceReference, faceLandmarksReference, poseFaceReference, shoulderReference, selectFaceReference, createFaceTracker, faceRelativePoint, avatarWrist, poseArmTargets } from "../recognition.js";

const hand = (x = 0.3) => Array.from({ length: 21 }, (_, i) => ({ x: x + i * 0.003, y: 0.6 - i * 0.008, z: i * -0.002 }));
const vector = landmarksToVector([hand()]);
const samples = (label, v = vector, count = 3) => Array.from({ length: count }, () => ({ label, vector: [...v] }));

test("legacy and versioned backups are accepted; malformed data is rejected", () => {
  assert.equal(validateSamples(samples("hello")).length, 3);
  assert.equal(validateSamples({ version: 1, samples: samples("hello") }).length, 3);
  assert.equal(validateSamples({ version: 3, samples: [] }).length, 0);
  for (const bad of [null, {}, { version: 4, samples: [] }, [{ label: "", vector }], [{ label: "a", vector: [1] }], [{ label: "a", vector: vector.map(() => NaN) }]]) {
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

test("motion sequences use DTW and require three recordings per label", () => {
  const motionVector = [...vector, 0, 0, 0, 0];
  const move = Array.from({ length: 6 }, (_, index) => motionVector.map((value, item) => item >= 126 ? value + index * 0.03 : value));
  const different = move.map((frame) => frame.map((value, item) => item >= 126 ? value + 0.7 : value));
  const sequenceSamples = (label, sequence, count) => Array.from({ length: count }, () => ({ label, sequence }));
  assert.equal(classifySequence(move, sequenceSamples("move", move, 2)).status, "untrained");
  assert.equal(classifySequence(move, sequenceSamples("move", move, 3)).label, "move");
  assert.ok(dtwDistance(move, move) < 1e-10);
  assert.ok(motionMagnitude(move[0], move[1]) > 0);
  assert.ok(dtwDistance(different, move) > dtwDistance(move, move));
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

test("mouth blendshapes extend shoulder-relative vectors without changing wrist motion", () => {
  const mouth = mouthFeatures({ categories: [
    { categoryName: "jawOpen", score: 0.75 },
    { categoryName: "mouthPucker", score: 0.25 },
    { categoryName: "mouthSmileLeft", score: 2 },
  ] });
  assert.deepEqual(mouth, [0.75, 0, 0, 0.25, 1, 0]);
  const mouthVector = landmarksToVector([hand()], face, mouth);
  assert.equal(mouthVector.length, 136);
  assert.equal(validateSamples([{ label: "mouth", vector: mouthVector }]).length, 1);
  const alteredMouth = [...mouthVector];
  alteredMouth[130] += 0.5;
  assert.ok(distance(mouthVector, alteredMouth) > 0);
  assert.equal(motionMagnitude(mouthVector, alteredMouth), 0);
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

test("pose reference rejects a face box far from the detected person", () => {
  const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.99, presence: 0.99 }));
  pose[0] = { x: 0.55, y: 0.2, visibility: 0.99, presence: 0.99 };
  pose[11] = { x: 0.72, y: 0.5, visibility: 0.99, presence: 0.99 };
  pose[12] = { x: 0.38, y: 0.5, visibility: 0.99, presence: 0.99 };
  const falsePositive = { boundingBox: { originX: 30, originY: 80, width: 120, height: 120 } };
  const selected = selectFaceReference([falsePositive], pose, 1000, 600);
  assert.equal(selected.source, "pose");
  assert.equal(selected.box, null);
});

test("face landmarks provide the primary face box when one face is tracked", () => {
  const landmarks = Array.from({ length: 468 }, (_, index) => ({
    x: 0.4 + (index % 12) * 0.018,
    y: 0.2 + (Math.floor(index / 12) % 12) * 0.02,
  }));
  const face = faceLandmarksReference([landmarks], 1000, 600);
  assert.equal(face.source, "landmarks");
  assert.ok(face.box.width > 100);
  assert.equal(faceLandmarksReference([], 1000, 600), null);
});

test("an empty landmark result does not use a weaker detector box", () => {
  const falsePositive = { boundingBox: { originX: 20, originY: 30, width: 180, height: 180 } };
  assert.equal(selectFaceReference([falsePositive], null, 1000, 600, []), null);
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
  assert.deepEqual(arms[0].wrist.source, pose[15]);
  pose[13].visibility = 0.1;
  assert.equal(poseArmTargets(pose, face), null);
});

test("pose arms attach their wrist endpoint to the matching hand landmark", () => {
  const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.99, presence: 0.99 }));
  pose[11] = { x: 0.72, y: 0.45, visibility: 0.99, presence: 0.99 };
  pose[13] = { x: 0.75, y: 0.6, visibility: 0.99, presence: 0.99 };
  pose[15] = { x: 0.76, y: 0.72, visibility: 0.99, presence: 0.99 };
  pose[12] = { x: 0.28, y: 0.45, visibility: 0.99, presence: 0.99 };
  pose[14] = { x: 0.25, y: 0.6, visibility: 0.99, presence: 0.99 };
  pose[16] = { x: 0.24, y: 0.72, visibility: 0.99, presence: 0.99 };
  const leftHand = hand(0.77), rightHand = hand(0.23);
  const arms = poseArmTargets(pose, face, [leftHand, rightHand]);
  assert.equal(arms[0].wrist.source, leftHand[0]);
  assert.equal(arms[1].wrist.source, rightHand[0]);
  assert.ok(arms.every((arm) => arm.wrist.handAttached));
});

test("hand wrists calibrate the pose arm position into the camera coordinate system", () => {
  const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.99, presence: 0.99 }));
  pose[11] = { x: 0.1, y: 0.25, visibility: 0.99, presence: 0.99 };
  pose[13] = { x: 0.14, y: 0.4, visibility: 0.99, presence: 0.99 };
  pose[15] = { x: 0.18, y: 0.55, visibility: 0.99, presence: 0.99 };
  pose[12] = { x: 0.25, y: 0.25, visibility: 0.99, presence: 0.99 };
  pose[14] = { x: 0.29, y: 0.4, visibility: 0.99, presence: 0.99 };
  pose[16] = { x: 0.33, y: 0.55, visibility: 0.99, presence: 0.99 };
  const wrist = { x: 0.68, y: 0.55, z: 0 };
  const arms = poseArmTargets(pose, face, [[wrist], null]);
  assert.equal(arms[0].wrist.source, wrist);
  assert.ok(arms[0].shoulder.source.x > 0.55);
});

test("avatar arm targets always begin at the avatar shoulder anchors", () => {
  const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.99, presence: 0.99 }));
  pose[11] = { x: 0.72, y: 0.45, visibility: 0.99, presence: 0.99 };
  pose[13] = { x: 0.76, y: 0.6, visibility: 0.99, presence: 0.99 };
  pose[15] = { x: 0.8, y: 0.38, visibility: 0.99, presence: 0.99 };
  pose[12] = { x: 0.28, y: 0.45, visibility: 0.99, presence: 0.99 };
  pose[14] = { x: 0.24, y: 0.6, visibility: 0.99, presence: 0.99 };
  pose[16] = { x: 0.2, y: 0.38, visibility: 0.99, presence: 0.99 };
  const arms = poseArmTargets(pose, face);
  assert.deepEqual(arms[0].shoulder, { x: -0.68, y: -0.12, z: -0.08, source: pose[11], handAttached: false });
  assert.deepEqual(arms[1].shoulder, { x: 0.68, y: -0.12, z: -0.08, source: pose[12], handAttached: false });
});

test("shoulder reference works without a face and scales with shoulder width", () => {
  const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.99, presence: 0.99 }));
  pose[11] = { x: 0.7, y: 0.45, visibility: 0.99, presence: 0.99 };
  pose[12] = { x: 0.3, y: 0.45, visibility: 0.99, presence: 0.99 };
  const reference = shoulderReference(pose, 1000, 600);
  assert.equal(reference.source, "shoulders");
  assert.equal(reference.x, 0.5);
  assert.ok(reference.scale > 0.39);
  pose[11].visibility = 0.1;
  assert.equal(shoulderReference(pose, 1000, 600), null);
});
