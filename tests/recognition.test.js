import test from "node:test";
import assert from "node:assert/strict";
import { validateSamples, landmarksToVector, distance, classify, classifySequence, motionMagnitude, dtwDistance, createStabilizer, assignHandSlots, faceReference, faceLandmarksReference, poseFaceReference, shoulderReference, selectFaceReference, createFaceTracker, faceRelativePoint, avatarWrist, poseArmTargets } from "../recognition.js";

const hand = (x = 0.3) => Array.from({ length: 21 }, (_, i) => ({ x: x + i * 0.003, y: 0.6 - i * 0.008, z: i * -0.002 }));
const vector = landmarksToVector([hand()]);

test("high-score held hands and count-only flicker never qualify as partial motion", () => {
  const pair=[...vector.slice(0,63),...vector.slice(0,63)];
  const rows=Array.from({length:3},()=>({label:"motion",sequence:Array.from({length:12},()=>pair)}));
  const held=classifySequence(Array.from({length:12},()=>vector),rows);
  assert.ok(held.similarity>=0.5);
  assert.equal(held.reason,"insufficient-motion");
  const flicker=classifySequence(Array.from({length:12},(_,i)=>i%2 ? pair : vector),rows);
  assert.equal(flicker.status,"unknown");
  assert.equal(flicker.reason,"insufficient-motion");
});

test("partial motion accepts 40 percent or more while weaker scores remain unknown", () => {
  const pair = [...vector.slice(0,63),...vector.slice(0,63)];
  const reference = Array.from({length:12},(_,t)=>pair.map((v,i)=>i===30 ? v+t*0.4 : v));
  const rows = Array.from({length:3},()=>({label:"motion",sequence:reference}));
  const input = offset => reference.map((frame,i)=>frame.map((v,j)=>i===0 && j>=63 ? 0 : v+offset));
  const above = classifySequence(input(0.42),rows);
  assert.ok(above.similarity>=0.4 && above.similarity<0.5);
  assert.equal(above.partial,true);
  assert.equal(above.status,"match");
  const below = classifySequence(input(0.44),rows);
  assert.ok(below.similarity<0.4);
  assert.equal(below.status,"unknown");
  const ambiguous = classifySequence(input(0.42),[...rows,...rows.map(row=>({...row,label:"other"}))]);
  assert.equal(ambiguous.status,"unknown");
  const longMissing = input(0.2).map(frame=>[...frame.slice(0,63),...Array(63).fill(0)]);
  assert.equal(classifySequence(longMissing,rows).status,"match");
  assert.equal(classify(input(0.32)[0],samples("motion",pair),{allowPartial:true}).status,"unknown");
});

test("partial hand distance follows the surviving hand and rejects absent or dissimilar hands", () => {
  const other = hand().map((p,i) => ({...p,x:p.x+i*0.012}));
  const pair = landmarksToVector([hand(),other]);
  assert.equal(distance(vector,pair),Infinity);
  assert.ok(Math.abs(distance(vector,pair,{allowPartial:true})-0.08)<1e-10);
  assert.ok(Math.abs(distance(landmarksToVector([other]),pair,{allowPartial:true})-0.08)<1e-10);
  assert.equal(distance(Array(126).fill(0),pair,{allowPartial:true}),Infinity);
  const far = vector.map((v,i) => i<63 ? v+2 : v);
  assert.equal(classify(far,samples("a",pair),{allowPartial:true}).status,"unknown");
  assert.equal(classify(vector,samples("a",pair),{allowPartial:true}).partial,true);
  assert.equal(classify(vector,[...samples("a",pair),...samples("b",pair)],{allowPartial:true}).status,"unknown");
});

test("DTW accepts short and sustained one-hand motion but withholds a held posture", () => {
  const other = hand().map((p,i) => ({...p,x:p.x+i*0.012}));
  const base = landmarksToVector([hand(),other]);
  const reference = Array.from({length:12}, (_,t) => base.map((v,i) => i===30 ? v+t*0.4 : v));
  const single = frame => [...frame.slice(0,63),...Array(63).fill(0)];
  const short = reference.map((frame,i) => [0,11].includes(i) ? single(frame) : [...frame]);
  const rows = Array.from({length:3},()=>({label:"motion",sequence:reference}));
  assert.equal(dtwDistance(short,reference),Infinity);
  const result = classifySequence(short,rows);
  assert.equal(result.label,"motion");
  assert.equal(result.partial,true);
  assert.ok(result.similarity<=0.8);
  const sustained = classifySequence(reference.map(single),rows);
  assert.equal(sustained.status,"match");
  const held = classifySequence(Array.from({length:12},()=>single(reference.at(-1))),rows);
  assert.equal(held.status,"unknown");
  assert.equal(held.reason,"insufficient-motion");
});

test("partial matches retain shoulder-relative position checks", () => {
  const single = [...vector,0.1,0.2,0,0];
  const pair = [...vector.slice(0,63),...vector.slice(0,63),0.1,0.2,3,3];
  assert.ok(Math.abs(distance(single,pair,{allowPartial:true})-0.08)<1e-10);
  const far = [...vector,9,9,0,0];
  assert.ok(distance(far,pair,{allowPartial:true})>0.12);
});

test("ambiguous samples expose a nearest label without confirming it", () => {
  const staticRows = ["a", "b"].flatMap(label => Array.from({length: 3}, () => ({label, vector})));
  const result = classify(vector, staticRows);
  assert.equal(result.status, "unknown");
  assert.equal(result.nearestLabel, "a");
  const sequence = Array.from({length: 6}, () => vector);
  const motionRows = ["a", "b"].flatMap(label => Array.from({length: 3}, () => ({label, sequence})));
  const motion = classifySequence(sequence, motionRows);
  assert.equal(motion.status, "unknown");
  assert.equal(motion.nearestLabel, "a");
  const incompatible = classifySequence(sequence.map(frame => [...frame,0,0,0,0]), motionRows);
  assert.equal(incompatible.nearestLabel, undefined);
});

test("motion detects finger changes without treating detector order changes as movement", () => {
  const changed = vector.map((v, i) => i === 30 ? v + 0.5 : v);
  assert.ok(motionMagnitude(vector, changed) > 0.035);
  const other = hand().map((p, i) => ({ ...p, x: p.x + i * 0.012 }));
  assert.equal(motionMagnitude(landmarksToVector([hand(), other]), landmarksToVector([other, hand()])), 0);
});
const samples = (label, v = vector, count = 3) => Array.from({ length: count }, () => ({ label, vector: [...v] }));

test("legacy and versioned backups are accepted; malformed data is rejected", () => {
  assert.equal(validateSamples(samples("hello")).length, 3);
  assert.equal(validateSamples({ version: 1, samples: samples("hello") }).length, 3);
  assert.equal(validateSamples({ version: 3, samples: [] }).length, 0);
  for (const bad of [null, {}, { version: 4, samples: [] }, [{ label: "", vector }], [{ label: "a", vector: [1] }], [{ label: "a", vector: vector.map(() => NaN) }]]) {
    assert.throws(() => validateSamples(bad));
  }
});

test("mouth-era backups retain hand data without modifying original samples", () => {
  const extended = [...vector, 0.1, 0.2, 0.3, 0.4, 1, 2, 3, 4, 5, 6];
  const original = { version: 3, samples: [{ label: "hello", vector: extended }, { label: "move", sequence: Array.from({ length: 6 }, () => [...extended]) }] };
  const before = JSON.stringify(original);
  const loaded = validateSamples(original);
  assert.deepEqual(loaded[0].vector, extended.slice(0, 130));
  assert.equal(loaded[1].sequence[0].length, 130);
  assert.equal(JSON.stringify(original), before);
  assert.doesNotThrow(() => classify(loaded[0].vector, loaded));
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

test("default word confirmation takes 200ms without accepting a changed candidate early", () => {
  const gate = createStabilizer();
  const match = { status: "match", label: "a", similarity: 1 };
  assert.equal(gate.update(match, 0).status, "pending");
  assert.equal(gate.update(match, 199).status, "pending");
  assert.equal(gate.update(match, 200).status, "match");
  assert.equal(gate.update({ ...match, label: "b" }, 201).status, "pending");
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
