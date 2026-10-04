import test from "node:test";
import assert from "node:assert/strict";
import { validateSamples, landmarksToVector, distance, classify, createStabilizer, assignHandSlots } from "../recognition.js";

const hand = (x = 0.3) => Array.from({ length: 21 }, (_, i) => ({ x: x + i * 0.003, y: 0.6 - i * 0.008, z: i * -0.002 }));
const vector = landmarksToVector([hand()]);
const samples = (label, v = vector, count = 3) => Array.from({ length: count }, () => ({ label, vector: [...v] }));

test("legacy and versioned backups are accepted; malformed data is rejected", () => {
  assert.equal(validateSamples(samples("hello")).length, 3);
  assert.equal(validateSamples({ version: 1, samples: samples("hello") }).length, 3);
  for (const bad of [null, {}, { version: 2, samples: [] }, [{ label: "", vector }], [{ label: "a", vector: [1] }], [{ label: "a", vector: vector.map(() => NaN) }]]) {
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
  assert.deepEqual(assignHandSlots({ landmarks: [left], handedness: [category("Left")] }), [null, left]);
  assert.deepEqual(assignHandSlots({ landmarks: [left, right], handedness: [category("Left"), category("Right")] }), [right, left]);
  assert.deepEqual(assignHandSlots({ landmarks: [right] }, [right, left]), [right, null]);
});
