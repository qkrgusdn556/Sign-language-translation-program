import test from "node:test";
import assert from "node:assert/strict";
import { parseLabels, softmax, classifyOutput, createStillnessGate } from "../yolo.js";

test("YOLO labels accept plain text and common JSON shapes", () => {
  assert.deepEqual(parseLabels("hello\nthank you\n"), ["hello", "thank you"]);
  assert.deepEqual(parseLabels('{"names":["a","b"]}'), ["a", "b"]);
  assert.throws(() => parseLabels(""));
  assert.throws(() => parseLabels('{"names":[]}'));
});

test("YOLO classifier output produces a normalized winning class", () => {
  const probabilities = softmax([1, 2, 3]);
  assert.ok(Math.abs(probabilities.reduce((sum, p) => sum + p, 0) - 1) < 1e-12);
  const result = classifyOutput(new Float32Array([0, 3, 1]), ["a", "b", "c"]);
  assert.equal(result.label, "b");
  assert.ok(result.confidence > 0.8);
  assert.throws(() => classifyOutput(new Float32Array([1]), ["a", "b"]));
});

test("YOLO runs only after hands remain still and resets when motion returns", () => {
  const gate = createStillnessGate(400, 0.09);
  assert.equal(gate.update(Infinity, 0), false);
  assert.equal(gate.update(0.05, 100), false);
  assert.equal(gate.update(0.05, 499), false);
  assert.equal(gate.update(0.05, 500), true);
  assert.equal(gate.update(0.12, 520), false);
  assert.equal(gate.update(0.05, 921), false);
});
