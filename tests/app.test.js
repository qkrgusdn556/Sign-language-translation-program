import test from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import { readFile } from "node:fs/promises";

const appSource = await readFile(new URL("../app.js", import.meta.url), "utf8");
const recognitionSource = await readFile(new URL("../recognition.js", import.meta.url), "utf8");
const yoloSource = await readFile(new URL("../yolo.js", import.meta.url), "utf8");
const vector = Array.from({ length: 126 }, (_, i) => i < 63 ? i / 63 : 0);
const sample = () => ({ label: "hello", vector: [...vector] });

async function setup({ saved = "[]", denied = false, quota = false, faces = [], hands = [] } = {}) {
  const elements = new Map();
  const ctx = { clearRect() {}, strokeRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, arc() {}, fill() {} };
  function element(id = "") {
    return { id, value: id === "#labelInput" ? "hello" : "", textContent: "", hidden: false, disabled: false,
      style: {}, dataset: {}, children: [], checked: true, files: [], currentTime: 1, readyState: 2,
      videoWidth: 960, videoHeight: 540, width: 960, height: 540,
      addEventListener() {}, setAttribute() {}, append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = items; }, querySelectorAll() { return []; },
      getContext() { return ctx; }, getBoundingClientRect() { return { width: 960, height: 540 }; },
      pause() {}, play() { return Promise.resolve(); }, close() {}, showModal() {}, click() {},
    };
  }
  let clock = 0, calls = 0, stopped = 0, detects = 0;
  const tracks = [{ stop() { stopped++; }, addEventListener() {} }];
  const storage = new Map([["sign-translator-samples", saved]]);
  const context = vm.createContext({
    console, performance: { now: () => clock }, devicePixelRatio: 1,
    requestAnimationFrame: () => 1, cancelAnimationFrame() {}, setTimeout, Blob, URL,
    window: { addEventListener() {} },
    document: { hidden: false, addEventListener() {}, createElement: element,
      querySelector(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); } },
    navigator: { mediaDevices: { async getUserMedia() {
      calls++;
      if (denied) throw Object.assign(new Error("denied"), { name: "NotAllowedError" });
      return { getTracks: () => tracks, getVideoTracks: () => tracks };
    } } },
    localStorage: { getItem: (key) => storage.get(key), setItem(key, value) {
      if (quota) throw Object.assign(new Error("full"), { name: "QuotaExceededError" });
      storage.set(key, value);
    } },
  });
  const recognition = new vm.SourceTextModule(recognitionSource, { context });
  await recognition.link(() => {});
  await recognition.evaluate();
  const yolo = new vm.SourceTextModule(yoloSource, { context });
  await yolo.link(() => {});
  await yolo.evaluate();
  const mockVision = new vm.SyntheticModule(["HandLandmarker", "FaceDetector", "FilesetResolver"], function () {
    this.setExport("HandLandmarker", { HAND_CONNECTIONS: [], createFromOptions: async () => ({
      detectForVideo() { detects++; return { landmarks: hands }; }, close() {},
    }) });
    this.setExport("FaceDetector", { createFromOptions: async () => ({
      detectForVideo() { return { detections: faces }; }, close() {},
    }) });
    this.setExport("FilesetResolver", { forVisionTasks: async () => ({}) });
  }, { context });
  await mockVision.link(() => {});
  await mockVision.evaluate();
  const app = new vm.SourceTextModule(appSource + "\nexport { state, startCamera, stopCamera, loop, addSample, recordSequence, captureFrame, finishRecording, commitSamples, importSamples, undoSamples };", {
    context, importModuleDynamically: async (specifier) => {
      if (specifier.includes("tasks-vision")) return mockVision;
      throw new Error("3D deliberately unavailable in lifecycle tests");
    },
  });
  await app.link((specifier) => specifier === "./recognition.js" ? recognition : yolo);
  await app.evaluate();
  return { api: app.namespace, elements, storage, metrics: () => ({ calls, stopped, detects }),
    setDenied: (value) => { denied = value; }, setFaces: (value) => { faces = value; }, setClock: (value) => { clock = value; } };
}

test("camera denial re-enables start and a retry can start then release the camera", async () => {
  const h = await setup({ denied: true });
  await h.api.startCamera();
  assert.equal(h.elements.get("#startBtn").disabled, false);
  assert.match(h.elements.get("#noticeText").textContent, /권한/);
  h.setDenied(false);
  await h.api.startCamera();
  assert.equal(h.api.state.running, true);
  h.api.stopCamera();
  assert.equal(h.metrics().stopped, 1);
  assert.equal(h.elements.get("#camera").srcObject, null);
});

test("same video frame is detected once and stopping cannot leave a fresh sample", async () => {
  const h = await setup();
  await h.api.startCamera();
  h.api.loop(100);
  h.api.loop(200);
  assert.equal(h.metrics().detects, 1);
  h.api.state.lastVector = vector;
  h.api.stopCamera();
  h.api.addSample("hello");
  assert.equal(h.api.state.samples.length, 0);
});

test("a stalled video invalidates the previous hand pose", async () => {
  const h = await setup();
  await h.api.startCamera();
  h.api.loop(100);
  h.api.state.lastVector = vector;
  h.api.loop(700);
  assert.equal(h.api.state.lastVector, null);
  assert.equal(h.elements.get("#sampleBtn").disabled, true);
});

test("recording saves only observed frames in a single undoable batch", async () => {
  const h = await setup();
  await h.api.startCamera();
  h.api.state.lastVector = vector;
  h.api.recordSequence();
  h.api.captureFrame(0);
  h.api.captureFrame(50);
  h.api.captureFrame(150);
  h.api.state.lastVector = null;
  h.api.captureFrame(300);
  h.api.finishRecording(true);
  assert.equal(h.api.state.samples.length, 2);
  h.api.undoSamples();
  assert.equal(h.api.state.samples.length, 0);
});

test("stopping during collection discards unfinished data", async () => {
  const h = await setup();
  await h.api.startCamera();
  h.api.state.lastVector = vector;
  h.api.recordSequence();
  h.api.captureFrame(100);
  h.api.stopCamera();
  h.api.finishRecording(true);
  assert.equal(h.api.state.recording, null);
  assert.equal(h.api.state.samples.length, 0);
});

test("backup import preserves repeated samples and reimport does not multiply them", async () => {
  const h = await setup();
  const file = { size: 2000, text: async () => JSON.stringify({ version: 1, samples: [sample(), sample(), sample()] }) };
  h.elements.get("#importFile").files = [file];
  await h.api.importSamples();
  assert.equal(h.api.state.samples.length, 3);
  await h.api.importSamples();
  assert.equal(h.api.state.samples.length, 3);
  h.elements.get("#importFile").files = [{ size: 1, text: async () => "{" }];
  await h.api.importSamples();
  assert.equal(h.api.state.samples.length, 3);
});

test("quota failures and invalid existing data cannot silently overwrite saved samples", async () => {
  const saved = JSON.stringify([sample()]);
  const h = await setup({ saved, quota: true });
  assert.equal(h.api.commitSamples([], "cleared"), false);
  assert.equal(h.api.state.samples.length, 1);
  assert.equal(h.storage.get("sign-translator-samples"), saved);
  const corrupt = await setup({ saved: "{broken" });
  assert.equal(corrupt.api.state.storageBlocked, true);
  assert.equal(corrupt.api.commitSamples([], "clear"), false);
  assert.equal(corrupt.storage.get("sign-translator-samples"), "{broken");
});

test("face is required for capture, and face loss invalidates samples immediately", async () => {
  const hand = Array.from({length:21}, (_,i)=>({x:0.7+i*0.002,y:0.5-i*0.005,z:0}));
  const h = await setup({ hands: [hand] });
  await h.api.startCamera();
  h.api.loop(100);
  assert.equal(h.api.state.lastVector, null);
  assert.equal(h.elements.get("#sampleBtn").disabled, true);
  h.setFaces([{boundingBox:{originX:380,originY:80,width:180,height:210}}]);
  h.elements.get("#camera").currentTime++;
  h.api.loop(200);
  assert.equal(h.api.state.lastVector.length, 130);
  assert.equal(h.elements.get("#sampleBtn").disabled, false);
  h.setFaces([]);
  h.elements.get("#camera").currentTime++;
  h.api.loop(300);
  assert.equal(h.api.state.lastVector, null);
  assert.equal(h.elements.get("#sampleBtn").disabled, true);
});
