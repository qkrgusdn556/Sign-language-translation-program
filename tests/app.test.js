import test from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import { readFile } from "node:fs/promises";

const appSource = await readFile(new URL("../app.js", import.meta.url), "utf8");
const recognitionSource = await readFile(new URL("../recognition.js", import.meta.url), "utf8");
const yoloSource = await readFile(new URL("../yolo.js", import.meta.url), "utf8");
const vector = Array.from({ length: 126 }, (_, i) => i < 63 ? i / 63 : 0);
const sample = () => ({ label: "hello", vector: [...vector] });

test("one visible hand keeps partial comparison enabled beyond 220ms", async () => {
  const h = await setup();
  const pair=[...vector.slice(0,63),...vector.slice(0,63)];
  h.api.state.samples=Array.from({length:3},()=>({label:"hello",vector:pair}));
  h.api.updatePartialTracking(pair,100);
  h.api.updatePartialTracking(vector,110);
  h.api.predict(vector,110);
  h.api.updatePartialTracking(vector,310);
  h.api.predict(vector,310);
  assert.equal(h.api.state.sentenceWords.length,1);
  h.api.clearSentence();
  h.api.updatePartialTracking(vector,400);
  h.api.predict(vector,400);
  assert.equal(h.api.state.sentenceWords.length,1);
  assert.equal(h.elements.get("#prediction").textContent,"hello");
  h.api.updatePartialTracking(vector,5000);
  assert.equal(h.api.state.partialMatchAllowed,true);
  h.api.updatePartialTracking(null,5100);
  assert.equal(h.api.state.partialMatchAllowed,false);
  h.api.stopCamera();
  assert.equal(h.api.state.partialMatchAllowed,false);
});

test("short tracking gaps retain observed frames without manufacturing coordinates or confirming", async () => {
  const h = await setup();
  const frames = Array.from({length: 8}, (_, i) => [...vector, i * 0.05, 0, 0, 0]);
  frames.forEach((frame, i) => h.api.updateMotionRecognition(frame, 100 + i * 90));
  const count = h.api.state.motion.frames.length;
  assert.equal(h.api.noteMotionGap(760), true);
  assert.equal(h.api.noteMotionGap(850), true);
  assert.equal(h.api.state.motion.frames.length, count);
  assert.equal(h.api.updateMotionRecognition(frames.at(-1), 940), null);
  assert.equal(h.api.state.motion.active, true);
  assert.equal(h.api.state.motion.frames.length, count + 1);
  assert.equal(h.api.state.motion.quietSince, 0);
  assert.equal(h.api.state.motion.deltas.length, 0);
  assert.equal(h.api.state.sentenceWords.length, 0);
  h.api.stopCamera();
  assert.equal(h.api.state.motion.frames.length, 0);
});

test("a long tracking gap cancels the old segment even if detection returns before another missing frame", async () => {
  const h = await setup();
  h.api.updateMotionRecognition([...vector,0,0,0,0], 100);
  h.api.updateMotionRecognition([...vector,0.1,0,0,0], 190);
  assert.equal(h.api.noteMotionGap(250), true);
  assert.equal(h.api.updateMotionRecognition([...vector,0.1,0,0,0], 500), null);
  assert.equal(h.api.state.motion.active, false);
  assert.equal(h.api.state.motion.frames.length, 0);
});

test("tracking loss cannot count as quiet time toward motion completion", async () => {
  const h = await setup();
  const frame = [...vector,0,0,0,0];
  h.api.state.motion = {active:true,lastVector:frame,lastAt:100,quietSince:100,frames:Array.from({length:6},()=>frame),deltas:[0,0,0]};
  h.api.noteMotionGap(190);
  assert.equal(h.api.updateMotionRecognition(frame, 370), null);
  assert.equal(h.api.state.motion.quietSince, 0);
  assert.equal(h.api.updateMotionRecognition(frame, 460), null);
  assert.equal(h.api.state.motion.active, true);
});

test("a single small motion spike does not restart the quiet timer when the recent mean stays low", async () => {
  const h = await setup();
  const frame = [...vector,0,0,0,0];
  h.api.state.motion = {active:true,lastVector:frame,lastAt:100,quietSince:100,frames:Array.from({length:6},()=>frame),deltas:[0.01,0.01,0.01]};
  h.api.state.samples = Array.from({length:3},()=>({label:"hello",sequence:Array.from({length:6},()=>frame)}));
  assert.equal(h.api.updateMotionRecognition([...vector,0.04,0,0,0], 190), null);
  assert.equal(h.api.state.motion.quietSince, 100);
  const result = h.api.updateMotionRecognition([...vector,0.05,0,0,0], 280);
  assert.equal(result.label, "hello");
  assert.equal(h.api.state.motion.active, false);
});

test("motion score updates before motion ends without committing the preview", async () => {
  const h = await setup();
  const frames = Array.from({length: 9}, (_, i) => [...vector, i * 0.05, 0, 0, 0]);
  h.api.state.samples = Array.from({length: 3}, () => ({label: "hello", sequence: frames}));
  frames.forEach((frame, i) => h.api.updateMotionRecognition(frame, 100 + i * 90));
  assert.equal(h.api.state.motion.active, true);
  assert.notEqual(h.elements.get("#similarity").textContent, "0%");
  assert.equal(h.elements.get("#nearestSample").textContent, "동작 후보: hello");
  assert.equal(h.api.state.sentenceWords.length, 0);
});

test("long one-hand motion previews and confirms after movement ends", async () => {
  const h = await setup();
  const frames = Array.from({length: 8}, (_, i) => [...vector, i * 0.05, 0, 0, 0]);
  const twoHands = frames.map(frame => frame.map((v, i) => i >= 63 && i < 126 ? 1 : v));
  h.api.state.samples = Array.from({length: 3}, () => ({label: "hello", sequence: twoHands}));
  frames.forEach((frame, i) => h.api.updateMotionRecognition(frame, 100 + i * 90));
  assert.equal(h.elements.get("#similarity").textContent, "80%");
  assert.equal(h.api.state.sentenceWords.length, 0);
  let result;
  for(let i=8;i<=12;i++) result=h.api.updateMotionRecognition(frames.at(-1),100+i*90)||result;
  assert.equal(result.status,"match");
  assert.equal(result.partial,true);
  h.api.updatePartialTracking(frames.at(-1),1180);
  h.api.applyRecognition(result,"동작");
  assert.equal(h.api.state.sentenceWords.length,1);
});

test("holding a final pose cannot confirm a word that has trained motion samples", async () => {
  const h=await setup();
  const pair=[...vector.slice(0,63),...vector.slice(0,63)];
  h.api.state.samples=[...Array.from({length:3},()=>({label:"hello",vector:pair})),...Array.from({length:3},()=>({label:"hello",sequence:Array.from({length:12},()=>pair)}))];
  h.api.updatePartialTracking(vector,1000);
  h.api.predict(vector,1000);
  h.api.predict(vector,1500);
  h.api.predict(vector,5000);
  assert.equal(h.api.state.sentenceWords.length,0);
  assert.equal(h.elements.get("#prediction").textContent,"동작 대기");
});

test("nearest sample remains a tentative candidate without appending an unknown word", async () => {
  const h = await setup();
  h.api.applyRecognition({status: "unknown", nearestLabel: "hello", similarity: 0.6}, "동작");
  assert.equal(h.elements.get("#nearestSample").textContent, "동작 후보: hello");
  assert.equal(h.elements.get("#similarity").textContent, "60%");
  assert.equal(h.api.state.sentenceWords.length, 0);
  h.api.applyRecognition({status: "match", label: "hello", similarity: 0.95}, "동작");
  assert.equal(h.elements.get("#nearestSample").textContent, "동작: hello");
  h.api.stopCamera();
  assert.equal(h.elements.get("#nearestSample").textContent, "비교 샘플 없음");
});

test("sampled motion is recognized and its result survives the next static frame", async () => {
  const h = await setup();
  const frames = Array.from({ length: 9 }, (_, i) => [...vector, i * 0.05, 0, 0, 0]);
  h.api.state.samples = Array.from({ length: 3 }, () => ({ label: "hello", sequence: frames }));
  frames.forEach((frame, i) => {
    h.api.updateMotionRecognition(frame, 100 + i * 90);
    h.api.updateMotionRecognition([...vector, i * 0.05 + 0.025, 0, 0, 0], 145 + i * 90);
  });
  let result;
  for (let i = 9; i <= 11; i++) result = h.api.updateMotionRecognition(frames.at(-1), 100 + i * 90) || result;
  assert.equal(result.status, "match");
  assert.equal(result.label, "hello");
  h.api.applyRecognition(result, "motion");
  h.api.predict(frames.at(-1), 1150);
  assert.equal(h.elements.get("#prediction").textContent, "hello");
  assert.equal([...h.api.state.sentenceWords].join(" "), "hello");
});

test("finger-only motion activates DTW with legacy 126-dimensional samples", async () => {
  const h = await setup();
  const frames = Array.from({ length: 8 }, (_, i) => vector.map((v, j) => j === 30 ? v + i * 0.4 : v));
  h.api.state.samples = Array.from({ length: 3 }, () => ({ label: "fingers", sequence: frames }));
  frames.forEach((frame, i) => h.api.updateMotionRecognition(frame, 100 + i * 90));
  assert.equal(h.api.state.motion.active, true);
  let result;
  for (let i = 8; i <= 10; i++) result = h.api.updateMotionRecognition(frames.at(-1), 100 + i * 90) || result;
  assert.equal(result.label, "fingers");
});

test("motion-only data shows motion guidance, not missing static sample guidance", async () => {
  const h = await setup();
  h.api.state.samples = [{ label: "hello", sequence: Array.from({ length: 6 }, () => vector) }];
  h.api.predict(vector, 100);
  assert.equal(h.elements.get("#prediction").textContent, "동작 샘플 부족");
  h.api.state.samples.push(...h.api.state.samples, ...h.api.state.samples);
  h.api.predict(vector, 200);
  assert.equal(h.elements.get("#prediction").textContent, "동작 대기");
});

async function setup({ saved = "[]", denied = false, quota = false, faceLandmarks = [], hands = [], poses = [], indexedStore = null, fileStore = null } = {}) {
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
  const indexed = indexedStore && { open() {
    const request = {};
    setTimeout(() => {
      request.result = { close() {}, transaction() {
        const transaction = { objectStore() { return {
          get(key) {
            const result = { result: indexedStore.get(key) };
            setTimeout(() => transaction.oncomplete(), 0);
            return result;
          },
          put(value, key) {
            const result = {};
            setTimeout(() => {
              if (indexedStore.fail) {
                transaction.error = Object.assign(new Error("full"), { name: "QuotaExceededError" });
                transaction.onabort();
              } else {
                indexedStore.set(key, JSON.parse(JSON.stringify(value)));
                transaction.oncomplete();
              }
            }, 0);
            return result;
          },
        }; } };
        return transaction;
      } };
      request.onsuccess();
    }, 0);
    return request;
  } };
  const context = vm.createContext({
    ...(fileStore ? { fetch: async (url, options = {}) => {
      assert.equal(url, "/api/samples");
      if (fileStore.fail) return { ok: false };
      if (options.method === "POST") fileStore.samples = JSON.parse(options.body).samples;
      return { ok: true, json: async () => ({ samples: fileStore.samples ?? null }) };
    } } : {}),
    ...(indexed ? { indexedDB: indexed } : {}),
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
  const mockVision = new vm.SyntheticModule(["HandLandmarker", "FaceLandmarker", "PoseLandmarker", "FilesetResolver"], function () {
    this.setExport("HandLandmarker", { HAND_CONNECTIONS: [], createFromOptions: async () => ({
      detectForVideo() { detects++; return { landmarks: hands }; }, close() {},
    }) });
    this.setExport("FaceLandmarker", { createFromOptions: async () => ({
      detectForVideo() { return { faceLandmarks }; }, close() {},
    }) });
    this.setExport("PoseLandmarker", { createFromOptions: async () => ({
      detectForVideo() { return { landmarks: poses }; }, close() {},
    }) });
    this.setExport("FilesetResolver", { forVisionTasks: async () => ({}) });
  }, { context });
  await mockVision.link(() => {});
  await mockVision.evaluate();
  const app = new vm.SourceTextModule(appSource + "\nexport { state, startCamera, stopCamera, loop, addSample, recordSequence, captureFrame, finishRecording, commitSamples, importSamples, undoSamples, predict, clearSentence, updateMotionRecognition, applyRecognition, noteMotionGap, updatePartialTracking };", {
    context, importModuleDynamically: async (specifier) => {
      if (specifier.includes("tasks-vision")) return mockVision;
      throw new Error("3D deliberately unavailable in lifecycle tests");
    },
  });
  await app.link((specifier) => specifier.startsWith("./recognition.js") ? recognition : yolo);
  await app.evaluate();
  return { api: app.namespace, elements, storage, metrics: () => ({ calls, stopped, detects }),
    setDenied: (value) => { denied = value; }, setFaceLandmarks: (value) => { faceLandmarks = value; }, setPoses: (value) => { poses = value; }, setClock: (value) => { clock = value; } };
}

test("computer file storage migrates browser samples and survives localStorage quota", async () => {
  const saved = JSON.stringify([sample()]);
  const fileStore = {};
  const indexedStore = new Map([["sign-translator-samples", [sample(), sample()]]]);
  const h = await setup({ saved, fileStore, indexedStore, quota: true });
  assert.equal(h.api.state.fileStorage, true);
  assert.equal(fileStore.samples.length, 2);
  assert.equal(await h.api.commitSamples([sample()], "saved"), true);
  assert.equal(fileStore.samples.length, 1);
  assert.equal(h.storage.get("sign-translator-samples"), saved);
  assert.equal(indexedStore.get("sign-translator-samples").length, 2);
});

test("existing computer file wins over browser data and failed writes preserve memory", async () => {
  const fileStore = { samples: [sample(), sample()] };
  const h = await setup({ saved: "{broken", fileStore });
  assert.equal(h.api.state.storageBlocked, false);
  assert.equal(h.api.state.samples.length, 2);
  fileStore.fail = true;
  assert.equal(await h.api.commitSamples([], "clear"), false);
  assert.equal(h.api.state.samples.length, 2);
  assert.equal(fileStore.samples.length, 2);
});

test("IndexedDB migration keeps legacy storage and saves despite localStorage quota", async () => {
  const indexedStore = new Map();
  const saved = JSON.stringify([sample()]);
  const h = await setup({ indexedStore, saved, quota: true });
  assert.equal(h.api.state.samples.length, 1);
  assert.equal(indexedStore.get("sign-translator-samples").length, 1);
  assert.equal(await h.api.commitSamples([sample(), sample()], "saved"), true);
  assert.equal(indexedStore.get("sign-translator-samples").length, 2);
  assert.equal(h.storage.get("sign-translator-samples"), saved);
  await h.api.undoSamples();
  assert.equal(h.api.state.samples.length, 1);
});

test("IndexedDB takes precedence on reload and failed writes retain saved samples", async () => {
  const indexedStore = new Map([["sign-translator-samples", [sample(), sample()]]]);
  const h = await setup({ indexedStore, saved: "{broken" });
  assert.equal(h.api.state.storageBlocked, false);
  assert.equal(h.api.state.samples.length, 2);
  indexedStore.fail = true;
  assert.equal(await h.api.commitSamples([], "cleared"), false);
  assert.equal(h.api.state.samples.length, 2);
  assert.equal(indexedStore.get("sign-translator-samples").length, 2);
  assert.equal(h.api.state.saving, false);
});

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

test("recording saves observed frames as one motion sequence in a single undoable batch", async () => {
  const h = await setup();
  await h.api.startCamera();
  h.api.state.lastVector = vector;
  h.api.recordSequence();
  h.api.captureFrame(0);
  h.api.captureFrame(150);
  h.api.captureFrame(300);
  h.api.captureFrame(450);
  h.api.captureFrame(600);
  h.api.captureFrame(750);
  h.api.state.lastVector = null;
  h.api.captureFrame(900);
  h.api.finishRecording(true);
  assert.equal(h.api.state.samples.length, 1);
  assert.equal(h.api.state.samples[0].sequence.length, 6);
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

test("shoulders are required for capture, and shoulder loss invalidates samples", async () => {
  const hand = Array.from({length:21}, (_,i)=>({x:0.7+i*0.002,y:0.5-i*0.005,z:0}));
  const h = await setup({ hands: [hand] });
  await h.api.startCamera();
  h.api.loop(100);
  assert.equal(h.api.state.lastVector, null);
  assert.equal(h.elements.get("#sampleBtn").disabled, true);
  const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.99, presence: 0.99 }));
  pose[11] = { x: 0.7, y: 0.45, visibility: 0.99, presence: 0.99 };
  pose[12] = { x: 0.3, y: 0.45, visibility: 0.99, presence: 0.99 };
  h.setPoses([pose]);
  h.elements.get("#camera").currentTime++;
  h.api.loop(200);
  assert.equal(h.api.state.lastVector.length, 130);
  assert.equal(h.elements.get("#sampleBtn").disabled, false);
  h.setPoses([]);
  h.elements.get("#camera").currentTime++;
  h.api.loop(300);
  assert.equal(h.api.state.lastVector, null);
  assert.equal(h.elements.get("#sampleBtn").disabled, true);
});

test("mouth-era stored samples do not lock backup import or overwrite storage on load", async () => {
  const saved = JSON.stringify([{ label: "hello", vector: [...vector, ...Array(10).fill(0.1)] }]);
  const h = await setup({ saved });
  assert.equal(h.api.state.storageBlocked, false);
  assert.equal(h.elements.get("#importBtn").disabled, false);
  assert.equal(h.api.state.samples[0].vector.length, 130);
  assert.equal(h.storage.get("sign-translator-samples"), saved);
});

test("stable predictions compose a sentence without repeating a held sign", async () => {
  const h = await setup();
  h.api.state.samples = [sample(), sample(), sample()];
  h.api.predict(vector, 0);
  h.api.predict(vector, 199);
  assert.equal(h.api.state.sentenceWords.length, 0);
  h.api.predict(vector, 200);
  h.api.predict(vector, 800);
  assert.equal([...h.api.state.sentenceWords].join(" "), "hello");
  assert.equal(h.elements.get("#sentenceText").textContent, "hello");
  assert.equal(h.elements.get("#polishedSentenceText").textContent, "hello.");
  h.api.clearSentence();
  assert.equal(h.api.state.sentenceWords.length, 0);
  assert.equal(h.elements.get("#clearSentenceBtn").disabled, true);
});

test("sentence polish keeps the original tokens and applies the selected speech style", async () => {
  const h = await setup();
  assert.equal(h.api.formatSentence(["나", "학교", "가다"], "polite"), "저는 학교 가요.");
  assert.equal(h.api.formatSentence(["나", "학교", "가다"], "casual"), "나는 학교 가.");
  assert.equal(h.api.formatSentence(["안녕하세요"], "polite"), "안녕하세요.");
});
