import { STORAGE_KEY, MAX_SAMPLES, validateSamples, landmarksToVector, classify, classifySequence, trackedHandCount, motionMagnitude, createStabilizer, assignHandSlots, shoulderReference, avatarWrist, poseArmTargets } from "./recognition.js?v=one-hand-40";
import { loadYolo11Classifier, inferYolo11, createStillnessGate } from "./yolo.js?v=pose-arm-1";

let THREE, createHuman, segment, connect, poseArms;
let HAND_CONNECTIONS = [];
let icons;

const els = {
  video: document.querySelector("#camera"),
  canvas: document.querySelector("#overlay"),
  prediction: document.querySelector("#prediction"),
  confidence: document.querySelector("#confidence"),
  confidenceBar: document.querySelector("#confidenceBar"),
  labelInput: document.querySelector("#labelInput"),
  startBtn: document.querySelector("#startBtn"),
  sampleBtn: document.querySelector("#sampleBtn"),
  recordBtn: document.querySelector("#recordBtn"),
  clearBtn: document.querySelector("#clearBtn"),
  sampleCount: document.querySelector("#sampleCount"),
  trackingState: document.querySelector("#trackingState"),
  labels: document.querySelector("#labels"),
  avatarCanvas: document.querySelector("#avatarCanvas"),
  avatarState: document.querySelector("#avatarState"),
  cameraEmpty: document.querySelector("#cameraEmpty"),
  notice: document.querySelector("#notice"),
  noticeText: document.querySelector("#noticeText"),
  undoBtn: document.querySelector("#undoBtn"),
  exportBtn: document.querySelector("#exportBtn"),
  importBtn: document.querySelector("#importBtn"),
  importFile: document.querySelector("#importFile"),
  clearDialog: document.querySelector("#clearDialog"),
  clearConfirm: document.querySelector("#clearConfirm"),
  captureProgress: document.querySelector("#captureProgress"),
  similarity: document.querySelector("#similarity"),
  nearestSample: document.querySelector("#nearestSample"),
  overlayToggle: document.querySelector("#overlayToggle"),
  faceState: document.querySelector("#faceState"),
  yoloModelFile: document.querySelector("#yoloModelFile"),
  yoloLabelsFile: document.querySelector("#yoloLabelsFile"),
  yoloLoadBtn: document.querySelector("#yoloLoadBtn"),
  yoloUnloadBtn: document.querySelector("#yoloUnloadBtn"),
  yoloState: document.querySelector("#yoloState"),
  yoloResult: document.querySelector("#yoloResult"),
  sentenceText: document.querySelector("#sentenceText"),
  clearSentenceBtn: document.querySelector("#clearSentenceBtn"),
  polishSentenceBtn: document.querySelector("#polishSentenceBtn"),
  polishedSentenceText: document.querySelector("#polishedSentenceText"),
  politeStyle: document.querySelector("#politeStyle"),
  casualStyle: document.querySelector("#casualStyle"),
};

const state = {
  landmarker: null,
  poseLandmarker: null,
  shoulders: null,
  pose: null,
  poseArms: null,
  avatar: null,
  running: false,
  starting: false,
  importing: false,
  stream: null,
  frameId: 0,
  lastVideoTime: -1,
  lastFrameAt: 0,
  lastPoseAt: 0,
  lastVector: null,
  slots: [null, null],
  samples: [],
  recording: null,
  undo: null,
  storageBlocked: false,
  storageLoading: typeof fetch !== "undefined" || typeof indexedDB !== "undefined",
  fileStorage: false,
  saving: false,
  database: null,
  stabilizer: createStabilizer(),
  yolo: null,
  yoloLoading: false,
  yoloBusy: false,
  yoloLastAt: 0,
  stillness: createStillnessGate(),
  lastWrists: null,
  sentenceWords: [],
  sentenceArmed: true,
  speechStyle: "polite",
  polishedSentence: "",
  motion: { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] },
  partialMatchAllowed: false,
};

els.startBtn.addEventListener("click", () => state.running ? stopCamera() : startCamera());
els.sampleBtn.addEventListener("click", () => addSample(els.labelInput.value));
els.recordBtn.addEventListener("click", recordSequence);
els.clearBtn.addEventListener("click", () => els.clearDialog.showModal());
els.clearConfirm.addEventListener("click", clearSamples);
document.querySelector("#clearCancel").addEventListener("click", () => els.clearDialog.close());
els.labelInput.addEventListener("input", updateControls);
els.exportBtn.addEventListener("click", exportSamples);
els.importBtn.addEventListener("click", () => els.importFile.click());
els.importFile.addEventListener("change", importSamples);
els.undoBtn.addEventListener("click", undoSamples);
els.overlayToggle.addEventListener("change", () => { els.canvas.hidden = !els.overlayToggle.checked; });
els.yoloLoadBtn.addEventListener("click", loadYolo);
els.yoloUnloadBtn.addEventListener("click", unloadYolo);
els.clearSentenceBtn.addEventListener("click", clearSentence);
els.polishSentenceBtn.addEventListener("click", polishCurrentSentence);
els.politeStyle.addEventListener("change", () => setSpeechStyle("polite"));
els.casualStyle.addEventListener("change", () => setSpeechStyle("casual"));
window.addEventListener("pagehide", () => stopCamera());
document.addEventListener("visibilitychange", () => {
  if (document.hidden && (state.running || state.recording)) stopCamera("화면을 벗어나 카메라를 중지했습니다.");
});

state.samples = loadSamples();
updateSampleUi();
renderSentence();
if (typeof fetch !== "undefined") await initializeFileStorage();
else if (state.storageLoading) await initializeSampleStorage();
loadAvatar();
loadIcons();

async function loadIcons() {
  try {
    icons = await import("https://cdn.jsdelivr.net/npm/lucide@0.468.0/+esm");
    for (const [id, name] of [["exportBtn", "Download"], ["importBtn", "Upload"], ["undoBtn", "Undo2"], ["clearBtn", "Trash2"]]) {
      els[id].prepend(icons.createElement(icons[name], { width: 16, height: 16, "aria-hidden": "true" }));
    }
    document.querySelector(".camera-symbol").replaceChildren(icons.createElement(icons.Camera, { width: 32, height: 32, "aria-hidden": "true" }));
    updateSampleUi();
  } catch { /* Text controls remain available if the optional icon CDN is offline. */ }
}

async function loadAvatar() {
  try {
    THREE = await import("https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js");
    ({ createHuman, segment, connect, poseArms } = await import("./humanoid.js?v=pose-arm-1"));
    initAvatar();
    requestAnimationFrame(animateAvatar);
    els.avatarState.textContent = "대기 중";
  } catch {
    els.avatarState.textContent = "3D 화면을 불러오지 못했습니다. 네트워크 연결 후 새로고침해 주세요.";
  }
}

async function startCamera() {
  if (state.starting || state.running) return;
  state.starting = true;
  updateControls();
  els.startBtn.textContent = "준비 중";
  els.trackingState.textContent = "모델 로딩";
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("카메라를 사용할 수 없는 브라우저입니다.");
    if (!state.landmarker || !state.poseLandmarker) {
      const { HandLandmarker, PoseLandmarker, FilesetResolver } = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304");
      HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;
      const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm");
      const options = {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO", numHands: 2,
        minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.6,
      };
      if (!state.landmarker) {
        try {
          state.landmarker = await HandLandmarker.createFromOptions(vision, options);
        } catch {
          options.baseOptions.delegate = "CPU";
          state.landmarker = await HandLandmarker.createFromOptions(vision, options);
        }
      }
      if (!state.poseLandmarker) {
        els.trackingState.textContent = "팔 모델 로딩";
        const poseOptions = {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO", numPoses: 1,
          minPoseDetectionConfidence: 0.35, minPosePresenceConfidence: 0.35, minTrackingConfidence: 0.35,
        };
        try {
          state.poseLandmarker = await PoseLandmarker.createFromOptions(vision, poseOptions);
        } catch {
          poseOptions.baseOptions.delegate = "CPU";
          state.poseLandmarker = await PoseLandmarker.createFromOptions(vision, poseOptions);
        }
      }
    }
    els.trackingState.textContent = "카메라 연결";
    state.stream = await navigator.mediaDevices.getUserMedia({ video: {
      width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user", frameRate: { ideal: 30, max: 30 },
    }, audio: false });
    if (document.hidden) { stopCamera(); return; }
    els.video.srcObject = state.stream;
    await els.video.play();
    if (document.hidden) { stopCamera(); return; }
    state.stream.getVideoTracks().forEach((track) => track.addEventListener("ended", () => stopCamera("카메라 연결이 끊어졌습니다."), { once: true }));
    state.running = true;
    state.lastVideoTime = -1;
    state.lastPoseAt = 0;
    els.cameraEmpty.hidden = true;
    els.trackingState.textContent = "손 대기";
    state.frameId = requestAnimationFrame(loop);
  } catch (error) {
    stopCamera();
    const messages = {
      NotAllowedError: "카메라 권한이 차단되었습니다. 주소창의 카메라 권한을 허용한 뒤 다시 시작해 주세요.",
      NotFoundError: "연결된 카메라를 찾지 못했습니다.",
      NotReadableError: "카메라를 열지 못했습니다. 다른 앱에서 사용 중인지 확인해 주세요.",
    };
    notify(messages[error.name] || "카메라 준비에 실패했습니다. 네트워크와 카메라 연결을 확인한 뒤 다시 시도해 주세요.", true);
  } finally {
    state.starting = false;
    updateControls();
  }
}

async function loadYolo() {
  if (state.yoloLoading) return;
  state.yoloLoading = true;
  updateYoloUi("모델 로딩 중");
  try {
    state.yolo = await loadYolo11Classifier(els.yoloModelFile.files[0], els.yoloLabelsFile.files[0]);
    state.stillness.reset();
    state.lastWrists = null;
    els.yoloResult.textContent = `${state.yolo.labels.length}개 라벨을 불러왔습니다. 손이 정지하면 분류합니다.`;
    updateYoloUi("연결됨");
  } catch (error) {
    state.yolo = null;
    updateYoloUi("연결 실패");
    els.yoloResult.textContent = error.message || "YOLO11 모델을 불러오지 못했습니다.";
  } finally {
    state.yoloLoading = false;
    updateYoloUi();
  }
}

function unloadYolo() {
  state.yolo = null;
  state.yoloBusy = false;
  state.stillness.reset();
  state.lastWrists = null;
  state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
  state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
  els.yoloModelFile.value = "";
  els.yoloLabelsFile.value = "";
  els.yoloResult.textContent = "YOLO11 연결을 해제했습니다.";
  updateYoloUi("모델 없음");
}

function updateYoloUi(status) {
  els.yoloState.textContent = status || (state.yolo ? "연결됨" : "모델 없음");
  els.yoloLoadBtn.disabled = state.yoloLoading;
  els.yoloUnloadBtn.disabled = !state.yolo || state.yoloLoading;
  els.yoloModelFile.disabled = state.yoloLoading;
  els.yoloLabelsFile.disabled = state.yoloLoading;
}

function stopCamera(message) {
  state.partialMatchAllowed = false;
  state.running = false;
  state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
  cancelAnimationFrame(state.frameId);
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  els.video.pause();
  els.video.srcObject = null;
  state.lastVector = null;
  state.shoulders = null;
  state.lastPoseAt = 0;
  state.pose = null;
  state.poseArms = null;
  state.stillness.reset();
  state.lastWrists = null;
  els.faceState.textContent = "어깨 대기";
  state.slots = [null, null];
  state.stabilizer.reset();
  finishRecording(false);
  els.canvas.getContext("2d").clearRect(0, 0, els.canvas.width, els.canvas.height);
  els.cameraEmpty.hidden = false;
  els.trackingState.textContent = "중지";
  if (state.avatar) { updateAvatar([null, null]); els.avatarState.textContent = "대기 중"; }
  showPrediction("대기 중", "저장된 손모양과 비교할 준비가 되었습니다.", 0);
  updateControls();
  if (message) notify(message);
}

function loop(now) {
  if (!state.running) return;
  try {
    if (els.video.readyState >= 2 && els.video.currentTime !== state.lastVideoTime && now - state.lastFrameAt >= 45) {
      state.lastVideoTime = els.video.currentTime;
      state.lastFrameAt = now;
      resizeCanvas();
      const result = state.landmarker.detectForVideo(els.video, now);
      if (now - state.lastPoseAt >= 90) {
        state.lastPoseAt = now;
        state.pose = state.poseLandmarker.detectForVideo(els.video, now).landmarks[0] || null;
        state.shoulders = shoulderReference(state.pose, els.video.videoWidth, els.video.videoHeight);
      }
      state.slots = assignHandSlots(result, state.slots);
      state.poseArms = poseArmTargets(state.pose, state.shoulders, state.slots);
      els.faceState.textContent = state.shoulders ? "어깨 기준 연결됨" : "어깨 대기";
      els.trackingState.textContent = state.poseArms ? "어깨·팔 추적" : state.shoulders ? "팔 대기" : "어깨 대기";
      drawLandmarks(result);
      updateAvatar(state.shoulders ? state.slots : [null, null], result);
      state.lastVector = state.shoulders ? landmarksToVector(result.landmarks, state.shoulders) : null;
      if (state.lastVector) {
        updatePartialTracking(state.lastVector, now);
        if (state.avatar) els.avatarState.textContent = state.poseArms ? `팔·손 ${result.landmarks.length}개 연결됨` : `손 ${result.landmarks.length}개 연결됨 · 팔 대기`;
        if (!state.recording) {
          const dynamicResult = updateMotionRecognition(state.lastVector, now);
          if (dynamicResult) applyRecognition(dynamicResult, "동작");
          else if (!state.motion.active) predict(state.lastVector, now);
          evaluateYolo(result.landmarks, now);
        }
      } else {
        state.partialMatchAllowed = false;
        state.stillness.reset();
        state.lastWrists = null;
        state.stabilizer.reset();
        const preservingMotion = noteMotionGap(now);
        els.trackingState.textContent = state.shoulders ? "손 없음" : "어깨 대기";
        if (state.avatar) els.avatarState.textContent = state.shoulders ? "손 대기" : "어깨 기준 대기";
        if (preservingMotion) {
          els.trackingState.textContent = "추적 복귀 대기";
          els.confidence.textContent = "짧은 추적 누락입니다. 동작을 잠시 보존하며 결과 확정은 멈춥니다.";
        } else showPrediction("대기 중", state.shoulders ? "손이 감지되지 않았습니다." : "양어깨와 손이 함께 보이도록 해 주세요.", 0);
      }
      captureFrame(now);
      updateControls();
    }
    if (now - state.lastFrameAt > 500 && (state.lastVector || state.shoulders)) {
      state.lastVector = null;
      state.shoulders = null;
      state.pose = null;
      state.poseArms = null;
      els.faceState.textContent = "어깨 대기";
      state.slots = [null, null];
      state.stabilizer.reset();
      state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
      updateAvatar(state.slots);
      showPrediction("대기 중", "새 카메라 프레임을 기다리고 있습니다.", 0);
      els.trackingState.textContent = "영상 대기";
      updateControls();
    }
    if (state.recording && now >= state.recording.endAt) finishRecording(true);
    state.frameId = requestAnimationFrame(loop);
  } catch {
    stopCamera();
    state.landmarker?.close();
    state.landmarker = null;
    state.poseLandmarker?.close();
    state.poseLandmarker = null;
    notify("손 추적이 중단되었습니다. 카메라를 다시 시작해 주세요.", true);
  }
}

function evaluateYolo(hands, now) {
  if (!state.yolo || state.yoloBusy) return;
  const wrists = hands.slice(0, 2).map((hand) => hand[0]);
  if (!wrists.length) return;
  let motion = Infinity;
  if (state.lastWrists?.length === wrists.length) {
    motion = wrists.reduce((total, wrist, index) => total + Math.hypot(wrist.x - state.lastWrists[index].x, wrist.y - state.lastWrists[index].y), 0) / wrists.length;
  }
  state.lastWrists = wrists.map((wrist) => ({ x: wrist.x, y: wrist.y }));
  if (!state.stillness.update(motion, now)) {
    els.yoloResult.textContent = "손이 정지하면 YOLO11 결과가 표시됩니다.";
    return;
  }
  if (now - state.yoloLastAt < 650) return;
  state.yoloLastAt = now;
  state.yoloBusy = true;
  inferYolo11(els.video, state.yolo).then((result) => {
    if (!state.yolo) return;
    els.yoloResult.textContent = result.confidence >= 0.6
      ? `YOLO11 정지 수어: ${result.label} (${Math.round(result.confidence * 100)}%)`
      : `YOLO11 판단 보류 (${Math.round(result.confidence * 100)}%)`;
  }).catch((error) => {
    els.yoloResult.textContent = `YOLO11 추론 오류: ${error.message || "모델을 확인해 주세요."}`;
  }).finally(() => { state.yoloBusy = false; });
}

function resizeCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  const width = Math.round(rect.width * devicePixelRatio);
  const height = Math.round(rect.height * devicePixelRatio);

  if (els.canvas.width !== width || els.canvas.height !== height) {
    els.canvas.width = width;
    els.canvas.height = height;
  }
}

function drawLandmarks(result) {
  const ctx = els.canvas.getContext("2d");
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (!els.overlayToggle.checked) return;
  // Match object-fit: contain, including letterboxing and the shared CSS mirror.
  const scale = Math.min(els.canvas.width / els.video.videoWidth, els.canvas.height / els.video.videoHeight);
  const width = els.video.videoWidth * scale;
  const height = els.video.videoHeight * scale;
  const x = (els.canvas.width - width) / 2;
  const y = (els.canvas.height - height) / 2;
  ctx.lineWidth = 2 * Math.min(devicePixelRatio, 2);
  ctx.strokeStyle = "#65dfb3";
  ctx.fillStyle = "#f5ce91";
  if (state.poseArms) {
    ctx.strokeStyle = "#bda9ff";
    ctx.lineWidth = 3 * Math.min(devicePixelRatio, 2);
    state.poseArms.forEach((arm) => {
      const points = [arm.shoulder.source, arm.elbow.source, arm.wrist.source];
      ctx.beginPath();
      points.forEach((point, index) => {
        const px = x + point.x * width;
        const py = y + point.y * height;
        if (index) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      });
      ctx.stroke();
    });
    ctx.strokeStyle = "#65dfb3";
  }
  for (const landmarks of result.landmarks) {
    ctx.beginPath();
    for (const connection of HAND_CONNECTIONS) {
      const a = landmarks[connection.start ?? connection[0]];
      const b = landmarks[connection.end ?? connection[1]];
      ctx.moveTo(x + a.x * width, y + a.y * height);
      ctx.lineTo(x + b.x * width, y + b.y * height);
    }
    ctx.stroke();
    for (const p of landmarks) {
      ctx.beginPath();
      ctx.arc(x + p.x * width, y + p.y * height, 3 * Math.min(devicePixelRatio, 2), 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function initAvatar() {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  const renderer = new THREE.WebGLRenderer({
    canvas: els.avatarCanvas,
    alpha: true,
    antialias: true,
  });

  camera.position.set(0, 0.3, 5.4);
  camera.lookAt(0, -0.05, 0);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const keyLight = new THREE.DirectionalLight(0xffffff, 2.1);
  keyLight.position.set(3, 4, 5);
  scene.add(keyLight);
  scene.add(new THREE.AmbientLight(0x9fb6aa, 1.4));

  const floor = new THREE.GridHelper(8, 16, 0x2c5541, 0x17251f);
  floor.position.y = -1.72;
  scene.add(floor);

  const { body, skin, arms } = createHuman(scene);
  const handRigs = [createHandRig(skin), createHandRig(skin)];
  handRigs[0].group.position.set(-1.15, -0.38, 0.75);
  handRigs[1].group.position.set(1.15, -0.38, 0.75);
  handRigs.forEach((rig) => scene.add(rig.group));
  seedIdleHands(handRigs);
  handRigs.forEach((rig) => {
    rig.targetPosition = rig.group.position.clone();
    rig.targetPoints = rig.joints.map((joint) => joint.position.clone());
    rig.points = rig.joints.map((joint) => joint.position.clone());
    rig.live = false;
  });

  state.avatar = {
    scene,
    camera,
    renderer,
    body,
    arms,
    handRigs,
    hasLiveHands: false,
    lastTime: 0,
  };
}

function createHandRig(material) {
  const group = new THREE.Group();
  const joints = Array.from({ length: 21 }, () => {
    const joint = new THREE.Mesh(new THREE.SphereGeometry(0.038, 16, 10), material);
    group.add(joint);
    return joint;
  });

  const connections = [];
  for (const start of [1, 5, 9, 13, 17]) {
    for (let i = start; i < start + 3; i++) connections.push([i, i + 1]);
  }
  connections.push([0, 1]);
  const bones = connections.map(() => {
    const bone = segment(material, 0.032, 0.04);
    group.add(bone);
    return bone;
  });
  const palmGeometry = new THREE.BufferGeometry();
  palmGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(36), 3));
  palmGeometry.setIndex([0,1,2,0,2,3,0,3,4,0,4,5,6,8,7,6,9,8,6,10,9,6,11,10,
    0,6,7,0,7,1,1,7,8,1,8,2,2,8,9,2,9,3,3,9,10,3,10,4,4,10,11,4,11,5,5,11,6,5,6,0]);
  const palm = new THREE.Mesh(palmGeometry, material);
  group.add(palm);
  return { group, joints, bones, connections, palm };
}

function seedIdleHands(handRigs) {
  const idleHands = [
    makeIdleHand(-0.22),
    makeIdleHand(0.22),
  ];
  handRigs.forEach((rig, index) => {
    rig.group.visible = true;
    const points = idleHands[index];
    if (index === 0) points.forEach((point) => { point.x *= -1; });
    setRigPoints(rig, points);
  });
}

function makeIdleHand(spread) {
  const points = Array.from({ length: 21 }, () => new THREE.Vector3());
  points[0].set(0, 0, 0);
  const fingerStarts = [1, 5, 9, 13, 17];
  const xOffsets = [-0.18, -0.12, -0.015, 0.09, 0.18];
  const heights = [0.34, 0.59, 0.65, 0.59, 0.48];

  fingerStarts.forEach((start, finger) => {
    for (let i = 0; i < 4; i += 1) {
      const t = (i + 1) / 4;
      points[start + i].set(
        xOffsets[finger] + spread * t * (finger - 2) * 0.08,
        finger === 0 ? heights[finger] * t : 0.26 + (heights[finger] - 0.26) * i / 3,
        0.018 * Math.sin(t * Math.PI + finger),
      );
    }
  });

  return points;
}

function updateAvatar(slots, result) {
  if (!state.avatar) return;
  state.avatar.hasLiveHands = slots.some(Boolean);
  for (let i = 0; i < state.avatar.handRigs.length; i += 1) {
    const rig = state.avatar.handRigs[i];
    const hand = slots[i];
    rig.live = Boolean(hand);
    if (hand && state.shoulders) {
      const wrist = hand[0];
      const target = state.poseArms?.[i]?.wrist || avatarWrist(wrist, state.shoulders);
      rig.targetPosition.set(target.x, target.y, target.z);
      const resultIndex = result?.landmarks.indexOf(hand);
      rig.targetPoints = handToAvatarPoints(result?.worldLandmarks?.[resultIndex] || hand);
    } else {
      rig.targetPosition.set(i === 0 ? -1.15 : 1.15, -0.38, 0.75);
      const points = makeIdleHand(i === 0 ? -0.22 : 0.22);
      if (i === 0) points.forEach((point) => { point.x *= -1; });
      rig.targetPoints = points;
    }
  }
}

function handToAvatarPoints(hand) {
  const wrist = hand[0];
  const middleBase = hand[9];
  const scale =
    Math.hypot(
      middleBase.x - wrist.x,
      middleBase.y - wrist.y,
      middleBase.z - wrist.z,
    ) || 1;

  return hand.map(
    (point) =>
      new THREE.Vector3(
        ((wrist.x - point.x) / scale) * 0.36,
        ((wrist.y - point.y) / scale) * 0.36,
        ((wrist.z - point.z) / scale) * 0.36,
      ),
  );
}

function setRigPoints(rig, points) {
  for (let i = 0; i < rig.joints.length; i += 1) {
    rig.joints[i].position.copy(points[i]);
  }

  rig.connections.forEach(([from, to], index) => connect(rig.bones[index], points[from], points[to]));
  const normal = new THREE.Vector3().subVectors(points[5], points[0])
    .cross(new THREE.Vector3().subVectors(points[17], points[0])).normalize().multiplyScalar(0.038);
  const attribute = rig.palm.geometry.attributes.position;
  [0, 1, 5, 9, 13, 17].forEach((pointIndex, index) => {
    const point = points[pointIndex];
    attribute.setXYZ(index, point.x + normal.x, point.y + normal.y, point.z + normal.z);
    attribute.setXYZ(index + 6, point.x - normal.x, point.y - normal.y, point.z - normal.z);
  });
  attribute.needsUpdate = true;
  rig.palm.geometry.computeVertexNormals();
  rig.palm.geometry.computeBoundingSphere();
}

function animateAvatar(time = 0) {
  if (!state.avatar) return;
  requestAnimationFrame(animateAvatar);
  if (document.hidden) return;
  const rect = els.avatarCanvas.getBoundingClientRect();
  if (rect.bottom < 0 || rect.top > innerHeight) return;
  resizeAvatar();
  const alpha = 1 - Math.exp(-Math.min(time - state.avatar.lastTime, 100) / 75);
  state.avatar.lastTime = time;
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  state.avatar.handRigs.forEach((rig, index) => {
    rig.group.position.lerp(rig.targetPosition, alpha);
    rig.points.forEach((point, i) => point.lerp(rig.targetPoints[i], alpha));
    const idleRotation = (index === 0 ? -1 : 1) * (0.18 + (reducedMotion ? 0 : Math.sin(time * 0.0015) * 0.025));
    rig.group.rotation.z = THREE.MathUtils.lerp(rig.group.rotation.z, rig.live ? 0 : idleRotation, alpha);
    setRigPoints(rig, rig.points);
  });
  poseArms(state.avatar.arms, state.avatar.handRigs, state.poseArms);
  state.avatar.renderer.render(state.avatar.scene, state.avatar.camera);
}

function resizeAvatar() {
  const { renderer, camera } = state.avatar;
  const rect = els.avatarCanvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));

  const pixelWidth = Math.round(width * renderer.getPixelRatio());
  const pixelHeight = Math.round(height * renderer.getPixelRatio());

  if (els.avatarCanvas.width !== pixelWidth || els.avatarCanvas.height !== pixelHeight) {
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.position.z = Math.max(5.4, 4.7 / camera.aspect);
    camera.updateProjectionMatrix();
  }
}

function addSample(rawLabel) {
  const label = rawLabel.trim();
  if (!label || label.length > 40 || !state.lastVector || state.recording || !state.running) return;
  commitSamples([...state.samples, { label, vector: [...state.lastVector] }], `${label}: 샘플 1개를 저장했습니다.`);
}

function recordSequence() {
  if (state.recording) { finishRecording(false); return; }
  const label = els.labelInput.value.trim();
  if (!label || label.length > 40 || !state.running || !state.lastVector) return;
  state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
  state.stabilizer.reset();
  state.recording = { label, frames: [], endAt: performance.now() + 2000, lastAt: -Infinity };
  els.captureProgress.hidden = false;
  els.captureProgress.value = 0;
  updateControls();
}

function captureFrame(now) {
  const recording = state.recording;
  if (!recording) return;
  els.captureProgress.value = Math.min(1, 1 - (recording.endAt - now) / 2000);
  if (now <= recording.endAt && state.lastVector && now - recording.lastAt >= 80) {
    recording.frames.push([...state.lastVector]);
    recording.lastAt = now;
  }
}

function finishRecording(save) {
  const recording = state.recording;
  if (!recording) return;
  state.recording = null;
  els.captureProgress.hidden = true;
  if (save && recording.frames.length >= 6) {
    commitSamples([...state.samples, { label: recording.label, sequence: recording.frames }], `${recording.label}: 동작 샘플 1개를 저장했습니다.`);
  } else notify(save ? "손이 감지되지 않아 저장된 샘플이 없습니다." : "샘플 수집을 취소했습니다.");
  updateControls();
}

function updatePartialTracking(vector, now) {
  const count = trackedHandCount(vector);
  state.partialMatchAllowed = count >= 1;
}

function updateMotionRecognition(vector, now) {
  const gap = state.motion.missingSince;
  if (gap !== undefined && now - gap > 220) {
    state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
    state.stabilizer.reset();
  }
  const motion = state.motion;
  const resumed = motion.missingSince !== undefined;
  if (resumed) {
    delete motion.missingSince;
    motion.lastVector = null;
    motion.lastAt = -Infinity;
    motion.quietSince = 0;
    motion.deltas = [];
  }
  if (now - motion.lastAt < 75) return null;
  const delta = motion.lastVector ? motionMagnitude(motion.lastVector, vector) : 0;
  const previous = motion.lastVector;
  const countChanged = previous && trackedHandCount(previous) !== trackedHandCount(vector);
  motion.lastVector = [...vector];
  motion.lastAt = now;
  if (!motion.active) {
    if (delta > 0.035) {
      motion.active = true;
      motion.frames = previous ? [[...previous]] : [];
      motion.previewAt = -Infinity;
      motion.deltas = [];
      state.stabilizer.reset();
      state.sentenceArmed = true;
      showPrediction("동작 비교 중", "저장된 동작 샘플과 비교하고 있습니다.", 0);
    } else return null;
  }
  motion.frames.push([...vector]);
  previewMotion(now);
  // A recovered coordinate is not evidence of motion or stillness across the gap.
  if (resumed || countChanged) {
    motion.quietSince = 0;
    motion.deltas = [];
    return null;
  }
  motion.deltas = [...(motion.deltas || []), delta].slice(-3);
  const meanMotion = motion.deltas.reduce((sum, value) => sum + value, 0) / motion.deltas.length;
  if (motion.deltas.length < 3 || meanMotion > 0.035) {
    motion.quietSince = 0;
    return null;
  }
  if (!motion.quietSince) motion.quietSince = now;
  if (now - motion.quietSince < 180) return null;
  const frames = motion.frames;
  motion.active = false;
  motion.frames = [];
  motion.displayUntil = now + 2000;
  return frames.length >= 6 ? classifySequence(frames, state.samples) : null;
}

function noteMotionGap(now) {
  const motion = state.motion;
  if (!motion.active) return false;
  motion.missingSince ??= now;
  motion.quietSince = 0;
  if (now - motion.missingSince <= 220) return true;
  state.motion = { lastVector: null, lastAt: 0, active: false, quietSince: 0, frames: [] };
  return false;
}

function previewMotion(now) {
  const motion = state.motion;
  if (motion.frames.length < 6 || now - (motion.previewAt ?? -Infinity) < 180) return;
  motion.previewAt = now;
  const result = classifySequence(motion.frames.slice(-32), state.samples);
  const message = result.reason === "incompatible-tracking"
    ? "감지된 손 개수 또는 좌표 형식이 저장 동작과 달라 비교할 수 없습니다. 양손이 보이는지 확인해 주세요."
    : result.reason === "insufficient-motion"
      ? "남은 손의 실제 움직임이 부족합니다. 마지막 자세만으로 동작을 확정하지 않습니다."
      : result.status === "untrained"
      ? "단어별 동작 샘플이 3개 이상 필요합니다."
      : "동작 중 임시 비교 점수입니다. 끝난 동작을 비교한 뒤 확정합니다.";
  showPrediction("동작 비교 중", message, result.similarity, result.label || result.nearestLabel, "동작", true);
}

function predict(vector, now) {
  if (now < (state.motion.displayUntil || 0)) return;
  const motionCounts = new Map();
  for (const sample of state.samples) if (sample.sequence) motionCounts.set(sample.label, (motionCounts.get(sample.label) || 0) + 1);
  // Motion-trained words must not bypass their sequence via a final static pose.
  const staticSamples = state.samples.filter(sample => (motionCounts.get(sample.label) || 0) < 3);
  const result = classify(vector, staticSamples, { allowPartial: true });
  if (result.status === "untrained" && state.samples.some(sample => sample.sequence)) {
    const counts = new Map();
    state.samples.filter(sample => sample.sequence).forEach(sample => counts.set(sample.label, (counts.get(sample.label) || 0) + 1));
    const ready = [...counts.values()].some(count => count >= 3);
    showPrediction(ready ? "동작 대기" : "동작 샘플 부족", ready
      ? "동작이 끝나면 저장된 동작 샘플과 비교합니다."
      : "DTW 동작 비교에는 단어별 동작 샘플이 3개 이상 필요합니다.", 0);
    return;
  }
  applyRecognition(state.stabilizer.update(result, now), "정지");
}

function applyRecognition(result, mode) {
  if (["match", "pending"].includes(result.status) && result.partial && !state.partialMatchAllowed) {
    result = { ...result, status: "unknown", nearestLabel: result.label, reason: "missing-hands" };
  }
  if (result.status === "untrained") {
    state.sentenceArmed = true;
    showPrediction("학습 필요", `${mode} 수어 이름별 샘플을 3개 이상 저장해 주세요.`, 0);
  } else if (result.status === "unknown") {
    state.sentenceArmed = true;
    showPrediction("판단 보류", result.reason === "missing-hands"
      ? "보이는 손이 없어 확정을 보류합니다."
      : result.reason === "insufficient-motion"
      ? "남은 손의 실제 움직임이 부족합니다. 마지막 자세만으로 동작을 확정하지 않습니다."
      : result.reason === "incompatible-tracking"
      ? "감지된 손 개수 또는 좌표 형식이 저장 동작과 달라 비교할 수 없습니다. 양손이 보이는지 확인해 주세요."
      : "저장된 손모양과 다르거나 후보가 비슷합니다.", result.similarity, result.nearestLabel, mode, true);
  } else if (result.status === "pending") {
    state.sentenceArmed = true;
    showPrediction("확인 중", "손모양을 비교하고 있습니다.", result.similarity, result.label, mode, true);
  } else {
    showPrediction(result.label, result.partial ? `${mode} 수어와 유사합니다. 한 손 누락을 보완한 부분 비교입니다.` : `${mode} 수어와 유사합니다.`, result.similarity, result.label, mode);
    if (state.sentenceArmed || state.sentenceWords.at(-1) !== result.label) {
      state.sentenceWords.push(result.label);
      state.sentenceArmed = false;
      state.polishedSentence = formatSentence(state.sentenceWords, state.speechStyle);
      renderSentence();
    }
  }
}

function clearSentence() {
  state.sentenceWords = [];
  state.sentenceArmed = true;
  state.polishedSentence = "";
  renderSentence();
}

function setSpeechStyle(style) {
  state.speechStyle = style;
  if (state.sentenceWords.length) polishCurrentSentence();
}

function polishCurrentSentence() {
  state.polishedSentence = formatSentence(state.sentenceWords, state.speechStyle);
  renderSentence();
}

export function formatSentence(words, speechStyle) {
  if (!words.length) return "";
  const opening = speechStyle === "polite" ? { "나": "저는" } : { "나": "나는" };
  const endings = speechStyle === "polite"
    ? { "가다": "가요", "오다": "와요", "먹다": "먹어요", "마시다": "마셔요", "하다": "해요", "보다": "봐요", "있다": "있어요", "없다": "없어요", "알다": "알아요", "모르다": "몰라요", "이다": "예요" }
    : { "가다": "가", "오다": "와", "먹다": "먹어", "마시다": "마셔", "하다": "해", "보다": "봐", "있다": "있어", "없다": "없어", "알다": "알아", "모르다": "몰라", "이다": "야" };
  const sentence = words.map((word, index) => index === 0 ? (opening[word] || word) : word);
  sentence[sentence.length - 1] = endings[sentence.at(-1)] || sentence.at(-1);
  const text = sentence.join(" ").replace(/[.?!]+$/, "");
  return `${text}.`;
}

function renderSentence() {
  els.sentenceText.textContent = state.sentenceWords.length ? state.sentenceWords.join(" ") : "대기 중";
  els.polishedSentenceText.textContent = state.polishedSentence || "대기 중";
  els.clearSentenceBtn.disabled = !state.sentenceWords.length;
  els.polishSentenceBtn.disabled = !state.sentenceWords.length;
}

function showPrediction(label, message, confidence, nearestLabel = null, mode = "", tentative = false) {
  els.nearestSample.textContent = nearestLabel ? `${mode}${tentative ? " 후보" : ""}: ${nearestLabel}` : "비교 샘플 없음";
  els.prediction.textContent = label;
  els.confidence.textContent = message;
  els.confidenceBar.style.width = `${Math.round(confidence * 100)}%`;
  els.similarity.textContent = `${Math.round(confidence * 100)}%`;
}

function updateControls() {
  const label = els.labelInput.value.trim();
  const busy = Boolean(state.recording || state.importing || state.saving || state.storageLoading);
  const canCapture = state.running && Boolean(state.lastVector) && label.length > 0 && label.length <= 40 && !state.storageBlocked;
  els.startBtn.disabled = state.starting;
  els.startBtn.textContent = state.starting ? "준비 중" : state.running ? "카메라 중지" : "카메라 시작";
  els.sampleBtn.disabled = !canCapture || busy;
  els.recordBtn.disabled = !state.recording && (!canCapture || busy);
  els.recordBtn.textContent = state.recording ? "수집 취소" : "2초 동작 수집";
  els.labelInput.disabled = busy;
  els.clearBtn.disabled = busy || !state.samples.length || state.storageBlocked;
  els.exportBtn.disabled = !state.samples.length;
  els.importBtn.disabled = busy || state.storageBlocked;
  els.undoBtn.disabled = busy || !state.undo;
  els.labels.querySelectorAll("button").forEach((button) => { button.disabled = busy || state.storageBlocked; });
}

function updateSampleUi() {
  els.sampleCount.textContent = String(state.samples.length);

  const counts = new Map();
  const shoulderCounts = new Map();
  const motionCounts = new Map();
  for (const sample of state.samples) {
    counts.set(sample.label, (counts.get(sample.label) || 0) + 1);
    if (sample.vector?.length === 130) shoulderCounts.set(sample.label, (shoulderCounts.get(sample.label) || 0) + 1);
    if (sample.sequence) motionCounts.set(sample.label, (motionCounts.get(sample.label) || 0) + 1);
  }

  els.labels.replaceChildren(
    ...[...counts.entries()].map(([label, count]) => {
      const item = document.createElement("div");
      item.className = "sample-row";
      const name = document.createElement("span");
      name.textContent = label;
      const amount = document.createElement("span");
      const shoulderCount = shoulderCounts.get(label) || 0;
      const motionCount = motionCounts.get(label) || 0;
      amount.className = shoulderCount >= 3 ? "sample-ready" : "sample-pending";
      amount.textContent = `${count}개 · 정지 ${shoulderCount}개 · 동작 ${motionCount}개`;
      const button = document.createElement("button");
      button.className = "icon-button";
      button.textContent = "×";
      if (icons) button.replaceChildren(icons.createElement(icons.X, { width: 16, height: 16, "aria-hidden": "true" }));
      button.title = `${label} 샘플 삭제`;
      button.setAttribute("aria-label", button.title);
      button.addEventListener("click", () => commitSamples(state.samples.filter((sample) => sample.label !== label), `${label} 샘플을 삭제했습니다.`));
      item.append(name, amount, button);
      return item;
    }),
  );
  if (!counts.size) {
    const empty = document.createElement("p");
    empty.className = "empty-samples";
    empty.textContent = "저장된 샘플 없음";
    els.labels.append(empty);
  }
  updateControls();
}

function commitSamples(next, message, remember = true) {
  if (state.saving || state.storageLoading) return false;
  if (state.storageBlocked) { notify("기존 저장 데이터를 읽지 못해 변경을 중지했습니다.", true); return false; }
  try {
    const checked = validateSamples(next);
    if (state.fileStorage || state.database) return commitDatabaseSamples(checked, message, remember);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(checked));
    if (remember) state.undo = state.samples;
    state.samples = checked;
    state.stabilizer.reset();
    showPrediction("대기 중", "샘플이 변경되었습니다.", 0);
    updateSampleUi();
    notify(message);
    return true;
  } catch (error) {
    notify(error.name === "QuotaExceededError" ? "저장 공간이 부족합니다. 백업 후 불필요한 샘플을 정리해 주세요." : error.message, true);
    return false;
  }
}

function databaseSamples(database, mode, samples) {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction("samples", mode);
    const store = transaction.objectStore("samples");
    const request = mode === "readonly" ? store.get(STORAGE_KEY) : store.put(samples, STORAGE_KEY);
    transaction.oncomplete = () => resolve(request.result);
    transaction.onabort = () => reject(transaction.error || new Error("샘플 저장을 완료하지 못했습니다."));
    transaction.onerror = () => reject(transaction.error || request.error);
  });
}

async function fileSamples(samples) {
  const response = await fetch("/api/samples", samples === undefined
    ? { cache: "no-store" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ samples }) });
  if (!response.ok) throw new Error("컴퓨터 파일 저장에 실패했습니다. 로컬 서버 실행과 디스크 공간을 확인해 주세요.");
  return response.json();
}

async function initializeFileStorage() {
  try {
    const saved = await fileSamples();
    if (saved.samples !== null) {
      state.samples = validateSamples(saved.samples);
      state.storageBlocked = false;
      els.notice.hidden = true;
    } else {
      if (typeof indexedDB !== "undefined") await initializeSampleStorage(true);
      if (state.storageBlocked) throw new Error("기존 데이터를 읽지 못했습니다. 백업을 보존해 주세요.");
      state.storageLoading = true;
      await fileSamples(state.samples);
    }
    state.fileStorage = true;
    const location = document.querySelector("#storageLocation");
    if (location) location.textContent = "컴퓨터 파일";
  } catch (error) {
    state.storageBlocked = true;
    notify(`${error.message} run_web.bat으로 실행해 주세요.`, true);
  } finally {
    state.storageLoading = false;
    updateSampleUi();
  }
}

async function initializeSampleStorage(readOnly = false) {
  try {
    const database = await new Promise((resolve, reject) => {
      const request = indexedDB.open("sign-translator", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("samples");
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error("다른 탭을 닫고 다시 열어 주세요."));
    });
    const saved = await databaseSamples(database, "readonly");
    if (saved !== undefined) {
      state.samples = validateSamples(saved);
      state.storageBlocked = false;
      els.notice.hidden = true;
    } else if (!state.storageBlocked && !readOnly) {
      // Keep the legacy copy untouched until the user explicitly changes data.
      await databaseSamples(database, "readwrite", state.samples);
    }
    state.database = database;
    database.onversionchange = () => {
      database.close();
      if (state.fileStorage) return;
      state.storageBlocked = true;
      updateControls();
      notify("저장소가 변경되었습니다. 페이지를 새로고침해 주세요.", true);
    };
  } catch (error) {
    state.storageBlocked = true;
    notify(`저장소를 열지 못했습니다. 기존 데이터는 유지됩니다. ${error.message}`, true);
  } finally {
    state.storageLoading = false;
    updateSampleUi();
  }
}

async function commitDatabaseSamples(checked, message, remember) {
  state.saving = true;
  updateControls();
  try {
    if (state.fileStorage) await fileSamples(checked);
    else await databaseSamples(state.database, "readwrite", checked);
    if (remember) state.undo = state.samples;
    state.samples = checked;
    state.stabilizer.reset();
    showPrediction("대기 중", "샘플이 변경되었습니다.", 0);
    notify(message);
    return true;
  } catch (error) {
    notify(error.name === "QuotaExceededError" ? "브라우저 저장 공간이 부족합니다. 백업을 저장해 주세요. 기존 샘플은 유지됩니다." : error.message, true);
    return false;
  } finally {
    state.saving = false;
    updateSampleUi();
  }
}

function loadSamples() {
  try {
    return validateSamples(JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"));
  } catch {
    state.storageBlocked = true;
    notify("기존 저장 데이터를 읽지 못했습니다. 원본 보호를 위해 저장·삭제를 잠갔습니다.", true);
    return [];
  }
}

function clearSamples() {
  els.clearDialog.close();
  if (state.recording || state.importing) return;
  commitSamples([], "모든 샘플을 삭제했습니다. 되돌리기로 복구할 수 있습니다.");
}

async function undoSamples() {
  if (!state.undo || state.recording || state.importing) return;
  if (await commitSamples(state.undo, "직전 변경을 되돌렸습니다.", false)) {
    state.undo = null;
    els.undoBtn.hidden = true;
    updateControls();
  }
}

function notify(message, error = false) {
  els.notice.hidden = false;
  els.notice.dataset.tone = error ? "error" : "info";
  els.noticeText.textContent = message;
  els.undoBtn.hidden = !state.undo;
}

function exportSamples() {
  const blob = new Blob([JSON.stringify({ version: 3, exportedAt: new Date().toISOString(), samples: state.samples }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `sign-samples-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  notify(`${state.samples.length}개 샘플의 백업 파일을 만들었습니다.`);
}

async function importSamples() {
  const file = els.importFile.files?.[0];
  if (!file || state.recording || state.importing) return;
  state.importing = true;
  updateControls();
  try {
    if (file.size > 64 * 1024 * 1024) throw new Error("64MB 이하의 샘플 파일만 불러올 수 있습니다.");
    const incoming = validateSamples(JSON.parse(await file.text()));
    const occurrences = new Map();
    state.samples.forEach((sample) => {
      const key = JSON.stringify(sample);
      occurrences.set(key, (occurrences.get(key) || 0) + 1);
    });
    const unique = incoming.filter((sample) => {
      const key = JSON.stringify(sample);
      const existing = occurrences.get(key) || 0;
      if (existing) { occurrences.set(key, existing - 1); return false; }
      return true;
    });
    if (state.samples.length + unique.length > MAX_SAMPLES) throw new Error(`샘플은 최대 ${MAX_SAMPLES}개까지 저장할 수 있습니다.`);
    if (!unique.length) notify("추가할 새 샘플이 없습니다.");
    else await commitSamples([...state.samples, ...unique], `${unique.length}개 샘플을 추가했습니다. 기존 샘플도 유지됩니다.`);
  } catch (error) {
    notify(error instanceof SyntaxError ? "JSON 파일을 읽지 못했습니다. 기존 데이터는 유지됩니다." : error.message, true);
  } finally {
    state.importing = false;
    els.importFile.value = "";
    updateControls();
  }
}
