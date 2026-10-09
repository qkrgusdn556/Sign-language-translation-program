const MAX_MODEL_BYTES = 128 * 1024 * 1024;

export function parseLabels(text) {
  const cleaned = text.trim();
  if (!cleaned) throw new Error("라벨 파일이 비어 있습니다.");
  let labels;
  try {
    const json = JSON.parse(cleaned);
    labels = Array.isArray(json) ? json : json.names || json.labels;
  } catch {
    labels = cleaned.split(/\r?\n/).map((label) => label.trim()).filter(Boolean);
  }
  if (!Array.isArray(labels) || !labels.length || labels.some((label) => typeof label !== "string" || !label.trim())) {
    throw new Error("라벨은 줄마다 하나씩 쓰거나 JSON 배열이어야 합니다.");
  }
  return labels.map((label) => label.trim());
}

export function softmax(scores) {
  const max = Math.max(...scores);
  const exp = scores.map((score) => Math.exp(score - max));
  const total = exp.reduce((sum, score) => sum + score, 0);
  return exp.map((score) => score / total);
}

export function classifyOutput(values, labels) {
  if (!values || values.length !== labels.length) throw new Error("모델 출력 수와 라벨 개수가 다릅니다.");
  const probabilities = softmax(Array.from(values));
  const index = probabilities.reduce((best, score, i) => score > probabilities[best] ? i : best, 0);
  return { label: labels[index], confidence: probabilities[index], index };
}

export function createStillnessGate(holdMs = 400, maxMotion = 0.09) {
  let stillSince = null;
  return {
    reset() { stillSince = null; },
    update(motion, now) {
      if (!Number.isFinite(motion) || motion > maxMotion) { stillSince = null; return false; }
      if (stillSince === null) stillSince = now;
      return now - stillSince >= holdMs;
    },
  };
}

export async function loadYolo11Classifier(modelFile, labelsFile) {
  if (!modelFile || !labelsFile) throw new Error("YOLO11 ONNX 모델과 라벨 파일을 모두 선택해 주세요.");
  if (modelFile.size > MAX_MODEL_BYTES) throw new Error("128MB 이하 모델만 브라우저에서 불러올 수 있습니다.");
  const labels = parseLabels(await labelsFile.text());
  const ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs");
  const session = await ort.InferenceSession.create(await modelFile.arrayBuffer(), {
    executionProviders: ["webgpu", "wasm"],
    graphOptimizationLevel: "all",
  });
  const inputName = session.inputNames[0];
  const dims = session.inputMetadata[inputName].dimensions;
  const height = Number(dims[dims.length - 2]) || 224;
  const width = Number(dims[dims.length - 1]) || 224;
  if (dims.length !== 4 || height > 1024 || width > 1024) throw new Error("[1, 3, H, W] 입력을 가진 YOLO11 분류 ONNX 모델만 지원합니다.");
  return { ort, session, inputName, width, height, labels, outputName: session.outputNames[0] };
}

export function imageToTensor(video, model) {
  const canvas = new OffscreenCanvas(model.width, model.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const scale = Math.min(model.width / video.videoWidth, model.height / video.videoHeight);
  const drawWidth = video.videoWidth * scale;
  const drawHeight = video.videoHeight * scale;
  ctx.fillStyle = "rgb(114, 114, 114)";
  ctx.fillRect(0, 0, model.width, model.height);
  ctx.drawImage(video, (model.width - drawWidth) / 2, (model.height - drawHeight) / 2, drawWidth, drawHeight);
  const pixels = ctx.getImageData(0, 0, model.width, model.height).data;
  const plane = model.width * model.height;
  const input = new Float32Array(plane * 3);
  for (let i = 0; i < plane; i++) {
    input[i] = pixels[i * 4] / 255;
    input[plane + i] = pixels[i * 4 + 1] / 255;
    input[plane * 2 + i] = pixels[i * 4 + 2] / 255;
  }
  return input;
}

export async function inferYolo11(video, model) {
  const input = imageToTensor(video, model);
  const outputs = await model.session.run({ [model.inputName]: new model.ort.Tensor("float32", input, [1, 3, model.height, model.width]) });
  return classifyOutput(outputs[model.outputName].data, model.labels);
}
