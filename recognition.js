export const STORAGE_KEY = "sign-translator-samples";
export const MAX_SAMPLES = 5000;
export const MIN_SEQUENCE_FRAMES = 6;
export const MAX_SEQUENCE_FRAMES = 32;

export function validateSamples(value) {
  const samples = Array.isArray(value) ? value : [1, 2, 3].includes(value?.version) ? value.samples : null;
  if (!Array.isArray(samples) || samples.length > MAX_SAMPLES) throw new Error("샘플 파일 형식 또는 개수를 확인해 주세요.");
  return samples.map((sample) => {
    if (typeof sample?.label !== "string" || !sample.label.trim() || sample.label.trim().length > 40) {
      throw new Error("유효하지 않은 샘플이 있습니다. 기존 데이터는 유지됩니다.");
    }
    const validVector = (vector) => Array.isArray(vector) && [126, 130, 136].includes(vector.length) &&
      vector.every((n) => typeof n === "number" && Number.isFinite(n) && Math.abs(n) <= 100);
    if (validVector(sample.vector)) return { label: sample.label.trim(), vector: [...sample.vector] };
    if (Array.isArray(sample.sequence) && sample.sequence.length >= MIN_SEQUENCE_FRAMES && sample.sequence.length <= MAX_SEQUENCE_FRAMES && sample.sequence.every(validVector)) {
      return { label: sample.label.trim(), sequence: sample.sequence.map((vector) => [...vector]) };
    }
    throw new Error("유효하지 않은 샘플이 있습니다. 기존 데이터는 유지됩니다.");
  });
}

export function landmarksToVector(hands, face = null, mouth = null) {
  if (!hands?.length) return null;
  const vector = hands.slice(0, 2).flatMap((hand) => {
    const wrist = hand[0];
    const base = hand[9];
    const scale = Math.hypot(base.x - wrist.x, base.y - wrist.y, base.z - wrist.z);
    if (scale < 0.0001) return [];
    return hand.flatMap((p) => [(p.x - wrist.x) / scale, (p.y - wrist.y) / scale, (p.z - wrist.z) / scale]);
  });
  if (!vector.length || !vector.every(Number.isFinite)) return null;
  while (vector.length < 126) vector.push(0);
  if (face) {
    const positions = hands.slice(0, 2).flatMap((hand) => faceRelativePoint(hand[0], face));
    while (positions.length < 4) positions.push(0);
    vector.push(...positions);
    if (mouth) vector.push(...mouth);
  }
  return vector;
}

function handCount(vector) {
  return Number(vector.slice(0, 63).some((n) => n !== 0)) + Number(vector.slice(63, 126).some((n) => n !== 0));
}

export function distance(a, b) {
  if (a.length !== b.length || handCount(a) !== handCount(b)) return Infinity;
  let direct = 0;
  let swapped = 0;
  for (let i = 0; i < 126; i++) {
    direct += (a[i] - b[i]) ** 2;
    swapped += (a[i] - b[(i + 63) % 126]) ** 2;
  }
  // Keep old 126-value samples compatible when the detector changes hand order.
  const dimensions = 63 * Math.max(1, handCount(a));
  direct /= dimensions;
  swapped /= dimensions;
  if (a.length >= 130) {
    for (let i = 0; i < 4; i++) {
      direct += 0.2 * (a[126 + i] - b[126 + i]) ** 2;
      swapped += 0.2 * (a[126 + i] - b[126 + (i + 2) % 4]) ** 2;
    }
    for (let i = 130; i < a.length; i++) {
      direct += 0.15 * (a[i] - b[i]) ** 2;
      swapped += 0.15 * (a[i] - b[i]) ** 2;
    }
  }
  return Math.sqrt(Math.min(direct, swapped));
}

export function mouthFeatures(blendshapes) {
  const categories = Array.isArray(blendshapes?.categories) ? blendshapes.categories : [];
  const scores = new Map(categories.map((category) => [category.categoryName, category.score]));
  const names = ["jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthSmileLeft", "mouthSmileRight"];
  return names.map((name) => Math.max(0, Math.min(1, Number(scores.get(name)) || 0)));
}

export function classify(vector, samples) {
  const counts = new Map();
  const compatible = samples.filter((sample) => sample.vector.length === vector.length);
  for (const { label } of compatible) counts.set(label, (counts.get(label) || 0) + 1);
  const eligible = compatible.filter((sample) => counts.get(sample.label) >= 3);
  if (!eligible.length) return { status: "untrained", similarity: 0 };
  const groups = new Map();
  for (const sample of eligible) {
    const d = distance(vector, sample.vector);
    if (!Number.isFinite(d)) continue;
    if (!groups.has(sample.label)) groups.set(sample.label, []);
    groups.get(sample.label).push(d);
  }
  const ranked = [...groups].map(([label, values]) => {
    values.sort((a, b) => a - b);
    const closest = values.slice(0, 3);
    return { label, distance: closest.reduce((a, b) => a + b, 0) / closest.length };
  }).sort((a, b) => a.distance - b.distance);
  if (!ranked.length) return { status: "unknown", similarity: 0 };
  const best = ranked[0];
  // Conservative prototype heuristics, not a calibrated probability of correctness.
  const similarity = Math.exp(-best.distance / 0.2);
  const ambiguous = ranked[1] && ranked[1].distance - best.distance < 0.055;
  if (best.distance > 0.3 || ambiguous) return { status: "unknown", similarity };
  return { status: "match", label: best.label, similarity };
}

export function motionMagnitude(previous, next) {
  if (!previous || !next || previous.length !== next.length || previous.length < 130) return 0;
  return Math.hypot(...previous.slice(126, 130).map((value, index) => next[126 + index] - value));
}

export function dtwDistance(sequenceA, sequenceB) {
  if (!sequenceA?.length || !sequenceB?.length) return Infinity;
  const rows = sequenceA.length + 1, columns = sequenceB.length + 1;
  const matrix = Array.from({ length: rows }, () => Array(columns).fill(Infinity));
  matrix[0][0] = 0;
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < columns; j += 1) {
      const cost = distance(sequenceA[i - 1], sequenceB[j - 1]);
      matrix[i][j] = cost + Math.min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]);
    }
  }
  return matrix[rows - 1][columns - 1] / (sequenceA.length + sequenceB.length);
}

export function classifySequence(sequence, samples) {
  const eligible = samples.filter((sample) => Array.isArray(sample.sequence) && sample.sequence.length >= MIN_SEQUENCE_FRAMES);
  const counts = new Map();
  eligible.forEach((sample) => counts.set(sample.label, (counts.get(sample.label) || 0) + 1));
  const trained = eligible.filter((sample) => counts.get(sample.label) >= 3);
  if (!trained.length) return { status: "untrained", similarity: 0 };
  const grouped = new Map();
  trained.forEach((sample) => {
    const value = dtwDistance(sequence, sample.sequence);
    if (!grouped.has(sample.label)) grouped.set(sample.label, []);
    grouped.get(sample.label).push(value);
  });
  const ranked = [...grouped].map(([label, values]) => ({ label, distance: Math.min(...values) })).sort((a, b) => a.distance - b.distance);
  const best = ranked[0];
  const similarity = Math.exp(-best.distance / 0.24);
  const ambiguous = ranked[1] && ranked[1].distance - best.distance < 0.04;
  return best.distance > 0.35 || ambiguous ? { status: "unknown", similarity } : { status: "match", label: best.label, similarity };
}

export function createStabilizer(holdMs = 350) {
  let candidate = null;
  let since = 0;
  return {
    reset() { candidate = null; since = 0; },
    update(result, now) {
      if (result.status !== "match") { this.reset(); return result; }
      if (candidate !== result.label) { candidate = result.label; since = now; }
      return now - since >= holdMs ? result : { ...result, status: "pending" };
    },
  };
}

export function assignHandSlots(result, previous = [null, null]) {
  const hands = result.landmarks || [];
  const slots = [null, null];
  const pending = [];
  hands.slice(0, 2).forEach((hand, i) => {
    const category = result.handedness?.[i]?.[0];
    // MediaPipe's selfie handedness on raw video: Left maps to mirrored screen-left.
    const slot = category?.categoryName === "Left" ? 0 : category?.categoryName === "Right" ? 1 : -1;
    if (slot >= 0 && category.score >= 0.65 && !slots[slot]) slots[slot] = hand;
    else pending.push(hand);
  });
  for (const hand of pending) {
    const available = [0, 1].filter((slot) => !slots[slot]);
    available.sort((a, b) => {
      const cost = (slot) => previous[slot]
        ? Math.hypot(hand[0].x - previous[slot][0].x, hand[0].y - previous[slot][0].y)
        : Math.abs(hand[0].x - (slot === 0 ? 0.75 : 0.25));
      return cost(a) - cost(b);
    });
    if (available.length) slots[available[0]] = hand;
  }
  return slots;
}

export function faceReference(detections, width, height) {
  // Multiple people make hand ownership ambiguous; do not choose a different person silently.
  if (detections.length !== 1 || !width || !height) return null;
  const box = detections[0].boundingBox;
  if (!box || ![box.originX, box.originY, box.width, box.height].every(Number.isFinite) || box.width < 20 || box.height < 20) return null;
  return { x: (box.originX + box.width / 2) / width, y: (box.originY + box.height / 2) / height,
    scale: box.width / width, aspect: height / width, box, source: "face" };
}

export function poseFaceReference(pose, width, height) {
  if (!Array.isArray(pose) || pose.length < 13 || !width || !height) return null;
  const [nose, leftShoulder, rightShoulder] = [pose[0], pose[11], pose[12]];
  const visible = (point) => point && (point.visibility ?? 1) >= 0.55 && (point.presence ?? 1) >= 0.55;
  if (![nose, leftShoulder, rightShoulder].every(visible)) return null;
  const shoulderWidth = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
  if (shoulderWidth < 0.06) return null;
  return {
    x: nose.x,
    y: nose.y + shoulderWidth * 0.09,
    scale: shoulderWidth * 0.42,
    aspect: height / width,
    box: null,
    source: "pose",
  };
}

export function shoulderReference(pose, width, height) {
  if (!Array.isArray(pose) || pose.length < 13 || !width || !height) return null;
  const [leftShoulder, rightShoulder] = [pose[11], pose[12]];
  const visible = (point) => point && (point.visibility ?? 1) >= 0.55 && (point.presence ?? 1) >= 0.55;
  if (![leftShoulder, rightShoulder].every(visible)) return null;
  const scale = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
  if (scale < 0.06) return null;
  return {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2,
    scale,
    aspect: height / width,
    box: null,
    source: "shoulders",
  };
}

export function faceLandmarksReference(faces, width, height) {
  if (!Array.isArray(faces) || faces.length !== 1 || !width || !height) return null;
  const landmarks = faces[0];
  if (!Array.isArray(landmarks) || landmarks.length < 100) return null;
  const xs = landmarks.map((point) => point.x).filter(Number.isFinite);
  const ys = landmarks.map((point) => point.y).filter(Number.isFinite);
  if (xs.length !== landmarks.length || ys.length !== landmarks.length) return null;
  const left = Math.max(0, Math.min(...xs));
  const right = Math.min(1, Math.max(...xs));
  const top = Math.max(0, Math.min(...ys));
  const bottom = Math.min(1, Math.max(...ys));
  const scale = right - left;
  if (scale < 0.06 || bottom - top < 0.06) return null;
  return {
    x: (left + right) / 2,
    y: (top + bottom) / 2,
    scale,
    aspect: height / width,
    box: { originX: left * width, originY: top * height, width: scale * width, height: (bottom - top) * height },
    source: "landmarks",
  };
}

export function selectFaceReference(detections, pose, width, height, landmarkFaces = null) {
  const faceCount = Array.isArray(landmarkFaces) ? landmarkFaces.length : detections.length;
  if (faceCount > 1) return null;
  const detected = Array.isArray(landmarkFaces)
    ? faceLandmarksReference(landmarkFaces, width, height)
    : faceReference(detections, width, height);
  const fromPose = poseFaceReference(pose, width, height);
  if (!detected || !fromPose) return detected || fromPose;

  // A face box far from the pose nose is usually a false positive (for example, a hand or background detail).
  const separation = Math.hypot(detected.x - fromPose.x, detected.y - fromPose.y);
  return separation <= Math.max(0.18, fromPose.scale * 1.4) ? detected : fromPose;
}

export function createFaceTracker({ maxLostMs = 420, smoothing = 0.35 } = {}) {
  let face = null;
  let lastSeen = -Infinity;
  return {
    reset() { face = null; lastSeen = -Infinity; },
    update(candidate, now) {
      if (candidate) {
        if (face) {
          face = {
            ...candidate,
            x: face.x + (candidate.x - face.x) * smoothing,
            y: face.y + (candidate.y - face.y) * smoothing,
            scale: face.scale + (candidate.scale - face.scale) * smoothing,
          };
        } else face = { ...candidate };
        lastSeen = now;
        return face;
      }
      return now - lastSeen <= maxLostMs ? { ...face, stale: true } : null;
    },
  };
}

export function faceRelativePoint(point, face) {
  return [(face.x - point.x) / face.scale, (face.y - point.y) * face.aspect / face.scale];
}

export function avatarWrist(point, face) {
  const [x, y] = faceRelativePoint(point, face);
  return { x: Math.max(-1.7, Math.min(1.7, x * 0.7)),
    y: Math.max(-1.25, Math.min(1.45, 0.82 + y * 0.7)), z: 0.65 };
}

export function poseArmTargets(pose, reference, hands = []) {
  if (!Array.isArray(pose) || pose.length < 17 || !reference) return null;
  const chains = [[11, 13, 15], [12, 14, 16]].map((indices) => {
    const points = indices.map((index) => pose[index]);
    if (points.some((point) => !point || (point.visibility ?? 1) < 0.5 || (point.presence ?? 1) < 0.5)) return null;
    return { shoulder: points[0], elbow: points[1], wrist: points[2] };
  }).filter(Boolean);
  if (chains.length !== 2) return null;
  // Avatar is displayed as a mirror, so the smaller avatar x is screen-left.
  // The camera is mirrored in the UI, so larger camera x is the avatar's screen-left arm.
  chains.sort((a, b) => b.shoulder.x - a.shoulder.x);

  const corrections = chains.flatMap((chain, index) => {
    const handWrist = hands[index]?.[0];
    return handWrist ? [{ x: handWrist.x - chain.wrist.x, y: handWrist.y - chain.wrist.y }] : [];
  });
  const correction = corrections.length
    ? corrections.reduce((total, item) => ({ x: total.x + item.x, y: total.y + item.y }), { x: 0, y: 0 })
    : { x: 0, y: 0 };
  correction.x /= Math.max(corrections.length, 1);
  correction.y /= Math.max(corrections.length, 1);

  const alignedPoint = (point) => ({
    ...point,
    x: Math.max(0, Math.min(1, point.x + correction.x)),
    y: Math.max(0, Math.min(1, point.y + correction.y)),
  });
  return chains.map((chain, index) => {
    const handWrist = hands[index]?.[0];
    const shoulder = alignedPoint(chain.shoulder);
    const anchor = { x: index === 0 ? -0.68 : 0.68, y: -0.12, z: -0.08 };
    const toTarget = (point, handAttached = false) => {
      const localX = (shoulder.x - point.x) / reference.scale;
      const localY = (shoulder.y - point.y) * reference.aspect / reference.scale;
      return {
        x: Math.max(-1.7, Math.min(1.7, anchor.x + localX * 0.9)),
        y: Math.max(-1.2, Math.min(1.3, anchor.y + localY * 0.9)),
        z: 0.55,
        source: point,
        handAttached,
      };
    };
    return {
      shoulder: { ...anchor, source: shoulder, handAttached: false },
      elbow: toTarget(alignedPoint(chain.elbow)),
      wrist: toTarget(handWrist || alignedPoint(chain.wrist), Boolean(handWrist)),
    };
  });
}
