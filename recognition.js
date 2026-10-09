export const STORAGE_KEY = "sign-translator-samples";
export const MAX_SAMPLES = 5000;

export function validateSamples(value) {
  const samples = Array.isArray(value) ? value : [1, 2].includes(value?.version) ? value.samples : null;
  if (!Array.isArray(samples) || samples.length > MAX_SAMPLES) throw new Error("샘플 파일 형식 또는 개수를 확인해 주세요.");
  return samples.map((sample) => {
    if (typeof sample?.label !== "string" || !sample.label.trim() || sample.label.trim().length > 40 ||
      !Array.isArray(sample.vector) || ![126, 130].includes(sample.vector.length) ||
      !sample.vector.every((n) => typeof n === "number" && Number.isFinite(n) && Math.abs(n) <= 100)) {
      throw new Error("유효하지 않은 샘플이 있습니다. 기존 데이터는 유지됩니다.");
    }
    return { label: sample.label.trim(), vector: [...sample.vector] };
  });
}

export function landmarksToVector(hands, face = null) {
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
  if (a.length === 130) {
    for (let i = 0; i < 4; i++) {
      direct += 0.2 * (a[126 + i] - b[126 + i]) ** 2;
      swapped += 0.2 * (a[126 + i] - b[126 + (i + 2) % 4]) ** 2;
    }
  }
  return Math.sqrt(Math.min(direct, swapped));
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
    scale: box.width / width, aspect: height / width, box };
}

export function faceRelativePoint(point, face) {
  return [(face.x - point.x) / face.scale, (face.y - point.y) * face.aspect / face.scale];
}

export function avatarWrist(point, face) {
  const [x, y] = faceRelativePoint(point, face);
  return { x: Math.max(-1.7, Math.min(1.7, x * 0.7)),
    y: Math.max(-1.25, Math.min(1.45, 0.82 + y * 0.7)), z: 0.65 };
}
