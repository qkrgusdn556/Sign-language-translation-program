export const STORAGE_KEY = "sign-translator-samples";
export const MAX_SAMPLES = 5000;

export function validateSamples(value) {
  const samples = Array.isArray(value) ? value : value?.version === 1 ? value.samples : null;
  if (!Array.isArray(samples) || samples.length > MAX_SAMPLES) throw new Error("샘플 파일 형식 또는 개수를 확인해 주세요.");
  return samples.map((sample) => {
    if (typeof sample?.label !== "string" || !sample.label.trim() || sample.label.trim().length > 40 ||
      !Array.isArray(sample.vector) || sample.vector.length !== 126 ||
      !sample.vector.every((n) => typeof n === "number" && Number.isFinite(n) && Math.abs(n) <= 100)) {
      throw new Error("유효하지 않은 샘플이 있습니다. 기존 데이터는 유지됩니다.");
    }
    return { label: sample.label.trim(), vector: [...sample.vector] };
  });
}

export function landmarksToVector(hands) {
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
  return vector;
}

function handCount(vector) {
  return Number(vector.slice(0, 63).some((n) => n !== 0)) + Number(vector.slice(63).some((n) => n !== 0));
}

export function distance(a, b) {
  if (handCount(a) !== handCount(b)) return Infinity;
  let direct = 0;
  let swapped = 0;
  for (let i = 0; i < 126; i++) {
    direct += (a[i] - b[i]) ** 2;
    swapped += (a[i] - b[(i + 63) % 126]) ** 2;
  }
  // Keep old 126-value samples compatible when the detector changes hand order.
  return Math.sqrt(Math.min(direct, swapped) / (63 * Math.max(1, handCount(a))));
}

export function classify(vector, samples) {
  const counts = new Map();
  for (const { label } of samples) counts.set(label, (counts.get(label) || 0) + 1);
  const eligible = samples.filter((sample) => counts.get(sample.label) >= 3);
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
    // Unmirrored camera input is displayed as a mirror: Right goes to screen-left.
    const slot = category?.categoryName === "Right" ? 0 : category?.categoryName === "Left" ? 1 : -1;
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
