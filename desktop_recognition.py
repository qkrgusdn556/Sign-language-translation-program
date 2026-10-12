"""Local sample storage and lightweight sign classification for the desktop app."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

VECTOR_SIZE = 136
MIN_SAMPLES_PER_LABEL = 3
MIN_SEQUENCE_FRAMES = 6
MAX_SEQUENCE_FRAMES = 32


def storage_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "SignLanguageDesktop"
    base.mkdir(parents=True, exist_ok=True)
    return base / "samples.json"


def load_samples() -> list[dict]:
    try:
        raw = json.loads(storage_path().read_text(encoding="utf-8"))
        samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
        return [sample for sample in samples if valid_sample(sample)]
    except (OSError, json.JSONDecodeError):
        return []


def save_samples(samples: list[dict]) -> None:
    storage_path().write_text(json.dumps({"version": 3, "samples": samples}, ensure_ascii=False), encoding="utf-8")


def valid_vector(vector) -> bool:
    return isinstance(vector, list) and len(vector) in (126, 130, 136) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and abs(value) <= 100 for value in vector)


def valid_sample(sample: object) -> bool:
    if not isinstance(sample, dict) or not isinstance(sample.get("label"), str) or not sample["label"].strip():
        return False
    if "vector" in sample:
        return valid_vector(sample["vector"])
    sequence = sample.get("sequence")
    return isinstance(sequence, list) and MIN_SEQUENCE_FRAMES <= len(sequence) <= MAX_SEQUENCE_FRAMES and all(
        valid_vector(frame) and len(frame) == len(sequence[0]) for frame in sequence)


def landmark_vector(left_hand, right_hand, pose=None, face=None) -> list[float] | None:
    """Match the web app's hand-first, shoulder/mouth-enhanced vector shape."""
    hands = [left_hand, right_hand]
    if not any(hands):
        return None
    vector: list[float] = []
    for hand in hands:
        if not hand:
            vector.extend([0.0] * 63)
            continue
        wrist, base = hand[0], hand[9]
        scale = math.sqrt((base.x - wrist.x) ** 2 + (base.y - wrist.y) ** 2 + (base.z - wrist.z) ** 2)
        if scale < 0.0001:
            vector.extend([0.0] * 63)
            continue
        for point in hand:
            vector.extend(((point.x - wrist.x) / scale, (point.y - wrist.y) / scale, (point.z - wrist.z) / scale))
    if pose and len(pose) > 12:
        left_shoulder, right_shoulder = pose[11], pose[12]
        shoulder_scale = math.hypot(left_shoulder.x - right_shoulder.x, left_shoulder.y - right_shoulder.y)
        center_x, center_y = (left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2
        if shoulder_scale > 0.0001:
            for hand in hands:
                vector.extend(((hand[0].x - center_x) / shoulder_scale, (hand[0].y - center_y) / shoulder_scale) if hand else (0.0, 0.0))
        else:
            vector.extend([0.0] * 4)
    else:
        vector.extend([0.0] * 4)
    vector.extend(mouth_features(face))
    return vector if len(vector) == VECTOR_SIZE and all(math.isfinite(value) for value in vector) else None


def mouth_features(face) -> list[float]:
    if not face or len(face) <= 405:
        return [0.0] * 6
    point = lambda index: face[index]
    width = max(math.hypot(point(61).x - point(291).x, point(61).y - point(291).y), 0.0001)
    height = math.hypot(point(13).x - point(14).x, point(13).y - point(14).y) / width
    inner_width = math.hypot(point(78).x - point(308).x, point(78).y - point(308).y) / width
    upper = math.hypot(point(0).x - point(17).x, point(0).y - point(17).y) / width
    return [min(height * 4, 1), min(inner_width, 1), min(upper * 4, 1), 0.0, 0.0, 0.0]


def hand_count(vector: list[float]) -> int:
    return int(any(vector[:63])) + int(any(vector[63:126]))


def distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or hand_count(a) != hand_count(b):
        return math.inf
    direct = sum((a[index] - b[index]) ** 2 for index in range(126))
    swapped = sum((a[index] - b[(index + 63) % 126]) ** 2 for index in range(126))
    scale = 63 * max(1, hand_count(a))
    direct, swapped = direct / scale, swapped / scale
    if len(a) >= 130:
        direct += sum(0.2 * (a[126 + index] - b[126 + index]) ** 2 for index in range(4))
        swapped += sum(0.2 * (a[126 + index] - b[126 + (index + 2) % 4]) ** 2 for index in range(4))
    if len(a) >= 136:
        direct += sum(0.15 * (a[index] - b[index]) ** 2 for index in range(130, 136))
        swapped += sum(0.15 * (a[index] - b[index]) ** 2 for index in range(130, 136))
    return math.sqrt(min(direct, swapped))


def classify(vector: list[float] | None, samples: list[dict]) -> dict:
    if not vector:
        return {"status": "waiting", "similarity": 0.0}
    compatible = [sample for sample in samples if "vector" in sample and len(sample["vector"]) == len(vector)]
    counts: dict[str, int] = {}
    for sample in compatible:
        counts[sample["label"]] = counts.get(sample["label"], 0) + 1
    eligible = [sample for sample in compatible if counts[sample["label"]] >= MIN_SAMPLES_PER_LABEL]
    if not eligible:
        return {"status": "untrained", "similarity": 0.0}
    groups: dict[str, list[float]] = {}
    for sample in eligible:
        value = distance(vector, sample["vector"])
        if math.isfinite(value):
            groups.setdefault(sample["label"], []).append(value)
    if not groups:
        return {"status": "unknown", "similarity": 0.0}
    ranked = []
    for label, values in groups.items():
        values.sort()
        ranked.append((sum(values[:3]) / min(len(values), 3), label))
    ranked.sort()
    best_distance, label = ranked[0]
    similarity = math.exp(-best_distance / 0.2)
    ambiguous = len(ranked) > 1 and ranked[1][0] - best_distance < 0.055
    if best_distance > 0.3 or ambiguous:
        return {"status": "unknown", "similarity": similarity}
    return {"status": "match", "label": label, "similarity": similarity}


def dtw_distance(a, b):
    if not a or not b:
        return math.inf
    previous = [0.0] + [math.inf]*len(b)
    for frame in a:
        row = [math.inf]
        for index, other in enumerate(b):
            row.append(distance(frame, other) + min(previous[index], previous[index+1], row[-1]))
        previous = row
    return previous[-1]/(len(a)+len(b))


def classify_sequence(sequence, samples):
    if len(sequence) < MIN_SEQUENCE_FRAMES:
        return {"status": "waiting", "similarity": 0.0}
    eligible = [s for s in samples if "sequence" in s and len(s["sequence"][0]) == len(sequence[0])]
    counts = {}
    for sample in eligible:
        counts[sample["label"]] = counts.get(sample["label"], 0)+1
    trained = [s for s in eligible if counts[s["label"]] >= MIN_SAMPLES_PER_LABEL]
    if not trained:
        return {"status": "untrained", "similarity": 0.0}
    groups = {}
    for sample in trained:
        value = dtw_distance(sequence, sample["sequence"])
        if math.isfinite(value):
            groups[sample["label"]] = min(groups.get(sample["label"], math.inf), value)
    if not groups:
        return {"status": "unknown", "similarity": 0.0}
    ranked = sorted((value,label) for label,value in groups.items())
    value, label = ranked[0]
    similarity = math.exp(-value/.24)
    ambiguous = len(ranked) > 1 and ranked[1][0]-value < .04
    return {"status": "unknown" if value > .35 or ambiguous else "match", "label": label, "similarity": similarity}


class MotionCapture:
    """Bounded two-second collection, using actual frame times instead of UI arrival times."""
    def __init__(self, label, started_at):
        self.label, self.started_at = label, started_at
        self.frames = []
        self.last_at = -math.inf
        self.first_at = None

    def append(self, vector, timestamp):
        if valid_vector(vector) and self.started_at <= timestamp <= self.started_at+2 and timestamp-self.last_at >= 1/16:
            if len(self.frames) < MAX_SEQUENCE_FRAMES:
                self.frames.append(vector[:])
                self.last_at = timestamp
                if self.first_at is None:
                    self.first_at = timestamp

    def sample(self):
        covered = self.first_at is not None and self.last_at-self.first_at >= 1
        return {"label": self.label, "sequence": self.frames} if len(self.frames) >= MIN_SEQUENCE_FRAMES and covered else None


def polish(words: list[str], polite: bool) -> str:
    if not words:
        return "아직 인식된 단어가 없습니다."
    text = " ".join(words)
    replacements = ({"나": "저", "고마워": "감사합니다", "안녕": "안녕하세요", "미안": "죄송합니다"} if polite
                    else {"저": "나", "감사합니다": "고마워", "안녕하세요": "안녕", "죄송합니다": "미안"})
    for before, after in replacements.items():
        text = text.replace(before, after)
    return text + "요" if polite and not text.endswith(("요", "니다", ".", "!", "?")) else text
