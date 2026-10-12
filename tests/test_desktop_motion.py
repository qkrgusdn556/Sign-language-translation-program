import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import desktop_recognition as recognition


def frame(position=0):
    return [.1]*63+[0.0]*63+[position, 0.0, 0.0, 0.0]+[0.0]*6


class MotionTests(unittest.TestCase):
    def test_old_static_and_new_motion_samples_round_trip(self):
        samples = [{"label": "static", "vector": frame()},
                   {"label": "motion", "sequence": [frame(i/10) for i in range(16)]}]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, patch.object(recognition, "storage_path", return_value=Path(directory)/"samples.json"):
            recognition.save_samples(samples)
            self.assertEqual(recognition.load_samples(), samples)
        self.assertTrue(recognition.valid_sample({"label": "old", "vector": frame()[:126]}))
        self.assertFalse(recognition.valid_sample({"label": "bad", "sequence": [frame()]*5}))

    def test_dtw_tolerates_speed_change_and_rejects_reverse_order(self):
        motion = [frame(i/10) for i in range(20)]
        slowed = [point for point in motion for _ in range(2)]
        self.assertEqual(recognition.dtw_distance(motion, slowed), 0)
        self.assertGreater(recognition.dtw_distance(motion, list(reversed(motion))), .2)
        samples = [{"label": "wave", "sequence": motion} for _ in range(3)]
        result = recognition.classify_sequence(slowed, samples)
        self.assertEqual(result["status"], "match")
        self.assertEqual(result["label"], "wave")
        self.assertEqual(recognition.classify_sequence(motion, samples[:2])["status"], "untrained")

    def test_mixed_samples_do_not_break_static_classification(self):
        samples = [{"label": "still", "vector": frame()} for _ in range(3)]
        samples.append({"label": "move", "sequence": [frame()]*10})
        self.assertEqual(recognition.classify(frame(), samples)["label"], "still")

    def test_capture_uses_two_second_window_and_bounded_frames(self):
        capture = recognition.MotionCapture("wave", 10)
        for i in range(101):
            capture.append(frame(i/100), 10+i*.025)
        self.assertLessEqual(len(capture.frames), 32)
        self.assertGreaterEqual(len(capture.frames), 6)
        self.assertLessEqual(capture.last_at, 12)
        self.assertIsNotNone(capture.sample())
        self.assertTrue(recognition.valid_sample(capture.sample()))

    def test_short_tracking_and_invalid_frames_are_not_saved(self):
        capture = recognition.MotionCapture("wave", 10)
        for i in range(7):
            capture.append(frame(), 10+i*.075)
        capture.append([math.nan]*136, 11)
        self.assertIsNone(capture.sample())


if __name__ == "__main__":
    unittest.main()
