"""Native Windows sign-language prototype: camera, local recognition, sentence output and avatar."""

from __future__ import annotations

import sys
import time
import traceback
import os
from collections import deque
from pathlib import Path

# PyInstaller keeps Qt DLLs in _internal\PySide6. Put that directory first before
# importing QtCore so another bundled library cannot satisfy the wrong Qt symbol.
if getattr(sys, "frozen", False):
    qt_directory = Path(sys._MEIPASS) / "PySide6"
    os.environ["PATH"] = f"{qt_directory}{os.pathsep}{os.environ.get('PATH', '')}"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(qt_directory)

try:
    import cv2
    import mediapipe as mp
    from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
    from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (QApplication, QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel,
                                   QLineEdit, QMainWindow, QPushButton, QProgressBar, QRadioButton, QVBoxLayout, QWidget)
except Exception:
    Path.cwd().joinpath("desktop-error.log").write_text(traceback.format_exc(), encoding="utf-8")
    raise

from desktop_recognition import (MIN_SAMPLES_PER_LABEL, MotionCapture, classify, classify_sequence,
                                 landmark_vector, load_samples, polish, save_samples)
from desktop_avatar import AvatarView

MOUTH_EDGES = ((61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291), (291, 61))


class CameraWorker(QObject):
    frame_ready = Signal(QImage)
    tracking_ready = Signal(object, object)
    status_changed = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._running = False

    @Slot()
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.status_changed.emit("카메라 연결 중")
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        capture.set(cv2.CAP_PROP_FPS, 30)
        if not capture.isOpened():
            self.failed.emit("카메라를 열 수 없습니다. 다른 프로그램에서 사용 중인지 확인해 주세요.")
            self._running = False
            self.finished.emit()
            return
        holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=2, smooth_landmarks=True,
            refine_face_landmarks=True, min_detection_confidence=0.35, min_tracking_confidence=0.35)
        drawing = mp.solutions.drawing_utils
        point_spec = drawing.DrawingSpec(color=(105, 223, 179), thickness=2, circle_radius=2)
        line_spec = drawing.DrawingSpec(color=(189, 169, 255), thickness=2, circle_radius=1)
        hand_spec = drawing.DrawingSpec(color=(245, 206, 145), thickness=2, circle_radius=3)
        last_fps_at, frames = time.monotonic(), 0
        try:
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    self.failed.emit("카메라 프레임을 읽지 못했습니다.")
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = holistic.process(rgb)
                captured_at = time.monotonic()
                rgb.flags.writeable = True
                pose = result.pose_landmarks.landmark if result.pose_landmarks else None
                left = result.left_hand_landmarks.landmark if result.left_hand_landmarks else None
                right = result.right_hand_landmarks.landmark if result.right_hand_landmarks else None
                face = result.face_landmarks.landmark if result.face_landmarks else None
                vector = landmark_vector(left, right, pose, face)
                if result.pose_landmarks:
                    drawing.draw_landmarks(rgb, result.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS, point_spec, line_spec)
                for hand in (result.left_hand_landmarks, result.right_hand_landmarks):
                    if hand:
                        drawing.draw_landmarks(rgb, hand, mp.solutions.hands.HAND_CONNECTIONS, hand_spec, point_spec)
                if face:
                    height, width = rgb.shape[:2]
                    for start, end in MOUTH_EDGES:
                        a, b = face[start], face[end]
                        cv2.line(rgb, (int(a.x * width), int(a.y * height)), (int(b.x * width), int(b.y * height)), (255, 155, 85), 2)
                joints = self._joints(pose, left, right)
                joints["aspect"] = rgb.shape[0] / rgb.shape[1]
                joints["captured_at"] = captured_at
                self.tracking_ready.emit(vector, joints)
                frames += 1
                now = time.monotonic()
                if now - last_fps_at >= 1:
                    self.status_changed.emit(f"추적 중 · {frames} FPS · Heavy 포즈 · 기기 내 처리")
                    frames, last_fps_at = 0, now
                height, width, channels = rgb.shape
                self.frame_ready.emit(QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy())
        except Exception as error:
            self.failed.emit(f"추적 오류: {error}")
        finally:
            holistic.close()
            capture.release()
            self._running = False
            self.finished.emit()

    @staticmethod
    def _joints(pose, left, right) -> dict:
        point = lambda item: (float(item.x), float(item.y), float(item.z))
        data = {"hands": {side: [point(p) for p in hand] for side, hand in (("left", left), ("right", right)) if hand}}
        if pose and len(pose) >= 17:
            for side, indices, hand in (("left", (11,13,15), left), ("right", (12,14,16), right)):
                if all(pose[i].visibility >= .35 for i in indices[:2]):
                    data[side] = (point(pose[indices[0]]), point(pose[indices[1]]), point(hand[0] if hand else pose[indices[2]]))
        return data

    @Slot()
    def stop(self) -> None:
        self._running = False


class SignLanguageWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("수어 인식 실험실 - 데스크톱")
        self.resize(1280, 900)
        self.thread: QThread | None = None
        self.worker: CameraWorker | None = None
        self.samples = load_samples()
        self.last_vector = None
        self.candidate, self.candidate_at = "", 0.0
        self.sentence_words: list[str] = []
        self.sentence_armed = True
        self.recording = None
        self.motion_frames = deque(maxlen=32)
        self.motion_result = {"status": "waiting", "similarity": 0.0}
        self.last_motion_check = 0.0
        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(30)
        self.capture_timer.timeout.connect(self.update_capture_progress)
        root = QWidget(); root.setStyleSheet("background:#101214;color:#f5f3ee;font-family:'Segoe UI';")
        self.setCentralWidget(root)
        page = QVBoxLayout(root); page.setContentsMargins(24, 20, 24, 24); page.setSpacing(16)
        header = QHBoxLayout(); title = QLabel("수어 인식 실험실"); title.setStyleSheet("font-size:25px;font-weight:700;")
        header.addWidget(title); header.addStretch(); self.status = QLabel("준비 · 기기 내 처리")
        self.status.setStyleSheet("color:#92d0b0;"); header.addWidget(self.status); page.addLayout(header)
        top = QHBoxLayout(); top.setSpacing(18)
        self.video = QLabel("카메라 대기 중"); self.video.setAlignment(Qt.AlignmentFlag.AlignCenter); self.video.setMinimumSize(720, 405)
        self.video.setStyleSheet("background:#050607;border:1px solid #303538;border-radius:8px;color:#7f9098;font-size:18px;"); top.addWidget(self.video, 2)
        panel = QFrame(); panel.setStyleSheet("QFrame{background:#15191b;border:1px solid #303538;border-radius:8px;} QLabel{border:0;}")
        side = QVBoxLayout(panel); side.setContentsMargins(18, 18, 18, 18); side.setSpacing(10)
        hint = QLabel("손·팔·입모양 인식"); hint.setStyleSheet("font-weight:700;color:#7fd8ba;"); side.addWidget(hint)
        self.prediction = QLabel("손 대기"); self.prediction.setStyleSheet("font-size:28px;font-weight:700;"); side.addWidget(self.prediction)
        self.similarity_bar = self._progress_bar("#7fd8ba")
        self.similarity_bar.setToolTip("저장 샘플과의 유사도")
        side.addWidget(self.similarity_bar)
        self.confidence = QLabel("유사도 0%"); self.confidence.setStyleSheet("color:#c8c5bd;"); side.addWidget(self.confidence)
        self.sample_state = QLabel(); self.sample_state.setWordWrap(True); self.sample_state.setStyleSheet("color:#9ea9ae;"); side.addWidget(self.sample_state)
        self.start_button = QPushButton("카메라 시작"); self.start_button.clicked.connect(self.toggle_camera); side.addWidget(self.start_button)
        side.addWidget(QLabel("수어 이름")); self.label_input = QLineEdit(); self.label_input.setPlaceholderText("예: 안녕하세요"); side.addWidget(self.label_input)
        self.sample_button = QPushButton("현재 자세 샘플 저장"); self.sample_button.clicked.connect(self.save_current_sample); side.addWidget(self.sample_button)
        self.record_button = QPushButton("2초 동작 수집")
        self.record_button.clicked.connect(self.start_recording)
        side.addWidget(self.record_button)
        self.capture_bar = self._progress_bar("#f0bb76")
        self.capture_bar.setToolTip("2초 동작 수집 진행")
        side.addWidget(self.capture_bar)
        self.capture_label = QLabel("0.0 / 2.0초")
        self.capture_label.setStyleSheet("color:#9ea9ae;font-size:11px;")
        side.addWidget(self.capture_label)
        self.clear_samples = QPushButton("저장 샘플 비우기"); self.clear_samples.clicked.connect(self.clear_all_samples); side.addWidget(self.clear_samples)
        side.addStretch(); top.addWidget(panel, 1); page.addLayout(top)
        sentence_panel = QFrame(); sentence_panel.setStyleSheet("QFrame{background:#15191b;border:1px solid #303538;border-radius:8px;} QLabel{border:0;}")
        sentence = QGridLayout(sentence_panel); sentence.setContentsMargins(18, 14, 18, 14)
        sentence.addWidget(QLabel("인식 문장"), 0, 0); self.sentence_label = QLabel("단어를 인식하면 여기에 연결됩니다."); self.sentence_label.setWordWrap(True); self.sentence_label.setStyleSheet("font-size:18px;font-weight:600;"); sentence.addWidget(self.sentence_label, 1, 0, 1, 4)
        self.polished = QLabel("문장 다듬기 대기"); self.polished.setWordWrap(True); self.polished.setStyleSheet("color:#92d0b0;"); sentence.addWidget(self.polished, 2, 0, 1, 4)
        self.polite = QRadioButton("존댓말"); self.casual = QRadioButton("반말"); self.polite.setChecked(True); styles = QButtonGroup(self); styles.addButton(self.polite); styles.addButton(self.casual)
        sentence.addWidget(self.polite, 3, 0); sentence.addWidget(self.casual, 3, 1)
        polish_button = QPushButton("문장 다듬기"); polish_button.clicked.connect(self.polish_sentence); clear_sentence = QPushButton("문장 지우기"); clear_sentence.clicked.connect(self.clear_sentence)
        sentence.addWidget(polish_button, 3, 2); sentence.addWidget(clear_sentence, 3, 3); page.addWidget(sentence_panel)
        page.addWidget(QLabel("실시간 아바타")); self.avatar = AvatarView(); page.addWidget(self.avatar, 1)
        self._update_sample_state()

    @Slot()
    def toggle_camera(self) -> None:
        if self.worker:
            self.cancel_recording()
            self.last_vector = None
            self.motion_frames.clear()
            self.status.setText("카메라 중지 중"); self.worker.stop(); return
        self.thread = QThread(self); self.worker = CameraWorker(); self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start); self.worker.frame_ready.connect(self.show_frame)
        self.worker.tracking_ready.connect(self.update_tracking); self.worker.status_changed.connect(self.update_camera_status)
        self.worker.failed.connect(self.show_error); self.worker.finished.connect(self.cleanup_camera); self.thread.start()
        self.start_button.setText("카메라 중지")

    @Slot(QImage)
    def show_frame(self, image: QImage) -> None:
        self.video.setPixmap(QPixmap.fromImage(image).scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    @Slot(object, object)
    def update_tracking(self, vector, joints) -> None:
        self.last_vector = vector; self.avatar.set_joints(joints)
        now = joints.get("captured_at", time.monotonic())
        if self.recording:
            self.recording.append(vector, now)
            return
        if not vector:
            self.motion_frames.clear()
            self.motion_result = {"status": "waiting", "similarity": 0.0}
        elif not self.motion_frames or now-self.motion_frames[-1][0] >= 1/16:
            self.motion_frames.append((now, vector[:]))
        while self.motion_frames and now-self.motion_frames[0][0] > 2:
            self.motion_frames.popleft()
        result = classify(vector, self.samples)
        if vector and now-self.last_motion_check >= .3:
            self.last_motion_check = now
            frames = [frame for _,frame in self.motion_frames]
            duration = self.motion_frames[-1][0]-self.motion_frames[0][0] if self.motion_frames else 0
            self.motion_result = classify_sequence(frames if duration >= 1.5 else [], self.samples)
        if vector and self.motion_result["status"] == "match" and self.motion_result["similarity"] > result["similarity"]:
            result = self.motion_result
        percentage = int(result["similarity"]*100)
        self.confidence.setText(f"유사도 {percentage}%")
        self.similarity_bar.setValue(percentage*10)
        status = result["status"]
        if status == "match":
            label, now = result["label"], time.monotonic(); self.prediction.setText(label)
            if self.candidate != label: self.candidate, self.candidate_at, self.sentence_armed = label, now, True
            elif self.sentence_armed and now - self.candidate_at >= 0.35:
                self.sentence_words.append(label); self.sentence_armed = False; self._render_sentence()
        elif status == "untrained": self.prediction.setText("학습 필요")
        elif status == "waiting":
            self.prediction.setText("손 대기"); self.candidate = ""; self.sentence_armed = True
        else: self.prediction.setText("알 수 없음"); self.candidate = ""; self.sentence_armed = True

    @Slot()
    def save_current_sample(self) -> None:
        if self.recording:
            return
        label = self.label_input.text().strip()
        if not label: self.status.setText("저장할 수어 이름을 입력해 주세요."); return
        if not self.last_vector: self.status.setText("손을 카메라에 보여준 뒤 저장해 주세요."); return
        self.samples.append({"label": label, "vector": self.last_vector[:]}); save_samples(self.samples); self._update_sample_state(); self.status.setText(f"'{label}' 샘플 저장됨")

    @Slot()
    def clear_all_samples(self) -> None:
        self.cancel_recording()
        self.samples.clear(); save_samples(self.samples); self._update_sample_state(); self.status.setText("저장 샘플을 비웠습니다.")
        self.motion_result = {"status": "waiting", "similarity": 0.0}

    @staticmethod
    def _progress_bar(color):
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(6)
        bar.setStyleSheet(f"QProgressBar{{border:0;background:#303538;border-radius:3px;}} QProgressBar::chunk{{background:{color};border-radius:3px;}}")
        return bar

    @Slot()
    def start_recording(self):
        if self.recording:
            return
        label = self.label_input.text().strip()
        if not label:
            self.status.setText("저장할 수어 이름을 입력해 주세요."); return
        if not self.worker or not self.last_vector:
            self.status.setText("카메라를 시작하고 손을 보여 주세요."); return
        self.recording = MotionCapture(label, time.monotonic())
        self.capture_bar.setValue(0)
        self.capture_label.setText("0.0 / 2.0초")
        self.record_button.setEnabled(False)
        self.sample_button.setEnabled(False)
        self.label_input.setEnabled(False)
        self.status.setText(f"'{label}' 동작 수집 중")
        self.candidate = ""
        self.capture_timer.start()

    @Slot()
    def update_capture_progress(self):
        if not self.recording:
            return
        elapsed = min(2.0, time.monotonic()-self.recording.started_at)
        self.capture_bar.setValue(int(elapsed/2*1000))
        self.capture_label.setText(f"{elapsed:.1f} / 2.0초")
        if elapsed >= 2:
            sample = self.recording.sample()
            self.cancel_recording(reset_progress=False)
            if sample:
                self.samples.append(sample)
                try:
                    save_samples(self.samples)
                except OSError as error:
                    self.samples.pop()
                    self.status.setText(f"샘플 저장 실패: {error}")
                    return
                self._update_sample_state()
                self.status.setText(f"'{sample['label']}' 2초 동작 저장됨")
            else:
                self.status.setText("손을 충분히 추적하지 못했습니다. 다시 수집해 주세요.")

    def cancel_recording(self, reset_progress=True):
        self.capture_timer.stop()
        self.recording = None
        self.record_button.setEnabled(True)
        self.sample_button.setEnabled(True)
        self.label_input.setEnabled(True)
        self.motion_frames.clear()
        if reset_progress:
            self.capture_bar.setValue(0)
            self.capture_label.setText("0.0 / 2.0초")

    @Slot()
    def polish_sentence(self) -> None: self.polished.setText(polish(self.sentence_words, self.polite.isChecked()))

    @Slot()
    def clear_sentence(self) -> None: self.sentence_words.clear(); self._render_sentence()

    def _render_sentence(self) -> None:
        self.sentence_label.setText(" ".join(self.sentence_words) or "단어를 인식하면 여기에 연결됩니다."); self.polished.setText("문장 다듬기 대기")

    def _update_sample_state(self) -> None:
        labels = {sample["label"] for sample in self.samples}
        motion_count = sum("sequence" in sample for sample in self.samples)
        self.sample_state.setText(f"저장 샘플 {len(self.samples)}개 · 단어 {len(labels)}개\n정지 {len(self.samples)-motion_count}개 · 동작 {motion_count}개")

    @Slot(str)
    def show_error(self, message: str) -> None: self.status.setText(message)

    @Slot(str)
    def update_camera_status(self, message):
        if not self.recording:
            self.status.setText(message)

    @Slot()
    def cleanup_camera(self) -> None:
        self.cancel_recording()
        self.last_vector = None
        self.similarity_bar.setValue(0)
        self.confidence.setText("유사도 0%")
        if self.thread: self.thread.quit(); self.thread.wait(1500); self.thread.deleteLater()
        self.thread = None; self.worker = None; self.start_button.setText("카메라 시작"); self.video.setPixmap(QPixmap()); self.video.setText("카메라 대기 중")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.cancel_recording()
        if self.worker: self.worker.stop()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv); window = SignLanguageWindow(); window.show(); return app.exec()


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception:
        Path.cwd().joinpath("desktop-error.log").write_text(traceback.format_exc(), encoding="utf-8"); raise
