"""Native Windows prototype for webcam-based sign-language landmark tracking."""

from __future__ import annotations

import sys
import time

import cv2
import mediapipe as mp
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CameraWorker(QObject):
    frame_ready = Signal(QImage)
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

        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        drawing = mp.solutions.drawing_utils
        styles = mp.solutions.drawing_styles
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
                rgb.flags.writeable = True

                if result.pose_landmarks:
                    drawing.draw_landmarks(rgb, result.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
                                           landmark_drawing_spec=styles.get_default_pose_landmarks_style())
                for hand in (result.left_hand_landmarks, result.right_hand_landmarks):
                    if hand:
                        drawing.draw_landmarks(rgb, hand, mp.solutions.hands.HAND_CONNECTIONS,
                                               styles.get_default_hand_landmarks_style(), styles.get_default_hand_connections_style())
                if result.face_landmarks:
                    mouth = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405),
                             (405, 321), (321, 375), (375, 291), (291, 61)]
                    height, width = rgb.shape[:2]
                    for start, end in mouth:
                        a, b = result.face_landmarks.landmark[start], result.face_landmarks.landmark[end]
                        cv2.line(rgb, (int(a.x * width), int(a.y * height)), (int(b.x * width), int(b.y * height)), (255, 155, 85), 2)

                frames += 1
                now = time.monotonic()
                if now - last_fps_at >= 1:
                    self.status_changed.emit(f"추적 중 · {frames} FPS · Heavy 포즈")
                    frames, last_fps_at = 0, now
                height, width, channels = rgb.shape
                image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(image)
        except Exception as error:  # Keep a model failure inside the native app, not the camera driver.
            self.failed.emit(f"추적 오류: {error}")
        finally:
            holistic.close()
            capture.release()
            self._running = False
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._running = False


class SignLanguageWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("수어 인식 실험실 - 데스크톱")
        self.resize(1180, 760)
        self.thread: QThread | None = None
        self.worker: CameraWorker | None = None

        root = QWidget()
        root.setStyleSheet("background:#101214;color:#f5f3ee;font-family:'Segoe UI';")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(22)

        self.video = QLabel("카메라 대기 중")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(720, 405)
        self.video.setStyleSheet("background:#050607;border:1px solid #303538;border-radius:8px;color:#7f9098;font-size:18px;")
        layout.addWidget(self.video, 1)

        side = QVBoxLayout()
        side.setSpacing(14)
        title = QLabel("수어 인식 실험실")
        title.setStyleSheet("font-size:25px;font-weight:700;")
        side.addWidget(title)
        note = QLabel("네이티브 카메라 · 기기 내 처리\nHeavy 포즈 + 손 + 입 윤곽")
        note.setStyleSheet("color:#92d0b0;line-height:1.5;")
        side.addWidget(note)

        self.status = QLabel("준비")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("padding:14px;background:#171c1f;border:1px solid #303538;border-radius:8px;color:#c8c5bd;")
        side.addWidget(self.status)
        self.start_button = QPushButton("카메라 시작")
        self.start_button.clicked.connect(self.toggle_camera)
        self.start_button.setStyleSheet("padding:12px;background:#f5f3ee;color:#101214;border-radius:8px;font-weight:700;")
        side.addWidget(self.start_button)

        side.addWidget(QLabel("수어 이름"))
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("예: 안녕하세요")
        self.label_input.setStyleSheet("padding:10px;background:#090b0c;border:1px solid #303538;border-radius:8px;")
        side.addWidget(self.label_input)
        self.sample_note = QLabel("다음 단계에서 이 창에 수어 샘플 저장·문장 조합을 연결합니다.")
        self.sample_note.setWordWrap(True)
        self.sample_note.setStyleSheet("color:#9ea9ae;line-height:1.5;")
        side.addWidget(self.sample_note)
        side.addStretch(1)
        layout.addLayout(side)

    @Slot()
    def toggle_camera(self) -> None:
        if self.worker:
            self.status.setText("카메라 중지 중")
            self.worker.stop()
            return
        self.thread = QThread(self)
        self.worker = CameraWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.status_changed.connect(self.status.setText)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.cleanup_camera)
        self.thread.start()
        self.start_button.setText("카메라 중지")

    @Slot(QImage)
    def show_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image).scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video.setPixmap(pixmap)

    @Slot(str)
    def show_error(self, message: str) -> None:
        self.status.setText(message)

    @Slot()
    def cleanup_camera(self) -> None:
        if self.thread:
            self.thread.quit()
            self.thread.wait(1500)
            self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self.start_button.setText("카메라 시작")
        if not self.status.text().startswith("추적 오류") and "열 수 없습니다" not in self.status.text():
            self.status.setText("중지")
        self.video.setPixmap(QPixmap())
        self.video.setText("카메라 대기 중")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self.worker:
            self.worker.stop()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = SignLanguageWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
