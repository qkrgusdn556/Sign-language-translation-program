"""Exercise collection/progress UI without opening a real webcam or modifying user samples."""
import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# UI simulation does not use the actual vision model.
sys.modules["mediapipe"] = types.SimpleNamespace()
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import desktop_app as desktop

app = QApplication([])
with patch.object(desktop, "load_samples", return_value=[]):
    window = desktop.SignLanguageWindow()
window.label_input.setText("테스트 동작")
window.worker = types.SimpleNamespace(stop=lambda: None)
vector = [.1]*63+[0.0]*63+[0.0]*10
window.last_vector = vector
window.start_recording()
started = window.recording.started_at
for index in range(27):
    window.recording.append(vector, started+index*.07)
with patch.object(desktop.time, "monotonic", return_value=started+1):
    window.update_capture_progress()
assert window.capture_bar.value() == 500
assert not window.sample_button.isEnabled()
with patch.object(desktop.time, "monotonic", return_value=started+2.01), patch.object(desktop, "save_samples") as save:
    window.update_capture_progress()
    assert save.call_count == 1
assert len(window.samples) == 1 and "sequence" in window.samples[0]
assert window.capture_bar.value() == 1000
assert window.record_button.isEnabled()
window.start_recording()
window.cancel_recording()
assert window.recording is None and window.capture_bar.value() == 0
assert len(window.samples) == 1
window.samples = [{"label": "테스트", "vector": vector} for _ in range(3)]
window.update_tracking(vector, {})
assert window.similarity_bar.value() == 1000
window.samples = []
window.similarity_bar.setValue(850)
window.confidence.setText("유사도 85%")
window.capture_bar.setValue(500)
window.capture_label.setText("1.0 / 2.0초")
window.resize(1280, 960)
window.show()
def capture():
    path = Path(__file__).resolve().parents[1]/"capture-ui-preview.png"
    assert window.grab().save(str(path))
    window.worker = None
    window.close()
    print("Collection, cancellation, similarity bar and UI checks passed", flush=True)
    app.quit()
QTimer.singleShot(2000, capture)
sys.exit(app.exec())
