"""Render a synthetic tracked hand to verify native 3D fingers without a camera."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from desktop_avatar import AvatarView

app = QApplication([])
avatar = AvatarView()
avatar.resize(1000, 440)
data = {"left": ((.35,.4,0),(.25,.55,-.1),(.3,.3,-.2)),
        "right": ((.65,.4,0),(.75,.55,-.1),(.7,.3,-.2)), "aspect": 9/16, "hands": {}}
for side, x in (("left", .3), ("right", .7)):
    points = [(x,.3,0)]*21
    for indices, dx, length in (((1,2,3,4),-.06,.065),((5,6,7,8),-.03,.14),
                               ((9,10,11,12),0,.16),((13,14,15,16),.03,.14),((17,18,19,20),.06,.11)):
        for i,index in enumerate(indices):
            points[index]=(x+dx,.27-length*i/3,-.005*i)
    data["hands"][side]=points
avatar.set_joints(data)
assert len(avatar.rigs["left"]["joints"]) == 21
assert len(avatar.rigs["right"]["fingers"]) == 16
before = avatar.rigs["left"]["joints"][8].translation()
points = data["hands"]["left"]
old_tip = points[8]
points[8] = (old_tip[0], old_tip[1]+.08, old_tip[2]-.03)
avatar.set_joints(data)
assert (avatar.rigs["left"]["joints"][8].translation()-before).length() > .05
points[8] = old_tip
avatar.set_joints(data)
avatar.show()
def capture():
    screen = avatar.view.screen()
    image = screen.grabWindow(avatar.view.winId())
    assert image.save(str(Path(__file__).resolve().parents[1]/"avatar-preview.png"))
    print("3D avatar preview saved", flush=True)
    app.quit()
QTimer.singleShot(4000, capture)
sys.exit(app.exec())
