"""Native Qt3D humanoid driven by shoulder-relative pose and hand landmarks."""

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QQuaternion, QVector3D
from PySide6.QtWidgets import QVBoxLayout, QWidget
from PySide6.Qt3DCore import Qt3DCore as Core
from PySide6.Qt3DExtras import Qt3DExtras as Extras
from PySide6.Qt3DRender import Qt3DRender as Render

FINGERS = ((0, 1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
           (13, 14, 15, 16), (17, 18, 19, 20))


class AvatarView(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(280)
        self.view = Extras.Qt3DWindow()
        self.view.defaultFrameGraph().setClearColor(QColor("#171d20"))
        self.root = Core.QEntity()
        self.transforms = []
        self.view.setRootEntity(self.root)
        container = QWidget.createWindowContainer(self.view, self)
        container.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(container)
        camera = self.view.camera()
        camera.lens().setPerspectiveProjection(38, 3, 0.1, 100)
        camera.setPosition(QVector3D(0, 0.2, 5.4))
        camera.setViewCenter(QVector3D(0, 0, 0))
        self.view.widthChanged.connect(self._aspect)
        self.view.heightChanged.connect(self._aspect)
        for position, intensity in (((-3, 4, 5), 1.1), ((3, 2, 3), 0.65)):
            entity = Core.QEntity(self.root)
            light = Render.QPointLight(entity)
            light.setColor(QColor("#ffffff"))
            light.setIntensity(intensity)
            transform = Core.QTransform(entity)
            transform.setTranslation(QVector3D(*position))
            entity.addComponent(light)
            entity.addComponent(transform)
        self.skin = self._material("#d7a98c")
        self.shirt = self._material("#48778c")
        self.hair = self._material("#282524")
        self.white = self._material("#f7f5f0")
        self.lip = self._material("#995e58")
        self._oval(self.shirt, (0, -.63, -.12), (.57, .68, .26))
        self._oval(self.shirt, (0, -.12, -.1), (.76, .24, .28))
        self._oval(self.skin, (0, .25, 0), (.13, .23, .13))
        self._oval(self.skin, (0, .69, 0), (.28, .36, .25))
        self._oval(self.hair, (0, .91, -.055), (.295, .19, .24))
        self._oval(self.skin, (0, .65, .26), (.04, .07, .05))
        self._oval(self.lip, (0, .54, .231), (.083, .013, .012))
        for side in (-1, 1):
            self._oval(self.skin, (side*.28, .67, 0), (.04, .075, .035))
            self._oval(self.white, (side*.103, .72, .225), (.063, .034, .025))
            self._oval(self.hair, (side*.103, .72, .248), (.024, .027, .011))
            self._oval(self.hair, (side*.103, .795, .208), (.065, .012, .016))
        self.rigs = {}
        for side in ("left", "right"):
            self.rigs[side] = {
                "upper": self._segment(self.shirt, .115),
                "lower": self._segment(self.skin, .075),
                "elbow": self._oval(self.skin, (0, 0, 0), (.085,)*3),
                "joints": [self._oval(self.skin, (0, 0, 0), (.017,)*3) for _ in range(21)],
                "fingers": [self._segment(self.skin, .014) for chain in FINGERS for _ in zip(chain, chain[1:])],
                "palm": [self._segment(self.skin, .024) for _ in range(7)],
                "palm_fill": self._oval(self.skin, (0,0,0), (.06,.08,.024)),
            }
        self.set_joints({})

    def _aspect(self, *_):
        self.view.camera().lens().setPerspectiveProjection(
            38, max(self.view.width(), 1)/max(self.view.height(), 1), .1, 100)

    def _material(self, color):
        material = Extras.QPhongMaterial(self.root)
        material.setDiffuse(QColor(color))
        material.setAmbient(QColor(color).darker(150))
        material.setSpecular(QColor("#121212"))
        material.setShininess(32)
        return material

    def _mesh(self, mesh, material):
        entity = Core.QEntity(self.root)
        transform = Core.QTransform(entity)
        entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)
        self.transforms.append(transform)
        return transform

    def _oval(self, material, position, scale):
        sphere = Extras.QSphereMesh(self.root)
        sphere.setRadius(1)
        sphere.setRings(12)
        sphere.setSlices(16)
        transform = self._mesh(sphere, material)
        transform.setTranslation(QVector3D(*position))
        transform.setScale3D(QVector3D(*scale))
        return transform

    def _segment(self, material, radius):
        mesh = Extras.QCylinderMesh(self.root)
        mesh.setRadius(radius)
        mesh.setLength(1)
        mesh.setRings(2)
        mesh.setSlices(10)
        return self._mesh(mesh, material)

    @staticmethod
    def _connect(transform, start, end):
        a, b = QVector3D(*start), QVector3D(*end)
        delta = b - a
        transform.setTranslation((a+b)*.5)
        transform.setScale3D(QVector3D(1, max(delta.length(), .001), 1))
        if delta.length() > .001:
            transform.setRotation(QQuaternion.rotationTo(QVector3D(0, 1, 0), delta.normalized()))

    @staticmethod
    def _rest_hand(wrist, sign):
        points = [wrist]*21
        for chain, x, length in zip(FINGERS, (-.08, -.047, -.012, .026, .059), (.13, .23, .25, .23, .18)):
            for step, index in enumerate(chain if chain[0] != 0 else chain[1:]):
                points[index] = (wrist[0]+sign*x, wrist[1]+.045+step*length/4, wrist[2]+.008*step)
        return points

    def set_joints(self, data):
        hands = data.get("hands", {})
        shoulders = [data[side][0] for side in ("left", "right") if side in data]
        scale = math.hypot(shoulders[0][0]-shoulders[1][0], shoulders[0][1]-shoulders[1][1]) if len(shoulders) == 2 else .3
        scale = max(scale, .08)
        aspect = data.get("aspect", 9/16)
        for side, rig in self.rigs.items():
            chain = data.get(side)
            # Input video is already mirrored. Keep its screen x direction unchanged.
            sign = (-1 if chain[0][0] < sum(p[0] for p in shoulders)/2 else 1) if chain and len(shoulders) == 2 else (-1 if side == "left" else 1)
            anchor = (sign*.61, -.12, 0)
            if chain:
                shoulder = chain[0]
                def local(point):
                    return (anchor[0]+max(-1.25, min(1.25, (point[0]-shoulder[0])/scale)) ,
                            anchor[1]+max(-1.4, min(1.5, (shoulder[1]-point[1])*aspect/scale)),
                            .18+max(-.35, min(.55, (shoulder[2]-point[2])*.65)))
                elbow, wrist = local(chain[1]), local(chain[2])
            else:
                elbow, wrist = (sign*.86, -.56, .16), (sign*.86, -.9, .26)
            self._connect(rig["upper"], anchor, elbow)
            self._connect(rig["lower"], elbow, wrist)
            rig["elbow"].setTranslation(QVector3D(*elbow))
            landmarks = hands.get(side)
            if landmarks:
                root = landmarks[0]
                points = [(wrist[0]+(p[0]-root[0])/scale,
                           wrist[1]-(p[1]-root[1])*aspect/scale,
                           wrist[2]-(p[2]-root[2])*.75) for p in landmarks]
            else:
                points = self._rest_hand(wrist, sign)
            for transform, point in zip(rig["joints"], points):
                transform.setTranslation(QVector3D(*point))
            origin, base = QVector3D(*points[0]), QVector3D(*points[9])
            across = QVector3D(*points[17])-QVector3D(*points[5])
            along = base-origin
            normal = QVector3D.crossProduct(across, along).normalized()
            across = QVector3D.crossProduct(along.normalized(), normal).normalized()
            palm = rig["palm_fill"]
            palm.setTranslation(origin+along*.56)
            palm.setScale3D(QVector3D(max((QVector3D(*points[17])-QVector3D(*points[5])).length()*.62,.025), max(along.length()*.66,.035),.023))
            if normal.length() > .5:
                palm.setRotation(QQuaternion.fromAxes(across, along.normalized(), normal))
            edges = [(a,b) for chain in FINGERS for a,b in zip(chain,chain[1:])]
            for transform, (a,b) in zip(rig["fingers"], edges):
                self._connect(transform, points[a], points[b])
            for transform, (a,b) in zip(rig["palm"], ((0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17))):
                self._connect(transform, points[a], points[b])
