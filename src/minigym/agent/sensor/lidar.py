from math import cos, sin

from pygame import Vector2
from minigym.physics.geometry import LineSegment


class LidarBeam:
    geom: LineSegment
    _pos : float
    _angle: float
    _length: float

    def __init__(self, pos: Vector2, angle: float, length: float = 200):
        self.geom = LineSegment(pos, Vector2())
        self._angle = angle
        self._length = length
        self._pos = pos
        self.update_geom()

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value: Vector2):
        self._pos = value
        self.update_geom()

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value: float):
        self._angle = value
        self.update_geom()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value: float):
        self._length = value
        self.update_geom()

    def update_geom_params(self, pos: Vector2 = None, angle: float = None, length: float = None):
        if pos is not None:
            self._pos = pos

        if angle is not None:
            self._angle = angle

        if length is not None:
            self._length = length

        self.update_geom()

    def update_geom(self):
        self.geom.start = self._pos
        self.geom.end = self._pos + self._length * Vector2(cos(self._angle), sin(self._angle))
